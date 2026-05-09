"""
finetune_medical.py
===================
Model-agnostic SFT fine-tuning script for the MIMIC-IV lab-value prediction
task using HypLoRA (or vanilla LoRA as a baseline).

Data format expected
--------------------
Each line of the JSONL file must be a JSON object with a ``messages`` key:

    {
        "messages": [
            {"role": "system",    "content": "<patient info + discharge summary>"},
            {"role": "user",      "content": "<time-ordered clinical events with [PREDICTION_TARGET]>"},
            {"role": "assistant", "content": "{\"lab\": [v1, v2, ...]}"}
        ]
    }

Usage
-----
    python finetune_medical.py \\
        --base_model  "Qwen/Qwen3-8B" \\
        --train_data_path  /path/to/train.jsonl \\
        --val_data_path    /path/to/val.jsonl \\
        --output_dir  ./trained_models/qwen3-8b-hyplora-medical \\
        --lora_type   "hyplora-0.5" \\
        --lora_r 32 --lora_alpha 128 \\
        --batch_size 16 --micro_batch_size 2 \\
        --num_epochs 3 --learning_rate 3e-4 \\
        --cutoff_len 2048
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the local modified PEFT (with HypLoRA) is on the path.
# model_adapter.py already does this, but we repeat it here so the script
# can also be run standalone from the medical/ directory.
# ---------------------------------------------------------------------------
_MEDICAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_MEDICAL_DIR)
# The local modified PEFT package lives directly at <repo>/peft/
_PEFT_SRC = _REPO_ROOT
if _PEFT_SRC not in sys.path:
    sys.path.insert(0, _PEFT_SRC)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

from model_adapter import get_adapter  # noqa


# ===========================================================================
# Data loading
# ===========================================================================

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_messages(record: dict) -> tuple[str, str, str]:
    """
    Extract (system, user, assistant) strings from a messages-format record.

    Raises
    ------
    ValueError
        If the record does not contain exactly the expected roles.
    """
    msgs = record["messages"]
    role_map = {m["role"]: m["content"] for m in msgs}
    system    = role_map.get("system", "")
    user      = role_map.get("user", "")
    assistant = role_map.get("assistant", "")
    return system, user, assistant


# ===========================================================================
# Training entry point
# ===========================================================================

def train(
    # ---- required ----
    base_model: str,
    train_data_path: str,
    output_dir: str,
    # ---- data ----
    val_data_path: Optional[str] = None,
    # ---- LoRA / HypLoRA ----
    lora_r: int = 32,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_type: str = "std",          # "std" for vanilla LoRA; "hyplora-K" for HypLoRA
    target_modules: Optional[List[str]] = None,
    use_dora: bool = False,
    # ---- training ----
    batch_size: int = 16,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    train_on_inputs: bool = False,   # False → only compute loss on assistant tokens
    use_gradient_checkpointing: bool = False,
    group_by_length: bool = False,
    eval_steps: int = 200,
    save_steps: int = 200,
    resume_from_checkpoint: Optional[str] = None,
    # ---- logging ----
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
):
    """
    Fine-tune a causal LM on the MIMIC-IV lab-value prediction task.

    All model-specific logic (chat template, dtype, tokenizer quirks) is
    delegated to the adapter returned by ``get_adapter(base_model)``.
    """
    # ------------------------------------------------------------------
    # Resolve adapter (model-agnostic interface)
    # ------------------------------------------------------------------
    adapter = get_adapter(base_model)

    print(
        f"\n{'='*60}\n"
        f"Fine-tuning: {base_model}\n"
        f"Adapter:     {adapter.__class__.__name__}\n"
        f"lora_type:   {lora_type}\n"
        f"lora_r:      {lora_r}  lora_alpha: {lora_alpha}\n"
        f"cutoff_len:  {cutoff_len}\n"
        f"epochs:      {num_epochs}  lr: {learning_rate}\n"
        f"output_dir:  {output_dir}\n"
        f"{'='*60}\n"
    )

    # ------------------------------------------------------------------
    # Distributed training setup
    # ------------------------------------------------------------------
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = bool(wandb_project) or bool(os.environ.get("WANDB_PROJECT", ""))
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # ------------------------------------------------------------------
    # Load tokenizer and base model
    # ------------------------------------------------------------------
    tokenizer = adapter.load_tokenizer(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=adapter.torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        **adapter.extra_load_kwargs,
    )

    # ------------------------------------------------------------------
    # Apply LoRA / HypLoRA
    # ------------------------------------------------------------------
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        use_dora=use_dora,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        lora_type=lora_type,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------
    use_bf16 = (adapter.torch_dtype == torch.bfloat16)

    def tokenize(text: str, add_eos_token: bool = True) -> dict:
        result = tokenizer(
            text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            add_eos_token
            and len(result["input_ids"]) < cutoff_len
            and result["input_ids"][-1] != tokenizer.eos_token_id
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def process_record(record: dict) -> dict:
        """
        Convert a messages-format record into tokenized training features.

        When ``train_on_inputs=False`` (default), the labels for the
        system+user prefix are masked to -100 so that the loss is computed
        only on the assistant response tokens.

        The mask boundary is determined by tokenizing the *prefix-only*
        prompt (``include_response=False``), which is fully model-agnostic
        because the adapter's ``build_prompt`` handles all template details.
        """
        system, user, assistant = parse_messages(record)

        full_prompt   = adapter.build_prompt(system, user, include_response=True,  response=assistant)
        prefix_prompt = adapter.build_prompt(system, user, include_response=False)

        tokenized_full   = tokenize(full_prompt,   add_eos_token=True)
        tokenized_prefix = tokenize(prefix_prompt, add_eos_token=False)

        if not train_on_inputs:
            prefix_len = len(tokenized_prefix["input_ids"])
            tokenized_full["labels"] = (
                [-100] * prefix_len
                + tokenized_full["labels"][prefix_len:]
            )

        return tokenized_full

    # ------------------------------------------------------------------
    # Build HuggingFace Dataset objects
    # ------------------------------------------------------------------
    print("Processing training data …")
    train_records = load_jsonl(train_data_path)
    train_features = [process_record(r) for r in tqdm(train_records, desc="Tokenising train")]
    train_dataset  = Dataset.from_list(train_features)

    val_dataset = None
    if val_data_path:
        print("Processing validation data …")
        val_records  = load_jsonl(val_data_path)
        val_features = [process_record(r) for r in tqdm(val_records, desc="Tokenising val")]
        val_dataset  = Dataset.from_list(val_features)

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel    = True

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=(not use_bf16),
        logging_steps=10,
        optim="adamw_torch",
        eval_strategy="steps" if val_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_dataset is not None else None,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=3,
        label_names=["labels"],
        load_best_model_at_end=(val_dataset is not None),
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name if use_wandb else None,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if torch.cuda.is_available():
        print(f"GPU memory allocated : {torch.cuda.memory_allocated()  / 1e9:.2f} GB")
        print(f"GPU memory reserved  : {torch.cuda.memory_reserved()   / 1e9:.2f} GB")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
