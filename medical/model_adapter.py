"""
model_adapter.py
================
Model-agnostic adapter layer for the medical fine-tuning pipeline.

All model-specific logic (chat template formatting, output extraction,
tokenizer quirks, dtype preferences) is encapsulated here.
To support a new model, simply add an entry to MODEL_REGISTRY and, if
needed, a custom subclass of BaseModelAdapter.

Usage
-----
    adapter = get_adapter("Qwen3-8B")
    prompt  = adapter.build_prompt(system, user, include_response=True,  response="...")
    prefix  = adapter.build_prompt(system, user, include_response=False)
    answer  = adapter.extract_response(raw_generated_text)
    tok, mdl = adapter.load_model(base_model_path, lora_weights_path, lora_type, lora_r, lora_alpha)
"""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup: use the local modified PEFT that contains HypLoRA extensions
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The local modified PEFT package lives directly at <repo>/peft/
# (no src/ subdirectory), so we add the repo root to sys.path so that
# `import peft` resolves to <repo>/peft/__init__.py.
_PEFT_SRC = _REPO_ROOT
if _PEFT_SRC not in sys.path:
    sys.path.insert(0, _PEFT_SRC)

from peft import PeftModel  # noqa: E402 (after sys.path patch)


# ===========================================================================
# Base adapter
# ===========================================================================

class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Subclasses must implement:
      - ``build_prompt``   – construct the tokenizer-ready string
      - ``extract_response`` – strip special tokens from raw generation output
      - ``torch_dtype``    – preferred dtype for this model family
      - ``extra_load_kwargs`` – extra kwargs passed to from_pretrained
    """

    # Override in subclasses when the model needs special tokenizer setup
    pad_token_from_eos: bool = False

    @abstractmethod
    def build_prompt(
        self,
        system: str,
        user: str,
        include_response: bool = True,
        response: str = "",
    ) -> str:
        """
        Build the full prompt string.

        Parameters
        ----------
        system : str
            System message content.
        user : str
            User message content.
        include_response : bool
            If True, append the assistant response (used during training).
            If False, stop just before the assistant response (used during
            inference and for computing the label-mask boundary).
        response : str
            The assistant response to append when ``include_response=True``.
        """

    @abstractmethod
    def extract_response(self, raw_output: str) -> str:
        """
        Extract only the assistant reply from the full decoded sequence.

        Parameters
        ----------
        raw_output : str
            The full string returned by ``tokenizer.decode(generated_ids)``.
        """

    @property
    def torch_dtype(self) -> torch.dtype:
        """Preferred floating-point dtype for this model family."""
        return torch.bfloat16

    @property
    def extra_load_kwargs(self) -> dict:
        """Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``."""
        return {}

    # ------------------------------------------------------------------
    # Shared model loading logic (model-agnostic)
    # ------------------------------------------------------------------

    def load_tokenizer(self, base_model: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.pad_token_from_eos or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(
        self,
        base_model: str,
        lora_weights: str,
        batch_size: int = 1,
    ):
        """
        Load the base model + LoRA adapter weights.

        Returns
        -------
        tokenizer, model
        """
        tokenizer = self.load_tokenizer(base_model)
        tokenizer.padding_side = "left" if batch_size > 1 else "right"

        device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **self.extra_load_kwargs,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
        )
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        return tokenizer, model


# ===========================================================================
# Concrete adapters
# ===========================================================================

class QwenChatAdapter(BaseModelAdapter):
    """
    Adapter for the Qwen model family (Qwen2.5, Qwen3, etc.).

    Both generations share the ``<|im_start|>`` / ``<|im_end|>`` chat
    template.  Qwen3 additionally supports a *thinking mode* that wraps
    internal reasoning in ``<think>…</think>`` tags.  We disable it by
    appending ``/no_think`` to the system prompt so that the model always
    produces a clean, parseable response during both training and inference.

    To re-enable thinking for a future experiment, pass
    ``disable_thinking=False`` when constructing this adapter.
    """

    def __init__(self, disable_thinking: bool = True):
        self.disable_thinking = disable_thinking

    def _system_content(self, system: str) -> str:
        if self.disable_thinking:
            return system.rstrip() + "\n/no_think"
        return system

    def build_prompt(
        self,
        system: str,
        user: str,
        include_response: bool = True,
        response: str = "",
    ) -> str:
        sys_content = self._system_content(system)
        prompt = (
            f"<|im_start|>system\n{sys_content}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if include_response:
            prompt += f"{response}<|im_end|>\n"
        return prompt

    def extract_response(self, raw_output: str) -> str:
        marker = "<|im_start|>assistant"
        if marker in raw_output:
            text = raw_output.split(marker, 1)[1]
            # Strip leading newline added by the template
            text = text.lstrip("\n")
            text = text.split("<|im_end|>", 1)[0]
            # Strip any residual <think>…</think> block (safety net)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            return text.strip()
        return raw_output.strip()

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16


class LlamaInstructAdapter(BaseModelAdapter):
    """
    Adapter for Meta-Llama-3 Instruct models.
    """

    pad_token_from_eos = True

    def build_prompt(
        self,
        system: str,
        user: str,
        include_response: bool = True,
        response: str = "",
    ) -> str:
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if include_response:
            prompt += f"{response}<|eot_id|>"
        return prompt

    def extract_response(self, raw_output: str) -> str:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in raw_output:
            text = raw_output.split(marker, 1)[1].lstrip("\n")
            text = text.split("<|eot_id|>", 1)[0]
            return text.strip()
        return raw_output.strip()

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16


class MistralInstructAdapter(BaseModelAdapter):
    """
    Adapter for Mistral / Mixtral Instruct models.
    Uses the [INST] … [/INST] format; system prompt is prepended to the
    first user turn as recommended by Mistral.
    """

    def build_prompt(
        self,
        system: str,
        user: str,
        include_response: bool = True,
        response: str = "",
    ) -> str:
        # Mistral does not have a dedicated system role; prepend to user turn.
        combined_user = f"{system}\n\n{user}" if system else user
        prompt = f"<s>[INST] {combined_user} [/INST]"
        if include_response:
            prompt += f" {response}</s>"
        return prompt

    def extract_response(self, raw_output: str) -> str:
        marker = "[/INST]"
        if marker in raw_output:
            text = raw_output.split(marker, 1)[1]
            text = text.split("</s>", 1)[0]
            return text.strip()
        return raw_output.strip()


class GemmaChatAdapter(BaseModelAdapter):
    """
    Adapter for Google Gemma / Gemma-2 / Gemma-3 chat models.
    """

    def build_prompt(
        self,
        system: str,
        user: str,
        include_response: bool = True,
        response: str = "",
    ) -> str:
        prompt = f"<|system|>\n{system}\n<|end_of_turn|>\n"
        prompt += f"<|user|>\n{user}\n<|end_of_turn|>\n"
        prompt += "<|assistant|>\n"
        if include_response:
            prompt += f"{response}<|end_of_turn|>\n"
        return prompt

    def extract_response(self, raw_output: str) -> str:
        marker = "<|assistant|>"
        if marker in raw_output:
            text = raw_output.split(marker, 1)[1].lstrip("\n")
            text = text.split("<|end_of_turn|>", 1)[0]
            return text.strip()
        return raw_output.strip()


# ===========================================================================
# Registry
# ===========================================================================

# Maps model name patterns (matched via ``str.lower()``) to adapter instances.
# Patterns are checked in order; the first match wins.
# Add new models here without touching any other file.
MODEL_REGISTRY: list[tuple[str, BaseModelAdapter]] = [
    ("qwen3",    QwenChatAdapter(disable_thinking=True)),
    ("qwen2.5",  QwenChatAdapter(disable_thinking=False)),
    ("qwen2",    QwenChatAdapter(disable_thinking=False)),
    ("qwen",     QwenChatAdapter(disable_thinking=False)),
    ("llama-3",  LlamaInstructAdapter()),
    ("llama3",   LlamaInstructAdapter()),
    ("mistral",  MistralInstructAdapter()),
    ("mixtral",  MistralInstructAdapter()),
    ("gemma",    GemmaChatAdapter()),
]


def get_adapter(model_name_or_path: str) -> BaseModelAdapter:
    """
    Return the appropriate adapter for the given model name or path.

    Parameters
    ----------
    model_name_or_path : str
        The model identifier, e.g. ``"Qwen/Qwen3-8B"``,
        ``"meta-llama/Meta-Llama-3-8B-Instruct"``, or a local path.

    Raises
    ------
    ValueError
        If no registered adapter matches the model name.
    """
    key = model_name_or_path.lower()
    for pattern, adapter in MODEL_REGISTRY:
        if pattern in key:
            return adapter
    raise ValueError(
        f"No adapter registered for model '{model_name_or_path}'.\n"
        f"Registered patterns: {[p for p, _ in MODEL_REGISTRY]}\n"
        f"Please add an entry to MODEL_REGISTRY in medical/model_adapter.py."
    )
