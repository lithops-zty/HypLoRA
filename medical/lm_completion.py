from openai import OpenAI
import httpx
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import uuid


class LMCompletion:
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 request_url: str = None,
                 timeout: int = 200,
                 log_mode: str = "file",
                 track_usage: bool = True):
        """
        Language Model Completion wrapper for OpenAI-compatible APIs.

        Args:
            model: Model name to use
            api_key: API key (defaults to OPENAI_API_KEY environment variable)
            request_url: Base URL for the API
            timeout: Request timeout in seconds
            log_mode: Logging behavior - "none", "terminal", "file", or "both"
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.request_url = request_url or os.getenv("OPENAI_BASE_URL")
        self.log_mode = log_mode.lower()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        } if track_usage else None
        self.messages = []

        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")

        self.openai_client = OpenAI(
            base_url=self.request_url,
            api_key=self.api_key,
            timeout=timeout,
            http_client=httpx.Client(
                base_url=self.request_url,
                follow_redirects=True,
            )
        )

        # Setup file logging if needed
        if self.log_mode in ["file", "both"]:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Setup file logging (独立 logger + FileHandler)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"lm_completion_{timestamp}.log"

        # 每个实例单独 logger，避免和全局 logging 混在一起
        self.logger = logging.getLogger(f"LMCompletion-{self.model}-{timestamp}")
        self.logger.setLevel(logging.INFO)

        # 避免重复加 handler（多次初始化会导致 log 重复写）
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log_to_file(self, content: str, log_type: str = "INFO"):
        """Log to file if file logging is enabled."""
        if hasattr(self, 'logger'):
            if log_type == "INFO":
                self.logger.info(content)
            elif log_type == "ERROR":
                self.logger.error(content)

    def _print_debug_info(self, action: str, model: str, content: str, uuid: Optional[str]=None):
        """Print formatted debug information to terminal with colors."""
        color_code = "31" if "prompt" in action.lower() else "32"
        print(f"""\033[{color_code}m
{uuid if uuid else '':>^100}
{action} <{model}>:
{content}
{"":<^100}\033[0m
""")

    def __call__(self,
                 data: str,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.1,
                 role="user",
                 **kwargs) -> str:
        """
        Generate completion for the given prompt.
        """
        # Generate UUID to visually match prompt and response in logs
        if role not in ['system', 'user']:
            raise ValueError("Role must be either 'system' or 'user'")

        short_uuid = str(uuid.uuid4())[:8]

        # Terminal logging
        if self.log_mode in ["terminal", "both"]:
            self._print_debug_info("PROMPT", self.model, data, short_uuid)

        # File logging
        if self.log_mode in ["file", "both"]:
            tmp = '\n'.join(f"Q| {line}" for line in data.split('\n'))
            self._log_to_file(f"""
{short_uuid:>^100}
PROMPT <{self.model}>:
{tmp}
{"":<^100}
""")
        self.messages.append({"role": role, "content": data})

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=self.messages,
                max_tokens=max_tokens,
                **kwargs
            )

            result = response.choices[0].message.content

            self.messages.append({"role": "assistant", "content": result})

            if self.usage:
                usage_info = getattr(response, 'usage', {})
                self.usage["prompt_tokens"] += getattr(usage_info, "prompt_tokens", 0)
                self.usage["completion_tokens"] += getattr(usage_info, "completion_tokens", 0)
                self.usage["total_tokens"] += getattr(usage_info, "total_tokens", 0)

            # Terminal logging
            if self.log_mode in ["terminal", "both"]:
                self._print_debug_info("RESPONSE", self.model, result, short_uuid)

            # File logging
            if self.log_mode in ["file", "both"]:
                tmp = '\n'.join(f"A| {line}" for line in result.split('\n'))
                self._log_to_file(f"""
{short_uuid:>^100}
RESPONSE <{self.model}>:
{tmp}
{"":<^100}
""")

            return result

        except Exception as e:
            error_msg = f"Error: {e}"
            if self.log_mode in ["terminal", "both"]:
                print(f"\033[31m{error_msg}\033[0m")
            if self.log_mode in ["file", "both"]:
                self._log_to_file(error_msg, "ERROR")
            raise

    def embed(self,
              texts: List[str],
              model: Optional[str] = None,
              batch_size: int = 512,
              **kwargs) -> List[List[float]]:
        """
        Call the /v1/embeddings endpoint for a list of texts.

        Texts are sent in batches to avoid exceeding API limits.
        Results are returned in the same order as the input.

        Args:
            texts:      List of strings to embed.
            model:      Embedding model name. Defaults to self.model.
            batch_size: Number of texts per API request (default 512).
            **kwargs:   Additional arguments forwarded to embeddings.create().

        Returns:
            List of embedding vectors (List[float]), one per input text.
        """
        if not texts:
            return []

        embed_model = model or self.model
        short_uuid = str(uuid.uuid4())[:8]

        if self.log_mode in ["terminal", "both"]:
            self._print_debug_info(
                "EMBED", embed_model,
                f"{len(texts)} text(s), first: {repr(texts[0][:80])}",
                short_uuid,
            )
        if self.log_mode in ["file", "both"]:
            self._log_to_file(
                f"{short_uuid:>^100}\nEMBED <{embed_model}>: {len(texts)} text(s)\n"
            )

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model=embed_model,
                    input=batch,
                    **kwargs,
                )
            except Exception as e:
                error_msg = f"Embed error (batch {i}–{i + len(batch)}): {e}"
                if self.log_mode in ["terminal", "both"]:
                    print(f"\033[31m{error_msg}\033[0m")
                if self.log_mode in ["file", "both"]:
                    self._log_to_file(error_msg, "ERROR")
                raise

            # Sort by index to guarantee order (API may reorder)
            batch_vectors = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            all_vectors.extend(batch_vectors)

            if self.usage is not None:
                usage_info = getattr(response, "usage", None)
                if usage_info is not None:
                    self.usage["prompt_tokens"] += getattr(usage_info, "prompt_tokens", 0)
                    self.usage["total_tokens"]  += getattr(usage_info, "total_tokens", 0)

        if self.log_mode in ["file", "both"]:
            dim = len(all_vectors[0]) if all_vectors else 0
            self._log_to_file(
                f"{short_uuid:>^100}\nEMBED DONE <{embed_model}>: "
                f"{len(all_vectors)} vectors, dim={dim}\n"
            )

        return all_vectors

    def get_log_file_path(self) -> Optional[str]:
        """Get the path to the current log file."""
        return getattr(self, 'log_file', None)

    def __del__(self):
        """Destructor to log usage statistics if tracking is enabled."""
        usage_summary = f"""
{'OPENAI API USAGE SUMMARY':=^100}
Model: {self.model}
Prompt Tokens: {self.usage['prompt_tokens']}
Completion Tokens: {self.usage['completion_tokens']}
Total Tokens: {self.usage['total_tokens']}
{'':=^100}
"""
        if self.usage and self.log_mode in ["file", "both"]:
            self._log_to_file(usage_summary)

        if self.usage and self.log_mode in ["terminal", "both"]:
            print(f"\033[34m{usage_summary}\033[0m")


if __name__ == '__main__':
    lm = LMCompletion(model="gpt-4o-mini")
    prompt = "Explain the theory of relativity in simple terms."
    response = lm(prompt, max_tokens=150, temperature=0.5)
    print("Final Response:", response)
    if lm.get_log_file_path():
        print("Log file saved at:", lm.get_log_file_path())

    del lm