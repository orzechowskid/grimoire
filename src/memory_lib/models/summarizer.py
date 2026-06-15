# SPDX-License-Identifier: MIT
"""LLM summarizer wrapper using LlamaCpp (GGUF).

Provides local LLM-based summarization for generating concise memory briefs.
Used by the observer pipeline to produce topic-focused summaries instead of
action-oriented last sentences.
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class Summarizer:
    """LlamaCpp wrapper for a GGUF instruct model.

    Provides LLM-based summarization suitable for generating
    concise briefs from agent messages.
    """

    PROMPT = (
        "Summarize the following text in a concise brief (max 80 words). "
        "Preserve specific details: file names, function names, specific items "
        "discussed, concrete outcomes, and any precise values or identifiers "
        "mentioned. Avoid generic topic labels like 'we discussed X' — instead "
        "say what specifically about X was discussed. "
        "Return only the summary, nothing else.\n\n"
        "{text}\n\nSummary:"
    )

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model: Llama | None = None
        self._executor: ThreadPoolExecutor | None = None

        t0 = time.monotonic()

        # Resolve model name for logging
        model_name = Path(model_path).name

        # Load GGUF model via llama_cpp
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=2,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

        t_done = time.monotonic()
        logger.info(
            "Summarizer loaded | model=%s | total=%.1fs",
            model_name,
            t_done - t0,
        )

        # Executor for async
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-sum"
        )

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text synchronously via chat completion API."""
        messages = [{"role": "user", "content": prompt}]
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        # Extract the assistant's message content
        return response["choices"][0]["message"]["content"]

    async def summarize(self, text: str, max_tokens: int = 150) -> str:
        """Generate a summary of the given text.

        Runs inference in a thread pool to avoid blocking the event loop.
        Returns an empty string on any failure or timeout.
        """
        if self.model is None:
            return ""

        prompt = self.PROMPT.format(text=text)

        try:
            loop = asyncio.get_running_loop()
            summary = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._generate, prompt, max_tokens),
                timeout=30.0,
            )
            return summary.strip()
        except Exception as exc:
            logger.warning("Summarize failed | error=%s", exc)
            return ""

    def close(self) -> None:
        """Release the model and executor."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self.model = None  # type: ignore[assignment]
        logger.debug("Summarizer unloaded")
