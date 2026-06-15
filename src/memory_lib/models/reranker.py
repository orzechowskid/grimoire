# SPDX-License-Identifier: MIT
"""Cross-encoder reranking using TinyBERT (ONNX).

Reranks Top-20 retrieved candidates into Top-5 based on precise
relevance computed by a cross-encoder model.
"""
import asyncio
import logging
import os
import threading

from concurrent.futures import ThreadPoolExecutor

import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class Reranker:
    """ONNX wrapper for TinyBERT-L2-v2 cross-encoder reranker.

    Takes a query and a list of candidate documents, returns them
    sorted by relevance score (highest first).
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.session: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None
        self._loaded = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _load(self) -> None:
        """Load the ONNX model and tokenizer from disk."""
        if self._loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Reranker ONNX not found at {self.model_path}")

        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(
                f"Reranker Tokenizer not found at {self.tokenizer_path}"
            )

        self.session = ort.InferenceSession(self.model_path)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self._loaded = True
        logger.info("Successfully loaded TinyBERT Reranker ONNX")

    def rerank(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Rerank candidates against a query.

        Returns list of (document_text, score) sorted by descending score.
        """
        if not candidates:
            return []

        if not self._loaded:
            self._load()

        scores: list[tuple[str, float]] = []
        with self._lock:
            for doc in candidates:
                # [CLS] query [SEP] doc [SEP]
                pair = f"[CLS] {query} [SEP] {doc} [SEP]"
                tokens = self.tokenizer.encode(pair)

                inputs = {
                    "input_ids": [tokens.ids],
                    "attention_mask": [tokens.attention_mask],
                    "token_type_ids": [tokens.type_ids],
                }

                # Model output is usually logits [1, 2] or regression score [1, 1]
                # Assuming standard cross-encoder softmax over binary classification
                outputs = self.session.run(None, inputs)[0][0]

                if len(outputs) > 1:
                    score = float(outputs[1])  # probability of relevant
                else:
                    score = float(outputs[0])  # regression raw score

                scores.append((doc, score))

        # Returns sorted pairs (highest score first)
        return sorted(scores, key=lambda x: -x[1])

    def close(self) -> None:
        """Release the ONNX session."""
        self.session = None
        self.tokenizer = None
        self._loaded = False
        self._executor.shutdown(wait=False)
        logger.debug("Reranker unloaded")

    async def arerank(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Async variant of rerank — runs inference on a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, candidates)
