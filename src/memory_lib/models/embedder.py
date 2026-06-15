# SPDX-License-Identifier: MIT
"""Embedding model wrapper using BGE-M3 (ONNX).

Provides dense vector embeddings for text via an ONNX runtime session.
Used by the observer pipeline for semantic search and anchor matching.
"""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class Embedder:
    """ONNX wrapper for a multilingual embedding model.

    Provides highly accurate document-level embeddings suitable for
    cosine-similarity search over session memories.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        dim: int = 384,
        max_length: int = 512,
        threads: int = 2,
        intra_threads: int = 2,
        query_prefix: str = "",
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._dim = dim
        self._max_length = max_length
        self._query_prefix = query_prefix
        self.session = None
        self.tokenizer: Tokenizer | None = None
        self._loaded = False
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

        t0 = time.monotonic()

        # ONNX session
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = threads
        sess_options.intra_op_num_threads = intra_threads
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options)

        # Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        t_done = time.monotonic()
        logger.info(
            "Embedder loaded | model=%s dim=%d | total=%.1fs",
            Path(model_path).parent.name,
            dim,
            t_done - t0,
        )

        # Executor for async
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx-emb")

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str, max_length: int | None = None) -> np.ndarray:
        """Encode text → normalized float16 vector (dim,).

        Attention-masked mean pooling (B04 fix).
        """
        t0 = time.monotonic()

        if not self._loaded:
            self._load()

        # Apply model-specific instructions (e.g. "query: ")
        if self._query_prefix:
            text = self._query_prefix + text

        length = max_length or self._max_length
        with self._lock:
            self.tokenizer.enable_truncation(max_length=length)
            encoded = self.tokenizer.encode(text)

            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

            feed: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Check if model requires token_type_ids (Xenova-style E5/BERT)
            sess_input_names = [i.name for i in self.session.get_inputs()]
            if "token_type_ids" in sess_input_names:
                feed["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

            outputs = self.session.run(None, feed)

            token_embeddings = outputs[0]  # (1, seq_len, dim)

            # Attention-masked mean pooling
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            mean_pooled = (sum_embeddings / sum_mask)[0]  # (dim,)

            # MRL Truncation if needed
            if len(mean_pooled) > self._dim:
                mean_pooled = mean_pooled[: self._dim]

            # L2 normalize
            norm = np.linalg.norm(mean_pooled)
            if norm > 0:
                mean_pooled = mean_pooled / norm

            result = mean_pooled.astype(np.float16)

        logger.debug(
            "encode | tokens=%d dim=%d latency=%.0fms",
            len(encoded.ids),
            self._dim,
            (time.monotonic() - t0) * 1000,
        )

        return result

    async def aencode(self, text: str, max_length: int | None = None) -> np.ndarray:
        """Non-blocking async encode."""
        if not self._loaded:
            self._load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.encode,
            text,
            max_length,
        )

    def _load(self) -> None:
        """Load the ONNX model and tokenizer from disk."""
        if self._loaded:
            return
        # Already loaded in __init__, just mark as loaded
        self._loaded = True

    def close(self) -> None:
        """Release the ONNX session."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self.session = None
        self.tokenizer = None
        self._loaded = False
        logger.debug("Embedder unloaded")
