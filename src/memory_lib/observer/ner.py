# SPDX-License-Identifier: MIT
"""NER extraction using an ONNX-based Named Entity Recognition model.

Provides a thin async wrapper around the model's entity extraction
capability, integrated into the observer pipeline.
"""
import logging
from typing import Any

from ..models.ner import NER as ModelNER
from .entities import EntityType

logger = logging.getLogger(__name__)


class NERExtractor:
    """NER extraction step for the observer pipeline.

    Wraps the underlying ONNX NER model and provides an async
    interface for the pipeline to call.
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self.model = ModelNER(model_path, tokenizer_path)
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            try:
                self.model._load()  # noqa: SLF001
                self._loaded = True
                logger.info("NERExtractor loaded")
            except Exception as e:
                logger.warning("NERExtractor failed to load: %s", e)
                # Mark as "loaded" but the model will fail gracefully
                self._loaded = True

    async def extract(self, text: str, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Extract entities from text.

        Returns list of dicts with keys: type, value, score.
        """
        await self._ensure_loaded()
        try:
            return await self.model.extract_entities(text, threshold=threshold)
        except Exception as e:
            logger.error("NER extraction failed: %s", e)
            return []

    def unload(self) -> None:
        """Release the ONNX session."""
        self.model.close()
        self._loaded = False
        logger.debug("NERExtractor unloaded")
