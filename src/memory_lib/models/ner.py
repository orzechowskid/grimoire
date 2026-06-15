# SPDX-License-Identifier: MIT
"""Named entity recognition using a BERT/DistilBERT ONNX model.

Extracts typed entities (PER, ORG, LOC, DATE, technology, decision,
prohibition, outcome) from raw text without requiring PyTorch or
the transformers library.
"""
import asyncio
import logging
import re
import threading
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Entity priority mapping (lower = higher priority)
ENTITY_PRIORITY: dict[str, int] = {
    "decision": 1,
    "prohibition": 1,
    "outcome": 1,
    "technology": 2,
    "PER": 3,
    "ORG": 3,
    "LOC": 3,
    "DATE": 4,
    "MONEY": 4,
}

# Technology names (language-independent)
TECH_PATTERN = re.compile(
    r"\b(Python|Rust|Go|Java|JavaScript|TypeScript|C\+\+|Ruby|PHP|Swift|Kotlin|"
    r"PostgreSQL|MySQL|MongoDB|Redis|SQLite|Cassandra|ClickHouse|"
    r"Docker|Kubernetes|Nginx|Apache|Linux|Windows|"
    r"React|Vue|Angular|FastAPI|Django|Flask|Spring|"
    r"ONNX|TensorFlow|PyTorch|LangChain|"
    r"Git|GitHub|GitLab|Jira|Slack|Telegram|"
    r"AWS|GCP|Azure|Vercel|Heroku)\b",
    re.IGNORECASE,
)

OUTCOME_PATTERNS: list[re.Pattern[str]] = [
    # EN: success/failure (flexible)
    re.compile(
        r"\b(?:successfully|works|worked|fixed|solved)\b(?:\s+[a-z]{1,30})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:failed|broken|crashes|doesn't work)\b(?:\s+[a-z]{1,30})?",
        re.IGNORECASE,
    ),
]

DECISION_PATTERNS: list[re.Pattern[str]] = [
    # EN
    re.compile(
        r"(?:decided to|chose|will use|switching to|agreed on|approved|"
        r"going with|selected|picked)[:\s]+(.{3,60}?)(?=[.,;!\n]|$)",
        re.IGNORECASE,
    ),
]

PROHIBITION_PATTERNS: list[re.Pattern[str]] = [
    # EN
    re.compile(
        r"(?:forbidden|must not|do not use|banned|prohibited|"
        r"deprecated|removed|blocked|no external)[:\s]+(.{3,60}?)(?=[.,;!\n]|$)",
        re.IGNORECASE,
    ),
]


class NER:
    """ONNX-based NER extractor.

    Combines BERT-based token classification (PER, ORG, LOC, DATE)
    with regex patterns (technology, decision, prohibition, outcome).

    Adheres to the No-Torch rule: runs entirely on ONNX Runtime
    with a tokenizers tokenizer. Optimized for resource-constrained
    environments (~700 MB budget).
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.session: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None
        self._loaded = False
        self._lock = threading.Lock()

        self._id2label: dict[str, str] = {
            "0": "O",
            "1": "B-DATE",
            "2": "I-DATE",
            "3": "B-PER",
            "4": "I-PER",
            "5": "B-ORG",
            "6": "I-ORG",
            "7": "B-LOC",
            "8": "I-LOC",
        }
        # Mapping from BIO tag type to output type name
        self._label_map: dict[str, str] = {
            "DATE": "DATE",
            "PER": "PER",
            "ORG": "ORG",
            "LOC": "LOC",
        }

    def _load(self) -> None:
        """Load the ONNX model and tokenizer from disk."""
        if self._loaded:
            return

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False

        self.session = ort.InferenceSession(self.model_path, opts)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self._loaded = True
        logger.info("NER loaded model: %s", self.model_path)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Standard Softmax over logits."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _predict_entities(self, text: str, threshold: float = 0.5) -> list[dict[str, Any]]:
        """BERT-based token classification for PER, ORG, LOC, DATE."""
        if self.session is None or self.tokenizer is None:
            self._load()

        with self._lock:
            encoded = self.tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

            feed: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Check if model requires token_type_ids
            sess_input_names = [i.name for i in self.session.get_inputs()]
            if "token_type_ids" in sess_input_names:
                feed["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

            outputs = self.session.run(None, feed)
            logits = outputs[0][0]  # [seq_len, num_labels]

            # Softmax + top-1
            probs = self._softmax(logits)
            predictions = np.argmax(probs, axis=-1)
            scores = np.max(probs, axis=-1)

            # Span reconstruction (BIO to spans)
            entities: list[dict[str, Any]] = []
            current_entity: dict[str, Any] | None = None

            for i, (pred_id, score) in enumerate(zip(predictions, scores)):
                # Skip special tokens [CLS], [SEP]
                if i == 0 or i == len(encoded.ids) - 1:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    continue

                label = self._id2label.get(str(pred_id), "O")

                if label == "O":
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    continue

                bio, ent_type = label.split("-")
                mapped_type = self._label_map.get(ent_type, ent_type)
                start, end = encoded.offsets[i]
                is_subword = encoded.tokens[i].startswith("##")

                if bio == "B" and not is_subword:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "type": mapped_type,
                        "value": text[start:end],
                        "score": float(score),
                        "start": int(start),
                        "end": int(end),
                    }
                elif bio == "I" and current_entity and current_entity["type"] == mapped_type:
                    current_entity["value"] = text[current_entity["start"] : end]
                    current_entity["end"] = int(end)
                    current_entity["score"] = min(current_entity["score"], float(score))
                elif bio == "I" and is_subword and current_entity:
                    current_entity["value"] = text[current_entity["start"] : end]
                    current_entity["end"] = int(end)
                    current_entity["score"] = min(current_entity["score"], float(score))
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            # Post-processing: filter by threshold and clean
            result: list[dict[str, Any]] = []
            for e in entities:
                if e["score"] < threshold:
                    continue
                e["value"] = e["value"].strip()
                if len(e["value"]) <= 1:
                    continue
                result.append(e)
            return result

    async def extract_entities(
        self, text: str, threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Extract entities from text.

        Returns a list of dicts with keys: type, value, score, start, end.
        """
        if not self._loaded:
            self._load()

        entities: list[dict[str, Any]] = []

        # 1. Model-based NER (PER, ORG, LOC, DATE) via executor
        try:
            loop = asyncio.get_running_loop()
            model_entities = await loop.run_in_executor(
                None, self._predict_entities, text, threshold
            )
            entities.extend(model_entities)
        except Exception as e:
            logger.error("BertNER prediction failed: %s", e)

        # 2. Regex: technologies
        for m in TECH_PATTERN.finditer(text):
            if not self._overlaps(entities, m.start(), m.end(), "technology"):
                entities.append(
                    {
                        "type": "technology",
                        "value": m.group(0),
                        "score": 1.0,
                        "start": m.start(),
                        "end": m.end(),
                    }
                )

        # 3. Regex: decisions
        for pattern in DECISION_PATTERNS:
            for m in pattern.finditer(text):
                val = m.group(1).strip()
                full_start = m.start(1)
                if not self._overlaps(entities, full_start, full_start + len(val), "decision"):
                    entities.append(
                        {
                            "type": "decision",
                            "value": val,
                            "score": 0.9,
                            "start": full_start,
                            "end": full_start + len(val),
                        }
                    )

        # 4. Regex: prohibitions
        for pattern in PROHIBITION_PATTERNS:
            for m in pattern.finditer(text):
                val = m.group(1).strip()
                full_start = m.start(1)
                if not self._overlaps(
                    entities, full_start, full_start + len(val), "prohibition"
                ):
                    entities.append(
                        {
                            "type": "prohibition",
                            "value": val,
                            "score": 0.9,
                            "start": full_start,
                            "end": full_start + len(val),
                        }
                    )

        # 5. Regex: outcomes
        for pattern in OUTCOME_PATTERNS:
            for m in pattern.finditer(text):
                val = m.group(0).strip()
                full_start = m.start()
                if not self._overlaps(entities, full_start, full_start + len(val), "outcome"):
                    entities.append(
                        {
                            "type": "outcome",
                            "value": val,
                            "score": 0.85,
                            "start": full_start,
                            "end": full_start + len(val),
                        }
                    )

        # Sort by position
        entities.sort(key=lambda e: e["start"])
        return entities

    @staticmethod
    def _overlaps(
        entities: list[dict[str, Any]], start: int, end: int, candidate_type: str
    ) -> bool:
        """Check if span overlaps with existing entities considering priority."""
        for e in entities:
            if start < e["end"] and end > e["start"]:
                prio_candidate = ENTITY_PRIORITY.get(candidate_type, 99)
                prio_existing = ENTITY_PRIORITY.get(e.get("type", ""), 99)
                if prio_candidate < prio_existing:
                    # Higher priority — allow coexistence
                    return False
                return True
        return False

    def close(self) -> None:
        """Release the ONNX session to free ~200 MB."""
        self.session = None
        self.tokenizer = None
        self._loaded = False
        logger.debug("NER unloaded (ONNX session released)")
