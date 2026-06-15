# SPDX-License-Identifier: MIT
"""Memory marker — decides what to save from incoming text.

Replaces the simple keep/discard filter with semantic classification.
Returns an Entity, Emotion, Atmosphere, or discard signal.
"""
import logging
import re
from typing import Any

import numpy as np

from .entities import (
    Atmosphere,
    Emotion,
    EmotionCharge,
    Entity,
    EntityType,
    Explicitness,
    MarkerAction,
    MarkerResult,
    SourceType,
    TemporalMarker,
    TemporalRelations,
    TimeRef,
)

logger = logging.getLogger(__name__)


# Anchor keyword sets for classification
ANCHORS: dict[str, str] = {
    EntityType.DECISION.value: "critical decision chosen selected rejected forbidden",
    EntityType.FACT.value: "important fact requirement dependency we use",
    EntityType.CODE.value: "code function class implementation artifact",
    EntityType.EVENT.value: "event happened occurred completed failed",
    EntityType.QUESTION.value: "question unclear need to check unknown",
    EntityType.RESULT.value: "result outcome success failure done finished",
    "principle": "never always rule non-negotiable architectural principle",
    "urgency": "deadline urgent today tomorrow asap by end of day",
    "emotion": "great terrible wrong not working finally works",
}

_POSITIVE_WORDS = re.compile(
    r"\b(great|works|finally|done|success|fixed|perfect|excellent|awesome)\b",
    re.IGNORECASE,
)

_NEGATIVE_WORDS = re.compile(
    r"\b(terrible|wrong|broken|failed|error|not working|crash|bug|issue|problem)\b",
    re.IGNORECASE,
)


def infer_temporal(text: str, chain: list[Entity]) -> TemporalMarker:
    """Infer time reference from text and conversation chain position."""
    past_re = re.compile(r"\b(was|were|had|been|before|yesterday|previously)\b", re.I)
    future_re = re.compile(r"\b(will|shall|going to|tomorrow|next week|soon)\b", re.I)

    if past_re.search(text):
        return TemporalMarker(TimeRef.PAST, TimeRef.PAST, Explicitness.EXPLICIT, 1.0)
    if future_re.search(text):
        return TemporalMarker(TimeRef.FUTURE, TimeRef.FUTURE, Explicitness.EXPLICIT, 1.0)

    if chain:
        return TemporalMarker(TimeRef.PRESENT, TimeRef.PAST, Explicitness.INFERRED, 0.8)

    return TemporalMarker(TimeRef.UNKNOWN, TimeRef.UNKNOWN, Explicitness.LOST, 0.3)


def _build_t_rel(temp: TemporalMarker, chain: list[Entity]) -> TemporalRelations:
    """Infer TemporalRelations from temporal marker and chain position."""
    if not chain:
        return TemporalRelations()
    last_id = chain[-1].id
    if temp.explicitness == Explicitness.INFERRED and temp.ref_time == TimeRef.PAST:
        return TemporalRelations(after=[last_id])
    if temp.explicitness == Explicitness.EXPLICIT:
        if temp.gram_time == TimeRef.PAST:
            return TemporalRelations(after=[last_id])
        if temp.gram_time == TimeRef.FUTURE:
            return TemporalRelations(before=[last_id])
    return TemporalRelations()


def structural_prefilter(text: str) -> bool:
    """Return True if text has enough structure for semantic analysis."""
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    return bool(re.search(r"\w{2,}", stripped))


def _detect_emotion_charge(text: str) -> tuple[EmotionCharge, float]:
    """Keyword-based emotion charge detection."""
    pos_count = len(_POSITIVE_WORDS.findall(text))
    neg_count = len(_NEGATIVE_WORDS.findall(text))
    if pos_count > neg_count:
        return EmotionCharge.POSITIVE, min(1.0, 0.4 + pos_count * 0.2)
    if neg_count > pos_count:
        return EmotionCharge.NEGATIVE, min(1.0, 0.4 + neg_count * 0.2)
    if pos_count == neg_count and pos_count > 0:
        return EmotionCharge.UNCERTAIN, 0.5
    if pos_count > 0:
        return EmotionCharge.UNCERTAIN, 0.5
    return EmotionCharge.NEUTRAL, 0.2


def _keyword_classify(text: str) -> str:
    """Lightweight keyword classifier — used when embedder is unavailable."""
    keyword_map = [
        (EntityType.DECISION.value, re.compile(r"\b(decided|chosen|rejected|forbidden)\b", re.I)),
        (EntityType.CODE.value, re.compile(r"\b(def |class |function|import |async )\b", re.I)),
        (EntityType.QUESTION.value, re.compile(r"[?]|\b(why|how|what|when|where)\b", re.I)),
        (EntityType.RESULT.value, re.compile(r"\b(done|complete|finished|failed)\b", re.I)),
        ("emotion", re.compile(r"\b(great|terrible|wrong|works|finally)\b", re.I)),
    ]
    for label, pattern in keyword_map:
        if pattern.search(text):
            return label
    return EntityType.FACT.value


def _importance_from_label(label: str) -> float:
    """Default importance score by anchor label."""
    return {
        EntityType.DECISION.value: 0.9,
        "principle": 1.0,
        "urgency": 0.85,
        EntityType.FACT.value: 0.6,
        EntityType.CODE.value: 0.7,
        EntityType.EVENT.value: 0.65,
        EntityType.QUESTION.value: 0.5,
        EntityType.RESULT.value: 0.75,
    }.get(label, 0.5)


async def marker(
    text: str,
    role: SourceType,
    session_id: str,
    ctx: Any,
    chain: list[Entity] | None = None,
    pending_emotions: list[Emotion] | None = None,
    embedding: np.ndarray | None = None,
) -> MarkerResult:
    """Classify incoming text and decide what to save.

    Args:
        text: Raw input text.
        role: Who produced the text.
        session_id: Current session identifier.
        ctx: System context.
        chain: Recent Entity objects from this session.
        pending_emotions: Pending emotions awaiting binding.
        embedding: Pre-computed embedding (optional).

    Returns:
        MarkerResult with the recommended action.
    """
    chain = chain or []
    pending_emotions = pending_emotions or []

    stripped = text.strip()
    if len(stripped) < 5:
        return MarkerResult.discard()

    # User path: always creates an Entity
    if role == SourceType.USER:
        temp = infer_temporal(text, chain)
        entity = Entity.create(
            what=stripped,
            entity_type=EntityType.FACT,
            source=SourceType.USER,
            temp=temp,
            importance=0.7,
            t_rel=_build_t_rel(temp, chain),
        )
        return MarkerResult(
            action=MarkerAction.CREATE_ENTITY,
            entity=entity,
            confidence=1.0,
        )

    # Agent / tool path: structural prefilter + semantic classification
    if not structural_prefilter(stripped):
        temp = infer_temporal(text, chain)
        if temp.explicitness != Explicitness.LOST:
            atm = Atmosphere.create(signals=stripped.split()[:10], noise_level=0.8)
            return MarkerResult(
                action=MarkerAction.CREATE_ATMOSPHERE,
                atmosphere=atm,
                confidence=0.5,
            )
        return MarkerResult.discard()

    # Fallback classification
    if embedding is not None:
        embedding = np.array(embedding, dtype=np.float32).flatten()
    else:
        logger.debug("marker: no embedding available, using keyword classification")

    top_label = _keyword_classify(text)
    confidence = 0.5

    temp = infer_temporal(text, chain)

    if top_label == "emotion":
        charge, intensity = _detect_emotion_charge(text)
        emotion = Emotion.create(charge=charge, intensity=intensity, pending=True)
        return MarkerResult(
            action=MarkerAction.CREATE_EMOTION,
            emotion=emotion,
            confidence=confidence,
        )

    try:
        entity_type = EntityType(top_label)
    except ValueError:
        if top_label == "principle":
            entity_type = EntityType.DECISION
        elif top_label == "urgency":
            entity_type = EntityType.EVENT
        else:
            entity_type = EntityType.FACT

    entity = Entity.create(
        what=stripped,
        entity_type=entity_type,
        source=role,
        temp=temp,
        importance=_importance_from_label(top_label),
        embedding=embedding,
        t_rel=_build_t_rel(temp, chain),
    )

    return MarkerResult(
        action=MarkerAction.CREATE_ENTITY,
        entity=entity,
        confidence=confidence,
    )
