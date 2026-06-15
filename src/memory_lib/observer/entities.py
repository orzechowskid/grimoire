# SPDX-License-Identifier: MIT
"""Memory Model v2 dataclasses — Entity, Emotion, Atmosphere.

Provides the core data types used by the observer pipeline for
tagging and classifying observed content.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Categories of entities that the observer can tag."""

    DECISION = "decision"
    FACT = "fact"
    CODE = "code"
    EVENT = "event"
    QUESTION = "question"
    RESULT = "result"


class SourceType(str, Enum):
    """Who produced the observed text."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


class ResultType(str, Enum):
    """Outcome of an action."""

    SUCCESS = "success"
    FAIL = "fail"
    PENDING = "pending"
    NONE = "none"


class TimeRef(str, Enum):
    """Temporal reference point."""

    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    UNKNOWN = "unknown"


class Explicitness(str, Enum):
    """How the temporal reference was determined."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    LOST = "lost"


class EmotionCharge(str, Enum):
    """Direction of an emotional signal."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class MarkerAction(str, Enum):
    """Action the marker() function recommends."""

    CREATE_ENTITY = "create_entity"
    CREATE_EMOTION = "create_emotion"
    CREATE_ATMOSPHERE = "create_atmosphere"
    DISCARD = "discard"


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


@dataclass
class TemporalMarker:
    """Time reference attached to every Entity."""

    gram_time: TimeRef = TimeRef.UNKNOWN
    ref_time: TimeRef = TimeRef.UNKNOWN
    explicitness: Explicitness = Explicitness.LOST
    confidence: float = 0.3

    @classmethod
    def unknown(cls) -> TemporalMarker:
        return cls(
            TimeRef.UNKNOWN,
            TimeRef.UNKNOWN,
            Explicitness.LOST,
            0.3,
        )


@dataclass
class TemporalRelations:
    """Directed links to other Entity IDs (uuid4 strings)."""

    after: list[str] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    caused_by: list[str] = field(default_factory=list)
    during: list[str] = field(default_factory=list)

    def all_ids(self) -> list[str]:
        return self.after + self.before + self.caused_by + self.during

    def is_empty(self) -> bool:
        return not any([self.after, self.before, self.caused_by, self.during])


# ---------------------------------------------------------------------------
# Core objects
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """Primary memory unit — anchors a fact, decision, code, event, etc."""

    id: str  # uuid4
    what: str  # content text
    type: EntityType
    source: SourceType
    t_abs: int  # unix ms
    temp: TemporalMarker

    t_rel: TemporalRelations = field(default_factory=TemporalRelations)

    result: ResultType | None = None
    atmosphere: str | None = None
    importance: float = 0.5  # 0.0–1.0
    embedding: np.ndarray | None = field(default=None, repr=False)

    last_accessed: int = field(
        default_factory=lambda: int(time.time() * 1000)
    )

    @classmethod
    def create(
        cls,
        what: str,
        entity_type: EntityType,
        source: SourceType,
        temp: TemporalMarker | None = None,
        **kwargs,
    ) -> Entity:
        now_ms = int(time.time() * 1000)
        return cls(
            id=str(uuid.uuid4()),
            what=what,
            type=entity_type,
            source=source,
            t_abs=now_ms,
            temp=temp or TemporalMarker.unknown(),
            last_accessed=now_ms,
            **kwargs,
        )


@dataclass
class Emotion:
    """User emotional signal — always attached to an Entity."""

    id: str  # uuid4
    charge: EmotionCharge
    intensity: float  # 0.0–1.0
    t_abs: int  # unix ms

    ref_entity_id: str | None = None
    ref_source: SourceType | None = None
    pending: bool = False

    @classmethod
    def create(
        cls,
        charge: EmotionCharge,
        intensity: float,
        ref_entity_id: str | None = None,
        ref_source: SourceType | None = None,
        pending: bool = False,
    ) -> Emotion:
        return cls(
            id=str(uuid.uuid4()),
            charge=charge,
            intensity=intensity,
            t_abs=int(time.time() * 1000),
            ref_entity_id=ref_entity_id,
            ref_source=ref_source,
            pending=pending,
        )


@dataclass
class Atmosphere:
    """Context placeholder — co-occurring signals without a primary entity yet."""

    entity_id: str | None = None
    signals: list[str] = field(default_factory=list)
    noise_level: float = 0.5
    pending: bool = True
    t_abs: int = 0

    @classmethod
    def create(cls, signals: list[str], noise_level: float = 0.5) -> Atmosphere:
        return cls(
            signals=signals,
            noise_level=noise_level,
            pending=True,
            t_abs=int(time.time() * 1000),
        )


# ---------------------------------------------------------------------------
# Marker result
# ---------------------------------------------------------------------------


@dataclass
class MarkerResult:
    """Return value of the marker() function."""

    action: MarkerAction
    entity: Entity | None = None
    emotion: Emotion | None = None
    atmosphere: Atmosphere | None = None
    confidence: float = 1.0

    @classmethod
    def discard(cls) -> MarkerResult:
        return cls(action=MarkerAction.DISCARD, confidence=1.0)
