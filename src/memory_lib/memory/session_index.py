# SPDX-License-Identifier: MIT
"""Session brief data model for the working memory index.

SessionBrief represents the compressed form of one agent session:
a brief summary, tags, importance level, composite score, resolution,
and vector embedding. Managed by the memory subsystem.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SessionBrief:
    """Single session in RAM memory index.

    Represents the compressed form of one agent session: brief summary,
    tags, importance level, score, and dissolution resolution.

    Attributes:
        session_id: Unique session identifier.
        brief: Compressed summary (configurable, default max 500 chars).
        tags: Semantic tags extracted by Observer, max 7.
        importance: Level — background / important / critical / principle.
        score: Ranking score = α×R + β×T + γ×I.
        resolution: Dissolution level 0.05–1.0, managed by Dissolver.
        created_at: Creation timestamp (UTC, unix seconds).
        use_count: Frequency of retrieval in active context.
        conflict_flag: Indicates semantic dissonance with other sessions.
        urgency: Level — none / deadline_h / deadline_d / deadline_w.
        deadline_ts: Optional deadline timestamp.
        urgency_expired: Whether the deadline has passed.
        layer: Current memory layer — RAM_HOT, RAM_WARM, etc.
        embedding: Vector representation (dim × 384 or similar).
        implicit_score: Feedback-driven quality score (0.0–1.0).
        intensity: Emotion intensity linked to this session (0.0 = none).
    """

    session_id: str
    brief: str
    tags: list[str]
    importance: str
    score: float
    resolution: float
    created_at: int

    # Optional / computed fields
    use_count: int = 0
    deep_use_count: int = 0
    last_use_ts: int | None = None
    conflict_flag: bool = False
    urgency: str = "none"
    deadline_ts: int | None = None
    urgency_active: bool = False
    urgency_expired: bool = False
    bare_entity: bool = False
    embedding_model_version: str = "multilingual-e5-small"

    layer: str = "RAM_HOT"
    embedding: np.ndarray | None = None
    implicit_score: float = 0.5
    intensity: float = 0.0

    # Content Branch classification (set by Observer PersistStep)
    session_type: str | None = None
    age_signal: str = "fresh"
