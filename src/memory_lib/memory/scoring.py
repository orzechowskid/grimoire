# SPDX-License-Identifier: MIT
"""Importance scoring for memory items.

Computes composite ranking scores from relevance, temporal decay,
importance weight, and implicit feedback signals.
"""
import time
from typing import Any

import numpy as np


def get_importance_weight(importance: str, config: Any) -> float:
    """Get the importance weight from configuration (γ signal)."""
    importance_settings = getattr(config, "importance", config)
    imp_map = {
        "background": getattr(importance_settings, "weight_background", 0.1),
        "important": getattr(importance_settings, "weight_important", 0.5),
        "critical": getattr(importance_settings, "weight_critical", 0.9),
        "principle": getattr(importance_settings, "weight_principle", 1.0),
    }
    return imp_map.get(importance, 0.1)


def calculate_score(
    relevance: float,
    created_at: int,
    importance: str,
    config: Any,
    profile: str = "write",
    urgency_expired: bool = False,
    implicit_score: float = 0.5,
) -> float:
    """Calculate ranking score using profile-based weights.

    Applies implicit feedback adjustment:
        R_adjusted = R × (0.7 + 0.3 × implicit_score)

    Profile A (Write): α=0.5, β=0.3, γ=0.2
    Profile B (Search): α=0.6, β=0.3, γ=0.1

    Args:
        relevance: Raw relevance score (0.0–1.0).
        created_at: Session creation Unix timestamp.
        importance: Level — background / important / critical / principle.
        config: Configuration object with score weights.
        profile: Scoring profile — "write" or "search".
        urgency_expired: Apply 50% penalty if deadline has passed.
        implicit_score: EMA-adjusted feedback score (0.0–1.0).

    Returns:
        Final composite score.
    """
    # Temporal decay
    age_days = (time.time() - created_at) / 86400
    score_settings = getattr(config, "score", config)
    decay_lambda = getattr(score_settings, "temporal_decay_lambda", 0.05)
    T = float(np.exp(-decay_lambda * age_days))

    # Importance weights (γ signal)
    I = get_importance_weight(importance, config)

    # Adjust relevance by implicit feedback
    r_adjusted = relevance * (0.7 + 0.3 * implicit_score)

    if profile == "search":
        alpha, beta, gamma = 0.6, 0.3, 0.1
    else:
        alpha, beta, gamma = 0.5, 0.3, 0.2

    score = alpha * r_adjusted + beta * T + gamma * I

    # Policy modifiers
    if urgency_expired:
        score *= 0.50  # URGENCY_EXPIRED_PENALTY
    if importance == "principle":
        score *= 1.30  # PRINCIPLE_BOOST

    return score
