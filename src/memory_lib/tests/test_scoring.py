# SPDX-License-Identifier: MIT
"""Tests for src/memory_lib/memory/scoring.py

Covers importance weights, composite scoring, urgency penalty,
principle boost, and edge cases.
"""
import time
from types import SimpleNamespace

import pytest

from memory_lib.memory.scoring import calculate_score, get_importance_weight


# ── helper ──────────────────────────────────────────────────────────────

def make_config(
    weight_background: float = 0.1,
    weight_important: float = 0.5,
    weight_critical: float = 0.9,
    weight_principle: float = 1.0,
    temporal_decay_lambda: float = 0.05,
) -> SimpleNamespace:
    """Build a nested config object matching what the scoring module expects."""
    importance = SimpleNamespace(
        weight_background=weight_background,
        weight_important=weight_important,
        weight_critical=weight_critical,
        weight_principle=weight_principle,
    )
    score = SimpleNamespace(temporal_decay_lambda=temporal_decay_lambda)
    return SimpleNamespace(importance=importance, score=score)


# ── get_importance_weight ───────────────────────────────────────────────

def test_importance_background_default():
    """background returns 0.1 by default."""
    config = make_config()
    assert get_importance_weight("background", config) == 0.1


def test_importance_important_default():
    """important returns 0.5 by default."""
    config = make_config()
    assert get_importance_weight("important", config) == 0.5


def test_importance_critical_default():
    """critical returns 0.9 by default."""
    config = make_config()
    assert get_importance_weight("critical", config) == 0.9


def test_importance_principle_default():
    """principle returns 1.0 by default."""
    config = make_config()
    assert get_importance_weight("principle", config) == 1.0


def test_importance_custom_weights():
    """Custom weights in config are respected."""
    config = make_config(
        weight_background=0.2,
        weight_important=0.6,
        weight_critical=0.95,
        weight_principle=1.0,
    )
    assert get_importance_weight("background", config) == 0.2
    assert get_importance_weight("important", config) == 0.6
    assert get_importance_weight("critical", config) == 0.95
    assert get_importance_weight("principle", config) == 1.0


def test_importance_unknown_returns_default():
    """An unrecognized importance level falls back to 0.1."""
    config = make_config()
    assert get_importance_weight("unknown_level", config) == 0.1


def test_importance_flat_config():
    """Works when config has importance attrs directly (no nesting)."""
    flat = SimpleNamespace(
        weight_background=0.3,
        weight_important=0.7,
        weight_critical=0.9,
        weight_principle=1.0,
    )
    assert get_importance_weight("background", flat) == 0.3
    assert get_importance_weight("important", flat) == 0.7


# ── calculate_score ─────────────────────────────────────────────────────

def test_score_write_profile_basic():
    """Basic write-profile scoring with relevance=1.0 and fresh item."""
    now = int(time.time())
    config = make_config()
    score = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="important",
        config=config,
        profile="write",
    )
    # r_adjusted = 1.0 * (0.7 + 0.3 * 0.5) = 0.85
    # T ≈ 1.0 (fresh item, negligible age)
    # alpha=0.5, beta=0.3, gamma=0.2, I=0.5
    # score ≈ 0.5 * 0.85 + 0.3 * 1.0 + 0.2 * 0.5 = 0.425 + 0.3 + 0.1 = 0.825
    assert 0.80 <= score <= 0.84


def test_score_search_profile():
    """Search profile uses alpha=0.6, beta=0.3, gamma=0.1."""
    now = int(time.time())
    config = make_config()
    score_search = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="important",
        config=config,
        profile="search",
    )
    score_write = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="important",
        config=config,
        profile="write",
    )
    # search puts more weight on relevance (alpha=0.6 vs 0.5), less on importance
    assert score_search > score_write


def test_score_implicit_feedback_adjustment():
    """implicit_score increases adjusted relevance."""
    now = int(time.time())
    config = make_config()
    score_low = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="important",
        config=config,
        profile="write",
        implicit_score=0.0,
    )
    score_high = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="important",
        config=config,
        profile="write",
        implicit_score=1.0,
    )
    assert score_high > score_low
    # r_adjusted for 0.0: 1.0 * 0.7 = 0.7
    # r_adjusted for 1.0: 1.0 * 1.0 = 1.0
    # diff in alpha*r_adjusted = 0.5 * 0.3 = 0.15
    assert abs((score_high - score_low) - 0.15) < 0.01


def test_score_urgency_expired_penalty():
    """urgency_expired halves the final score."""
    now = int(time.time())
    config = make_config()
    score_no_penalty = calculate_score(
        relevance=0.8,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
        urgency_expired=False,
    )
    score_with_penalty = calculate_score(
        relevance=0.8,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
        urgency_expired=True,
    )
    assert score_with_penalty == pytest.approx(score_no_penalty * 0.5, rel=1e-9)


def test_score_principle_boost():
    """principle importance boosts the final score by 30%."""
    now = int(time.time())
    config = make_config(
        weight_background=0.1,
        weight_important=0.5,
        weight_critical=0.9,
        weight_principle=1.0,
    )
    # Score a principle item without urgency penalty
    score_principle = calculate_score(
        relevance=0.7,
        created_at=now,
        importance="principle",
        config=config,
        profile="write",
        urgency_expired=False,
    )
    # Compare to a critical item (same everything else) — principle should be higher
    score_critical = calculate_score(
        relevance=0.7,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
        urgency_expired=False,
    )
    assert score_principle > score_critical
    # The boost factor should be exactly 1.3 on the pre-boost base
    # base_principle = alpha*r_adj + beta*T + gamma*I_principle
    # base_critical  = alpha*r_adj + beta*T + gamma*I_critical
    # score_principle = base_principle * 1.3
    # Verify by checking that (score_principle / 1.3) matches the pre-boost calculation
    base_principle = score_principle / 1.3
    # Reconstruct: r_adj = 0.7 * (0.7 + 0.3 * 0.5) = 0.7 * 0.85 = 0.595
    r_adj = 0.7 * (0.7 + 0.3 * 0.5)
    T = 1.0  # fresh
    I = 1.0
    expected_base = 0.5 * r_adj + 0.3 * T + 0.2 * I
    assert base_principle == pytest.approx(expected_base, rel=1e-2)


def test_score_principle_and_urgency_combined():
    """principle boost and urgency penalty stack multiplicatively."""
    # Use a timestamp slightly in the past to avoid negative age issues
    # from int() floor truncation vs time.time() float.
    config = make_config()
    ts = int(time.time()) - 1  # 1 second old
    score = calculate_score(
        relevance=0.9,
        created_at=ts,
        importance="principle",
        config=config,
        profile="write",
        urgency_expired=True,
    )
    # Reconstruct the exact base score from the returned value:
    # score = base * 1.3 * 0.5  =>  base = score / (1.3 * 0.5)
    base = score / (1.3 * 0.5)
    # r_adj = 0.9 * (0.7 + 0.3 * 0.5) = 0.9 * 0.85 = 0.765
    r_adj = 0.9 * (0.7 + 0.3 * 0.5)
    I = 1.0  # principle
    # base = 0.5 * r_adj + 0.3 * T + 0.2 * I  =>  T = (base - 0.5*r_adj - 0.2*I) / 0.3
    T = (base - 0.5 * r_adj - 0.2 * I) / 0.3
    # Verify T is a valid decay value in (0, 1]
    assert 0 < T <= 1.0


# ── temporal decay ──────────────────────────────────────────────────────

def test_temporal_decay_fresh_item():
    """Fresh items have T ≈ 1.0 (exp of ≈ 0)."""
    now = int(time.time())
    config = make_config()
    score = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="background",
        config=config,
        profile="write",
    )
    # With fresh item, T≈1, I=0.1, r_adj=0.85
    expected = 0.5 * 0.85 + 0.3 * 1.0 + 0.2 * 0.1
    assert score == pytest.approx(expected, rel=1e-2)


def test_temporal_decay_old_item():
    """Old items have reduced T, lowering the score."""
    config = make_config()
    # 1000 days old
    old_ts = int(time.time()) - 1000 * 86400
    score_old = calculate_score(
        relevance=1.0,
        created_at=old_ts,
        importance="background",
        config=config,
        profile="write",
    )
    # Fresh item for comparison
    score_fresh = calculate_score(
        relevance=1.0,
        created_at=int(time.time()),
        importance="background",
        config=config,
        profile="write",
    )
    assert score_old < score_fresh


def test_temporal_decay_custom_lambda():
    """Higher decay_lambda reduces scores faster."""
    now = int(time.time())
    old_ts = now - 30 * 86400  # 30 days
    config_fast = make_config(temporal_decay_lambda=0.2)
    config_slow = make_config(temporal_decay_lambda=0.01)
    score_fast = calculate_score(
        relevance=1.0,
        created_at=old_ts,
        importance="background",
        config=config_fast,
        profile="write",
    )
    score_slow = calculate_score(
        relevance=1.0,
        created_at=old_ts,
        importance="background",
        config=config_slow,
        profile="write",
    )
    assert score_fast < score_slow


# ── edge cases ──────────────────────────────────────────────────────────

def test_edge_zero_relevance():
    """Zero relevance should still produce a positive score from T and I."""
    now = int(time.time())
    config = make_config()
    score = calculate_score(
        relevance=0.0,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
    )
    # r_adj = 0.0 * ... = 0
    # score = 0.3 * T + 0.2 * 0.9 = 0.3 + 0.18 = 0.48
    assert score > 0
    assert score == pytest.approx(0.3 + 0.2 * 0.9, rel=1e-2)


def test_edge_max_relevance():
    """Max relevance (1.0) produces the highest score for a given profile."""
    now = int(time.time())
    config = make_config()
    score_high = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
    )
    score_low = calculate_score(
        relevance=0.5,
        created_at=now,
        importance="critical",
        config=config,
        profile="write",
    )
    assert score_high > score_low


def test_edge_negative_temporal_decay():
    """Negative decay_lambda (unusual) yields T > 1 (score boost for recent items)."""
    now = int(time.time())
    config = make_config(temporal_decay_lambda=-0.1)
    score = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="background",
        config=config,
        profile="write",
    )
    # With lambda=-0.1 and age≈0, T=exp(0)=1, so no boost at t=0
    # But with a small positive age, T > 1
    old_ts = now - 10 * 86400
    score_old = calculate_score(
        relevance=1.0,
        created_at=old_ts,
        importance="background",
        config=config,
        profile="write",
    )
    # With negative lambda, older items actually get higher T
    # T = exp(-(-0.1) * 10) = exp(1) ≈ 2.718
    assert score_old > score


def test_edge_relevance_clamped_implicit():
    """implicit_score outside [0, 1] still works (no clamping in code)."""
    now = int(time.time())
    config = make_config()
    # implicit_score > 1 increases score
    score_over = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="background",
        config=config,
        profile="write",
        implicit_score=2.0,
    )
    score_normal = calculate_score(
        relevance=1.0,
        created_at=now,
        importance="background",
        config=config,
        profile="write",
        implicit_score=1.0,
    )
    assert score_over > score_normal


def test_edge_importance_none_fallback():
    """All four importance levels produce increasing weights."""
    config = make_config()
    weights = [
        get_importance_weight("background", config),
        get_importance_weight("important", config),
        get_importance_weight("critical", config),
        get_importance_weight("principle", config),
    ]
    assert weights == sorted(weights)
    assert weights[0] < weights[1] < weights[2] <= weights[3]
