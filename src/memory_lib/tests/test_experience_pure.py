"""Tests for memory/experience.py: maturity, decay, update, signals, serialization."""

import time
from dataclasses import fields

import pytest

from memory_lib.memory.experience import (
    ExperienceCluster,
    ExperienceIndex,
    SCORE_USE,
    SCORE_DEEP_USE,
    SCORE_CONFLICT,
    SCORE_IGNORE,
    _EMOTION_MIN_SAMPLES,
    _compute_maturity,
)


# ── _compute_maturity ──────────────────────────────────────────────────

class TestComputeMaturity:
    """_compute_maturity(session_count, thresholds) → str."""

    @pytest.mark.parametrize(
        "count,thresholds,expected",
        [
            (0, (100, 30, 10, 5), "novice"),
            (3, (100, 30, 10, 5), "novice"),
            (5, (100, 30, 10, 5), "novice"),  # 5 is not > 5 → novice
            (6, (100, 30, 10, 5), "apprentice"),
            (10, (100, 30, 10, 5), "apprentice"),  # 10 > 5 → apprentice
            (11, (100, 30, 10, 5), "practitioner"),
            (30, (100, 30, 10, 5), "practitioner"),  # 30 is not > 30 → practitioner
            (31, (100, 30, 10, 5), "expert"),
            (100, (100, 30, 10, 5), "expert"),  # 100 is not > 100 → expert
            (101, (100, 30, 10, 5), "master"),
            # Custom thresholds
            (0, (10, 5, 1, 0), "novice"),
            (1, (10, 5, 1, 0), "apprentice"),  # 1 > 0 → apprentice
            (2, (10, 5, 1, 0), "practitioner"),  # 2 > 1 → practitioner
            (6, (10, 5, 1, 0), "expert"),
            (11, (10, 5, 1, 0), "master"),
            # Edge: large numbers
            (99999, (100, 30, 10, 5), "master"),
        ],
    )
    def test_levels(self, count, thresholds, expected):
        assert _compute_maturity(count, thresholds) == expected

    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError):
            _compute_maturity(5, ())

    def test_non_tuple_raises(self):
        with pytest.raises(ValueError):
            _compute_maturity(5, [])


# ── ExperienceCluster defaults & properties ────────────────────────────

class TestExperienceClusterDefaults:
    """Default values and property calculations on ExperienceCluster."""

    def test_default_values(self):
        cluster = ExperienceCluster(tag="test")
        assert cluster.tag == "test"
        assert cluster.session_count == 0
        assert cluster.score_sum == 0.0
        assert cluster.conflict_count == 0
        assert cluster.emotion_positive == 0
        assert cluster.emotion_negative == 0
        assert cluster.emotion_intensity_sum == 0.0

    def test_defaults_are_reproducible(self):
        """Two clusters with same tag should share same defaults (except last_updated)."""
        c1 = ExperienceCluster(tag="x")
        c2 = ExperienceCluster(tag="x")
        # last_updated is based on time.time() so allow small difference
        assert abs(c1.last_updated - c2.last_updated) <= 2
        assert c1.session_count == c2.session_count
        assert c1.score_sum == c2.score_sum

    def test_default_thresholds(self):
        c = ExperienceCluster(tag="x")
        assert c._thresholds == (100, 30, 10, 5)

    # ── maturity ────────────────────────────────────────────────────────

    def test_maturity_novice(self):
        c = ExperienceCluster(tag="x", session_count=0)
        assert c.maturity == "novice"

    def test_maturity_apprentice(self):
        c = ExperienceCluster(tag="x", session_count=6)
        assert c.maturity == "apprentice"

    def test_maturity_practitioner(self):
        c = ExperienceCluster(tag="x", session_count=15)
        assert c.maturity == "practitioner"

    def test_maturity_expert(self):
        c = ExperienceCluster(tag="x", session_count=50)
        assert c.maturity == "expert"

    def test_maturity_master(self):
        c = ExperienceCluster(tag="x", session_count=200)
        assert c.maturity == "master"

    def test_custom_thresholds_affect_maturity(self):
        c = ExperienceCluster(tag="x", session_count=7, _thresholds=(10, 5, 2, 1))
        assert c.maturity == "expert"  # 7 > 5 → expert

    # ── avg_score ───────────────────────────────────────────────────────

    def test_avg_score_zero_sessions(self):
        c = ExperienceCluster(tag="x")
        assert c.avg_score == 0.0

    def test_avg_score_positive(self):
        c = ExperienceCluster(tag="x")
        c.score_sum = 10.0
        c.session_count = 2
        assert c.avg_score == 5.0

    def test_avg_score_negative(self):
        c = ExperienceCluster(tag="x")
        c.score_sum = -6.0
        c.session_count = 3
        assert c.avg_score == -2.0

    def test_avg_score_float_rounding(self):
        c = ExperienceCluster(tag="x")
        c.score_sum = 1.0
        c.session_count = 3
        assert c.avg_score == pytest.approx(1.0 / 3)

    # ── conflict_rate ──────────────────────────────────────────────────

    def test_conflict_rate_zero_sessions(self):
        c = ExperienceCluster(tag="x")
        assert c.conflict_rate == 0.0

    def test_conflict_rate_all(self):
        c = ExperienceCluster(tag="x", session_count=5, conflict_count=5)
        assert c.conflict_rate == 1.0

    def test_conflict_rate_half(self):
        c = ExperienceCluster(tag="x", session_count=4, conflict_count=2)
        assert c.conflict_rate == 0.5

    # ── emotion properties ─────────────────────────────────────────────

    def test_emotion_count(self):
        c = ExperienceCluster(tag="x", emotion_positive=3, emotion_negative=2)
        assert c.emotion_count == 5

    def test_emotion_valence_zero(self):
        c = ExperienceCluster(tag="x", emotion_positive=0, emotion_negative=0)
        assert c.emotion_valence == 0.0

    def test_emotion_valence_positive(self):
        c = ExperienceCluster(tag="x", emotion_positive=4, emotion_negative=1)
        assert c.emotion_valence == pytest.approx(0.6)

    def test_emotion_valence_negative(self):
        c = ExperienceCluster(tag="x", emotion_positive=1, emotion_negative=4)
        assert c.emotion_valence == pytest.approx(-0.6)

    def test_emotion_valence_balanced(self):
        c = ExperienceCluster(tag="x", emotion_positive=2, emotion_negative=2)
        assert c.emotion_valence == 0.0

    # ── emotion_signal ─────────────────────────────────────────────────

    def test_emotion_signal_insufficient(self):
        """Fewer than _EMOTION_MIN_SAMPLES → None."""
        c = ExperienceCluster(tag="x", emotion_positive=2, emotion_negative=0)
        assert c.emotion_signal is None

    def test_emotion_signal_attract(self):
        c = ExperienceCluster(
            tag="x",
            emotion_positive=10,
            emotion_negative=0,
        )
        assert c.emotion_signal == "ATTRACT"

    def test_emotion_signal_repel(self):
        c = ExperienceCluster(
            tag="x",
            emotion_positive=0,
            emotion_negative=10,
        )
        assert c.emotion_signal == "REPEL"

    def test_emotion_signal_ambivalent(self):
        """Ambivalent requires abs(valence) ≤ 0.3 AND total ≥ 6."""
        c = ExperienceCluster(
            tag="x",
            emotion_positive=3,
            emotion_negative=3,
        )
        assert c.emotion_signal == "AMBIVALENT"

    def test_emotion_signal_ambivalent_not_enough(self):
        """Too few samples → None even with balanced."""
        c = ExperienceCluster(
            tag="x",
            emotion_positive=1,
            emotion_negative=1,
        )
        assert c.emotion_signal is None

    def test_emotion_signal_inbetween(self):
        """Valence between 0.3 and 0.6 (or -0.6 and -0.3) → None."""
        c = ExperienceCluster(
            tag="x",
            emotion_positive=4,
            emotion_negative=2,
        )
        # valence = (4-2)/6 = 0.333...  → 0.3 < 0.333 < 0.6 → None
        assert c.emotion_signal is None

    def test_emotion_signal_attract_exact_boundary(self):
        """Valence exactly 0.6 → ATTRACT."""
        # 8 positive, 2 negative → valence = (8-2)/10 = 0.6
        c = ExperienceCluster(
            tag="x",
            emotion_positive=8,
            emotion_negative=2,
        )
        assert c.emotion_signal == "ATTRACT"

    def test_emotion_signal_repel_exact_boundary(self):
        """Valence exactly -0.6 → REPEL."""
        # 2 positive, 8 negative → valence = (2-8)/10 = -0.6
        c = ExperienceCluster(
            tag="x",
            emotion_positive=2,
            emotion_negative=8,
        )
        assert c.emotion_signal == "REPEL"


# ── ExperienceCluster.record / record_emotion / decay ──────────────────

class TestExperienceClusterRecord:
    """record(), record_emotion(), decay() on ExperienceCluster."""

    def test_record_basic(self):
        c = ExperienceCluster(tag="x")
        c.record(SCORE_USE)
        assert c.session_count == 1
        assert c.score_sum == SCORE_USE
        assert c.conflict_count == 0

    def test_record_continuation(self):
        c = ExperienceCluster(tag="x")
        c.record(SCORE_DEEP_USE)
        assert c.session_count == 1
        assert c.score_sum == SCORE_DEEP_USE

    def test_record_conflict(self):
        c = ExperienceCluster(tag="x")
        c.record(SCORE_CONFLICT, is_conflict=True)
        assert c.session_count == 1
        assert c.conflict_count == 1
        assert c.score_sum == SCORE_CONFLICT

    def test_record_multiple(self):
        c = ExperienceCluster(tag="x")
        c.record(SCORE_USE)
        c.record(SCORE_USE)
        c.record(SCORE_USE)
        assert c.session_count == 3
        assert c.score_sum == pytest.approx(3 * SCORE_USE)

    def test_record_updates_timestamp(self):
        c = ExperienceCluster(tag="x")
        before = c.last_updated
        # Small sleep not needed; we just check the property changes
        c.record(SCORE_USE)
        assert c.last_updated >= before

    def test_record_emotion_positive(self):
        c = ExperienceCluster(tag="x")
        c.record_emotion("positive", 0.8)
        assert c.emotion_positive == 1
        assert c.emotion_intensity_sum == pytest.approx(0.8)

    def test_record_emotion_negative(self):
        c = ExperienceCluster(tag="x")
        c.record_emotion("negative", 0.5)
        assert c.emotion_negative == 1
        assert c.emotion_intensity_sum == pytest.approx(0.5)

    def test_record_emotion_updates_timestamp(self):
        c = ExperienceCluster(tag="x")
        before = c.last_updated
        c.record_emotion("positive", 0.3)
        assert c.last_updated >= before

    # ── decay ──────────────────────────────────────────────────────────

    def test_decay_positive_score(self):
        c = ExperienceCluster(tag="x", score_sum=10.0, last_updated=int(time.time()) - 86400 * 100)
        c.decay(days_inactive=100, rate=0.01)
        assert c.score_sum == pytest.approx(10.0 - 0.01 * 100)  # 9.0

    def test_decay_negative_score(self):
        """Decay should not reduce the magnitude of negative scores."""
        c = ExperienceCluster(tag="x", score_sum=-10.0, last_updated=int(time.time()) - 86400 * 100)
        before = c.score_sum
        c.decay(days_inactive=100, rate=0.01)
        assert c.score_sum <= before  # score_sum can only decrease (become more negative or stay)
        # -10 - 1.0 = -11, max(-10, -11) = -10 so stays at -10
        assert c.score_sum == -10.0

    def test_decay_zero_days(self):
        c = ExperienceCluster(tag="x", score_sum=10.0)
        c.decay(days_inactive=0, rate=0.01)
        assert c.score_sum == 10.0

    def test_decay_zero_rate(self):
        c = ExperienceCluster(tag="x", score_sum=10.0, last_updated=int(time.time()) - 86400 * 100)
        c.decay(days_inactive=100, rate=0.0)
        assert c.score_sum == 10.0

    def test_decay_updates_timestamp(self):
        c = ExperienceCluster(tag="x", score_sum=10.0, last_updated=int(time.time()) - 86400 * 100)
        before = c.last_updated
        c.decay(days_inactive=100, rate=0.01)
        assert c.last_updated >= before


# ── ExperienceIndex.update ─────────────────────────────────────────────

class TestExperienceIndexUpdate:
    """update() on ExperienceIndex."""

    def test_update_single_tag(self):
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        cluster = idx.get("coding")
        assert cluster is not None
        assert cluster.session_count == 1
        assert cluster.score_sum == SCORE_USE

    def test_update_multiple_tags(self):
        idx = ExperienceIndex()
        idx.update(["coding", "design"], is_continuation=False, is_conflict=False)
        assert idx.get("coding").session_count == 1
        assert idx.get("design").session_count == 1

    def test_update_continuation(self):
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=True, is_conflict=False)
        cluster = idx.get("coding")
        assert cluster.score_sum == SCORE_DEEP_USE

    def test_update_conflict(self):
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=False, is_conflict=True)
        cluster = idx.get("coding")
        # SCORE_USE + SCORE_CONFLICT = 1.0 + (-1.5) = -0.5
        assert cluster.score_sum == pytest.approx(SCORE_USE + SCORE_CONFLICT)
        assert cluster.conflict_count == 1

    def test_update_continuation_and_conflict(self):
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=True, is_conflict=True)
        cluster = idx.get("coding")
        # SCORE_DEEP_USE + SCORE_CONFLICT = 2.5 + (-1.5) = 1.0
        assert cluster.score_sum == pytest.approx(SCORE_DEEP_USE + SCORE_CONFLICT)
        assert cluster.conflict_count == 1

    def test_update_multiple_sessions(self):
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        cluster = idx.get("coding")
        assert cluster.session_count == 3
        assert cluster.score_sum == pytest.approx(3 * SCORE_USE)

    def test_update_creates_new_clusters(self):
        idx = ExperienceIndex()
        idx.update(["a"], is_continuation=False, is_conflict=False)
        idx.update(["b"], is_continuation=False, is_conflict=False)
        idx.update(["c"], is_continuation=False, is_conflict=False)
        assert set(idx._clusters.keys()) == {"a", "b", "c"}

    def test_update_empty_tags(self):
        idx = ExperienceIndex()
        idx.update([], is_continuation=False, is_conflict=False)
        assert len(idx._clusters) == 0


# ── ExperienceIndex.update_emotion ─────────────────────────────────────

class TestExperienceIndexUpdateEmotion:
    """update_emotion() on ExperienceIndex."""

    def test_emotion_creates_cluster(self):
        idx = ExperienceIndex()
        idx.update_emotion(["design"], "positive", 0.8)
        cluster = idx.get("design")
        assert cluster is not None
        assert cluster.emotion_positive == 1

    def test_emotion_updates_existing(self):
        idx = ExperienceIndex()
        idx.update(["design"], is_continuation=False, is_conflict=False)
        idx.update_emotion(["design"], "positive", 0.8)
        cluster = idx.get("design")
        assert cluster.session_count == 1
        assert cluster.emotion_positive == 1

    def test_emotion_multiple_tags(self):
        idx = ExperienceIndex()
        idx.update_emotion(["a", "b"], "negative", 0.5)
        assert idx.get("a").emotion_negative == 1
        assert idx.get("b").emotion_negative == 1

    def test_emotion_charge_negative(self):
        idx = ExperienceIndex()
        idx.update_emotion(["x"], "negative", 0.3)
        assert idx.get("x").emotion_negative == 1


# ── ExperienceIndex.apply_decay ────────────────────────────────────────

class TestExperienceIndexApplyDecay:
    """apply_decay() on ExperienceIndex."""

    def test_decay_active_cluster(self):
        """Cluster active within threshold → not decayed."""
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        decayed = idx.apply_decay(threshold_days=90)
        assert decayed == []

    def test_decay_inactive_cluster(self):
        """Cluster older than threshold → decayed."""
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=False, is_conflict=False)
        # Manually set last_updated to 100 days ago
        old_time = int(time.time()) - 86400 * 100
        idx._clusters["coding"].last_updated = old_time

        decayed = idx.apply_decay(threshold_days=90, rate=0.01)
        assert "coding" in decayed
        assert idx._clusters["coding"].score_sum < SCORE_USE

    def test_decay_returns_list(self):
        idx = ExperienceIndex()
        idx.update(["a"], is_continuation=False, is_conflict=False)
        idx._clusters["a"].last_updated = int(time.time()) - 86400 * 100
        decayed = idx.apply_decay(threshold_days=90, rate=0.01)
        assert isinstance(decayed, list)
        assert "a" in decayed

    def test_decay_no_clusters(self):
        idx = ExperienceIndex()
        decayed = idx.apply_decay(threshold_days=90, rate=0.01)
        assert decayed == []

    def test_decay_mixed(self):
        idx = ExperienceIndex()
        idx.update(["active"], is_continuation=False, is_conflict=False)
        idx.update(["old"], is_continuation=False, is_conflict=False)
        idx._clusters["old"].last_updated = int(time.time()) - 86400 * 200

        decayed = idx.apply_decay(threshold_days=90, rate=0.01)
        assert "old" in decayed
        assert "active" not in decayed

    def test_decay_custom_rate(self):
        idx = ExperienceIndex()
        idx.update(["x"], is_continuation=False, is_conflict=False)
        idx._clusters["x"].last_updated = int(time.time()) - 86400 * 100

        idx.apply_decay(threshold_days=90, rate=0.05)  # 5% per day
        assert idx._clusters["x"].score_sum < SCORE_USE


# ── ExperienceIndex.intuition_signals ──────────────────────────────────

class TestExperienceIndexIntuitionSignals:
    """intuition_signals() on ExperienceIndex."""

    def test_empty_no_signals(self):
        idx = ExperienceIndex()
        signals = idx.intuition_signals(["coding"])
        assert signals == []

    def test_tension_signal(self):
        """High conflict rate on expert/master → TENSION."""
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=40,  # expert
            score_sum=40.0,    # avg 1.0
            conflict_count=15, # conflict_rate = 15/40 = 0.375 > 0.3
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "TENSION" for s in signals)

    def test_do_this_signal(self):
        """Expert/master with high avg → DO_THIS."""
        idx = ExperienceIndex(signal_threshold=0.75)
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=40,  # expert
            score_sum=40.0,    # avg 1.0 >= 0.75
            conflict_count=0,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "DO_THIS" for s in signals)

    def test_avoid_this_signal(self):
        """Practitioner+ with negative avg → AVOID_THIS."""
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=15,  # practitioner
            score_sum=-5.0,    # avg negative
            conflict_count=0,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "AVOID_THIS" for s in signals)

    def test_no_signal_novice(self):
        """Novice/low maturity → no signal."""
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=3,   # apprentice (< practitioner)
            score_sum=3.0,
            conflict_count=0,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        # Novice/apprentice should not generate TENSION, DO_THIS, or AVOID_THIS
        assert all(s["type"] not in ("TENSION", "DO_THIS", "AVOID_THIS") for s in signals)

    def test_emotion_attract_signal(self):
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=10,
            score_sum=10.0,
            emotion_positive=10,
            emotion_negative=0,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "ATTRACT" for s in signals)

    def test_emotion_repel_signal(self):
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=10,
            score_sum=10.0,
            emotion_positive=0,
            emotion_negative=10,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "REPEL" for s in signals)

    def test_emotion_ambivalent_signal(self):
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=10,
            score_sum=10.0,
            emotion_positive=3,
            emotion_negative=3,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert any(s["type"] == "AMBIVALENT" for s in signals)

    def test_signal_cap(self):
        """Max 5 signals."""
        idx = ExperienceIndex()
        for i in range(20):
            idx._clusters[f"tag{i}"] = ExperienceCluster(
                tag=f"tag{i}",
                session_count=40,
                score_sum=40.0,
                _thresholds=(100, 30, 10, 5),
            )
        signals = idx.intuition_signals([f"tag{i}" for i in range(20)])
        assert len(signals) <= 5

    def test_signal_has_expected_keys(self):
        idx = ExperienceIndex()
        idx._clusters["coding"] = ExperienceCluster(
            tag="coding",
            session_count=40,
            score_sum=40.0,
            _thresholds=(100, 30, 10, 5),
        )
        signals = idx.intuition_signals(["coding"])
        assert signals
        for s in signals:
            assert "type" in s
            assert "tag" in s
            assert "message" in s

    def test_no_signal_unknown_tag(self):
        idx = ExperienceIndex()
        signals = idx.intuition_signals(["unknown_tag"])
        assert signals == []


# ── ExperienceIndex serialization ──────────────────────────────────────

class TestExperienceIndexSerialization:
    """to_dict() / from_dict() on ExperienceIndex."""

    def test_cluster_to_dict(self):
        c = ExperienceCluster(
            tag="coding",
            session_count=5,
            score_sum=5.0,
            conflict_count=1,
            emotion_positive=3,
            emotion_negative=1,
        )
        d = c.to_dict()
        assert d["tag"] == "coding"
        assert d["session_count"] == 5
        assert d["score_sum"] == pytest.approx(5.0)
        assert d["conflict_count"] == 1
        assert d["maturity"] == "novice"  # 5 sessions → apprentice boundary, 5>5 is false, so novice
        assert d["avg_score"] == pytest.approx(1.0)
        assert "emotion_positive" in d
        assert "emotion_negative" in d
        assert "emotion_valence" in d
        assert "emotion_signal" in d

    def test_cluster_to_dict_rounding(self):
        c = ExperienceCluster(
            tag="x",
            session_count=3,
            score_sum=1.0,
        )
        d = c.to_dict()
        # avg_score = 1/3 = 0.3333..., rounded to 4 decimals
        assert d["avg_score"] == pytest.approx(0.3333)

    def test_cluster_to_dict_no_emotion(self):
        c = ExperienceCluster(tag="x")
        d = c.to_dict()
        assert d["emotion_positive"] == 0
        assert d["emotion_negative"] == 0
        assert d["emotion_valence"] == 0.0
        assert d["emotion_signal"] is None

    def test_experience_index_load(self):
        idx = ExperienceIndex()
        rows = [
            {
                "tag": "coding",
                "session_count": 10,
                "score_sum": 10.0,
                "conflict_count": 1,
                "emotion_positive": 5,
                "emotion_negative": 2,
            },
            {
                "tag": "design",
                "session_count": 3,
                "score_sum": 3.0,
                "conflict_count": 0,
                "emotion_positive": 0,
                "emotion_negative": 0,
            },
        ]
        idx.load(rows)
        assert idx.get("coding").session_count == 10
        assert idx.get("design").session_count == 3

    def test_experience_index_load_defaults(self):
        idx = ExperienceIndex()
        idx.load([{"tag": "x"}])
        c = idx.get("x")
        assert c is not None
        assert c.session_count == 0
        assert c.score_sum == 0.0
        assert c.conflict_count == 0

    def test_experience_index_all_clusters(self):
        idx = ExperienceIndex()
        idx.update(["a", "b", "c"], is_continuation=False, is_conflict=False)
        clusters = idx.all_clusters()
        assert len(clusters) == 3
        assert {c.tag for c in clusters} == {"a", "b", "c"}

    def test_experience_index_get_none(self):
        idx = ExperienceIndex()
        assert idx.get("nonexistent") is None

    def test_full_roundtrip(self):
        """Create index → load from dicts → verify."""
        idx = ExperienceIndex()
        idx.update(["coding"], is_continuation=True, is_conflict=False)
        idx.update(["design"], is_continuation=False, is_conflict=True)
        idx.update_emotion(["coding"], "positive", 0.8)

        # Get clusters as dicts
        cluster_dicts = [c.to_dict() for c in idx.all_clusters()]

        # Create new index and load
        idx2 = ExperienceIndex()
        # Convert to DB-row format (load expects this format)
        rows = []
        for cd in cluster_dicts:
            rows.append({
                "tag": cd["tag"],
                "session_count": cd["session_count"],
                "score_sum": cd["score_sum"],
                "conflict_count": cd["conflict_count"],
                "emotion_positive": cd["emotion_positive"],
                "emotion_negative": cd["emotion_negative"],
                "emotion_intensity_sum": cd["emotion_intensity_sum"],
            })
        idx2.load(rows)

        for tag in ["coding", "design"]:
            orig = idx.get(tag)
            loaded = idx2.get(tag)
            assert loaded is not None
            assert loaded.session_count == orig.session_count
            assert loaded.score_sum == pytest.approx(orig.score_sum, abs=0.001)
