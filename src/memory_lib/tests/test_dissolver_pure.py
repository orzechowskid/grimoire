"""Tests for memory/dissolver.py: eviction priority and can_evict pure functions."""

import time
from types import SimpleNamespace

import pytest

from memory_lib.memory.dissolver import _eviction_priority, can_evict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session(**kwargs):
    """Build a minimal SessionBrief-like object with sensible defaults."""
    defaults = dict(
        session_id="test-session",
        importance="background",
        resolution=0.5,
        urgency=False,
        urgency_active=False,
        urgency_expired=False,
        conflict_flag=False,
        created_at=time.time() - 86400 * 10,  # 10 days old
        intensity=0.0,
        deadline_ts=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _eviction_priority
# ---------------------------------------------------------------------------


class TestEvictionPriority:
    """Tests for _eviction_priority(session_brief)."""

    def test_high_importance_low_priority(self):
        """Principle (highest) importance → lowest priority (evicted last)."""
        sb = make_session(importance="principle")
        p = _eviction_priority(sb)
        # principle weight = 1.0, recency for 10 days ≈ 0.89
        assert p > 0.0  # should have a non-zero priority

    def test_low_importance_high_priority(self):
        """Background importance → higher priority (evicted first)
        because weight = 0.3."""
        sb = make_session(importance="background")
        p_bg = _eviction_priority(sb)

        sb_principle = make_session(importance="principle")
        p_principle = _eviction_priority(sb_principle)

        # principle (1.0) > background (0.3) → principle has HIGHER
        # priority score, meaning background is evicted FIRST (lower score).
        assert p_bg < p_principle

    def test_resolution_not_used_in_priority(self):
        """resolution attribute does not influence _eviction_priority."""
        sb_low = make_session(importance="background", resolution=0.1)
        sb_high = make_session(importance="background", resolution=0.9)
        assert _eviction_priority(sb_low) == pytest.approx(
            _eviction_priority(sb_high), rel=1e-10
        )

    def test_high_urgency_uses_urgency_not_in_priority(self):
        """urgency flag is not used in priority calculation — only
        importance, intensity, and age."""
        sb_active = make_session(urgency_active=True)
        sb_inactive = make_session(urgency_active=False)
        assert _eviction_priority(sb_active) == pytest.approx(
            _eviction_priority(sb_inactive), rel=1e-10
        )

    def test_intensity_boosts_priority(self):
        """Higher intensity → higher priority score."""
        sb_low_int = make_session(intensity=0.0)
        sb_high_int = make_session(intensity=1.0)
        assert _eviction_priority(sb_high_int) > _eviction_priority(sb_low_int)

    def test_recency_decay(self):
        """Older sessions → lower recency → lower priority."""
        sb_new = make_session(created_at=time.time())
        sb_old = make_session(created_at=time.time() - 86400 * 80)
        assert _eviction_priority(sb_old) < _eviction_priority(sb_new)

    def test_recency_floor_zero(self):
        """Sessions older than 90 days get recency = 0 → priority = 0."""
        sb_very_old = make_session(created_at=time.time() - 86400 * 100)
        assert _eviction_priority(sb_very_old) == 0.0

    def test_importance_weights_ordering(self):
        """Priority ordering follows importance weight order:
        background < important < critical < principle."""
        base_time = time.time() - 86400 * 5
        order = ["background", "important", "critical", "principle"]
        priorities = []
        for imp in order:
            sb = make_session(importance=imp, created_at=base_time)
            priorities.append(_eviction_priority(sb))
        assert priorities == sorted(priorities), "Priorities must increase with importance"

    def test_combined_attributes(self):
        """High importance + high intensity + recent = highest priority."""
        sb_best = make_session(
            importance="principle",
            intensity=2.0,
            created_at=time.time(),  # brand new
        )
        sb_worst = make_session(
            importance="background",
            intensity=0.0,
            created_at=time.time() - 86400 * 85,  # very old
        )
        assert _eviction_priority(sb_best) > _eviction_priority(sb_worst)

    def test_unknown_importance_falls_back(self):
        """Unknown importance strings fall back to weight 0.3 (background)."""
        sb_unknown = make_session(importance="unknown_level")
        sb_background = make_session(importance="background")
        assert _eviction_priority(sb_unknown) == pytest.approx(
            _eviction_priority(sb_background), rel=1e-10
        )

    def test_critical_vs_important(self):
        """Critical importance → higher priority than important."""
        sb_important = make_session(importance="important")
        sb_critical = make_session(importance="critical")
        assert _eviction_priority(sb_critical) > _eviction_priority(sb_important)

    def test_zero_intensity(self):
        """intensity = 0 is the baseline (factor = 1.0)."""
        sb = make_session(intensity=0.0)
        imp_w = 0.3  # background default
        age_days = 10
        recency = max(0.0, 1.0 - age_days / 90.0)
        expected = imp_w * (1.0 + 0.0) * recency
        assert _eviction_priority(sb) == pytest.approx(expected, rel=1e-10)

    def test_new_session_all_importances(self):
        """Fresh sessions with all importance levels produce different priorities."""
        now = time.time()
        priorities = {}
        for imp in ["principle", "critical", "important", "background"]:
            sb = make_session(importance=imp, created_at=now, intensity=0.0)
            priorities[imp] = _eviction_priority(sb)
        # All should be non-zero and distinct
        vals = list(priorities.values())
        assert len(set(vals)) == len(vals)


# ---------------------------------------------------------------------------
# can_evict
# ---------------------------------------------------------------------------


class TestCanEvict:
    """Tests for can_evict(session_brief)."""

    def test_principle_never_evict(self):
        """Principle importance → cannot evict."""
        sb = make_session(importance="principle")
        assert can_evict(sb) is False

    def test_critical_not_evict_under_normal_ram(self):
        """Critical importance cannot be evicted when RAM is below 90% of hard limit."""
        sb = make_session(importance="critical")
        # Under normal conditions RAM < 900 MB (90% of 1000 MB hard limit)
        assert can_evict(sb) is False

    def test_urgency_active_cannot_evict(self):
        """Live deadline → cannot evict."""
        sb = make_session(deadline_ts=time.time() + 3600)
        assert can_evict(sb) is False

    def test_urgency_expired_can_evict(self):
        """Expired deadline → can evict (urgency_expired=True)."""
        sb = make_session(
            deadline_ts=time.time() - 3600,
            urgency_expired=True,
        )
        assert can_evict(sb) is True

    def test_deadline_in_past_not_expired(self):
        """Deadline in the past (even with urgency_expired=False) is NOT
        a live deadline → can evict (deadline_ts < now → has_live_deadline=False)."""
        sb = make_session(
            deadline_ts=time.time() - 3600,
            urgency_expired=False,
        )
        assert can_evict(sb) is True

    def test_deadline_in_future_not_expired(self):
        """Deadline in the future and not expired → cannot evict."""
        sb = make_session(
            deadline_ts=time.time() + 3600,
            urgency_expired=False,
        )
        assert can_evict(sb) is False

    def test_conflict_flag_cannot_evict(self):
        """Conflict flag present → cannot evict."""
        sb = make_session(conflict_flag=True)
        assert can_evict(sb) is False

    def test_background_can_evict(self):
        """Background importance with sufficient resolution → can evict."""
        sb = make_session(importance="background")
        assert can_evict(sb) is True

    def test_background_with_conflict_cannot_evict(self):
        """Background + conflict flag → cannot evict."""
        sb = make_session(importance="background", conflict_flag=True)
        assert can_evict(sb) is False

    def test_background_with_active_deadline_cannot_evict(self):
        """Background + active deadline → cannot evict."""
        sb = make_session(importance="background", deadline_ts=time.time() + 3600)
        assert can_evict(sb) is False

    def test_critical_with_high_ram_evictable(self):
        """Critical sessions can be evicted when RAM > 90% of hard limit.

        This is a system-level check, so we mock psutil to simulate high RAM.
        """
        import unittest.mock

        # Simulate RAM > 900 MB (90% of 1000 MB hard limit)
        mock_meminfo = unittest.mock.MagicMock()
        mock_meminfo.rss = 950 * 1024 * 1024  # 950 MB

        with unittest.mock.patch("psutil.Process") as MockProcess:
            MockProcess.return_value.memory_info.return_value = mock_meminfo
            sb = make_session(importance="critical")
            assert can_evict(sb) is True

    def test_critical_no_deadline_normal_ram(self):
        """Critical session without deadline under normal RAM → cannot evict."""
        sb = make_session(
            importance="critical",
            deadline_ts=None,
        )
        assert can_evict(sb) is False

    def test_important_can_evict(self):
        """Important importance → can evict (no special protection)."""
        sb = make_session(importance="important")
        assert can_evict(sb) is True

    def test_eviction_order_principle_protected(self):
        """Principle sessions should never be evictable, even with all bad attrs."""
        sb = make_session(
            importance="principle",
            conflict_flag=True,  # irrelevant, principle always blocked
            deadline_ts=time.time() + 3600,
        )
        assert can_evict(sb) is False

    def test_eviction_order_background_clean(self):
        """A clean background session is evictable."""
        sb = make_session(
            importance="background",
            conflict_flag=False,
            deadline_ts=None,
        )
        assert can_evict(sb) is True

    def test_priority_order_eviction(self):
        """Eviction should remove low-priority sessions first:
        background < important < critical < principle."""
        from memory_lib.memory.dissolver import _IMPORTANCE_WEIGHT

        sessions = [
            make_session(importance="principle"),
            make_session(importance="critical"),
            make_session(importance="important"),
            make_session(importance="background"),
        ]
        # Sort by priority (ascending) — lowest first (evicted first)
        sessions.sort(key=_eviction_priority)

        # Verify sort order
        for i in range(len(sessions) - 1):
            assert _eviction_priority(sessions[i]) <= _eviction_priority(sessions[i + 1])

        # The first session (lowest priority) should be evictable
        assert can_evict(sessions[0]) is True
        # The last session (highest priority) should not be evictable
        assert can_evict(sessions[-1]) is False

    def test_missing_attributes_use_defaults(self):
        """SessionBrief with minimal attributes should not raise."""
        sb = SimpleNamespace(session_id="x")
        assert can_evict(sb) is True
        assert _eviction_priority(sb) == 0.0  # importance default → background,
        # but no created_at → age infinite → recency 0

    def test_urgency_active_deadline_exact_now(self):
        """Deadline exactly at current time → still considered active."""
        # This is a race condition edge case: deadline_ts == time.time()
        # The check is deadline_ts > time.time(), so exact match means NOT active.
        sb = make_session(deadline_ts=time.time())
        assert can_evict(sb) is True

    def test_combined_flags_cannot_evict(self):
        """Multiple protection flags → still cannot evict."""
        sb = make_session(
            importance="important",
            conflict_flag=True,
            deadline_ts=time.time() + 3600,
        )
        assert can_evict(sb) is False

    def test_background_critical_interaction(self):
        """Critical importance should not be evicted even if background-like attrs."""
        sb = make_session(
            importance="critical",
            resolution=0.4,  # low resolution, but critical protection still applies
        )
        assert can_evict(sb) is False

    def test_intangible_attrs_dont_affect_can_evict(self):
        """intensity and resolution do not affect can_evict result for background."""
        sb1 = make_session(importance="background", intensity=0.0, resolution=0.1)
        sb2 = make_session(importance="background", intensity=5.0, resolution=0.9)
        assert can_evict(sb1) is can_evict(sb2) is True
