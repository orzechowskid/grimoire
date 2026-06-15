# SPDX-License-Identifier: MIT
"""Unit tests for the Dreamer idle-reassessment worker."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_lib.subconscious.anchor import Anchor
from memory_lib.subconscious.dreamer import Dreamer


# --- Fixtures ---

def _make_anchor(
    anchor_id: str = "a1",
    session_id: str = "s1",
    anchor_type: str = "entity",
    outcome: str = "pending",
    decay_level: int = 0,
    access_count: int = 1,
    continuation_of: str | None = None,
) -> Anchor:
    """Helper to create an Anchor with test flags."""
    flags = {"outcome": outcome}
    if continuation_of:
        flags["continuation_of"] = continuation_of
    return Anchor(
        anchor_id=anchor_id,
        session_id=session_id,
        anchor_type=anchor_type,
        brief="test anchor",
        key_facts=[],
        flags=flags,
        decay_level=decay_level,
        access_count=access_count,
        last_accessed_at=1000,
        t_rel={"after": [], "before": [], "caused_by": [], "during": []},
        created_at=1000,
        updated_at=1000,
    )


def _make_config(overrides: dict | None = None) -> MagicMock:
    """Helper to create a mock config with DreamerSettings defaults."""
    config = MagicMock()
    config.dreamer.enabled = True
    config.dreamer.interval_min = 15
    config.dreamer.max_anchors_per_cycle = 20
    config.dreamer.resurface_threshold = 3
    config.dreamer.disk_scan_enabled = True
    config.dreamer.disk_scan_page_size = 50
    if overrides:
        for k, v in overrides.items():
            setattr(config.dreamer, k, v)
    return config


# --- Tests ---

def test_reassess_outcome_resolves_pending_to_success():
    """Pending anchor + milestone continuation → outcome becomes 'success'."""
    anchor_index = MagicMock()
    pending = _make_anchor(anchor_id="pending1", outcome="pending")
    milestone = _make_anchor(
        anchor_id="ms1", anchor_type="milestone", outcome="success",
        continuation_of="pending1"
    )
    anchor_index.all.return_value = [pending, milestone]
    anchor_index._anchors = {"pending1": pending, "ms1": milestone}

    config = _make_config()
    persistence = MagicMock()
    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )

    result = dreamer._reassess_outcome(pending)
    assert result is True
    assert pending.flags["outcome"] == "success"


def test_reassess_outcome_resolves_pending_to_failure():
    """Pending anchor + failure continuation → outcome propagates failure."""
    anchor_index = MagicMock()
    pending = _make_anchor(anchor_id="pending1", outcome="pending")
    follower = _make_anchor(
        anchor_id="f1", outcome="failure",
        continuation_of="pending1"
    )
    anchor_index.all.return_value = [pending, follower]
    anchor_index._anchors = {"pending1": pending, "f1": follower}

    config = _make_config()
    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=MagicMock(), config=config
    )

    result = dreamer._reassess_outcome(pending)
    assert result is True
    assert pending.flags["outcome"] == "failure"


def test_reassess_outcome_skips_already_resolved():
    """Anchor with outcome != 'pending' → _reassess_outcome returns False (no-op)."""
    anchor_index = MagicMock()
    resolved = _make_anchor(anchor_id="r1", outcome="success")
    anchor_index.all.return_value = [resolved]
    anchor_index._anchors = {"r1": resolved}

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=MagicMock(), config=_make_config()
    )

    result = dreamer._reassess_outcome(resolved)
    assert result is False
    assert resolved.flags["outcome"] == "success"


def test_reassess_outcome_no_continuation():
    """Pending anchor with no continuations → returns False."""
    anchor_index = MagicMock()
    pending = _make_anchor(anchor_id="pending1", outcome="pending")
    anchor_index.all.return_value = [pending]
    anchor_index._anchors = {"pending1": pending}

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=MagicMock(), config=_make_config()
    )

    result = dreamer._reassess_outcome(pending)
    assert result is False
    assert pending.flags["outcome"] == "pending"


@pytest.mark.asyncio
async def test_resurface_high_access_anchor():
    """Anchor with access_count >= threshold AND decay_level < 2 → touch() called."""
    anchor_index = MagicMock()
    hot_anchor = _make_anchor(
        anchor_id="hot1", outcome="success",
        decay_level=0, access_count=5
    )
    anchor_index.all.return_value = [hot_anchor]
    anchor_index._anchors = {"hot1": hot_anchor}
    # Make anchor_index pass the guard check (len > 0)
    anchor_index.__len__ = MagicMock(return_value=1)
    anchor_index.__bool__ = MagicMock(return_value=True)

    persistence = AsyncMock()
    config = _make_config(
        overrides={"resurface_threshold": 3, "disk_scan_enabled": False}
    )

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )

    # Patch Anchor.touch to track calls
    with patch.object(Anchor, 'touch') as mock_touch:
        stats = await dreamer.dream()

    # hot_anchor has access_count=5 >= 3 AND decay_level=0 < 2 → should be resurfaced
    # touch was called on hot_anchor
    mock_touch.assert_called()
    assert stats["resurfaced"] >= 1


@pytest.mark.asyncio
async def test_dream_cycle_returns_stats():
    """Full dream cycle returns properly-shaped stats dict."""
    anchor_index = MagicMock()
    anchor = _make_anchor(
        anchor_id="a1", outcome="success",
        decay_level=0, access_count=1
    )
    anchor_index.all.return_value = [anchor]
    anchor_index._anchors = {"a1": anchor}
    # Make anchor_index pass the guard check (len > 0)
    anchor_index.__len__ = MagicMock(return_value=1)
    anchor_index.__bool__ = MagicMock(return_value=True)

    persistence = AsyncMock()
    config = _make_config(
        overrides={
            "max_anchors_per_cycle": 10,
            "resurface_threshold": 3,
            "disk_scan_enabled": False,
        }
    )

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )
    stats = await dreamer.dream()

    assert "anchors_checked" in stats
    assert "outcomes_updated" in stats
    assert "resurfaced" in stats
    assert "disk_anchors_checked" in stats
    assert "disk_outcomes_updated" in stats
    assert "duration_ms" in stats
    assert stats["anchors_checked"] == 1


@pytest.mark.asyncio
async def test_dreamer_start_stop():
    """Verify _running flag and _task lifecycle."""
    anchor_index = MagicMock()
    anchor_index.all.return_value = []
    anchor_index._anchors = {}
    config = _make_config(overrides={"interval_min": 1})
    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=AsyncMock(), config=config
    )

    await dreamer.start()
    assert dreamer._running is True
    assert dreamer._task is not None

    await dreamer.stop()
    assert dreamer._running is False
    # Task is cancelled but not cleared to None by stop()
    assert dreamer._task is not None
    assert dreamer._task.cancelled()


@pytest.mark.asyncio
async def test_disk_scan_resolves_pending_anchor():
    """Mock persistence returning a pending anchor with a resolver in RAM → disk scan resolves it."""
    anchor_index = MagicMock()
    resolver_milestone = _make_anchor(
        anchor_id="ms1", anchor_type="milestone", outcome="success"
    )
    anchor_index.all.return_value = [resolver_milestone]
    anchor_index._anchors = {"ms1": resolver_milestone}

    # The dreamer looks for anchors where OTHER anchors have continuation_of = disk_anchor's id
    # Set up: resolver_milestone has continuation_of = "disk1"
    resolver_milestone.flags["continuation_of"] = "disk1"

    persistence = AsyncMock()
    disk_anchor = _make_anchor(
        anchor_id="disk1", outcome="pending",
        continuation_of=None
    )
    persistence.find_anchors_by_flags.return_value = [disk_anchor]

    config = _make_config(
        overrides={
            "disk_scan_page_size": 50,
            "disk_scan_enabled": True,
            "max_anchors_per_cycle": 10,
            "resurface_threshold": 3,
        }
    )

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )

    stats = await dreamer._disk_scan()

    assert stats["disk_anchors_checked"] >= 1
    assert stats["disk_outcomes_updated"] >= 1
    assert disk_anchor.flags["outcome"] == "success"


@pytest.mark.asyncio
async def test_disk_scan_resets_offset_on_empty_page():
    """Two sequential empty pages with pass_resolved > 0 → offset advances each time.
    
    The code only resets offset when pass_resolved == 0 AND window < 6000.
    When pass_resolved > 0, empty pages just advance the offset.
    """
    anchor_index = MagicMock()
    anchor_index.all.return_value = []
    anchor_index._anchors = {}

    persistence = AsyncMock()
    persistence.find_anchors_by_flags.return_value = []

    config = _make_config(overrides={"disk_scan_page_size": 10})

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )
    dreamer._disk_offset = 20
    dreamer._disk_pass_resolved = 5

    stats = await dreamer._disk_scan()

    # First empty page advances offset (pass_resolved > 0 → else branch)
    assert dreamer._disk_offset == 30
    # Second empty page also advances offset
    stats2 = await dreamer._disk_scan()
    assert dreamer._disk_offset == 40


@pytest.mark.asyncio
async def test_disk_scan_window_expansion():
    """Full pass with 0 resolutions → window expands x3."""
    anchor_index = MagicMock()
    anchor_index.all.return_value = []
    anchor_index._anchors = {}

    persistence = AsyncMock()
    persistence.find_anchors_by_flags.return_value = []

    config = _make_config(overrides={"disk_scan_page_size": 50})

    dreamer = Dreamer(
        ram_index={}, anchor_index=anchor_index,
        persistence=persistence, config=config
    )
    dreamer._disk_pass_resolved = 0
    dreamer._disk_window = 1000

    # Simulate scanning through enough pages to complete a full pass with 0 resolutions
    # The window starts at 1000, page_size is 50, so 20 pages = 1000
    # We need to simulate the pass completing with 0 resolutions
    # Simplified: manually trigger the expansion condition
    dreamer._disk_offset = dreamer._disk_window  # Simulate reaching end of window

    stats = await dreamer._disk_scan()

    # Should expand window when pass_resolved == 0 and window < 6000
    assert dreamer._disk_window == 3000
    assert dreamer._disk_offset == 0
