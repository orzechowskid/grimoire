"""Tests for subconscious/anchor_index.py: AnchorIndex operations"""

import time
from unittest.mock import patch

import pytest

from memory_lib.subconscious.anchor import Anchor
from memory_lib.subconscious.anchor_index import AnchorIndex, FACT_PRIORITY, DEFAULT_MAX_CAPACITY


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_anchor(
    anchor_id: str = "test-1",
    session_id: str = "sess-1",
    brief: str = "test brief",
    anchor_type: str = "observation",
    last_accessed_at: int = 0,
    created_at: int = 0,
    key_facts: list | None = None,
    flags: dict | None = None,
) -> Anchor:
    """Create a minimal Anchor for testing."""
    now = int(time.time())
    return Anchor(
        anchor_id=anchor_id,
        session_id=session_id,
        brief=brief,
        anchor_type=anchor_type,
        key_facts=key_facts or [],
        flags=flags or {},
        last_accessed_at=last_accessed_at or now,
        created_at=created_at or now,
    )


# ── 1. get() ─────────────────────────────────────────────────────────────────

class TestGet:
    def test_lookup_by_id(self):
        idx = AnchorIndex()
        anchor = make_anchor(anchor_id="a1")
        idx.put(anchor)
        result = idx.get("a1")
        assert result is anchor

    def test_lookup_by_id_missing_returns_none(self):
        idx = AnchorIndex()
        assert idx.get("nonexistent") is None

    def test_get_returns_none_for_empty_index(self):
        idx = AnchorIndex()
        assert idx.get("anything") is None

    def test_get_does_not_update_access(self):
        """get() is a read, should not touch the anchor."""
        anchor = make_anchor(anchor_id="a1", last_accessed_at=1000)
        idx = AnchorIndex()
        idx.put(anchor)
        _ = idx.get("a1")
        assert anchor.last_accessed_at == 1000
        assert anchor.access_count == 0


# ── 2. put() ─────────────────────────────────────────────────────────────────

class TestPut:
    def test_insertion(self):
        idx = AnchorIndex()
        anchor = make_anchor(anchor_id="a1")
        evicted = idx.put(anchor)
        assert evicted is None
        assert idx.get("a1") is anchor

    def test_update_existing_anchor(self):
        """Putting an anchor that already exists should not trigger eviction."""
        idx = AnchorIndex(max_capacity=2)
        a1 = make_anchor(anchor_id="a1", last_accessed_at=100)
        a2 = make_anchor(anchor_id="a2", last_accessed_at=200)
        idx.put(a1)
        idx.put(a2)
        # Replace a1
        a1_new = make_anchor(anchor_id="a1", brief="updated", last_accessed_at=300)
        evicted = idx.put(a1_new)
        assert evicted is None  # no eviction, just replacement
        assert idx.get("a1").brief == "updated"

    def test_capacity_based_eviction(self):
        """When capacity is exceeded, oldest anchor is evicted."""
        idx = AnchorIndex(max_capacity=2)
        a1 = make_anchor(anchor_id="a1", last_accessed_at=100)
        a2 = make_anchor(anchor_id="a2", last_accessed_at=200)
        a3 = make_anchor(anchor_id="a3", last_accessed_at=300)

        idx.put(a1)
        idx.put(a2)
        evicted = idx.put(a3)

        assert evicted is not None
        assert evicted.anchor_id == "a1"  # oldest
        assert idx.get("a1") is None      # evicted
        assert idx.get("a2") is a2
        assert idx.get("a3") is a3

    def test_eviction_removes_one_per_put(self):
        idx = AnchorIndex(max_capacity=2)
        idx.put(make_anchor(anchor_id="a1", last_accessed_at=100))
        idx.put(make_anchor(anchor_id="a2", last_accessed_at=200))
        evicted = idx.put(make_anchor(anchor_id="a3", last_accessed_at=300))
        assert evicted.anchor_id == "a1"
        evicted = idx.put(make_anchor(anchor_id="a4", last_accessed_at=400))
        assert evicted.anchor_id == "a2"
        assert len(idx) == 2

    def test_put_resurfaces_via_touch(self):
        """put() simply overwrites the dict entry; it does NOT call touch()."""
        idx = AnchorIndex()
        anchor = make_anchor(anchor_id="a1", last_accessed_at=100)
        idx.put(anchor)
        anchor.last_accessed_at = 100
        anchor.access_count = 0
        idx.put(anchor)  # re-insert — no touch, just replacement
        assert anchor.access_count == 0  # touch() not called by put()


# ── 3. query_by_type() ───────────────────────────────────────────────────────

class TestQueryByType:
    def test_filter_exact_match(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1", anchor_type="decision"))
        idx.put(make_anchor(anchor_id="a2", anchor_type="observation"))
        idx.put(make_anchor(anchor_id="a3", anchor_type="decision"))
        results = idx.query_by_type("decision")
        assert len(results) == 2
        ids = {r.anchor_id for r in results}
        assert ids == {"a1", "a3"}

    def test_filter_no_match(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1", anchor_type="decision"))
        results = idx.query_by_type("milestone")
        assert results == []

    def test_filter_empty_index(self):
        idx = AnchorIndex()
        results = idx.query_by_type("any")
        assert results == []

    def test_filter_all_types(self):
        types = ["decision", "constraint", "milestone", "event", "observation"]
        idx = AnchorIndex()
        for t in types:
            idx.put(make_anchor(anchor_id=f"a-{t}", anchor_type=t))
        for t in types:
            results = idx.query_by_type(t)
            assert len(results) == 1
            assert results[0].anchor_type == t


# ── 4. query_by_flag() ───────────────────────────────────────────────────────

class TestQueryByFlag:
    def test_filter_flag_true(self):
        idx = AnchorIndex()
        a1 = make_anchor(anchor_id="a1", flags={"is_new_entity": True})
        a2 = make_anchor(anchor_id="a2", flags={"is_new_entity": False})
        idx.put(a1)
        idx.put(a2)
        results = idx.query_by_flag("is_new_entity", True)
        assert len(results) == 1
        assert results[0].anchor_id == "a1"

    def test_filter_flag_false(self):
        idx = AnchorIndex()
        a1 = make_anchor(anchor_id="a1", flags={"is_new_entity": False})
        a2 = make_anchor(anchor_id="a2", flags={"is_new_entity": True})
        idx.put(a1)
        idx.put(a2)
        results = idx.query_by_flag("is_new_entity", False)
        assert len(results) == 1
        assert results[0].anchor_id == "a1"

    def test_filter_flag_default_true(self):
        """When flag_value is omitted, defaults to True."""
        idx = AnchorIndex()
        a1 = make_anchor(anchor_id="a1", flags={"user_pin": True})
        a2 = make_anchor(anchor_id="a2", flags={"user_pin": False})
        idx.put(a1)
        idx.put(a2)
        results = idx.query_by_flag("user_pin")
        assert len(results) == 1
        assert results[0].anchor_id == "a1"

    def test_filter_missing_flag(self):
        """Anchors without the flag should not match."""
        idx = AnchorIndex()
        a1 = make_anchor(anchor_id="a1", flags={"user_pin": True})
        a2 = make_anchor(anchor_id="a2", flags={})
        idx.put(a1)
        idx.put(a2)
        results = idx.query_by_flag("user_pin", True)
        assert len(results) == 1
        assert results[0].anchor_id == "a1"

    def test_filter_no_match(self):
        idx = AnchorIndex()
        a = make_anchor(anchor_id="a1", flags={"outcome": "success"})
        idx.put(a)
        results = idx.query_by_flag("outcome", "pending")
        assert results == []

    def test_filter_empty_index(self):
        idx = AnchorIndex()
        results = idx.query_by_flag("anything")
        assert results == []


# ── 5. all() ─────────────────────────────────────────────────────────────────

class TestAll:
    def test_iteration_over_all_anchors(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1"))
        idx.put(make_anchor(anchor_id="a2"))
        idx.put(make_anchor(anchor_id="a3"))
        results = idx.all()
        assert len(results) == 3
        ids = {r.anchor_id for r in results}
        assert ids == {"a1", "a2", "a3"}

    def test_empty_index(self):
        idx = AnchorIndex()
        assert idx.all() == []

    def test_returns_copy_not_reference(self):
        """all() should return a list, not the internal dict."""
        idx = AnchorIndex()
        result = idx.all()
        assert isinstance(result, list)
        # Adding to the returned list should not affect the index
        result.append(make_anchor(anchor_id="fake"))
        assert len(idx) == 0


# ── 6. _evict_oldest() ───────────────────────────────────────────────────────

class TestEvictOldest:
    def test_oldest_first_eviction(self):
        idx = AnchorIndex(max_capacity=3)
        a1 = make_anchor(anchor_id="a1", last_accessed_at=100)
        a2 = make_anchor(anchor_id="a2", last_accessed_at=300)
        a3 = make_anchor(anchor_id="a3", last_accessed_at=200)
        idx.put(a1)
        idx.put(a2)
        idx.put(a3)
        evicted = idx._evict_oldest()
        assert evicted.anchor_id == "a1"
        assert idx.get("a1") is None

    def test_eviction_with_equal_timestamps(self):
        """When timestamps are equal, any anchor may be evicted (stable pick)."""
        idx = AnchorIndex(max_capacity=2)
        a1 = make_anchor(anchor_id="a1", last_accessed_at=500)
        a2 = make_anchor(anchor_id="a2", last_accessed_at=500)
        idx.put(a1)
        idx.put(a2)
        evicted = idx._evict_oldest()
        assert evicted.anchor_id in ("a1", "a2")
        assert len(idx) == 1

    def test_evict_removes_and_returns(self):
        idx = AnchorIndex()
        anchor = make_anchor(anchor_id="a1", last_accessed_at=100)
        idx.put(anchor)
        evicted = idx._evict_oldest()
        assert evicted is anchor
        assert len(idx) == 0

    def test_evict_empty_index(self):
        idx = AnchorIndex()
        assert idx._evict_oldest() is None


# ── 7. build_key_facts() ─────────────────────────────────────────────────────

class TestBuildKeyFacts:
    def test_facts_sorted_by_priority_then_score(self):
        entities = [
            {"type": "date", "value": "2024-01-01", "score": 0.8},
            {"type": "decision", "value": "Use Python", "score": 0.5},
            {"type": "person", "value": "Alice", "score": 0.9},
            {"type": "tech", "value": "FastAPI", "score": 0.3},
        ]
        result = AnchorIndex.build_key_facts(entities, max_facts=3)
        # decision (priority 1) comes first, then person (priority 2), then tech (priority 3)
        assert result[0]["type"] == "decision"
        assert result[1]["type"] == "person"
        assert result[2]["type"] == "tech"

    def test_same_priority_sorted_by_score_desc(self):
        entities = [
            {"type": "decision", "value": "A", "score": 0.3},
            {"type": "decision", "value": "B", "score": 0.9},
            {"type": "decision", "value": "C", "score": 0.7},
        ]
        result = AnchorIndex.build_key_facts(entities, max_facts=3)
        assert [f["value"] for f in result] == ["B", "C", "A"]

    def test_max_facts_limits_output(self):
        entities = [
            {"type": "decision", "value": f"fact-{i}", "score": 0.5}
            for i in range(10)
        ]
        result = AnchorIndex.build_key_facts(entities, max_facts=3)
        assert len(result) == 3

    def test_empty_entities(self):
        result = AnchorIndex.build_key_facts([], max_facts=5)
        assert result == []

    def test_unknown_type_gets_low_priority(self):
        """Unknown types get priority 5 (from FACT_PRIORITY.get fallback)."""
        entities = [
            {"type": "unknown", "value": "x", "score": 0.99},
            {"type": "decision", "value": "y", "score": 0.1},
        ]
        result = AnchorIndex.build_key_facts(entities, max_facts=5)
        # decision (priority 1) beats unknown (priority 5)
        assert result[0]["type"] == "decision"

    def test_all_fact_priorities(self):
        """Verify known fact types appear in correct priority order."""
        entities = [
            {"type": "date", "value": "2024-01-01", "score": 0.5},
            {"type": "decision", "value": "d", "score": 0.1},
            {"type": "person", "value": "p", "score": 0.1},
            {"type": "technology", "value": "t", "score": 0.1},
        ]
        result = AnchorIndex.build_key_facts(entities, max_facts=10)
        types = [f["type"] for f in result]
        # decision (1) < person (2) < technology (3) < date (4)
        assert types == ["decision", "person", "technology", "date"]

    def test_facts_include_score_and_priority(self):
        entities = [{"type": "decision", "value": "v", "score": 0.75}]
        result = AnchorIndex.build_key_facts(entities)
        assert result[0]["score"] == 0.75
        assert result[0]["priority"] == 1

    def test_facts_missing_score_defaults_zero(self):
        entities = [{"type": "person", "value": "v"}]
        result = AnchorIndex.build_key_facts(entities)
        assert result[0]["score"] == 0.0

    def test_facts_missing_type_defaults_to_empty(self):
        entities = [{"value": "v"}]
        result = AnchorIndex.build_key_facts(entities)
        assert result[0]["type"] == ""
        assert result[0]["priority"] == 5  # unknown type


# ── 8. infer_anchor_type() ───────────────────────────────────────────────────

class TestInferAnchorType:
    def test_decision_entity(self):
        entities = [{"type": "decision", "value": "Use Rust"}]
        assert AnchorIndex.infer_anchor_type("critical", entities) == "decision"

    def test_prohibition_entity(self):
        entities = [{"type": "prohibition", "value": "No mutable state"}]
        assert AnchorIndex.infer_anchor_type("low", entities) == "constraint"

    def test_ban_entity(self):
        entities = [{"type": "ban", "value": "No Python 2"}]
        assert AnchorIndex.infer_anchor_type("low", entities) == "constraint"

    def test_critical_importance(self):
        entities = [{"type": "person", "value": "Alice"}]
        assert AnchorIndex.infer_anchor_type("critical", entities) == "milestone"

    def test_date_entity(self):
        entities = [{"type": "date", "value": "2024-01-01"}]
        assert AnchorIndex.infer_anchor_type("low", entities) == "event"

    def test_default_observation(self):
        entities = [{"type": "person", "value": "Bob"}]
        assert AnchorIndex.infer_anchor_type("low", entities) == "observation"

    def test_decision_takes_precedence(self):
        """decision should win over critical which would be milestone."""
        entities = [{"type": "decision", "value": "d"}, {"type": "date", "value": "2024-01-01"}]
        assert AnchorIndex.infer_anchor_type("critical", entities) == "decision"

    def test_empty_entities(self):
        """With no entities but critical importance, returns milestone."""
        assert AnchorIndex.infer_anchor_type("critical", []) == "milestone"

    def test_empty_entities_no_critical(self):
        """With no entities and non-critical importance, defaults to observation."""
        assert AnchorIndex.infer_anchor_type("low", []) == "observation"

    def test_prohibition_precedes_critical(self):
        """prohibition → constraint wins over critical → milestone."""
        entities = [{"type": "prohibition", "value": "x"}]
        assert AnchorIndex.infer_anchor_type("critical", entities) == "constraint"

    def test_date_precedes_observation(self):
        """date → event wins over observation fallback."""
        entities = [{"type": "person", "value": "p"}]
        entities.append({"type": "date", "value": "2024-06-01"})
        assert AnchorIndex.infer_anchor_type("low", entities) == "event"


# ── 9. __len__ and __contains__ ──────────────────────────────────────────────

class TestDunderMethods:
    def test_len_empty(self):
        idx = AnchorIndex()
        assert len(idx) == 0

    def test_len_after_insertion(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1"))
        assert len(idx) == 1
        idx.put(make_anchor(anchor_id="a2"))
        assert len(idx) == 2

    def test_len_after_eviction(self):
        idx = AnchorIndex(max_capacity=2)
        idx.put(make_anchor(anchor_id="a1"))
        idx.put(make_anchor(anchor_id="a2"))
        idx.put(make_anchor(anchor_id="a3"))  # evicts a1
        assert len(idx) == 2

    def test_contains_existing(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1"))
        assert "a1" in idx

    def test_contains_missing(self):
        idx = AnchorIndex()
        idx.put(make_anchor(anchor_id="a1"))
        assert "a2" not in idx

    def test_contains_after_eviction(self):
        idx = AnchorIndex(max_capacity=2)
        idx.put(make_anchor(anchor_id="a1"))
        idx.put(make_anchor(anchor_id="a2"))
        idx.put(make_anchor(anchor_id="a3"))  # evicts a1
        assert "a1" not in idx
        assert "a2" in idx
        assert "a3" in idx


# ── Additional integration / edge cases ──────────────────────────────────────

class TestIntegration:
    def test_full_lifecycle(self):
        """put → get → query → all → evict lifecycle."""
        idx = AnchorIndex(max_capacity=3)
        a1 = make_anchor(anchor_id="a1", anchor_type="decision", last_accessed_at=100)
        a2 = make_anchor(anchor_id="a2", anchor_type="observation", last_accessed_at=200)
        a3 = make_anchor(anchor_id="a3", anchor_type="decision", last_accessed_at=300)

        # Insert
        idx.put(a1)
        idx.put(a2)
        idx.put(a3)
        assert len(idx) == 3

        # Query by type
        decisions = idx.query_by_type("decision")
        assert len(decisions) == 2

        # Query by flag
        idx2 = AnchorIndex()
        a = make_anchor(anchor_id="a1", flags={"user_pin": True})
        idx2.put(a)
        pinned = idx2.query_by_flag("user_pin")
        assert len(pinned) == 1

        # All
        all_anchors = idx.all()
        assert len(all_anchors) == 3

        # Evict oldest
        evicted = idx._evict_oldest()
        assert evicted.anchor_id == "a1"
        assert len(idx) == 2

        # Key facts building
        facts = AnchorIndex.build_key_facts(
            [{"type": "decision", "value": "d", "score": 0.9}], max_facts=5
        )
        assert len(facts) == 1

        # Type inference
        atype = AnchorIndex.infer_anchor_type("critical", [{"type": "decision", "value": "d"}])
        assert atype == "decision"

    def test_default_capacity(self):
        idx = AnchorIndex()
        assert idx._max_capacity == DEFAULT_MAX_CAPACITY

    def test_custom_capacity(self):
        idx = AnchorIndex(max_capacity=50)
        assert idx._max_capacity == 50

    def test_anchors_property_returns_dict(self):
        idx = AnchorIndex()
        result = idx.anchors
        assert isinstance(result, dict)

    def test_fact_priority_constants(self):
        """Verify the FACT_PRIORITY mapping exists and has expected entries."""
        assert "decision" in FACT_PRIORITY
        assert FACT_PRIORITY["decision"] == 1
        assert "person" in FACT_PRIORITY
        assert FACT_PRIORITY["person"] == 2
        assert "date" in FACT_PRIORITY
        assert FACT_PRIORITY["date"] == 4
