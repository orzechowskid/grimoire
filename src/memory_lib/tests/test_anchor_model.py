"""Tests for subconscious/anchor.py: Anchor dataclass"""

import time

import numpy as np
import pytest

from memory_lib.subconscious.anchor import Anchor


class TestAnchorCreation:
    """Test Anchor creation with default values."""

    def test_create_anchor_defaults(self):
        """All fields have correct defaults when creating an Anchor."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
        )

        assert anchor.anchor_id == "sess_001"
        assert anchor.session_id == "sess_001"
        assert anchor.brief == "Test brief"
        assert anchor.anchor_type == "observation"
        assert anchor.key_facts == []
        assert anchor.decay_level == 0
        assert anchor.access_count == 0
        assert anchor.last_accessed_at == 0
        assert anchor.t_rel == {"after": [], "before": [], "caused_by": [], "during": []}
        assert anchor.created_at == 0
        assert anchor.updated_at == 0
        assert anchor.embedding is None

    def test_flags_default_values(self):
        """Default flags have the correct initial values."""
        anchor = Anchor(
            anchor_id="sess_002",
            session_id="sess_002",
            brief="Test brief",
            anchor_type="decision",
        )

        assert anchor.flags["is_new_entity"] is True
        assert anchor.flags["continuation_of"] is None
        assert anchor.flags["continuation_depth"] == 0
        assert anchor.flags["mention_type"] == "focus"
        assert anchor.flags["outcome"] == "pending"
        assert anchor.flags["user_pin"] is False
        assert anchor.flags["multi_session"] is False

    def test_create_with_custom_values(self):
        """Anchor can be created with non-default values."""
        key_facts = [{"type": "entity", "value": "Python", "score": 0.9}]
        t_rel = {"after": ["sess_000"], "before": [], "caused_by": [], "during": []}
        embedding = np.array([0.1, 0.2, 0.3])

        anchor = Anchor(
            anchor_id="sess_100",
            session_id="sess_100",
            brief="Custom brief",
            anchor_type="milestone",
            key_facts=key_facts,
            flags={"user_pin": True},
            decay_level=1,
            access_count=5,
            last_accessed_at=1700000000,
            t_rel=t_rel,
            created_at=1699999000,
            updated_at=1699999500,
            embedding=embedding,
        )

        assert anchor.key_facts == key_facts
        assert anchor.flags == {"user_pin": True}
        assert anchor.decay_level == 1
        assert anchor.access_count == 5
        assert anchor.last_accessed_at == 1700000000
        assert anchor.t_rel == t_rel
        assert anchor.created_at == 1699999000
        assert anchor.updated_at == 1699999500
        assert np.array_equal(anchor.embedding, embedding)

    def test_default_flags_is_independent_copy(self):
        """Default flags dict must not be shared between instances."""
        a1 = Anchor(
            anchor_id="a1", session_id="a1", brief="b1", anchor_type="event"
        )
        a2 = Anchor(
            anchor_id="a2", session_id="a2", brief="b2", anchor_type="event"
        )

        a1.flags["user_pin"] = True
        assert a2.flags["user_pin"] is False

    def test_default_t_rel_is_independent_copy(self):
        """Default t_rel dict must not be shared between instances."""
        a1 = Anchor(
            anchor_id="a1", session_id="a1", brief="b1", anchor_type="event"
        )
        a2 = Anchor(
            anchor_id="a2", session_id="a2", brief="b2", anchor_type="event"
        )

        a1.t_rel["after"].append("foo")
        assert a2.t_rel["after"] == []


class TestAnchorTouch:
    """Test Anchor.touch() behavior."""

    def test_touch_increments_access_count(self):
        """touch() increments access_count by 1."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
        )
        assert anchor.access_count == 0
        anchor.touch()
        assert anchor.access_count == 1

    def test_touch_updates_last_accessed_at(self):
        """touch() sets last_accessed_at to current time."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
        )
        anchor.last_accessed_at = 0

        before = int(time.time())
        anchor.touch()
        after = int(time.time())

        assert anchor.last_accessed_at >= before
        assert anchor.last_accessed_at <= after

    def test_touch_updates_updated_at(self):
        """touch() sets updated_at equal to last_accessed_at."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
        )

        anchor.touch()
        assert anchor.updated_at == anchor.last_accessed_at

    def test_touch_from_nonzero_state(self):
        """touch() works correctly when starting from non-zero state."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
            access_count=10,
            last_accessed_at=1699000000,
            updated_at=1699000000,
        )

        old_count = anchor.access_count
        old_last = anchor.last_accessed_at

        anchor.touch()

        assert anchor.access_count == old_count + 1
        assert anchor.last_accessed_at > old_last
        assert anchor.updated_at == anchor.last_accessed_at


class TestAnchorToDict:
    """Test Anchor.to_dict() serialization."""

    def test_to_dict_includes_all_fields(self):
        """to_dict() returns a dict with all expected keys."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
            key_facts=[{"type": "entity", "value": "x", "score": 0.5}],
            flags={"user_pin": True},
            decay_level=1,
            access_count=3,
            last_accessed_at=1700000000,
            t_rel={"after": ["sess_000"]},
            created_at=1699999000,
            updated_at=1699999500,
        )

        result = anchor.to_dict()

        assert set(result.keys()) == {
            "anchor_id",
            "session_id",
            "brief",
            "anchor_type",
            "key_facts",
            "flags",
            "decay_level",
            "access_count",
            "last_accessed_at",
            "t_rel",
            "created_at",
            "updated_at",
        }
        assert len(result) == 12

    def test_to_dict_excludes_embedding(self):
        """to_dict() does not include the embedding field."""
        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Test brief",
            anchor_type="observation",
            embedding=np.array([0.1, 0.2, 0.3]),
        )

        result = anchor.to_dict()
        assert "embedding" not in result

    def test_to_dict_preserves_values(self):
        """to_dict() correctly serializes all field values."""
        key_facts = [{"type": "entity", "value": "Python", "score": 0.9}]
        t_rel = {"after": ["s1"], "before": ["s2"], "caused_by": [], "during": []}

        anchor = Anchor(
            anchor_id="id1",
            session_id="sess_1",
            brief="hello",
            anchor_type="decision",
            key_facts=key_facts,
            flags={"is_new_entity": False, "user_pin": True},
            decay_level=2,
            access_count=7,
            last_accessed_at=1700100000,
            t_rel=t_rel,
            created_at=1700000000,
            updated_at=1700050000,
        )

        result = anchor.to_dict()

        assert result["anchor_id"] == "id1"
        assert result["session_id"] == "sess_1"
        assert result["brief"] == "hello"
        assert result["anchor_type"] == "decision"
        assert result["key_facts"] == key_facts
        assert result["flags"] == {"is_new_entity": False, "user_pin": True}
        assert result["decay_level"] == 2
        assert result["access_count"] == 7
        assert result["last_accessed_at"] == 1700100000
        assert result["t_rel"] == t_rel
        assert result["created_at"] == 1700000000
        assert result["updated_at"] == 1700050000


class TestAnchorFromDict:
    """Test Anchor.from_dict() deserialization."""

    def test_from_dict_full_data(self):
        """from_dict() correctly deserializes a complete data dict."""
        data = {
            "anchor_id": "sess_100",
            "session_id": "sess_100",
            "brief": "Full data test",
            "anchor_type": "milestone",
            "key_facts": [{"type": "goal", "value": "deploy", "score": 1.0}],
            "flags": {"outcome": "success", "user_pin": True},
            "decay_level": 0,
            "access_count": 42,
            "last_accessed_at": 1700200000,
            "t_rel": {"after": ["old_sess"], "before": [], "caused_by": [], "during": []},
            "created_at": 1700100000,
            "updated_at": 1700150000,
        }

        anchor = Anchor.from_dict(data)

        assert anchor.anchor_id == "sess_100"
        assert anchor.session_id == "sess_100"
        assert anchor.brief == "Full data test"
        assert anchor.anchor_type == "milestone"
        assert anchor.key_facts == [{"type": "goal", "value": "deploy", "score": 1.0}]
        assert anchor.flags == {"outcome": "success", "user_pin": True}
        assert anchor.decay_level == 0
        assert anchor.access_count == 42
        assert anchor.last_accessed_at == 1700200000
        assert anchor.t_rel == {"after": ["old_sess"], "before": [], "caused_by": [], "during": []}
        assert anchor.created_at == 1700100000
        assert anchor.updated_at == 1700150000

    def test_from_dict_missing_t_rel(self):
        """from_dict() provides default t_rel when field is missing."""
        data = {
            "anchor_id": "sess_200",
            "session_id": "sess_200",
            "brief": "Missing t_rel",
            "anchor_type": "observation",
            "created_at": 1700000000,
        }

        anchor = Anchor.from_dict(data)

        assert anchor.t_rel == {"after": [], "before": [], "caused_by": [], "during": []}

    def test_from_dict_extra_fields_ignored(self):
        """from_dict() does not crash when extra unknown fields are present."""
        data = {
            "anchor_id": "sess_300",
            "session_id": "sess_300",
            "brief": "Extra fields test",
            "anchor_type": "event",
            "unknown_field": "should not crash",
            "another_extra": 42,
        }

        # This should not raise an error — extra kwargs will be ignored
        # Note: dataclass **data will fail if extra keys are present
        # We need to check the actual behavior
        anchor = Anchor.from_dict(data)

        assert anchor.anchor_id == "sess_300"
        assert anchor.session_id == "sess_300"
        assert anchor.brief == "Extra fields test"
        assert anchor.anchor_type == "event"

    def test_from_dict_minimal_data(self):
        """from_dict() works with only the required fields."""
        data = {
            "anchor_id": "minimal",
            "session_id": "minimal",
            "brief": "min",
            "anchor_type": "observation",
        }

        anchor = Anchor.from_dict(data)

        assert anchor.anchor_id == "minimal"
        assert anchor.session_id == "minimal"
        assert anchor.brief == "min"
        assert anchor.anchor_type == "observation"
        assert anchor.key_facts == []
        assert anchor.decay_level == 0
        assert anchor.access_count == 0
        assert anchor.last_accessed_at == 0
        assert anchor.created_at == 0
        assert anchor.updated_at == 0
        assert anchor.embedding is None


class TestAnchorRoundtrip:
    """Test to_dict() / from_dict() roundtrip."""

    def test_roundtrip_preserves_all_data(self):
        """Creating an Anchor, serializing, and deserializing preserves data."""
        key_facts = [
            {"type": "entity", "value": "Python", "score": 0.95},
            {"type": "goal", "value": "test", "score": 0.8},
        ]
        flags = {"is_new_entity": False, "outcome": "success", "user_pin": True}
        t_rel = {
            "after": ["sess_prev"],
            "before": ["sess_next"],
            "caused_by": ["sess_origin"],
            "during": ["sess_overlap"],
        }

        original = Anchor(
            anchor_id="rt_001",
            session_id="sess_rt_001",
            brief="Roundtrip test brief",
            anchor_type="decision",
            key_facts=key_facts,
            flags=flags,
            decay_level=1,
            access_count=15,
            last_accessed_at=1700300000,
            t_rel=t_rel,
            created_at=1700000000,
            updated_at=1700200000,
        )

        # Serialize
        d = original.to_dict()

        # Deserialize
        restored = Anchor.from_dict(d)

        # Compare all fields
        assert restored.anchor_id == original.anchor_id
        assert restored.session_id == original.session_id
        assert restored.brief == original.brief
        assert restored.anchor_type == original.anchor_type
        assert restored.key_facts == original.key_facts
        assert restored.flags == original.flags
        assert restored.decay_level == original.decay_level
        assert restored.access_count == original.access_count
        assert restored.last_accessed_at == original.last_accessed_at
        assert restored.t_rel == original.t_rel
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_roundtrip_excludes_embedding(self):
        """Embedding is not preserved through roundtrip (intentional)."""
        original = Anchor(
            anchor_id="rt_emb",
            session_id="sess_emb",
            brief="embedding test",
            anchor_type="observation",
            embedding=np.array([1.0, 2.0, 3.0]),
        )

        d = original.to_dict()
        assert "embedding" not in d

        restored = Anchor.from_dict(d)
        assert restored.embedding is None

    def test_roundtrip_empty_values(self):
        """Roundtrip with empty/minimal values preserves them."""
        original = Anchor(
            anchor_id="rt_empty",
            session_id="sess_empty",
            brief="",
            anchor_type="observation",
        )

        d = original.to_dict()
        restored = Anchor.from_dict(d)

        assert restored.brief == ""
        assert restored.key_facts == []
        assert restored.t_rel == {"after": [], "before": [], "caused_by": [], "during": []}


class TestAnchorMultipleTouches:
    """Test multiple touch() calls increment access_count each time."""

    def test_multiple_touches_increment_access_count(self):
        """Each touch() call increments access_count by exactly 1."""
        anchor = Anchor(
            anchor_id="multi_touch",
            session_id="sess_multi",
            brief="Multiple touches",
            anchor_type="observation",
        )

        for i in range(1, 6):
            anchor.touch()
            assert anchor.access_count == i

    def test_multiple_touches_update_timestamps(self):
        """Each touch() updates last_accessed_at and updated_at."""
        anchor = Anchor(
            anchor_id="multi_ts",
            session_id="sess_multi_ts",
            brief="Timestamp touches",
            anchor_type="observation",
        )

        timestamps = []
        for _ in range(3):
            anchor.touch()
            timestamps.append(anchor.last_accessed_at)

        # Timestamps should be monotonically non-decreasing
        assert timestamps[0] <= timestamps[1] <= timestamps[2]
        assert anchor.access_count == 3
        assert anchor.updated_at == anchor.last_accessed_at

    def test_touch_does_not_mutate_other_fields(self):
        """touch() only modifies access_count, last_accessed_at, and updated_at."""
        anchor = Anchor(
            anchor_id="no_mutation",
            session_id="sess_nm",
            brief="No mutation test",
            anchor_type="decision",
            key_facts=[{"type": "x", "value": "y", "score": 0.5}],
            flags={"user_pin": True},
            decay_level=2,
            t_rel={"after": ["a"]},
        )

        original_dict = anchor.to_dict()
        original_dict.pop("last_accessed_at")
        original_dict.pop("updated_at")
        original_dict.pop("access_count")

        anchor.touch()
        current_dict = anchor.to_dict()
        current_dict.pop("last_accessed_at")
        current_dict.pop("updated_at")
        current_dict.pop("access_count")

        assert current_dict == original_dict
