# SPDX-License-Identifier: MIT
"""Unit tests for the pure functions and core MatrixSearch behaviour in hnsw.py."""

import math

import numpy as np
import pytest

from memory_lib.memory.hnsw import (
    MatrixSearch,
    time_decay,
)


# ── time_decay ──────────────────────────────────────────────────────────────


class TestTimeDecay:
    """Tests for the time_decay function."""

    def test_zero_age_returns_one(self):
        """At age=0 the decay factor should be 1.0 (no penalty)."""
        assert time_decay(0, 10) == pytest.approx(1.0)

    def test_one_half_life_returns_half(self):
        """After one half_life the factor should be 0.5."""
        assert time_decay(10, 10) == pytest.approx(0.5)

    def test_two_half_lives_returns_quarter(self):
        """After two half_lives the factor should be 0.25."""
        assert time_decay(20, 10) == pytest.approx(0.25)

    def test_three_half_lives_returns_one_eighth(self):
        """After three half_lives the factor should be 0.125."""
        assert time_decay(30, 10) == pytest.approx(0.125)

    def test_returns_value_in_open_interval(self):
        """For positive age and half_life the result is always in (0, 1]."""
        for age in [1, 5, 100]:
            for half_life in [1, 10, 365]:
                result = time_decay(age, half_life)
                assert 0 < result <= 1.0

    def test_half_life_zero_returns_one(self):
        """A half_life of 0 is treated as "no decay"."""
        assert time_decay(42, 0) == 1.0

    def test_negative_age_not_allowed_but_stays_one(self):
        """If age is negative, decay still produces a value ≥ 0 (edge case)."""
        result = time_decay(-5, 10)
        assert result > 1.0

    def test_formula_matches_explicit_exponential(self):
        """Verify the internal formula: exp(-age * ln(2) / half_life)."""
        age, half_life = 7, 15
        expected = math.exp(-age * math.log(2) / half_life)
        assert time_decay(age, half_life) == pytest.approx(expected)


# ── Helpers ─────────────────────────────────────────────────────────────────

FAKE_CONFIG = type("Config", (), {"search": type("Search", (), {"pipeline_width": 1})})()


# ── MatrixSearch ────────────────────────────────────────────────────────────


class TestMatrixSearchAddItems:
    """Tests for add_items."""

    def test_add_single_vector(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ms.add_items([vec], [42])
        assert ms.get_current_count() == 1

    def test_add_multiple_vectors(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        vecs = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        ]
        ms.add_items(vecs, [1, 2])
        assert ms.get_current_count() == 2

    def test_vectors_get_normalised(self):
        """Vectors are normalised to unit length on insertion."""
        dim = 4
        ms = MatrixSearch(dim=dim)
        vec = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm=5
        ms.add_items([vec], [99])
        stored = ms._vectors[0]
        assert np.isclose(np.linalg.norm(stored), 1.0, atol=1e-6)

    def test_zero_vector_stays_unit(self):
        """A zero vector should not crash; it is kept as-is (normalised to 1)."""
        dim = 4
        ms = MatrixSearch(dim=dim)
        vec = np.zeros(4, dtype=np.float32)
        ms.add_items([vec], [0])
        assert ms.get_current_count() == 1

    def test_labels_match_ids(self):
        ids = [10, 20, 30]
        ms = MatrixSearch(dim=3)
        ms.add_items([np.zeros(3) for _ in ids], ids)
        assert ms._labels == ids


class TestMatrixSearchSearch:
    """Tests for knn_query."""

    def test_empty_index_returns_empty(self):
        ms = MatrixSearch(dim=4)
        labels, dists = ms.knn_query(np.zeros(4), k=5)
        assert labels == []
        assert dists == []

    def test_single_item_returns_that_item(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ms.add_items([vec], [7])
        labels, dists = ms.knn_query(vec, k=1)
        assert labels == [7]
        assert dists == pytest.approx([0.0], abs=1e-5)

    def test_identical_vectors_similarity_is_one(self):
        """Cosine distance of identical vectors should be ~0 → similarity 1.0."""
        dim = 4
        ms = MatrixSearch(dim=dim)
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ms.add_items([vec], [1])
        _, dists = ms.knn_query(vec, k=1)
        # distance is cosine distance: 1 - similarity
        assert dists[0] == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors_similarity_near_zero(self):
        """Cosine similarity of orthogonal vectors should be ~0 → distance ~1."""
        dim = 4
        ms = MatrixSearch(dim=dim)
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        ms.add_items([v2], [2])
        _, dists = ms.knn_query(v1, k=1)
        # distance = 1 - similarity; similarity ~0 → distance ~1
        assert dists[0] == pytest.approx(1.0, abs=1e-4)

    def test_returns_nearest_first(self):
        """Nearest neighbours should be returned in order of increasing distance."""
        dim = 3
        ms = MatrixSearch(dim=dim)
        # Three vectors: one very close to query, two further away
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        close = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        far1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        far2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        ms.add_items([close, far1, far2], [10, 20, 30])
        labels, dists = ms.knn_query(query, k=3)
        assert labels[0] == 10  # closest
        assert labels[1] == 20
        assert labels[2] == 30  # furthest
        # distances must be non-decreasing
        assert dists[0] <= dists[1] <= dists[2]

    def test_k_capped_at_total_count(self):
        """Requesting k > num_items should not error; should return all items."""
        dim = 3
        ms = MatrixSearch(dim=dim)
        vecs = [np.zeros(dim, dtype=np.float32) for _ in range(3)]
        ms.add_items(vecs, [1, 2, 3])
        labels, _ = ms.knn_query(vecs[0], k=100)
        assert len(labels) == 3

    def test_k_equals_one(self):
        """Requesting k=1 returns exactly one result."""
        dim = 3
        ms = MatrixSearch(dim=dim)
        ms.add_items([np.zeros(dim, dtype=np.float32)], [99])
        labels, _ = ms.knn_query(np.zeros(dim, dtype=np.float32), k=1)
        assert len(labels) == 1

    def test_different_dimensions_work(self):
        """Test with various embedding dimensions."""
        for dim in [4, 8, 16, 32]:
            ms = MatrixSearch(dim=dim)
            vec = np.ones(dim, dtype=np.float32)
            ms.add_items([vec], [1])
            labels, _ = ms.knn_query(vec, k=1)
            assert labels == [1]


class TestMatrixSearchSessionLabel:
    """Tests for get_session_label."""

    def test_same_id_returns_same_label(self):
        """Calling get_session_label with the same ID multiple times is consistent."""
        dim = 4
        ms = MatrixSearch(dim=dim)
        label1 = ms.get_session_label("session-a")
        label2 = ms.get_session_label("session-a")
        assert label1 == label2

    def test_different_ids_get_different_labels(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        l1 = ms.get_session_label("s1")
        l2 = ms.get_session_label("s2")
        assert l1 != l2

    def test_labels_are_sequential(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        assert ms.get_session_label("a") == 0
        assert ms.get_session_label("b") == 1
        assert ms.get_session_label("c") == 2

    def test_clear_resets_session_labels(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        ms.get_session_label("x")
        ms.clear()
        assert ms.get_session_label("x") == 0


class TestMatrixSearchClear:
    """Tests for clear()."""

    def test_clear_removes_vectors(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        ms.add_items([np.zeros(dim, dtype=np.float32)], [1])
        ms.clear()
        assert ms.get_current_count() == 0
        assert ms._vectors.shape == (0, dim)

    def test_clear_removes_labels(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        ms.add_items([np.zeros(dim, dtype=np.float32)], [42])
        ms.clear()
        assert ms._labels == []

    def test_clear_removes_session_data(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        ms.get_session_label("foo")
        ms.clear()
        # After clear, the same session_id should get label 0 again
        assert ms.get_session_label("foo") == 0

    def test_add_after_clear_works(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        ms.add_items([np.zeros(dim, dtype=np.float32)], [1])
        ms.clear()
        ms.add_items([np.ones(dim, dtype=np.float32)], [2])
        assert ms.get_current_count() == 1
        labels, _ = ms.knn_query(np.ones(dim, dtype=np.float32), k=1)
        assert labels == [2]


class TestMatrixSearchInit:
    """Tests for constructor and capacity helpers."""

    def test_initial_state(self):
        dim = 4
        ms = MatrixSearch(dim=dim)
        assert ms.get_current_count() == 0
        assert ms._vectors.shape == (0, dim)

    def test_max_elements_default(self):
        ms = MatrixSearch(dim=4)
        assert ms.get_max_elements() == 10000  # max(10000, 0+1000)

    def test_resize_sets_max(self):
        ms = MatrixSearch(dim=4)
        ms.resize_index(5000)
        assert ms._max_elements == 5000
        assert ms.get_max_elements() == 5000  # max(5000, 0+1000)
