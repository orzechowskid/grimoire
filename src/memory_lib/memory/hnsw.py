# SPDX-License-Identifier: MIT
"""Matrix cosine search — replaces hnswlib.

Language-agnostic: e5-small embeddings work across all languages.
Stores L2-normalised vectors in a numpy matrix and returns cosine
nearest-neighbour results via dot-product similarity.
"""
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def time_decay(age_days: float, half_life: float) -> float:
    """Exponential recency decay factor.

    At age=0 returns 1.0 (no penalty).
    At age=half_life returns 0.5.
    At age=2*half_life returns 0.25.

    Args:
        age_days: Age of the item in days.
        half_life: Days until score is halved.

    Returns:
        Multiplier in (0, 1].
    """
    if half_life <= 0:
        return 1.0
    return math.exp(-age_days * math.log(2) / half_life)


class MatrixSearch:
    """Cosine nearest-neighbour search over a float32 embedding matrix.

    Stores L2-normalised vectors; knn_query returns cosine distances
    (1 - similarity) to match the hnswlib cosine-space convention.
    """

    def __init__(self, dim: int, max_elements: int = 10000, **kwargs: Any) -> None:
        self.dim = dim
        self._max_elements = max_elements
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._labels: list[int] = []
        self._session_label_counter: int = 0
        self._sid_to_id: dict[str, int] = {}
        self._label_to_row_index: dict[int, int] = {}  # label -> row index in _vectors

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_items(self, vectors: list[np.ndarray], ids: list[int]) -> None:
        """Append normalised vectors and their integer labels."""
        vecs = np.array(vectors, dtype=np.float32).reshape(-1, self.dim)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms
        self._vectors = (
            np.vstack([self._vectors, vecs]) if len(self._vectors) else vecs
        )
        start_row = len(self._labels)
        self._labels.extend(ids)
        for i, label in enumerate(ids):
            self._label_to_row_index[label] = start_row + i

    def update_item(self, label: int, vector: np.ndarray) -> None:
        """Replace the vector for an existing label in-place.

        If the label already exists in the index, replaces its vector.
        If not, falls back to add_items (appending a new entry).

        Args:
            label: The integer label to update.
            vector: The new vector (will be L2-normalised).
        """
        row_idx = self._label_to_row_index.get(label)
        if row_idx is not None and row_idx < len(self._vectors):
            vec = np.array(vector, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            self._vectors[row_idx] = vec
        else:
            # Label not found — fall back to appending
            self.add_items([vector], [label])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def knn_query(
        self,
        vector: np.ndarray,
        k: int = 5,
        time_weighted: bool = False,
        half_life_days: float = 30.0,
        exempt_importance: tuple = ("critical", "principle"),
        label_meta: dict | None = None,
    ) -> tuple[list[int], list[float]]:
        """Return (labels, distances) for the k nearest neighbours.

        Distances are cosine distances: 0 = identical, 2 = opposite.
        Callers using ``1.0 - dist`` stay correct.
        """
        count = len(self._labels)
        if count == 0:
            return [], []

        k = min(k, count)
        vec = np.array(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm

        sims = self._vectors @ vec  # shape [N]

        if time_weighted and label_meta:
            weighted = sims.copy()
            now = time.time()
            for i, label in enumerate(self._labels):
                sb = label_meta.get(label)
                if sb is None:
                    continue
                importance = getattr(sb, "importance", "normal")
                if importance in exempt_importance:
                    continue
                age_days = (now - sb.created_at) / 86400
                weighted[i] *= time_decay(age_days, half_life_days)
            dists = 1.0 - weighted
        else:
            dists = 1.0 - sims

        idx = np.argsort(dists)[:k]
        labels = [self._labels[i] for i in idx]
        distances = dists[idx].tolist()
        return labels, distances

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset matrix and labels."""
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._labels = []
        self._session_label_counter = 0
        self._sid_to_id = {}
        self._label_to_row_index = {}

    def get_session_label(self, session_id: str) -> int:
        """Get existing or assign a new deterministic label for a session.
        
        The same session_id always returns the same label.
        """
        if session_id in self._sid_to_id:
            return self._sid_to_id[session_id]
        label = self._session_label_counter
        self._session_label_counter += 1
        self._sid_to_id[session_id] = label
        return label

    def get_current_count(self) -> int:
        return len(self._labels)

    def get_max_elements(self) -> int:
        return max(self._max_elements, len(self._labels) + 1000)

    def resize_index(self, new_max: int) -> None:
        """No-op — numpy matrix grows dynamically."""
        self._max_elements = new_max

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        base = str(path)
        np.save(base + ".vecs.npy", self._vectors)
        np.save(base + ".labels.npy", np.array(self._labels, dtype=np.int64))

    def load_index(self, path: str | Path, max_elements: int = 0) -> None:
        base = str(path)
        vecs_path = Path(base + ".vecs.npy")
        labels_path = Path(base + ".labels.npy")
        if vecs_path.exists() and labels_path.exists():
            self._vectors = np.load(str(vecs_path))
            self._labels = np.load(str(labels_path)).tolist()
            if max_elements:
                self._max_elements = max_elements
            # Rebuild label-to-row mapping
            self._label_to_row_index = {}
            for i, label in enumerate(self._labels):
                self._label_to_row_index[label] = i


# Aliases kept for backward compatibility
HNSWIndex = MatrixSearch


def init_session_index(config: Any) -> MatrixSearch:
    """Initialize matrix search index for sessions from config."""
    return MatrixSearch(
        dim=config.search.embedding_dim,
        max_elements=10000,
    )


def init_content_index(config: Any) -> MatrixSearch:
    """Initialize matrix search index for content blocks from config."""
    return MatrixSearch(
        dim=config.search.embedding_dim,
        max_elements=5000,
    )
