# SPDX-License-Identifier: MIT

"""RAM-to-SQLite persistence layer."""

import asyncio
import logging
from typing import Any

from .sqlite import DatabaseManager

logger = logging.getLogger("memory_lib.persistence")


class PersistenceLayer:
    """Disk memory layer interface.

    Single point of contact between WorkingMemory components
    (pipeline, consolidation, dreamer, dissolver) and SQLite storage.
    Eliminates direct DatabaseManager coupling and hasattr guards.

    Uses two explicit write paths:

      Path 1 — enqueue_session():  queued, 5-second batch cycle.
                Sessions can tolerate up to 5-second loss on crash.

      Path 2 — save_anchor() / save_experience():  guaranteed immediate await.
                These are sacred writes — NEVER lost, pipeline blocks until done.

    Args:
        db_manager: Initialized DatabaseManager backend.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background flush worker."""
        await self._db.start()

    async def stop(self) -> None:
        """Gracefully stop worker and flush remaining queue."""
        await self._db.stop()

    # ------------------------------------------------------------------
    # Write path 1 — queued session writes (5-sec batch, loss ≤5s OK)
    # ------------------------------------------------------------------

    def enqueue_session(self, sb: Any) -> None:
        """Add a SessionBrief to the async flush queue.

        Non-blocking. On QueueFull logs ERROR (RAM⊆DISK invariant violation).
        """
        self._db.queue_write(sb)

    # ------------------------------------------------------------------
    # Write path 2 — guaranteed immediate writes (NEVER lost)
    # ------------------------------------------------------------------

    async def save_anchor(self, anchor: Any) -> None:
        """Persist anchor to SQLite immediately.

        Sacred write: pipeline blocks until committed.
        Anchors must never be lost.
        """
        await self._db.save_anchor(anchor)

    async def save_experience(
        self,
        tag: str,
        session_count: int,
        score_sum: float,
        conflict_count: int,
        last_updated: int,
        emotion_positive: int = 0,
        emotion_negative: int = 0,
        emotion_intensity_sum: float = 0.0,
    ) -> None:
        """Persist experience cluster to SQLite immediately.

        Sacred write: experience markers encode success/failure patterns
        and must not be lost.
        """
        await self._db.upsert_experience(
            tag=tag,
            session_count=session_count,
            score_sum=score_sum,
            conflict_count=conflict_count,
            last_updated=last_updated,
            emotion_positive=emotion_positive,
            emotion_negative=emotion_negative,
            emotion_intensity_sum=emotion_intensity_sum,
        )

    # ------------------------------------------------------------------
    # Flush / sync
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Drain the pending session queue immediately."""
        await self._db.flush()

    async def sync(self, checkpoint_mode: str = "PASSIVE") -> dict[str, Any]:
        """Force-flush queue then WAL checkpoint.

        Steps:
        1. Drain queue — wait until worker persists everything (timeout 10s).
        2. WAL checkpoint (PASSIVE = non-blocking, TRUNCATE = shutdown mode).

        Returns:
            dict: flushed_sessions, wal_pages, checkpoint_mode.
        """
        queue_size_before = self._db.queue.qsize()
        try:
            await asyncio.wait_for(self._db.queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                "PersistenceLayer.sync: queue drain timed out after 10s (partial flush)"
            )

        wal_pages = -1
        try:
            async with self._db.db.execute(
                f"PRAGMA wal_checkpoint({checkpoint_mode})"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    wal_pages = row[1] if len(row) > 1 else -1
        except Exception as e:
            logger.error("PersistenceLayer.sync: WAL checkpoint failed: %s", e)

        stats = {
            "flushed_sessions": queue_size_before,
            "wal_pages": wal_pages,
            "checkpoint_mode": checkpoint_mode,
        }
        logger.info("PersistenceLayer.sync complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Hydration reads — startup only (WorkingMemory bootstrap)
    # ------------------------------------------------------------------

    async def get_all_session_briefs(self) -> list[Any]:
        """Load all session metadata for ram_index hydration."""
        return await self._db.get_all_session_briefs()

    async def get_all_embeddings(self, expected_dim: int) -> list[Any]:
        """Load all session embeddings for MatrixSearch hydration."""
        return await self._db.get_all_embeddings(expected_dim)

    async def load_anchors(self, limit: int = 1000) -> list[Any]:
        """Load most-recently-accessed anchors for anchor_index hydration."""
        return await self._db.load_anchors(limit)

    async def load_experience(self) -> list[dict]:
        """Load all experience clusters for ExperienceIndex hydration."""
        return await self._db.load_experience()

    # ------------------------------------------------------------------
    # Point reads — on-demand cold load
    # ------------------------------------------------------------------

    async def get_session_by_id(self, session_id: str) -> Any | None:
        """Cold-load a single session from SQLite (ctx.load() path)."""
        return await self._db.get_session_by_id(session_id)

    async def get_anchor(self, anchor_id: str) -> Any | None:
        """Load a single anchor by ID."""
        return await self._db.get_anchor(anchor_id)

    async def find_anchors_by_flags(
        self,
        outcome: str | None = None,
        multi_session: bool | None = None,
        anchor_type: str | None = None,
        session_id: str | None = None,
        decay_level_max: int = 3,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """Query anchors from disk by flag patterns."""
        return await self._db.find_anchors_by_flags(
            outcome=outcome,
            multi_session=multi_session,
            anchor_type=anchor_type,
            session_id=session_id,
            decay_level_max=decay_level_max,
            limit=limit,
            offset=offset,
        )

    def pending_writes(self) -> int:
        """Return current queue depth (pending session writes)."""
        return self._db.queue.qsize()
