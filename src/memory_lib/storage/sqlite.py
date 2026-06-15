# SPDX-License-Identifier: MIT

"""Async SQLite backend for the memory library."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

from .schemas import ALL_SCHEMAS

logger = logging.getLogger("memory_lib.storage")


async def init_db(db_path: str | Path) -> aiosqlite.Connection:
    """Initialize SQLite database with WAL mode and schemas.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Active aiosqlite connection.
    """
    # Ensure parent directory exists for the database file
    db_dir = Path(db_path).parent
    if db_dir != Path(".") and not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created database directory %s", db_dir)

    db = await aiosqlite.connect(db_path)

    # WAL journal mode and reasonable defaults
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA cache_size=-8000")   # 8MB
    await db.execute("PRAGMA mmap_size=134217728")  # 128MB

    # Apply all table schemas and indices
    for schema in ALL_SCHEMAS:
        await db.execute(schema)

    # Migrations — add missing columns gracefully
    await check_anchor_schema(db)
    await check_session_schema(db)

    await db.commit()
    logger.info("Database initialized at %s", db_path)
    return db


async def check_anchor_schema(db: aiosqlite.Connection) -> None:
    """Add missing columns to anchors table (migrations)."""
    # t_rel — added in v1.6
    try:
        await db.execute(
            "ALTER TABLE anchors ADD COLUMN "
            "t_rel TEXT NOT NULL DEFAULT '{\"after\":[],\"before\":[],\"caused_by\":[],\"during\":[]}'"
        )
        await db.commit()
    except Exception:
        pass
    # embedding — added in Phase 11 (Guardian/Surfacing semantic match)
    try:
        await db.execute("ALTER TABLE anchors ADD COLUMN embedding BLOB")
        await db.commit()
    except Exception:
        pass
    # Expression index — silently skip on older SQLite versions
    try:
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_anchors_flags_outcome "
            "ON anchors(json_extract(flags, '$.outcome'))"
        )
        await db.commit()
    except Exception:
        pass


async def check_session_schema(db: aiosqlite.Connection) -> None:
    """Migrate sessions table for v1.7.1 (resolution, intensity) and later versions."""
    for col_sql in [
        "ALTER TABLE sessions ADD COLUMN resolution REAL DEFAULT 1.0",
        "ALTER TABLE sessions ADD COLUMN intensity REAL DEFAULT 0.0",
        "ALTER TABLE sessions ADD COLUMN session_type TEXT",
        "ALTER TABLE sessions ADD COLUMN urgency TEXT DEFAULT 'none'",
        "ALTER TABLE sessions ADD COLUMN deadline_ts INTEGER",
        "ALTER TABLE sessions ADD COLUMN urgency_active INTEGER DEFAULT 0",
        "ALTER TABLE sessions ADD COLUMN urgency_expired INTEGER DEFAULT 0",
        "ALTER TABLE sessions ADD COLUMN bare_entity INTEGER DEFAULT 0",
        "ALTER TABLE sessions ADD COLUMN embedding_model_version TEXT DEFAULT 'multilingual-e5-small'",
        "ALTER TABLE sessions ADD COLUMN embedding BLOB",
        "ALTER TABLE sessions ADD COLUMN use_count INTEGER DEFAULT 0",
        "ALTER TABLE sessions ADD COLUMN deep_use_count INTEGER DEFAULT 0",
        "ALTER TABLE sessions ADD COLUMN last_use_ts INTEGER",
        "ALTER TABLE sessions ADD COLUMN score REAL DEFAULT 0.0",
    ]:
        try:
            await db.execute(col_sql)
            await db.commit()
        except Exception:
            pass
    # session_type index
    try:
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type)")
        await db.commit()
    except Exception:
        pass


class DatabaseManager:
    """Manager for async persistence to SQLite.

    Uses an internal asyncio.Queue to batch writes and avoid blocking
    the main Observer pipeline.

    Args:
        db: Active aiosqlite connection.
    """

    FLUSH_INTERVAL: float = 5.0
    BATCH_SIZE: int = 20

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Start the async flush worker."""
        if self._worker_task is None:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("DatabaseManager worker started")

    async def stop(self) -> None:
        """Gracefully stop the worker and flush remaining data."""
        self._running = False
        if self._worker_task:
            await self.queue.put(None)  # Sentinel
            await self._worker_task
            self._worker_task = None
            logger.info("DatabaseManager worker stopped")

    async def flush(self) -> None:
        """Drain the pending write queue to SQLite immediately.

        Safe to call while _worker is running. Consumes all queued items
        synchronously via get_nowait(), bypassing the worker's batch timer.
        """
        batch: list[Any] = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                self.queue.task_done()
                if item is None:          # stop sentinel — return it so stop() works
                    await self.queue.put(None)
                    break
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._flush_batch(batch)

    def queue_write(self, session: Any) -> None:
        """Add a SessionBrief to the persistence queue — sync, never blocks."""
        try:
            self.queue.put_nowait(session)
        except asyncio.QueueFull:
            sid = getattr(session, 'session_id', '?')
            logger.error(
                "StorePipe queue full — session NOT persisted (RAM⊆DISK violated): %s", sid
            )

    async def check_embedding_dim(self, expected_dim: int) -> None:
        """Wipe stale embeddings if stored dimension != expected."""
        try:
            async with self.db.execute(
                "SELECT embedding FROM sessions WHERE embedding IS NOT NULL LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row is None or row[0] is None:
                    return

                stored_blob = row[0]
                stored_dim = len(stored_blob) // 2  # float16 = 2 bytes per dim

                if stored_dim == expected_dim:
                    return

                logger.warning(
                    "dim_migration | stored_dim=%d expected_dim=%d -> wiping stale data",
                    stored_dim, expected_dim,
                )
                await self.db.execute("DELETE FROM sessions")
                await self.db.commit()
                logger.warning("dim_migration | wipe complete")

        except aiosqlite.Error as e:
            logger.error("dim_migration check failed (SQLite): %s", e)
        except Exception as e:
            logger.error("dim_migration check failed: %s", e)

    async def get_all_embeddings(self, expected_dim: int = 768) -> list[tuple[str, Any]]:
        """Retrieve all session embeddings from SQLite for HNSW hydration."""
        results: list[tuple[str, Any]] = []
        try:
            import numpy as np

            async with self.db.execute(
                "SELECT session_id, embedding FROM sessions WHERE embedding IS NOT NULL"
            ) as cursor:
                async for row in cursor:
                    session_id, raw_bytes = row
                    try:
                        vec = np.frombuffer(raw_bytes, dtype=np.float16)
                        if vec.shape[0] != expected_dim:
                            logger.warning(
                                "Dimension mismatch for %s: expected %d, got %d",
                                session_id, expected_dim, vec.shape[0],
                            )
                            continue
                        results.append((session_id, vec))
                    except Exception as e:
                        logger.error(
                            "Failed to deserialize embedding for %s: %s", session_id, e
                        )
        except Exception as e:
            logger.error("Error fetching embeddings for hydration: %s", e)

        return results

    async def get_all_session_briefs(self) -> list[Any]:
        """Retrieve all session metadata for ram_index hydration."""
        from ..memory.session_index import SessionBrief

        results: list[Any] = []
        try:
            import numpy as np

            async with self.db.execute(
                """SELECT session_id, created_at, importance, tags, brief,
                          conflict, urgency, deadline_ts, urgency_expired, bare_entity,
                          implicit_score, resolution, intensity, embedding,
                          use_count, deep_use_count, last_use_ts, score FROM sessions"""
            ) as cursor:
                async for row in cursor:
                    (
                        session_id, created_at, importance, tags_json, brief,
                        conflict, urgency, deadline_ts, urgency_expired, bare_entity,
                        implicit_score, resolution, intensity, embed_bytes,
                        use_count, deep_use_count, last_use_ts, score,
                    ) = row

                    embedding = None
                    if embed_bytes:
                        embedding = np.frombuffer(embed_bytes, dtype=np.float16)

                    sb = SessionBrief(
                        session_id=session_id,
                        brief=brief,
                        tags=json.loads(tags_json),
                        importance=importance,
                        score=score,
                        resolution=resolution,
                        created_at=created_at,
                        conflict_flag=bool(conflict),
                        urgency=urgency,
                        deadline_ts=deadline_ts,
                        urgency_expired=bool(urgency_expired),
                        bare_entity=bool(bare_entity),
                        embedding=embedding,
                        implicit_score=implicit_score,
                        intensity=intensity,
                        embedding_model_version="multilingual-e5-small",
                        use_count=use_count,
                        deep_use_count=deep_use_count,
                        last_use_ts=last_use_ts,
                    )
                    results.append(sb)
        except Exception as e:
            logger.error("Error fetching session briefs for hydration: %s", e)

        return results

    async def get_session_by_id(self, session_id: str) -> Any | None:
        """Load a single session from SQLite by session_id (cold load for ctx.load())."""
        from ..memory.session_index import SessionBrief

        try:
            import numpy as np

            async with self.db.execute(
                """SELECT session_id, created_at, importance, tags, brief,
                          conflict, urgency, deadline_ts, urgency_expired, bare_entity,
                          implicit_score, resolution, intensity, embedding,
                          use_count, deep_use_count, last_use_ts, score
                   FROM sessions WHERE session_id = ?""",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None

                (
                    session_id_, created_at, importance, tags_json, brief,
                    conflict, urgency, deadline_ts, urgency_expired, bare_entity,
                    implicit_score, resolution, intensity, embed_bytes,
                    use_count, deep_use_count, last_use_ts, score,
                ) = row

                embedding = None
                if embed_bytes:
                    embedding = np.frombuffer(embed_bytes, dtype=np.float16)

                return SessionBrief(
                    session_id=session_id_,
                    brief=brief,
                    tags=json.loads(tags_json),
                    importance=importance,
                    score=score,
                    resolution=resolution,
                    created_at=created_at,
                    conflict_flag=bool(conflict),
                    urgency=urgency,
                    deadline_ts=deadline_ts,
                    urgency_expired=bool(urgency_expired),
                    bare_entity=bool(bare_entity),
                    embedding=embedding,
                    implicit_score=implicit_score,
                    intensity=intensity,
                    embedding_model_version="multilingual-e5-small",
                    use_count=use_count,
                    deep_use_count=deep_use_count,
                    last_use_ts=last_use_ts,
                )
        except Exception as e:
            logger.error("get_session_by_id(%s): %s", session_id, e)
            return None

    async def save_anchor(self, anchor: Any) -> None:
        """Persist anchor to SQLite."""
        data = anchor.to_dict()
        import numpy as np

        emb_bytes = None
        if anchor.embedding is not None:
            emb_bytes = anchor.embedding.astype(np.float16).tobytes()

        try:
            await self.db.execute(
                """INSERT OR REPLACE INTO anchors
                   (anchor_id, session_id, anchor_type, brief, key_facts, flags,
                    decay_level, access_count, last_accessed_at, t_rel, created_at, updated_at,
                    embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["anchor_id"],
                    data["session_id"],
                    data["anchor_type"],
                    data["brief"],
                    json.dumps(data["key_facts"], ensure_ascii=False),
                    json.dumps(data["flags"], ensure_ascii=False),
                    data["decay_level"],
                    data["access_count"],
                    data["last_accessed_at"],
                    json.dumps(
                        data.get(
                            "t_rel",
                            {"after": [], "before": [], "caused_by": [], "during": []},
                        ),
                        ensure_ascii=False,
                    ),
                    data["created_at"],
                    data["updated_at"],
                    emb_bytes,
                ),
            )
            await self.db.commit()
        except Exception as e:
            logger.error("Failed to save anchor %s: %s", getattr(anchor, 'anchor_id', '?'), e)

    async def load_anchors(self, limit: int = 1000) -> list:
        """Load most recently accessed anchors for RAM index hydration."""
        import numpy as np

        anchors: list = []
        try:
            from ..subconscious.anchor import Anchor

            async with self.db.execute(
                """SELECT anchor_id, session_id, anchor_type, brief, key_facts, flags,
                          decay_level, access_count, last_accessed_at, t_rel, created_at, updated_at,
                          embedding
                   FROM anchors
                   ORDER BY last_accessed_at DESC
                   LIMIT ?""",
                (limit,),
            ) as cursor:
                async for row in cursor:
                    emb = None
                    if row[12] is not None:
                        try:
                            emb = np.frombuffer(row[12], dtype=np.float16).astype(np.float32)
                        except Exception:
                            pass
                    anchors.append(Anchor(
                        anchor_id=row[0],
                        session_id=row[1],
                        anchor_type=row[2],
                        brief=row[3],
                        key_facts=json.loads(row[4]),
                        flags=json.loads(row[5]),
                        decay_level=row[6],
                        access_count=row[7],
                        last_accessed_at=row[8],
                        t_rel=json.loads(row[9]),
                        created_at=row[10],
                        updated_at=row[11],
                        embedding=emb,
                    ))
        except Exception as e:
            logger.error("Error loading anchors: %s", e)
        return anchors

    async def get_anchor(self, anchor_id: str) -> Any | None:
        """Load single anchor from SQLite."""
        import numpy as np

        try:
            from ..subconscious.anchor import Anchor

            async with self.db.execute(
                """SELECT anchor_id, session_id, anchor_type, brief, key_facts, flags,
                          decay_level, access_count, last_accessed_at, t_rel, created_at, updated_at,
                          embedding
                   FROM anchors WHERE anchor_id = ?""",
                (anchor_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None

                emb = None
                if row[12] is not None:
                    try:
                        emb = np.frombuffer(row[12], dtype=np.float16).astype(np.float32)
                    except Exception:
                        pass

                return Anchor(
                    anchor_id=row[0],
                    session_id=row[1],
                    anchor_type=row[2],
                    brief=row[3],
                    key_facts=json.loads(row[4]),
                    flags=json.loads(row[5]),
                    decay_level=row[6],
                    access_count=row[7],
                    last_accessed_at=row[8],
                    t_rel=json.loads(row[9]),
                    created_at=row[10],
                    updated_at=row[11],
                    embedding=emb,
                )
        except Exception as e:
            logger.error("Error getting anchor %s: %s", anchor_id, e)
            return None

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
        """Query anchors from SQLite by flag patterns with pagination.

        Used by the Dreamer disk-scan to find pending anchors evicted from RAM.
        """
        import numpy as np

        from ..subconscious.anchor import Anchor

        # Build dynamic WHERE clause
        conditions: list[str] = []
        params: list[Any] = []

        if outcome is not None:
            conditions.append("json_extract(flags, '$.outcome') = ?")
            params.append(outcome)
        if multi_session is not None:
            conditions.append("json_extract(flags, '$.multi_session') = ?")
            params.append(1 if multi_session else 0)
        if anchor_type is not None:
            conditions.append("anchor_type = ?")
            params.append(anchor_type)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)

        # Always filter by decay level
        conditions.append("decay_level <= ?")
        params.append(decay_level_max)

        where_clause = " AND ".join(conditions)
        query = f"""SELECT anchor_id, session_id, anchor_type, brief, key_facts, flags,
                          decay_level, access_count, last_accessed_at, t_rel, created_at, updated_at,
                          embedding
                   FROM anchors
                   WHERE {where_clause}
                   ORDER BY last_accessed_at DESC
                   LIMIT ? OFFSET ?"""
        params.extend([limit, offset])

        anchors: list = []
        try:
            async with self.db.execute(query, params) as cursor:
                async for row in cursor:
                    emb = None
                    if row[12] is not None:
                        try:
                            emb = np.frombuffer(row[12], dtype=np.float16).astype(np.float32)
                        except Exception:
                            pass
                    anchors.append(Anchor(
                        anchor_id=row[0],
                        session_id=row[1],
                        anchor_type=row[2],
                        brief=row[3],
                        key_facts=json.loads(row[4]),
                        flags=json.loads(row[5]),
                        decay_level=row[6],
                        access_count=row[7],
                        last_accessed_at=row[8],
                        t_rel=json.loads(row[9]),
                        created_at=row[10],
                        updated_at=row[11],
                        embedding=emb,
                    ))
        except Exception as e:
            logger.error("Error in find_anchors_by_flags: %s", e)

        return anchors

    async def upsert_experience(
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
        """Upsert a single experience cluster row."""
        try:
            await self.db.execute(
                """INSERT INTO experience_metrics
                       (tag, session_count, score_sum, conflict_count, last_updated,
                        emotion_positive, emotion_negative, emotion_intensity_sum)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(tag) DO UPDATE SET
                       session_count=excluded.session_count,
                       score_sum=excluded.score_sum,
                       conflict_count=excluded.conflict_count,
                       last_updated=excluded.last_updated,
                       emotion_positive=excluded.emotion_positive,
                       emotion_negative=excluded.emotion_negative,
                       emotion_intensity_sum=excluded.emotion_intensity_sum""",
                (
                    tag,
                    session_count,
                    round(score_sum, 4),
                    conflict_count,
                    last_updated,
                    emotion_positive,
                    emotion_negative,
                    round(emotion_intensity_sum, 4),
                ),
            )
            await self.db.commit()
        except Exception as e:
            logger.error("Failed to upsert experience for tag '%s': %s", tag, e)

    async def load_experience(self) -> list[dict]:
        """Load all experience clusters for RAM hydration on bootstrap."""
        rows: list[dict] = []
        try:
            async with self.db.execute(
                """SELECT tag, session_count, score_sum, conflict_count, last_updated,
                          emotion_positive, emotion_negative, emotion_intensity_sum
                   FROM experience_metrics"""
            ) as cursor:
                async for row in cursor:
                    rows.append(
                        {
                            "tag": row[0],
                            "session_count": row[1],
                            "score_sum": row[2],
                            "conflict_count": row[3],
                            "last_updated": row[4],
                            "emotion_positive": row[5] if row[5] is not None else 0,
                            "emotion_negative": row[6] if row[6] is not None else 0,
                            "emotion_intensity_sum": (
                                row[7] if row[7] is not None else 0.0
                            ),
                        }
                    )
        except Exception as e:
            logger.error("Failed to load experience metrics: %s", e)
        return rows

    async def _worker(self) -> None:
        """Background worker that flushes batched sessions to SQLite."""
        batch: list[Any] = []
        last_flush: float = time.time()

        while True:
            try:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.FLUSH_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    item = "TIMEOUT"

                if item is None:  # Shutdown sentinel
                    if batch:
                        await self._flush_batch(batch)
                    self.queue.task_done()
                    break

                if item != "TIMEOUT":
                    batch.append(item)
                    self.queue.task_done()

                now = time.time()
                should_flush = (
                    len(batch) >= self.BATCH_SIZE
                    or (now - last_flush) >= self.FLUSH_INTERVAL
                )
                if batch and should_flush:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = now

            except aiosqlite.Error as e:
                logger.error("SQLite error in worker: %s", e, exc_info=True)
            except Exception as e:
                logger.error("Error in DatabaseManager worker: %s", e, exc_info=True)

    async def _flush_batch(self, batch: list[Any]) -> None:
        """Persist a batch of SessionBrief objects to the sessions table."""
        import numpy as np

        now_ts = int(time.time())

        for item in batch:
            try:
                # SessionBrief path: has session_id + brief attributes
                if hasattr(item, 'session_id') and hasattr(item, 'brief'):
                    session = item

                    urgency_active = (
                        bool(getattr(session, 'deadline_ts', None))
                        and not session.urgency_expired
                        and (session.deadline_ts or 0) > now_ts
                    )

                    # Serialize embedding with per-item error handling so
                    # individual failures don't break the entire batch.
                    emb_bytes: bytes | None = None
                    if session.embedding is not None:
                        logger.debug(
                            "Session %s: embedding present, shape=%s, dtype=%s",
                            session.session_id,
                            session.embedding.shape if hasattr(session.embedding, 'shape') else "unknown",
                            session.embedding.dtype if hasattr(session.embedding, 'dtype') else "unknown",
                        )
                        try:
                            emb_bytes = session.embedding.astype(np.float16).tobytes()
                            logger.debug(
                                "Session %s: serialized to %d bytes",
                                session.session_id, len(emb_bytes),
                            )
                        except Exception as e:
                            logger.warning(
                                "Embedding serialization failed for session %s: %s",
                                session.session_id, e,
                            )
                    else:
                        logger.warning(
                            "Session %s: embedding is None — will be stored without embedding",
                            session.session_id,
                        )

                    await self.db.execute(
                        """INSERT OR REPLACE INTO sessions
                        (session_id, created_at, importance, tags, brief,
                         conflict, urgency, deadline_ts, urgency_active, urgency_expired,
                         bare_entity, embedding_model_version, embedding,
                         use_count, deep_use_count, last_use_ts, implicit_score,
                         resolution, intensity, score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            session.session_id,
                            session.created_at,
                            session.importance,
                            json.dumps(session.tags),
                            session.brief,
                            1 if session.conflict_flag else 0,
                            session.urgency,
                            session.deadline_ts,
                            1 if urgency_active else 0,
                            1 if session.urgency_expired else 0,
                            1 if session.bare_entity else 0,
                            session.embedding_model_version,
                            emb_bytes,
                            getattr(session, 'use_count', 0),
                            getattr(session, 'deep_use_count', 0),
                            getattr(session, 'last_use_ts', None),
                            getattr(session, 'implicit_score', 0.5),
                            getattr(session, 'resolution', 1.0),
                            getattr(session, 'intensity', 0.0),
                            getattr(session, 'score', 0.0),
                        ),
                    )

            except aiosqlite.Error as e:
                logger.error(
                    "SQLite error flushing batch item: %s", e, exc_info=True
                )

        await self.db.commit()
        # Summary logging
        logger.info(
            "Flushed %d sessions to SQLite",
            len(batch),
        )


# Backward compatibility alias
SQLiteStorage = DatabaseManager
