# SPDX-License-Identifier: MIT
"""Memory dissolver — RAM eviction manager.

Ensures RAM usage stays within configured limits by evicting
low-score sessions to SQLite storage.
"""
import asyncio
import logging
import os
import time
from typing import Any

import psutil

from ..storage.persistence import PersistenceLayer

logger = logging.getLogger("memory_lib.dissolver")

# Importance → float weight for eviction priority formula
_IMPORTANCE_WEIGHT = {
    "principle": 1.0,
    "critical": 0.9,
    "important": 0.6,
    "background": 0.3,
}

# Hardcoded limits (removed ResourcesConfig dependency)
SESSION_WINDOW_SIZE = 1000
RAM_SOFT_LIMIT_MB = 800


class Dissolver:
    """RAM eviction manager.

    Ensures RAM usage stays within limits by moving low-score sessions
    to SQLite. Monitors session count and RSS memory pressure.
    """

    def __init__(self, ram_index: dict, persistence: PersistenceLayer, config: Any) -> None:
        self._ram_index = ram_index
        self._persistence = persistence
        self._config = config
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_blocked_log: float = 0.0

    async def start(self) -> None:
        """Start the background eviction loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Dissolver background loop started")

    async def stop(self) -> None:
        """Stop the background eviction loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Dissolver background loop stopped")

    async def check_and_evict(self) -> None:
        """Check RAM limits and trigger eviction if necessary.

        Rules:
        1. If session count > window_size * 0.8 -> evict N to bring down to window_size * 0.7.
        2. If RAM usage (RSS) > soft_limit_mb -> evict 10% of sessions.
        """
        # Rule 1: count-based
        current_count = len(self._ram_index)
        limit = SESSION_WINDOW_SIZE

        if current_count > limit * 0.8:
            n_to_evict = int(current_count - limit * 0.7)
            await self.evict_n_oldest(n_to_evict)
            return  # after count-based eviction, RAM is likely reduced enough for now

        # Rule 2: RAM-based (simple RSS check — excludes ONNX model weights)
        soft_limit = RAM_SOFT_LIMIT_MB
        process = psutil.Process(os.getpid())
        ram_used_mb = process.memory_info().rss / 1024 / 1024

        if ram_used_mb > soft_limit:
            n_to_evict = max(1, int(current_count * 0.10))
            await self.evict_n_oldest(n_to_evict)

    async def evict_n_oldest(self, n: int) -> int:
        """Evict N sessions with the lowest eviction priority.

        Priority formula:
            priority = importance_weight × (1 + intensity) × recency_factor
        Sessions with min(priority) are evicted first.

        Returns the number of sessions actually evicted.
        """
        if n <= 0:
            return 0

        # P1: flush pending writes before eviction — guarantee SQLite persistence
        if self._persistence:
            await self._persistence.flush()

        sessions = list(self._ram_index.values())
        sessions.sort(key=_eviction_priority)

        evicted_count = 0
        for sb in sessions:
            if evicted_count >= n:
                break
            if not can_evict(sb):
                continue
            del self._ram_index[sb.session_id]
            evicted_count += 1

        # Rate-limit: when all sessions are protected, log at most once per 10 min
        if evicted_count == 0 and n > 0:
            now = time.time()
            if now - self._last_blocked_log < 600:
                return evicted_count
            self._last_blocked_log = now

        logger.info("Dissolver evicted %d sessions from RAM.", evicted_count)
        return evicted_count

    async def _run_loop(self) -> None:
        """Periodic check for RAM limits."""
        interval = self._config.dissolver.consolidation_interval_sec
        while self._running:
            try:
                await self.check_and_evict()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in Dissolver loop: %s", e, exc_info=True)
                await asyncio.sleep(interval)


def _eviction_priority(sb: Any) -> float:
    """Compute eviction priority (ascending = evict first).

    priority = importance_weight × (1 + intensity) × recency_factor

    recency_factor: 1.0 for brand-new, decays to 0.0 at 90 days.
    Sessions with LOW priority are evicted first.
    """
    imp_w = _IMPORTANCE_WEIGHT.get(getattr(sb, "importance", "background"), 0.3)
    intensity = float(getattr(sb, "intensity", 0.0))
    age_days = (time.time() - getattr(sb, "created_at", 0)) / 86400
    recency = max(0.0, 1.0 - age_days / 90.0)
    return imp_w * (1.0 + intensity) * recency


def can_evict(sb: Any) -> bool:
    """Check if a specific session can be evicted according to spec rules.

    Rules (in priority order):
    - Principle: NEVER evict (resolution floor = 0.80, always RAM Hot/Warm).
    - Active urgency (live deadline): protect from eviction.
    - Conflict flag: keep in RAM for resolution.
    - Critical: protect unless RAM > 90% of hard limit.
    """
    if getattr(sb, "importance", "") == "principle":
        return False

    # Urgency active = deadline exists, in the future, and not expired
    has_live_deadline = (
        getattr(sb, "deadline_ts", None) is not None
        and not getattr(sb, "urgency_expired", False)
        and (getattr(sb, "deadline_ts", 0) or 0) > time.time()
    )
    if has_live_deadline:
        return False

    if getattr(sb, "conflict_flag", False):
        return False

    # Protection for critical sessions
    if getattr(sb, "importance", "") == "critical":
        try:
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / 1024 / 1024
            hard_limit = RAM_SOFT_LIMIT_MB * 1.25  # 800 * 1.25 = 1000
            # Evict critical only if we are above 90% of hard limit
            if ram_mb < hard_limit * 0.90:
                return False
        except Exception:
            pass

    return True
