# SPDX-License-Identifier: MIT
"""Dreamer — background idle-reassessment worker.

Tasks:
  1. Reassess pending anchor outcome flags (pending → success/failure/abandoned)
     using 1-hop continuation chain analysis.
  2. Resurface high-access anchors to reset decay timers.
  3. Scan SQLite for pending anchors evicted from RAM.
"""
import asyncio
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("memory_lib.subconscious.dreamer")


class Dreamer:
    """Background worker for idle anchor reassessment."""

    def __init__(
        self,
        ram_index: dict,
        anchor_index: Any,
        persistence: Any,
        config: Any,
    ) -> None:
        self._ram_index = ram_index
        self._anchor_index = anchor_index
        self._persistence = persistence
        self._config = config
        self._running: bool = False
        self._task: asyncio.Task | None = None
        # Disk scan state
        self._disk_offset: int = 0
        self._disk_window: int = 1000
        self._disk_pass_resolved: int = 0

    async def start(self) -> None:
        """Start the Dreamer background loop."""
        if self._running:
            logger.warning("Dreamer already running.")
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Dreamer started.")

    async def stop(self) -> None:
        """Stop the Dreamer background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Dreamer stopped.")

    async def _run(self) -> None:
        """Periodic dream loop."""
        interval = self._config.dreamer.interval_min * 60
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.dream()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Error in Dreamer: %s", e, exc_info=True)

    async def dream(self) -> Dict[str, Any]:
        """Main reassessment cycle.

        Returns a stats dict with keys:
          anchors_checked, outcomes_updated, resurfaced,
          disk_anchors_checked, disk_outcomes_updated, duration_ms
        """
        start = time.monotonic()
        stats: Dict[str, Any] = {
            "anchors_checked": 0,
            "outcomes_updated": 0,
            "resurfaced": 0,
            "disk_anchors_checked": 0,
            "disk_outcomes_updated": 0,
            "duration_ms": 0.0,
        }

        # Guard: no anchor index available
        if not self._anchor_index or len(self._anchor_index) == 0:
            stats["duration_ms"] = (time.monotonic() - start) * 1000
            logger.info(
                "Dreamer cycle: %s",
                ", ".join(f"{k}={v}" for k, v in stats.items()),
            )
            return stats

        max_anchors = getattr(
            self._config.dreamer, "max_anchors_per_cycle", 20
        )

        # Sort by access_count descending, take top max_anchors
        all_anchors = sorted(
            self._anchor_index.all(),
            key=lambda a: a.access_count,
            reverse=True,
        )
        selected = all_anchors[:max_anchors]

        # --- Outcome reassessment ---
        for anchor in selected:
            stats["anchors_checked"] += 1
            if self._reassess_outcome(anchor):
                if self._persistence:
                    await self._persistence.save_anchor(anchor)
                stats["outcomes_updated"] += 1
                logger.debug(
                    "dream.outcome | id=%s new_outcome=%s",
                    anchor.anchor_id,
                    anchor.flags.get("outcome"),
                )

        # --- Resurface high-access anchors ---
        resurface_threshold = getattr(
            self._config.dreamer, "resurface_threshold", 3
        )
        for anchor in selected:
            if (
                anchor.access_count >= resurface_threshold
                and anchor.decay_level < 2
            ):
                anchor.touch()
                if self._persistence:
                    await self._persistence.save_anchor(anchor)
                stats["resurfaced"] += 1
                logger.debug(
                    "dream.resurface | id=%s access_count=%d",
                    anchor.anchor_id,
                    anchor.access_count,
                )

        # --- Disk scan (pending evicted anchors) ---
        if getattr(self._config.dreamer, "disk_scan_enabled", True):
            disk_stats = await self._disk_scan()
            stats["disk_anchors_checked"] = disk_stats.get(
                "disk_anchors_checked", 0
            )
            stats["disk_outcomes_updated"] = disk_stats.get(
                "disk_outcomes_updated", 0
            )

        stats["duration_ms"] = (time.monotonic() - start) * 1000
        logger.info(
            "Dreamer cycle: %s",
            ", ".join(f"{k}={v}" for k, v in stats.items()),
        )
        return stats

    def _reassess_outcome(self, anchor) -> bool:
        """1-hop continuation chain outcome reassessment.

        Returns True if the outcome was changed.
        """
        if anchor.flags.get("outcome") != "pending":
            return False

        # Check all RAM anchors for continuations pointing to this anchor
        for other in self._anchor_index.all():
            if other.flags.get("continuation_of") == anchor.anchor_id:
                if other.anchor_type == "milestone":
                    anchor.flags["outcome"] = "success"
                    anchor.updated_at = int(time.time())
                    return True
                other_outcome = other.flags.get("outcome")
                if other_outcome in ("failure", "abandoned"):
                    anchor.flags["outcome"] = other_outcome
                    anchor.updated_at = int(time.time())
                    return True

        return False

    async def _disk_scan(self) -> Dict[str, Any]:
        """Paginated SQLite scan for pending evicted anchors."""
        stats = {
            "disk_anchors_checked": 0,
            "disk_outcomes_updated": 0,
        }

        page_size = getattr(
            self._config.dreamer, "disk_scan_page_size", 50
        )

        results = await self._persistence.find_anchors_by_flags(
            outcome="pending",
            multi_session=True,
            decay_level_max=2,
            limit=page_size,
            offset=self._disk_offset,
        )

        if not results:
            # No more pending anchors on this page
            if (
                self._disk_pass_resolved == 0
                and self._disk_window < 6000
            ):
                # Expand window x3, reset for another full pass
                self._disk_window *= 3
                self._disk_offset = 0
                self._disk_pass_resolved = 0
            else:
                self._disk_offset += page_size
            return stats

        for anchor in results:
            # Skip anchors already in RAM
            if anchor.anchor_id in self._anchor_index._anchors:
                continue

            stats["disk_anchors_checked"] += 1

            if self._reassess_outcome(anchor):
                if self._persistence:
                    await self._persistence.save_anchor(anchor)
                stats["disk_outcomes_updated"] += 1
                logger.debug(
                    "dream.disk_resolved | id=%s new_outcome=%s",
                    anchor.anchor_id,
                    anchor.flags.get("outcome"),
                )

            self._disk_pass_resolved += 1

        self._disk_offset += page_size
        return stats
