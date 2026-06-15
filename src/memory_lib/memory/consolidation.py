# SPDX-License-Identifier: MIT
"""Background consolidation worker for memory maintenance.

Tasks:
  1. Recalculate scores for all RAM sessions.
  2. Update age_signal (fresh → actual → stale → archive).
  3. Check expired deadlines.
  4. Trigger Dissolver eviction.
  5. Run experience and anchor decay.
"""
import asyncio
import logging
import time
from typing import Any

from .scoring import calculate_score

logger = logging.getLogger("memory_lib.consolidation")

# Fact priority for key_facts selection during decay
_DECAY_FACTS_LIMIT = {0: 5, 1: 3, 2: 1, 3: 0}


class ConsolidationWorker:
    """Background worker for offline memory maintenance."""

    def __init__(
        self,
        ram_index: dict,
        dissolver: Any,
        experience_index: Any,
        anchor_index: Any,
        persistence: Any,
        config: Any,
    ) -> None:
        self._ram_index = ram_index
        self._dissolver = dissolver
        self._experience_index = experience_index
        self._anchor_index = anchor_index
        self._persistence = persistence
        self._config = config
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_anchor_decay: float = 0.0

    async def start(self) -> None:
        """Start the consolidation background loop."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("ConsolidationWorker started.")

    async def stop(self) -> None:
        """Stop the consolidation background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ConsolidationWorker stopped.")

    async def _run(self) -> None:
        """Periodic consolidation loop."""
        interval = self._config.dissolver.consolidation_interval_sec
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ConsolidationWorker: %s", e, exc_info=True)

    async def consolidate(self) -> None:
        """Main consolidation loop.

        Tasks:
        1. Update age_signal for all RAM sessions based on age_days.
        2. Check expired deadlines — set urgency_expired flag.
        3. Recalculate scores using calculate_score().
        4. Trigger dissolver eviction.
        5. Run experience decay.
        6. Run anchor decay.
        7. Flush persistence.
        """
        now = time.time()

        # 1. Update age_signal and check deadlines for all RAM sessions
        for sid, sb in list(self._ram_index.items()):
            age_days = (now - sb.created_at) / 86400

            # Update age_signal
            if age_days > 90:
                sb.age_signal = "archive"
            elif age_days > 30:
                sb.age_signal = "stale"
            elif age_days > 7:
                sb.age_signal = "actual"
            else:
                sb.age_signal = "fresh"

            # Check deadlines
            if sb.deadline_ts and sb.deadline_ts < now and not sb.urgency_expired:
                sb.urgency_active = False
                sb.urgency_expired = True
                logger.warning("Deadline expired for session %s", sid)

            # Persist changes to SQLite
            if self._persistence:
                self._persistence.enqueue_session(sb)
                # For critical state changes like urgency expiry, flush immediately
                if sb.urgency_expired:
                    await self._persistence.flush()

        # 2. Recalculate scores with background relevance (0.5)
        for sid, sb in self._ram_index.items():
            sb.score = calculate_score(
                relevance=0.5,
                created_at=sb.created_at,
                importance=sb.importance,
                config=self._config,
            )

        # 3. Trigger dissolver eviction
        if self._dissolver:
            await self._dissolver.check_and_evict()

        # 4. Experience decay
        if self._experience_index:
            threshold = getattr(self._config, "experience", None)
            if threshold:
                threshold_days = getattr(threshold, "exp_decay_days_threshold", 90)
                rate = getattr(threshold, "exp_decay_rate", 0.01)
            else:
                threshold_days = 90
                rate = 0.01
            decayed_tags = self._experience_index.apply_decay(threshold_days, rate)
            if decayed_tags and self._persistence:
                for tag in decayed_tags:
                    cluster = self._experience_index.get(tag)
                    if cluster:
                        await self._persistence.save_experience(
                            tag=cluster.tag,
                            session_count=cluster.session_count,
                            score_sum=cluster.score_sum,
                            conflict_count=cluster.conflict_count,
                            last_updated=cluster.last_updated,
                            emotion_positive=cluster.emotion_positive,
                            emotion_negative=cluster.emotion_negative,
                            emotion_intensity_sum=cluster.emotion_intensity_sum,
                        )

        # 5. Anchor decay
        cfg_ad = getattr(self._config, "anchor_decay", None)
        if cfg_ad and getattr(cfg_ad, "enabled", True):
            interval_sec = cfg_ad.interval_min * 60
            if now - self._last_anchor_decay >= interval_sec:
                decayed = await self._run_anchor_decay(now)
                self._last_anchor_decay = now
                if decayed:
                    logger.info("Anchor decay: %d anchors decayed.", decayed)

        # 6. Flush persistence
        if self._persistence:
            await self._persistence.flush()

        logger.info(
            "Consolidation completed for %d RAM sessions.",
            len(self._ram_index),
        )

    async def _run_anchor_decay(self, now: float) -> int:
        """Advance decay_level for inactive, unpinned anchors.

        Decay levels:
          0 = full    (key_facts ≤ 5)
          1 = partial (key_facts ≤ 3)
          2 = skeleton (key_facts ≤ 1)
          3 = bedrock (key_facts = 0, brief only — anchor never deleted)

        Trigger: days_since_last_access >= threshold_days AND NOT user_pin
        """
        if not self._anchor_index:
            return 0

        cfg = self._config.anchor_decay
        decayed = 0

        for anchor in self._anchor_index.all():
            # Skip pinned anchors
            if anchor.flags.get("user_pin"):
                continue
            # Skip anchors already at decay_level >= 3
            if anchor.decay_level >= 3:
                continue

            last_access = anchor.last_accessed_at or anchor.created_at
            days_inactive = (now - last_access) / 86400

            if days_inactive < cfg.threshold_days:
                continue

            # Advance one level at a time
            anchor.decay_level = min(3, anchor.decay_level + 1)
            anchor.updated_at = int(now)

            # Trim key_facts to budget for new level
            limit = _DECAY_FACTS_LIMIT[anchor.decay_level]
            if len(anchor.key_facts) > limit:
                anchor.key_facts = anchor.key_facts[:limit]

            # Persist change to SQLite
            if self._persistence:
                await self._persistence.save_anchor(anchor)

            logger.debug(
                "anchor.decay | id=%s level=%d facts=%d inactive_days=%.1f",
                anchor.anchor_id,
                anchor.decay_level,
                len(anchor.key_facts),
                days_inactive,
            )
            decayed += 1

        return decayed
