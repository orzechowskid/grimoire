# SPDX-License-Identifier: MIT
"""Main observation pipeline.

Coordinates the sequential steps of the observer pipeline:
filter → parallel NER + embed → marker → score → persist.
"""
import asyncio
import logging
import time
from typing import Any

import numpy as np

from ..memory.scoring import calculate_score
from ..memory.session_index import SessionBrief
from ..subconscious.anchor import Anchor
from ..subconscious.anchor_index import AnchorIndex
from .entities import MarkerAction, SourceType
from .filter import detect_urgency, deterministic_filter
from .marker import marker as _marker
from .ner import NERExtractor
from .utils import compress_text

IMPORTANCE_LEVELS = {
    "background": 0,
    "important": 1,
    "critical": 2,
    "principle": 3,
}

logger = logging.getLogger("memory_lib.observer")


def _float_to_importance(importance: float) -> str:
    """Map a numeric importance score to a string label."""
    if importance >= 1.0:
        return "principle"
    if importance >= 0.88:
        return "critical"
    if importance >= 0.55:
        return "important"
    return "background"


async def observer_pipeline(
    text: str,
    session_id: str,
    ner_extractor: NERExtractor | None,
    embedder: Any | None,
    ram_index: dict[str, SessionBrief],
    session_index: Any,  # MatrixSearch
    anchor_index: Any,   # AnchorIndex
    experience_index: Any | None,  # ExperienceIndex
    persistence: Any,     # PersistenceLayer
    config: Any,          # Settings from config.py
    summarizer: Any | None = None,
    intent_vector: np.ndarray | None = None,
) -> SessionBrief | None:
    """Process agent output through the modular observation pipeline.

    Steps:
      1. Filter — classify text importance
      2. Parallel NER + Embedding
      3. (Skipped — Phase 11.A feature)
      4. Marker — decide what to save
      4.5. Summarize — optionally generate LLM brief (if configured)
      5. Score — compute composite score
      6. Persist — write to RAM index and enqueue for SQLite flush

    Args:
        text: Input text to process.
        session_id: Current session identifier.
        ner_extractor: NER model for entity extraction.
        embedder: Embedding model for text vectors.
        ram_index: In-memory dict of session briefs.
        session_index: MatrixSearch for embedding index.
        anchor_index: AnchorIndex for anchor storage.
        experience_index: ExperienceIndex for behavioral tracking.
        persistence: PersistenceLayer for SQLite writes.
        config: Settings object from config.py.
        summarizer: Optional Summarizer for LLM-based brief generation.
        intent_vector: Optional pre-computed intent embedding.

    Returns:
        Created SessionBrief, or None if the text was discarded.
    """
    start_time = time.time()

    # ── Step 1: Filter ───────────────────────────────────────────────────
    filter_result = deterministic_filter(text)
    importance = filter_result["importance"]
    conflict = filter_result["conflict"]
    urgency, deadline_val = filter_result["urgency"], filter_result["deadline_val"]

    # ── Step 1.5: Discard monologue noise ───────────────────────────────
    if filter_result.get("discard"):
        logger.debug("Pipeline discarded monologue session %s", session_id)
        return None

    # ── Step 2: Parallel NER + Embed ─────────────────────────────────────
    entities: list[dict[str, Any]] = []
    embedding: np.ndarray | None = None

    async def _run_ner() -> list[dict[str, Any]]:
        """Run NER extraction if available and needed."""
        if not ner_extractor or not filter_result.get("needs_ner"):
            return []
        return await ner_extractor.extract(text, threshold=0.5)

    async def _run_embed() -> np.ndarray | None:
        """Run embedding with random-fallback on failure."""
        if not embedder:
            return None
        try:
            return await embedder.aencode(text)
        except Exception as e:
            logger.warning("Embedding failed, using fallback: %s", e)
            dim = getattr(config, "search", None)
            dim = getattr(dim, "embedding_dim", 384)
            v = np.random.rand(dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                return v / norm
            return v

    entities, embedding = await asyncio.gather(
        _run_ner(), _run_embed()
    )

    # ── Step 3: Anchor Guardian & Surfacing ──────────────────────────────
    # Phase 11.A/C/E feature — skipped in Phase 10

    # ── Step 4: Marker ──────────────────────────────────────────────────
    mark_result = await _marker(
        text,
        SourceType.AGENT,
        session_id,
        None,  # ctx removed in refactored pipeline
        chain=[],
        pending_emotions=[],
        embedding=embedding,
    )

    if mark_result.action == MarkerAction.DISCARD:
        logger.debug("Pipeline discarded session %s", session_id)
        return None

    # ── Step 4.5: Summarize ─────────────────────────────────────────────
    summarizer_brief = ""
    if summarizer is not None:
        observer_cfg = getattr(config, "observer", None)
        threshold_str = (
            observer_cfg.summarize_threshold
            if observer_cfg is not None and hasattr(observer_cfg, "summarize_threshold")
            else "important"
        )
        threshold_level = IMPORTANCE_LEVELS.get(threshold_str, 1)
        current_level = IMPORTANCE_LEVELS.get(importance, 0)

        if current_level >= threshold_level:
            try:
                brief = await summarizer.summarize(text)
                if brief and brief.strip():
                    summarizer_brief = brief.strip()
                    logger.info(
                        "Summarizer brief | session=%s | brief=%s",
                        session_id,
                        summarizer_brief,
                    )
                else:
                    logger.debug("Summarizer returned empty brief for session %s", session_id)
            except Exception as e:
                logger.warning("Summarization failed for session %s: %s", session_id, e)

    # ── Step 5: Score ───────────────────────────────────────────────────
    created_at = int(time.time())

    # Determine if urgency deadline has expired
    urgency_expired = False
    if urgency != "none" and deadline_val is not None:
        urgency_expired = time.time() > deadline_val

    score = calculate_score(
        relevance=0.5,
        created_at=created_at,
        importance=importance,
        config=config,
        profile="write",
        urgency_expired=urgency_expired,
        implicit_score=0.5,
    )

    # ── Step 6: Persist ─────────────────────────────────────────────────

    # Compress text into brief and tags
    observer_cfg = getattr(config, "observer", None)
    brief_max_length = observer_cfg.brief_max_length if observer_cfg is not None else 500

    # If summarizer produced a brief, use it; otherwise generate both
    if summarizer_brief:
        brief = summarizer_brief
        # Still need tags — extract from original text
        _, tags = compress_text(
            text, entities,
            precision_items=filter_result.get("precision_items"),
            max_length=brief_max_length,
        )
    else:
        brief, tags = compress_text(
            text, entities,
            precision_items=filter_result.get("precision_items"),
            max_length=brief_max_length,
        )

    tags_max_limit = observer_cfg.tags_max_per_session if (observer_cfg is not None and hasattr(observer_cfg, "tags_max_per_session")) else 7
    tags = tags[:tags_max_limit]

    # Create SessionBrief
    sb = SessionBrief(
        session_id=session_id,
        brief=brief,
        tags=tags,
        importance=importance,
        score=score,
        resolution=1.0,
        created_at=created_at,
        conflict_flag=conflict,
        urgency=urgency,
        deadline_ts=deadline_val,
        urgency_expired=urgency_expired,
        bare_entity=(importance == "background"),
        embedding=embedding,
        embedding_model_version="multilingual-e5-small",
    )

    # Add to RAM index
    ram_index[session_id] = sb

    # Add embedding to session_index if available
    if session_index is not None and embedding is not None:
        try:
            vec_f32 = embedding.astype(np.float32).flatten()
            label = session_index.get_session_label(session_id)
            session_index.add_items([vec_f32], [label])
        except Exception as e:
            logger.warning("Session index add failed: %s", e)

    # Create Anchor
    anchor_type = AnchorIndex.infer_anchor_type(importance, entities)
    key_facts = AnchorIndex.build_key_facts(entities, max_facts=5)

    anchor = Anchor(
        anchor_id=session_id,
        session_id=session_id,
        brief=brief,
        anchor_type=anchor_type,
        key_facts=key_facts,
        flags={
            "is_new_entity": True,
            "continuation_of": None,
            "continuation_depth": 0,
            "mention_type": "focus",
            "outcome": "pending",
            "user_pin": False,
            "multi_session": False,
        },
        t_rel={"after": [], "before": [], "caused_by": [], "during": []},
        decay_level=0,
        access_count=0,
        last_accessed_at=created_at,
        created_at=created_at,
        updated_at=created_at,
        embedding=embedding,
    )

    # Save anchor via persistence
    if persistence is not None:
        await persistence.save_anchor(anchor)

    # Put anchor in anchor_index
    if anchor_index is not None:
        anchor_index.put(anchor)

    # Update experience index
    if experience_index is not None:
        experience_index.update(tags, is_continuation=False, is_conflict=conflict)
        # Persist experience immediately
        if persistence is not None:
            for tag in tags:
                cluster = experience_index.get(tag)
                if cluster:
                    await persistence.save_experience(
                        tag=cluster.tag,
                        session_count=cluster.session_count,
                        score_sum=cluster.score_sum,
                        conflict_count=cluster.conflict_count,
                        last_updated=cluster.last_updated,
                        emotion_positive=cluster.emotion_positive,
                        emotion_negative=cluster.emotion_negative,
                        emotion_intensity_sum=cluster.emotion_intensity_sum,
                    )

    # Enqueue session for SQLite flush
    if persistence is not None:
        persistence.enqueue_session(sb)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "Observer processed session %s in %.2fms", session_id, elapsed_ms
    )

    return sb
