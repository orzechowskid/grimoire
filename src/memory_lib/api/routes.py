# SPDX-License-Identifier: MIT

"""FastAPI HTTP API endpoints for the memory library.

Provides REST endpoints for injecting memory context, observing agent
output, semantic search, and graceful shutdown.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from memory_lib.config import Settings

logger = logging.getLogger("memory_lib.api")


# ── Pydantic request/response models ───────────────────────────────────

class InjectRequest(BaseModel):
    text: str
    top_k: int = 10
    time_weighted: bool = False
    min_similarity: float = 0.0


class InjectResult(BaseModel):
    session_id: str
    brief: str
    tags: list[str]
    importance: str
    score: float
    similarity: float
    key_facts: list[str] = []  # NEW: key facts from associated anchor


class ObserveRequest(BaseModel):
    text: str
    session_id: str | None = None
    source: str | None = None  # NEW: "user" or "agent"


class ObserveResponse(BaseModel):
    session_id: str
    brief: str
    tags: list[str]
    importance: str
    score: float
    discarded: bool


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_similarity: float = 0.0
    time_weighted: bool = False


class SearchResult(BaseModel):
    session_id: str
    brief: str
    tags: list[str]
    importance: str
    score: float
    created_at: int
    similarity: float
    key_facts: list[str] = []  # NEW: key facts from associated anchor


class IntuitionSignal(BaseModel):
    """Intuition signal from the experience layer.

    Surfaces topic-specific alerts (tensions, recommendations, emotional
    triggers) directly into the agent's system prompt.
    """
    type: str        # "TENSION", "DO_THIS", "AVOID_THIS", "ATTRACT", "REPEL", "AMBIVALENT"
    tag: str         # The topic tag this signal relates to
    message: str     # Human-readable signal description


# ── Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from memory_lib.bootstrap import bootstrap, shutdown

    config_path = os.environ.get("GRIMOIRE_CONFIG", "config.json")
    if os.path.exists(config_path):
        config = Settings.from_json_file(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        config = Settings()
        logger.warning("No config.json at %s — using defaults", config_path)
    ctx = await bootstrap(config)
    app.state.ctx = ctx

    # Build label → session_id mapping by reversing the session_index's
    # internal _sid_to_id mapping. This ensures labels match exactly what
    # MatrixSearch assigned during bootstrap (via get_session_label).
    try:
        sid_to_id = getattr(ctx.session_index, "_sid_to_id", {})
        ctx._label_to_sid = {v: k for k, v in sid_to_id.items()}
        logger.info("Built label_to_session_id mapping with %d entries", len(ctx._label_to_sid))
    except Exception as e:
        logger.warning("Failed to build label_to_session_id mapping: %s", e)
        ctx._label_to_sid = {}

    yield

    # Shutdown
    await shutdown(ctx)


app = FastAPI(title="Memory Library", lifespan=lifespan)


# ── Helper: map knn results to response objects ────────────────────────

async def _map_knn_results(
    ctx: Any,
    labels: list[int],
    distances: list[float],
    min_similarity: float = 0.0,
    include_created_at: bool = False,
    include_key_facts: bool = False,  # NEW
) -> tuple[list[dict[str, Any]], int]:
    """Convert knn_query (labels, distances) to response dicts.

    Args:
        ctx: MemoryContext with ram_index and _label_to_sid.
        labels: Integer labels from MatrixSearch knn_query.
        distances: Cosine distances (1 - similarity).
        min_similarity: Filter threshold (similarity = 1 - distance).
        include_created_at: Whether to include created_at in output.

    Returns:
        (results_list, total_candidates)
    """
    # Use live _sid_to_id from session_index — updated by get_session_label()
    # during both bootstrap and observer_pipeline, so new sessions are always included.
    sid_to_id = getattr(ctx.session_index, "_sid_to_id", {})
    label_to_sid = {v: k for k, v in sid_to_id.items()}
    ram_index = ctx.ram_index
    results: list[dict[str, Any]] = []

    total_candidates = len(labels)

    for label, distance in zip(labels, distances):
        similarity = 1.0 - distance

        # Apply min_similarity filter
        if similarity < min_similarity:
            continue

        # Map label to session_id
        session_id = label_to_sid.get(label)

        if session_id is None:
            continue
        elif session_id not in ram_index:
            # Session was evicted from RAM, try cold-load
            sb = await ctx.persistence.get_session_by_id(session_id)
            if sb is None:
                continue
        else:
            sb = ram_index[session_id]

        result: dict[str, Any] = {
            "session_id": sb.session_id,
            "brief": sb.brief,
            "tags": sb.tags,
            "importance": sb.importance,
            "score": sb.score,
            "similarity": round(similarity, 6),
        }
        if include_created_at:
            result["created_at"] = sb.created_at

        if include_key_facts:
            # Look up associated anchor and extract key_facts
            anchor = ctx.anchor_index.get(session_id)
            if anchor and anchor.key_facts:
                result["key_facts"] = [
                    fact.get("value", "") for fact in anchor.key_facts
                    if fact.get("value")
                ]
            else:
                result["key_facts"] = []

        results.append(result)

    return results, total_candidates


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Health check endpoint.

    Returns status of the memory system and loaded components.
    """
    ctx = request.app.state.ctx
    if ctx is None:
        return {"status": "starting"}

    return {
        "status": "ok",
        "sessions_in_ram": len(ctx.ram_index),
        "anchors_in_ram": len(ctx.anchor_index.all()),
        "experience_clusters": len(ctx.experience_index.all_clusters()),
        "embedder": "loaded" if ctx.embedder else "unavailable",
        "ner": "loaded" if ctx.ner_extractor else "unavailable",
        "reranker": "loaded" if ctx.reranker else "unavailable",
    }


@app.post("/inject")
async def inject(request: InjectRequest, req: Request):
    """Memory context injection for user messages.

    Embeds the query text and returns the top-k most similar sessions.
    """
    try:
        ctx = req.app.state.ctx
        if ctx is None:
            return JSONResponse(
                status_code=503,
                content={"error": "memory system not ready"},
            )

        # Check embedder availability
        if ctx.embedder is None:
            return {
                "results": [],
                "total_candidates": 0,
                "warning": "embedder_unavailable",
                "intuition_signals": [],
            }

        # Embed the query text
        vec = await ctx.embedder.aencode(request.text)
        vec_f32 = vec.astype(np.float32)

        # Build label_meta for time-weighted scoring
        label_meta: dict[int, Any] | None = None
        if request.time_weighted:
            label_meta = {
                label: ctx.ram_index[sid]
                for label, sid in getattr(ctx, "_label_to_sid", {}).items()
                if sid in ctx.ram_index
            }

        # Perform nearest neighbour search
        labels, distances = ctx.session_index.knn_query(
            vec_f32,
            k=request.top_k,
            time_weighted=request.time_weighted,
            label_meta=label_meta,
        )

        if not labels:
            return {"results": [], "total_candidates": 0, "intuition_signals": []}

        # Map results to response
        results, total_candidates = await _map_knn_results(
            ctx,
            labels,
            distances,
            min_similarity=0.0,
            include_created_at=False,
            include_key_facts=True,  # NEW
        )

        # Apply similarity floor (config wins as minimum)
        effective_min = max(request.min_similarity, ctx.config.search.inject_min_similarity)
        results = [r for r in results if r["similarity"] >= effective_min]

        # Reranker re-scoring (best-effort)
        if (
            results
            and ctx.reranker is not None
            and ctx.config.search.inject_rerank
        ):
            try:
                briefs = [r["brief"] for r in results]
                reranked = await ctx.reranker.arerank(request.text, briefs)
                # reranked is [(doc, score), ...] sorted by descending score
                # Build mapping from brief text back to result dict
                brief_to_result = {r["brief"]: r for r in results}
                reranked_results = []
                for doc, _score in reranked:
                    if doc in brief_to_result:
                        reranked_results.append(brief_to_result[doc])
                results = reranked_results
            except Exception as e:
                logger.warning("Reranker error during inject: %s — falling back to cosine ordering", e)

        # Cap at top 5 after reranking (or 10 if no reranker)
        if ctx.reranker is not None and ctx.config.search.inject_rerank:
            results = results[:5]
        else:
            results = results[:10]

        # Collect unique tags from retrieved results for intuition signals
        all_tags: set[str] = set()
        for r in results:
            for tag in r.get("tags", []):
                all_tags.add(tag)

        # Query experience index for intuition signals
        intuition_signals: list[IntuitionSignal] = []
        if all_tags and ctx.experience_index is not None:
            raw_signals = ctx.experience_index.intuition_signals(list(all_tags))
            for sig in raw_signals:
                intuition_signals.append(IntuitionSignal(
                    type=sig.get("type", ""),
                    tag=sig.get("tag", ""),
                    message=sig.get("message", ""),
                ))

        return {
            "results": results,
            "total_candidates": total_candidates,
            "intuition_signals": [
                {"type": s.type, "tag": s.tag, "message": s.message}
                for s in intuition_signals
            ],
        }

    except Exception as e:
        logger.error("Inject error: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/observe")
async def observe(request: ObserveRequest, req: Request):
    """Feed agent response for memory processing.

    Processes text through the observer pipeline and stores results.
    """
    try:
        ctx = req.app.state.ctx
        if ctx is None:
            return JSONResponse(
                status_code=503,
                content={"error": "memory system not ready"},
            )

        # Generate session_id if not provided
        session_id = request.session_id or uuid.uuid4().hex[:16]

        logger.info("Observing source=%s session=%s", request.source or "agent", session_id)

        # Run the observer pipeline
        from memory_lib.observer.pipeline import observer_pipeline

        sb = await observer_pipeline(
            text=request.text,
            session_id=session_id,
            ner_extractor=ctx.ner_extractor,
            embedder=ctx.embedder,
            ram_index=ctx.ram_index,
            session_index=ctx.session_index,
            anchor_index=ctx.anchor_index,
            experience_index=ctx.experience_index,
            persistence=ctx.persistence,
            config=ctx.config,
            summarizer=ctx.summarizer,
        )

        if sb is None:
            return ObserveResponse(
                session_id=session_id,
                brief="",
                tags=[],
                importance="background",
                score=0.0,
                discarded=True,
            )

        return ObserveResponse(
            session_id=sb.session_id,
            brief=sb.brief,
            tags=sb.tags,
            importance=sb.importance,
            score=sb.score,
            discarded=False,
        )

    except Exception as e:
        logger.error("Observe error: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/search")
async def search(request: SearchRequest, req: Request):
    """Semantic search over stored memory.

    Returns sessions matching the query, filtered by min_similarity.
    """
    try:
        ctx = req.app.state.ctx
        if ctx is None:
            return JSONResponse(
                status_code=503,
                content={"error": "memory system not ready"},
            )

        # Check embedder availability
        if ctx.embedder is None:
            return {"results": [], "total_candidates": 0}

        # Embed the query text
        vec = await ctx.embedder.aencode(request.query)
        vec_f32 = vec.astype(np.float32)

        # Build label_meta for time-weighted scoring
        label_meta: dict[int, Any] | None = None
        if request.time_weighted:
            label_meta = {
                label: ctx.ram_index[sid]
                for label, sid in getattr(ctx, "_label_to_sid", {}).items()
                if sid in ctx.ram_index
            }

        # Perform nearest neighbour search
        labels, distances = ctx.session_index.knn_query(
            vec_f32,
            k=request.top_k,
            time_weighted=request.time_weighted,
            label_meta=label_meta,
        )

        if not labels:
            return {"results": [], "total_candidates": 0}

        # Map results to response (with created_at for search)
        results, total_candidates = await _map_knn_results(
            ctx,
            labels,
            distances,
            min_similarity=request.min_similarity,
            include_created_at=True,
            include_key_facts=True,  # NEW
        )

        return {
            "results": results,
            "total_candidates": total_candidates,
        }

    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/shutdown")
async def shutdown_endpoint(request: Request):
    """Acknowledge shutdown signal.

    Actual cleanup is handled by uvicorn's lifespan shutdown when
    SIGTERM is received, which calls shutdown(ctx) via the
    lifespan context manager.
    """
    return {"status": "shutting_down"}
