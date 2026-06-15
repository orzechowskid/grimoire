# SPDX-License-Identifier: MIT

"""Initialization and bootstrap logic for all memory components."""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any

from .models.resolver import resolve_model

logger = logging.getLogger("memory_lib.bootstrap")


@dataclass
class MemoryContext:
    """Initialized memory system — everything the pipeline needs."""

    # Configuration
    config: Any

    # Models
    embedder: Any | None
    ner_extractor: Any | None
    reranker: Any | None
    summarizer: Any | None

    # Storage
    db: Any | None  # aiosqlite connection
    db_manager: Any | None  # DatabaseManager
    persistence: Any | None  # PersistenceLayer

    # RAM indices
    ram_index: dict  # session_id -> SessionBrief
    session_index: Any  # MatrixSearch
    anchor_index: Any  # AnchorIndex
    experience_index: Any  # ExperienceIndex

    # Workers
    dissolver: Any | None  # Dissolver
    consolidation_worker: Any | None  # ConsolidationWorker
    dreamer: Any | None  # Dreamer worker

    # Shutdown
    _shutdown_event: asyncio.Event


async def bootstrap(config: Any | None = None) -> MemoryContext:
    """Initialize the entire memory system.

    Args:
        config: Optional Settings instance. Uses defaults if not provided.

    Returns:
        MemoryContext with all components initialized.
    """
    # 1. Load config
    if config is None:
        from .config import Settings

        config = Settings()

    # 2. Initialize database
    from .storage.sqlite import init_db

    db = await init_db(config.storage.db_path)

    # 3. Initialize DatabaseManager
    from .storage.sqlite import DatabaseManager

    db_manager = DatabaseManager(db)
    await db_manager.start()

    # 4. Initialize PersistenceLayer
    from .storage.persistence import PersistenceLayer

    persistence = PersistenceLayer(db_manager)
    await persistence.start()

    # 5. Hydrate RAM from SQLite
    ram_index: dict[str, Any] = {}
    embeddings: list[tuple[str, Any]] = []

    try:
        all_briefs = await persistence.get_all_session_briefs()
        for sb in all_briefs:
            ram_index[sb.session_id] = sb
            logger.info("Hydrated session %s into RAM", sb.session_id)

        embeddings = await persistence.get_all_embeddings(config.search.embedding_dim)
        logger.info("Fetched %d embeddings for hydration", len(embeddings))
    except Exception as e:
        logger.warning("Hydration from SQLite failed: %s — starting fresh", e)

    # 6. Initialize MatrixSearch (session_index)
    from .memory.hnsw import MatrixSearch

    session_index = MatrixSearch(
        dim=config.search.embedding_dim, max_elements=10000
    )
    # Add hydrated embeddings to the session index.
    # IMPORTANT: use session_index.get_session_label(session_id) to assign labels.
    # This populates the internal _sid_to_id mapping so that label-to-session
    # resolution works correctly after bootstrap (matches observer_pipeline behavior).
    if embeddings:
        for session_id, vec in embeddings:
            try:
                vec_f32 = vec.astype(np.float32).flatten()
                label = session_index.get_session_label(session_id)
                session_index.add_items([vec_f32], [label])
            except Exception as e:
                logger.warning("Failed to add embedding for %s: %s", session_id, e)

    # 7. Initialize AnchorIndex
    from .subconscious.anchor_index import AnchorIndex

    anchor_index = AnchorIndex()
    try:
        loaded_anchors = await persistence.load_anchors()
        for anchor in loaded_anchors:
            anchor_index.put(anchor)
        logger.info("Hydrated %d anchors into AnchorIndex", len(loaded_anchors))
    except Exception as e:
        logger.warning("Anchor hydration failed: %s — starting fresh", e)

    # 8. Initialize ExperienceIndex
    from .memory.experience import ExperienceIndex

    experience_index = ExperienceIndex()
    try:
        loaded_experience = await persistence.load_experience()
        experience_index.load(loaded_experience)
        logger.info("Hydrated %d experience clusters", len(loaded_experience))
    except Exception as e:
        logger.warning("Experience hydration failed: %s — starting fresh", e)

    # 9. Resolve all model paths via the unified resolver
    embed_paths = None
    ner_paths = None
    rerank_paths = None
    summ_paths = None

    # Embedding model
    if config.models.embedding_id:
        try:
            embed_paths = resolve_model(config.models.embedding_id, "embedding")
            logger.info("Embedding model resolved to %s", embed_paths["model_path"])
        except Exception as e:
            logger.warning("Embedder resolution failed: %s — will be unavailable", e)

    # NER model
    if config.models.ner_id:
        try:
            ner_paths = resolve_model(config.models.ner_id, "ner")
            logger.info("NER model resolved to %s", ner_paths["model_path"])
        except Exception as e:
            logger.warning("NER resolution failed: %s — will be unavailable", e)

    # Reranker model
    if config.models.reranker_id:
        try:
            rerank_paths = resolve_model(config.models.reranker_id, "reranker")
            logger.info("Reranker model resolved to %s", rerank_paths["model_path"])
        except Exception as e:
            logger.warning("Reranker resolution failed: %s — will be unavailable", e)

    # Summarizer model
    if config.summarizer.model_id:
        try:
            summ_paths = resolve_model(config.summarizer.model_id, "observer")
            logger.info("Summarizer model resolved to %s", summ_paths["model_path"])
        except Exception as e:
            logger.warning("Summarizer resolution failed: %s — will be unavailable", e)

    # 10. Load models using resolved paths (best-effort)
    embedder = None
    ner_extractor = None
    reranker = None
    summarizer = None

    if embed_paths:
        try:
            from .models.embedder import Embedder
            embedder = Embedder(
                model_path=embed_paths["model_path"],
                tokenizer_path=embed_paths["tokenizer_path"],
            )
            logger.info("Embedder loaded successfully")
        except Exception as e:
            logger.warning("Embedder not available: %s — will use fallback vectors", e)

    if ner_paths:
        try:
            from .observer.ner import NERExtractor
            ner_extractor = NERExtractor(
                model_path=ner_paths["model_path"],
                tokenizer_path=ner_paths["tokenizer_path"],
            )
            logger.info("NER extractor loaded successfully")
        except Exception as e:
            logger.warning("NER extractor not available: %s", e)

    if rerank_paths:
        try:
            from .models.reranker import Reranker
            reranker = Reranker(
                model_path=rerank_paths["model_path"],
                tokenizer_path=rerank_paths["tokenizer_path"],
            )
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.warning("Reranker not available: %s", e)

    if summ_paths:
        try:
            from .models.summarizer import Summarizer
            summarizer = Summarizer(model_path=summ_paths["model_path"])
            logger.info("Summarizer loaded successfully")
        except Exception as e:
            logger.warning("Summarizer not available: %s — will use local fallback", e)

    # 10. Initialize Dissolver
    from .memory.dissolver import Dissolver

    dissolver = Dissolver(
        ram_index=ram_index,
        persistence=persistence,
        config=config,
    )
    await dissolver.start()

    # 11. Initialize ConsolidationWorker
    from .memory.consolidation import ConsolidationWorker

    consolidation_worker = ConsolidationWorker(
        ram_index=ram_index,
        dissolver=dissolver,
        experience_index=experience_index,
        anchor_index=anchor_index,
        persistence=persistence,
        config=config,
    )
    await consolidation_worker.start()

    # 11.5. Initialize Dreamer
    from .subconscious.dreamer import Dreamer

    dreamer = None
    if config.dreamer.enabled:
        dreamer = Dreamer(
            ram_index=ram_index,
            anchor_index=anchor_index,
            persistence=persistence,
            config=config,
        )
        await dreamer.start()

    # 12. Create shutdown event
    shutdown_event = asyncio.Event()

    # 13. Startup summary logging
    logger.info(
        "Bootstrap complete: %d sessions, %d anchors, %d experience clusters, embedder=%s, ner=%s, reranker=%s, summarizer=%s, dreamer=%s",
        len(ram_index),
        len(anchor_index.all()),
        len(experience_index.all_clusters()),
        "loaded" if embedder else "unavailable",
        "loaded" if ner_extractor else "unavailable",
        "loaded" if reranker else "unavailable",
        "loaded" if summarizer else "unavailable",
        "active" if dreamer else "disabled",
    )

    # 14. Return MemoryContext
    return MemoryContext(
        config=config,
        embedder=embedder,
        ner_extractor=ner_extractor,
        reranker=reranker,
        summarizer=summarizer,
        db=db,
        db_manager=db_manager,
        persistence=persistence,
        ram_index=ram_index,
        session_index=session_index,
        anchor_index=anchor_index,
        experience_index=experience_index,
        dissolver=dissolver,
        consolidation_worker=consolidation_worker,
        dreamer=dreamer,
        _shutdown_event=shutdown_event,
    )


async def shutdown(ctx: MemoryContext) -> None:
    """Gracefully shut down the memory system."""
    logger.info("Shutting down memory system...")

    # Signal shutdown event
    ctx._shutdown_event.set()

    # Stop consolidation worker
    if ctx.consolidation_worker:
        await ctx.consolidation_worker.stop()

    # Stop dreamer
    if ctx.dreamer:
        await ctx.dreamer.stop()

    # Stop dissolver
    if ctx.dissolver:
        await ctx.dissolver.stop()

    # Flush pending writes to SQLite
    if ctx.persistence:
        await ctx.persistence.flush()

    # Stop database manager
    if ctx.db_manager:
        await ctx.db_manager.stop()

    # Close summarizer
    if ctx.summarizer:
        ctx.summarizer.close()

    logger.info("Memory system shut down complete")
