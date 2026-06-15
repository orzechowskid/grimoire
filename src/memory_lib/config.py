# SPDX-License-Identifier: MIT
"""Configuration system for the memory library.

Provides a Pydantic-based Settings class with sensible defaults for
model paths, storage, search parameters, server, scoring, importance,
temporal thresholds, dissolver, and anchor decay.
"""
import json
import logging
import os

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelSettings(BaseModel):
    """HuggingFace repo IDs for models."""

    embedding_id: str = ""
    embedding_dim: int = 384
    ner_id: str = ""
    reranker_id: str = ""


class SummarizerSettings(BaseModel):
    """GGUF summarizer model configuration."""

    model_id: str = ""


class StorageSettings(BaseModel):
    """SQLite database configuration."""

    db_path: str = "memory.db"
    sqlite_synchronous: str = "NORMAL"
    sqlite_auto_vacuum: str = "INCREMENTAL"
    sqlite_compress_threshold_mb: int = 50
    sqlite_archive_cutoff_years: int = 5


class ScoreSettings(BaseModel):
    """Score weights for relevance, temporal decay, and importance."""

    weight_relevance: float = 0.5
    weight_temporal: float = 0.3
    weight_importance: float = 0.2
    temporal_decay_lambda: float = 0.01
    weight_relevance_search: float = 0.6
    weight_temporal_search: float = 0.25
    weight_importance_search: float = 0.15


class ImportanceSettings(BaseModel):
    """Importance weights and anchor TTL multipliers."""

    weight_critical: float = 1.0
    weight_important: float = 0.7
    weight_background: float = 0.3
    weight_principle: float = 0.9
    ner_score_threshold: float = 0.7
    tag_verification_threshold: float = 0.65
    anchor_ttl_importance_multiplier_critical: float = 2.0
    anchor_ttl_importance_multiplier_important: float = 1.5
    anchor_ttl_importance_multiplier_background: float = 0.5


class TemporalSettings(BaseModel):
    """Age thresholds for freshness, staleness, and archival."""

    age_threshold_fresh_days: int = 7
    age_threshold_actual_days: int = 30
    age_threshold_stale_days: int = 90
    age_threshold_archive_days: int = 365


class DissolverSettings(BaseModel):
    """Dissolver lambdas and consolidation interval."""

    lambda_critical: float = 0.005
    lambda_important: float = 0.01
    lambda_background: float = 0.02
    lambda_principle: float = 0.003
    consolidation_interval_sec: int = 300
    content_hot_protect_hours: int = 24
    content_active_protect: bool = True


class AnchorDecaySettings(BaseModel):
    """Anchor decay configuration."""

    enabled: bool = True
    interval_min: int = 60
    threshold_days: int = 30
    rate: float = 0.1


class DreamerSettings(BaseModel):
    """Dreamer background reassessment worker configuration."""

    enabled: bool = True
    interval_min: int = 15           # minutes between dream cycles
    max_anchors_per_cycle: int = 20  # max RAM anchors reassessed per cycle
    resurface_threshold: int = 3     # access_count threshold to resurface
    disk_scan_enabled: bool = True
    disk_scan_page_size: int = 50


class ObserverSettings(BaseModel):
    """Observer pipeline configuration."""

    brief_max_length: int = Field(
        default_factory=lambda: int(os.environ.get("GRIMOIRE_BRIEF_MAX_LENGTH", "500"))
    )
    summarize_threshold: str = "important"


class SearchSettings(BaseModel):
    """Matrix search, scoring, and pipeline configuration."""

    top_k_candidates: int = 100
    top_n_results: int = 10
    embedding_dim: int = 384
    matrix_dtype: str = "float32"
    pipeline_width: int = 4
    half_life_days: float = 30.0
    temporal_retrieval_ctx_recent_enabled: bool = True
    temporal_retrieval_time_weighted_search: bool = False
    temporal_retrieval_time_weight_exempt_importance: list[str] = ["critical", "principle"]
    inject_min_similarity: float = 0.0
    inject_rerank: bool = True


class ServerSettings(BaseModel):
    """HTTP server settings."""

    host: str = "127.0.0.1"
    port: int = 8766


class Settings(BaseModel):
    """Top-level configuration for the memory library."""

    models: ModelSettings = ModelSettings()
    storage: StorageSettings = StorageSettings()
    score: ScoreSettings = ScoreSettings()
    importance: ImportanceSettings = ImportanceSettings()
    temporal: TemporalSettings = TemporalSettings()
    dissolver: DissolverSettings = DissolverSettings()
    anchor_decay: AnchorDecaySettings = AnchorDecaySettings()
    dreamer: DreamerSettings = DreamerSettings()
    search: SearchSettings = SearchSettings()
    server: ServerSettings = ServerSettings()
    observer: ObserverSettings = ObserverSettings()
    summarizer: SummarizerSettings = SummarizerSettings()

    @classmethod
    def from_json_file(cls, path: str) -> "Settings":
        """Create a *Settings* instance from a JSON configuration file.

        A default *Settings* instance is created first so that every field
        has a value.  Only the keys present in the JSON file override those
        defaults.

        Model identifiers from the ``"models"`` section are stored as-is
        (repo IDs or paths). Resolution into concrete file paths happens
        in bootstrap.

        Parameters
        ----------
        path : str
            Path to the JSON configuration file.

        Returns
        -------
        Settings
            A configured *Settings* instance.
        """
        logger.info("Loading configuration from %s", path)
        with open(path, "r") as fh:
            data: dict = json.load(fh)

        instance = cls()  # start with all defaults

        # --- models -----------------------------------------------------------
        if "models" in data:
            models_data = data["models"]

            # embedding
            if "embedding" in models_data:
                emb = models_data["embedding"]
                instance.models.embedding_id = emb.get("repo_id", emb.get("path", ""))
                if "dim" in emb:
                    instance.models.embedding_dim = emb["dim"]

            # ner
            if "ner" in models_data:
                ner_data = models_data["ner"]
                instance.models.ner_id = ner_data.get("repo_id", ner_data.get("path", ""))

            # reranker
            if "reranker" in models_data:
                rerank_data = models_data["reranker"]
                instance.models.reranker_id = rerank_data.get("repo_id", rerank_data.get("path", ""))

            # observer (summarizer)
            if "observer" in models_data:
                obs_data = models_data["observer"]
                instance.summarizer.model_id = obs_data.get("repo_id", obs_data.get("path", ""))

        # --- search -------------------------------------------------------
        if "search" in data:
            search_data = data["search"]
            if "embedding_dim" in search_data:
                instance.search.embedding_dim = search_data["embedding_dim"]
            if "pipeline_width" in search_data:
                instance.search.pipeline_width = search_data["pipeline_width"]
            if "half_life_days" in search_data:
                instance.search.half_life_days = search_data["half_life_days"]

        # --- server -------------------------------------------------------
        if "server" in data:
            server_data = data["server"]
            if "host" in server_data:
                instance.server.host = server_data["host"]
            if "port" in server_data:
                instance.server.port = server_data["port"]

        # --- storage ------------------------------------------------------
        if "storage" in data:
            storage_data = data["storage"]
            if "db_path" in storage_data:
                instance.storage.db_path = os.path.expanduser(storage_data["db_path"])

        # --- observer -------------------------------------------------------
        if "observer" in data:
            observer_data = data["observer"]
            if "brief_max_length" in observer_data:
                instance.observer.brief_max_length = observer_data["brief_max_length"]
            if "summarize_threshold" in observer_data:
                instance.observer.summarize_threshold = observer_data["summarize_threshold"]

        # --- dreamer --------------------------------------------------------
        if "dreamer" in data:
            dreamer_data = data["dreamer"]
            if "enabled" in dreamer_data:
                instance.dreamer.enabled = dreamer_data["enabled"]
            if "interval_min" in dreamer_data:
                instance.dreamer.interval_min = dreamer_data["interval_min"]
            if "max_anchors_per_cycle" in dreamer_data:
                instance.dreamer.max_anchors_per_cycle = dreamer_data["max_anchors_per_cycle"]
            if "resurface_threshold" in dreamer_data:
                instance.dreamer.resurface_threshold = dreamer_data["resurface_threshold"]
            if "disk_scan_enabled" in dreamer_data:
                instance.dreamer.disk_scan_enabled = dreamer_data["disk_scan_enabled"]
            if "disk_scan_page_size" in dreamer_data:
                instance.dreamer.disk_scan_page_size = dreamer_data["disk_scan_page_size"]

        logger.info("Configuration loaded successfully from %s", path)
        return instance
