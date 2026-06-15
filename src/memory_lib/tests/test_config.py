# SPDX-License-Identifier: MIT

"""Tests for the Settings configuration system.

Verifies that Settings() instantiates with correct defaults and that
from_json_file() correctly loads and overrides configuration values from
JSON files on disk.
"""

import json
import os

import pytest

from memory_lib.config import Settings


# ── TestSettingsDefaults ───────────────────────────────────────────────

class TestSettingsDefaults:
    """Verify default values on a bare Settings instance."""

    def test_default_embedding_dim(self):
        assert Settings().models.embedding_dim == 384

    def test_default_server(self):
        settings = Settings()
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8766

    def test_default_db_path(self):
        assert Settings().storage.db_path == "memory.db"

    def test_default_score_weights(self):
        s = Settings().score
        assert s.weight_relevance == 0.5
        assert s.weight_temporal == 0.3
        assert s.weight_importance == 0.2

    def test_all_sub_configs_exist(self):
        s = Settings()
        assert s.models is not None
        assert s.storage is not None
        assert s.score is not None
        assert s.importance is not None
        assert s.temporal is not None
        assert s.dissolver is not None
        assert s.anchor_decay is not None
        assert s.dreamer is not None
        assert s.search is not None
        assert s.server is not None
        assert s.observer is not None
        assert s.summarizer is not None


# ── TestFromJsonFileFull ───────────────────────────────────────────────

class TestFromJsonFileFull:
    """Test loading a complete configuration file."""

    def test_loads_full_config(self, tmp_path):
        config = {
            "models": {
                "embedding": {"repo_id": "my-model", "dim": 512},
                "ner": {"repo_id": "my-ner"},
                "reranker": {"repo_id": "my-reranker"},
                "observer": {"repo_id": "my-observer"},
            },
            "server": {"host": "0.0.0.0", "port": 9999},
            "storage": {"db_path": "/custom/path.db"},
            "search": {"embedding_dim": 512, "pipeline_width": 8, "half_life_days": 60},
            "observer": {"brief_max_length": 2000},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        assert settings.models.embedding_id == "my-model"
        assert settings.models.embedding_dim == 512
        assert settings.models.ner_id == "my-ner"
        assert settings.models.reranker_id == "my-reranker"
        assert settings.summarizer.model_id == "my-observer"
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 9999
        assert settings.storage.db_path == "/custom/path.db"
        assert settings.search.embedding_dim == 512
        assert settings.search.pipeline_width == 8
        assert settings.search.half_life_days == 60.0
        assert settings.observer.brief_max_length == 2000


# ── TestFromJsonFilePartial ────────────────────────────────────────────

class TestFromJsonFilePartial:
    """Test that missing sections fall back to defaults."""

    def test_partial_config_uses_defaults_for_missing_sections(self, tmp_path):
        config = {"server": {"port": 9999}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        assert settings.server.port == 9999
        assert settings.models.embedding_dim == 384
        assert settings.storage.db_path == "memory.db"

    def test_empty_json_uses_all_defaults(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({}))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        assert settings.server.port == 8766
        assert settings.models.embedding_dim == 384

    def test_partial_models_only_embedding(self, tmp_path):
        config = {
            "models": {
                "embedding": {"repo_id": "my-model"},
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        assert settings.models.embedding_id == "my-model"
        assert settings.models.ner_id == ""


# ── TestFromJsonFileStorage ────────────────────────────────────────────

class TestFromJsonFileStorage:
    """Test storage-specific behaviour."""

    def test_db_path_expands_tilde(self, tmp_path):
        config = {"storage": {"db_path": "~/.grimoire/memory.db"}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        expected = os.path.expanduser("~/.grimoire/memory.db")
        assert settings.storage.db_path == expected


# ── TestFromJsonFileDreamer ────────────────────────────────────────────

class TestFromJsonFileDreamer:
    """Test dreamer configuration loading."""

    def test_dreamer_settings_from_json(self, tmp_path):
        config = {
            "dreamer": {
                "enabled": False,
                "interval_min": 30,
                "max_anchors_per_cycle": 50,
                "resurface_threshold": 5,
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        settings = Settings.from_json_file(str(tmp_path / "config.json"))

        assert settings.dreamer.enabled is False
        assert settings.dreamer.interval_min == 30
        assert settings.dreamer.max_anchors_per_cycle == 50
        assert settings.dreamer.resurface_threshold == 5


# ── TestFromJsonFileEdgeCases ──────────────────────────────────────────

class TestFromJsonFileEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Settings.from_json_file(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises_json_decode_error(self, tmp_path):
        (tmp_path / "config.json").write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            Settings.from_json_file(str(tmp_path / "config.json"))
