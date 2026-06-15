# SPDX-License-Identifier: MIT

"""Tests for the memory library API routes.

Verifies that the /inject endpoint correctly includes intuition_signals
from the ExperienceIndex in its JSON response.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from memory_lib.api.routes import IntuitionSignal, inject, _map_knn_results
from memory_lib.memory.session_index import SessionBrief
from memory_lib.subconscious.anchor import Anchor


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_ctx():
    """Create a mock MemoryContext with all required attributes."""
    ctx = MagicMock()
    ctx.embedder = AsyncMock()
    ctx.embedder.aencode = AsyncMock(return_value=np.array([0.1] * 384))
    ctx.session_index = MagicMock()
    ctx.session_index.knn_query = MagicMock(return_value=([1, 2], [0.1, 0.2]))
    ctx.reranker = None
    ctx.config = MagicMock()
    ctx.config.search.inject_min_similarity = 0.0
    ctx.config.search.inject_rerank = False
    ctx.persistence = AsyncMock()
    ctx.persistence.get_session_by_id = AsyncMock(return_value=None)
    ctx.anchor_index = MagicMock()
    ctx.anchor_index.get = MagicMock(return_value=None)
    return ctx


@pytest.fixture
def sample_briefs():
    """Session briefs for testing."""
    return {
        "sess_001": SessionBrief(
            session_id="sess_001",
            brief="Refactored database schema",
            tags=["database", "sqlite"],
            importance="critical",
            score=0.9,
            resolution=0.5,
            created_at=1000,
        ),
        "sess_002": SessionBrief(
            session_id="sess_002",
            brief="Fixed authentication bug",
            tags=["auth", "security"],
            importance="important",
            score=0.7,
            resolution=0.3,
            created_at=2000,
        ),
    }


# ── IntuitionSignal model tests ────────────────────────────────────────

class TestIntuitionSignalModel:
    """Test the IntuitionSignal Pydantic model."""

    def test_create_intuition_signal(self):
        signal = IntuitionSignal(
            type="TENSION",
            tag="coding",
            message="Topic has contradictions",
        )
        assert signal.type == "TENSION"
        assert signal.tag == "coding"
        assert signal.message == "Topic has contradictions"

    def test_intuition_signal_serialization(self):
        signal = IntuitionSignal(type="DO_THIS", tag="cli", message="Verified pattern")
        data = signal.model_dump()
        assert data == {"type": "DO_THIS", "tag": "cli", "message": "Verified pattern"}

    def test_intuition_signal_defaults(self):
        signal = IntuitionSignal(type="", tag="", message="")
        assert signal.type == ""
        assert signal.tag == ""
        assert signal.message == ""


# ── _map_knn_results tests ─────────────────────────────────────────────

class TestMapKnnResults:
    """Test _map_knn_results helper function."""

    @pytest.mark.asyncio
    async def test_map_knn_results_basic(self, mock_ctx, sample_briefs):
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}

        results, total = await _map_knn_results(
            mock_ctx, [1, 2], [0.1, 0.2],
            min_similarity=0.0,
            include_created_at=False,
            include_key_facts=False,
        )
        assert total == 2
        assert len(results) == 2
        assert results[0]["session_id"] == "sess_001"
        assert results[0]["similarity"] == pytest.approx(0.9)
        assert results[1]["session_id"] == "sess_002"

    @pytest.mark.asyncio
    async def test_map_knn_results_min_similarity_filter(self, mock_ctx, sample_briefs):
        """Results below min_similarity are excluded."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}

        results, total = await _map_knn_results(
            mock_ctx, [1, 2], [0.1, 0.2],
            min_similarity=0.15,  # 0.2 similarity threshold
            include_created_at=False,
            include_key_facts=False,
        )
        # 1 - 0.2 = 0.8 >= 0.15 ✓, but 1 - 0.1 = 0.9 >= 0.15 ✓
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_map_knn_results_strict_min_similarity(self, mock_ctx, sample_briefs):
        """Higher min_similarity filters out more results."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}

        results, total = await _map_knn_results(
            mock_ctx, [1, 2], [0.1, 0.2],
            min_similarity=0.95,
            include_created_at=False,
            include_key_facts=False,
        )
        # 0.9 < 0.95 and 0.8 < 0.95 → both filtered out
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_map_knn_results_includes_key_facts(self, mock_ctx, sample_briefs):
        """When include_key_facts=True, anchor key_facts are included."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}

        anchor = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Refactored database schema",
            anchor_type="decision",
            key_facts=[
                {"type": "code", "value": "Replaced sync with aiosqlite", "score": 0.9, "priority": 1},
                {"type": "config", "value": "Set journal mode to WAL", "score": 0.8, "priority": 2},
            ],
        )
        mock_ctx.anchor_index.get = MagicMock(side_effect=lambda sid: anchor if sid == "sess_001" else None)

        results, total = await _map_knn_results(
            mock_ctx, [1, 2], [0.1, 0.2],
            min_similarity=0.0,
            include_created_at=False,
            include_key_facts=True,
        )
        assert len(results) == 2
        assert results[0]["session_id"] == "sess_001"
        assert "key_facts" in results[0]
        assert len(results[0]["key_facts"]) == 2
        assert results[0]["key_facts"][0] == "Replaced sync with aiosqlite"
        assert results[0]["key_facts"][1] == "Set journal mode to WAL"
        # sess_002 has no anchor → empty key_facts
        assert results[1]["key_facts"] == []

    @pytest.mark.asyncio
    async def test_map_knn_results_includes_created_at(self, mock_ctx, sample_briefs):
        """When include_created_at=True, created_at is included."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1}

        results, total = await _map_knn_results(
            mock_ctx, [1], [0.1],
            min_similarity=0.0,
            include_created_at=True,
            include_key_facts=False,
        )
        assert len(results) == 1
        assert results[0]["created_at"] == 1000

    @pytest.mark.asyncio
    async def test_map_knn_results_missing_label(self, mock_ctx, sample_briefs):
        """Label not in _sid_to_id → skipped."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1}

        results, total = await _map_knn_results(
            mock_ctx, [99], [0.1],
            min_similarity=0.0,
            include_created_at=False,
            include_key_facts=False,
        )
        assert len(results) == 0
        assert total == 1

    @pytest.mark.asyncio
    async def test_map_knn_results_session_not_in_ram(self, mock_ctx):
        """Session evicted from RAM → cold-load via persistence."""
        mock_ctx.session_index._sid_to_id = {"evicted_sess": 1}
        cold_loaded = SessionBrief(
            session_id="evicted_sess",
            brief="Cold loaded",
            tags=["cold"],
            importance="background",
            score=0.5,
            resolution=0.5,
            created_at=500,
        )
        ctx = MagicMock()
        ctx.session_index = mock_ctx.session_index
        ctx.ram_index = {}  # empty RAM
        ctx.persistence.get_session_by_id = AsyncMock(return_value=cold_loaded)
        ctx.anchor_index.get = MagicMock(return_value=None)

        results, total = await _map_knn_results(
            ctx, [1], [0.1],
            min_similarity=0.0,
            include_created_at=False,
            include_key_facts=False,
        )
        assert len(results) == 1
        assert results[0]["session_id"] == "evicted_sess"


# ── /inject endpoint tests ─────────────────────────────────────────────

class TestInjectEndpoint:
    """Test /inject endpoint intuition_signals integration."""

    @pytest.mark.asyncio
    async def test_inject_returns_503_when_ctx_none(self):
        """When ctx is None, endpoint returns 503 error."""
        from fastapi import Request
        from fastapi.datastructures import State
        from memory_lib.api.routes import InjectRequest

        mock_request = MagicMock(spec=Request)
        mock_request.app.state = State()
        mock_request.app.state.ctx = None

        inject_req = InjectRequest(text="test query", top_k=10)
        response = await inject(inject_req, mock_request)

        assert response.status_code == 503
        assert response.body == b'{"error":"memory system not ready"}'

    @pytest.mark.asyncio
    async def test_inject_empty_results_no_experience_tags(self, mock_ctx, sample_briefs):
        """When results are empty, response contains intuition_signals: []."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.session_index.knn_query = MagicMock(return_value=([], []))

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "intuition_signals" in response
        assert response["intuition_signals"] == []
        assert response["results"] == []

    @pytest.mark.asyncio
    async def test_inject_backward_compatible_no_signals(self, mock_ctx, sample_briefs):
        """When ExperienceIndex returns no signals, response still contains intuition_signals: []."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "intuition_signals" in response
        assert response["intuition_signals"] == []
        assert "results" in response

    @pytest.mark.asyncio
    async def test_inject_returns_tension_signal(self, mock_ctx, sample_briefs):
        """When a tag has high conflict rate, response contains TENSION signal."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[
            {"type": "TENSION", "tag": "database", "message": "Topic 'database' has unresolved contradictions (5 conflicts)."},
        ])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "intuition_signals" in response
        assert len(response["intuition_signals"]) == 1
        assert response["intuition_signals"][0]["type"] == "TENSION"
        assert response["intuition_signals"][0]["tag"] == "database"

    @pytest.mark.asyncio
    async def test_inject_returns_do_this_signal(self, mock_ctx, sample_briefs):
        """When a tag has expert maturity and high avg_score, response contains DO_THIS."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[
            {"type": "DO_THIS", "tag": "auth", "message": "'auth' is a verified pattern (40 sessions, score 0.95)."},
        ])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "intuition_signals" in response
        assert len(response["intuition_signals"]) == 1
        assert response["intuition_signals"][0]["type"] == "DO_THIS"

    @pytest.mark.asyncio
    async def test_inject_collects_unique_tags_from_results(self, mock_ctx, sample_briefs):
        """Tags from multiple results are deduplicated before querying ExperienceIndex."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        await inject(inject_req, mock_request)

        # Verify intuition_signals was called with unique tags from both sessions
        # Both sessions contribute tags: database, sqlite, auth, security
        mock_ctx.experience_index.intuition_signals.assert_called_once()
        called_tags = mock_ctx.experience_index.intuition_signals.call_args[0][0]
        assert set(called_tags) == {"database", "sqlite", "auth", "security"}

    @pytest.mark.asyncio
    async def test_inject_includes_key_facts_in_results(self, mock_ctx, sample_briefs):
        """Results from /inject include key_facts from anchors."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}

        anchor1 = Anchor(
            anchor_id="sess_001",
            session_id="sess_001",
            brief="Refactored database schema",
            anchor_type="decision",
            key_facts=[
                {"type": "code", "value": "Replaced sync with aiosqlite", "score": 0.9, "priority": 1},
            ],
        )
        mock_ctx.anchor_index.get = MagicMock(side_effect=lambda sid: anchor1 if sid == "sess_001" else None)
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "results" in response
        assert len(response["results"]) >= 1
        # First result (sess_001) should have key_facts
        sess001_result = next(r for r in response["results"] if r["session_id"] == "sess_001")
        assert "key_facts" in sess001_result
        assert "Replaced sync with aiosqlite" in sess001_result["key_facts"]

    @pytest.mark.asyncio
    async def test_inject_response_has_all_required_fields(self, mock_ctx, sample_briefs):
        """Response contains all expected top-level fields."""
        mock_ctx.ram_index = sample_briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert "results" in response
        assert "total_candidates" in response
        assert "intuition_signals" in response
        assert isinstance(response["results"], list)
        assert isinstance(response["intuition_signals"], list)
        assert isinstance(response["total_candidates"], int)

    @pytest.mark.asyncio
    async def test_inject_embedder_unavailable(self, mock_ctx):
        """When embedder is None, returns warning and empty intuition_signals."""
        mock_ctx.embedder = None
        mock_ctx.experience_index = MagicMock()

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        response = await inject(inject_req, mock_request)

        assert response["results"] == []
        assert response["total_candidates"] == 0
        assert response.get("warning") == "embedder_unavailable"
        assert response["intuition_signals"] == []

    @pytest.mark.asyncio
    async def test_inject_caps_results_at_10(self, mock_ctx):
        """Without reranker, results capped at 10."""
        mock_ctx.ram_index = {}
        mock_ctx.session_index._sid_to_id = {f"s{i}": i for i in range(20)}

        # Create 20 session briefs
        briefs = {}
        for i in range(20):
            briefs[f"s{i}"] = SessionBrief(
                session_id=f"s{i}",
                brief=f"Brief {i}",
                tags=[f"tag{i}"],
                importance="background",
                score=0.5,
                resolution=0.5,
                created_at=i,
            )
        mock_ctx.ram_index = briefs

        # Return 20 results with decreasing similarity
        labels = list(range(20))
        distances = [0.0 + i * 0.01 for i in range(20)]  # 0.0, 0.01, ..., 0.19
        mock_ctx.session_index.knn_query = MagicMock(return_value=(labels, distances))
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=20)

        response = await inject(inject_req, mock_request)

        assert len(response["results"]) <= 10

    @pytest.mark.asyncio
    async def test_inject_tag_deduplication(self, mock_ctx):
        """Duplicate tags across sessions are deduplicated before querying experience index."""
        briefs = {
            "sess_001": SessionBrief(
                session_id="sess_001", brief="A", tags=["common", "unique_a"],
                importance="background", score=0.5, resolution=0.5, created_at=1,
            ),
            "sess_002": SessionBrief(
                session_id="sess_002", brief="B", tags=["common", "unique_b"],
                importance="background", score=0.5, resolution=0.5, created_at=2,
            ),
        }
        mock_ctx.ram_index = briefs
        mock_ctx.session_index._sid_to_id = {"sess_001": 1, "sess_002": 2}
        mock_ctx.experience_index = MagicMock()
        mock_ctx.experience_index.intuition_signals = MagicMock(return_value=[])

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.ctx = mock_ctx

        from memory_lib.api.routes import InjectRequest
        inject_req = InjectRequest(text="test query", top_k=10)

        await inject(inject_req, mock_request)

        called_tags = mock_ctx.experience_index.intuition_signals.call_args[0][0]
        # "common" appears in both sessions but should only be passed once
        assert called_tags.count("common") == 1
        assert set(called_tags) == {"common", "unique_a", "unique_b"}
