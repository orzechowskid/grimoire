# SPDX-License-Identifier: MIT

"""Standalone tests for the memory library API.

These tests verify that the memory library works as a complete system
by exercising the FastAPI endpoints via TestClient (no running server needed).

Key constraints:
- Model files (ONNX embeddings, NER, reranker) are not present in this
  environment, so embedder/ner/reranker will be "unavailable".
- The observe endpoint handles missing models gracefully.
- The inject/search endpoints return empty results when embedder is unavailable.

Run with:
    /home/dan/repos/grimoire/venv/bin/python -m pytest src/memory_lib/test_standalone.py -v
"""

import pytest
from fastapi.testclient import TestClient

from memory_lib.api.routes import app


# ---------------------------------------------------------------------------
# Fixture: shared TestClient instance
# ---------------------------------------------------------------------------
# We use a module-level fixture so that bootstrap/shutdown happens once
# per test module, which is much faster than per-test.
#
# Note: TestClient triggers the lifespan (bootstrap on enter, shutdown on
# exit), so each test run boots and tears down the memory system once.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers bootstrap/shutdown once."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Test: GET /health
# ---------------------------------------------------------------------------
def test_health(client):
    """Health endpoint returns system status.

    Since model files are not present, embedder/ner/reranker will report
    'unavailable', but the system should still report 'ok' status after
    successful bootstrap.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    # Should have operational fields
    assert "sessions_in_ram" in data
    assert "anchors_in_ram" in data
    assert "experience_clusters" in data
    assert "embedder" in data
    assert "ner" in data
    assert "reranker" in data


# ---------------------------------------------------------------------------
# Test: POST /observe with explicit session_id
# ---------------------------------------------------------------------------
def test_observe_creates_session(client):
    """Observe endpoint processes text and stores a session.

    Verifies that:
    - The response contains the provided session_id.
    - The session is not discarded.
    - A brief summary is generated.
    """
    resp = client.post(
        "/observe",
        json={"text": "The quick brown fox jumps over the lazy dog", "session_id": "test_session_abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test_session_abc123"
    assert data["discarded"] is False
    assert data["brief"] == "The quick brown fox jumps over the lazy dog"
    assert isinstance(data["tags"], list)
    assert isinstance(data["importance"], str)
    assert isinstance(data["score"], float)


# ---------------------------------------------------------------------------
# Test: POST /observe auto-generates session_id
# ---------------------------------------------------------------------------
def test_observe_auto_session_id(client):
    """Observe endpoint auto-generates session_id when not provided."""
    resp = client.post(
        "/observe",
        json={"text": "Testing auto session ID generation"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] is not None
    assert len(data["session_id"]) > 0
    assert data["discarded"] is False


# ---------------------------------------------------------------------------
# Test: POST /observe with empty text
# ---------------------------------------------------------------------------
def test_observe_empty_text(client):
    """Observe endpoint handles empty text gracefully."""
    resp = client.post(
        "/observe",
        json={"text": "", "session_id": "empty_test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Empty text may or may not be discarded depending on the pipeline
    # Either way, it should return a valid response
    assert "session_id" in data
    assert "discarded" in data


# ---------------------------------------------------------------------------
# Test: POST /search with embedder unavailable
# ---------------------------------------------------------------------------
def test_search_embedder_unavailable(client):
    """Search returns empty results when embedder is not available."""
    resp = client.post(
        "/search",
        json={"query": "test query", "top_k": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    # When embedder is unavailable, search returns early with empty results
    assert data["results"] == []
    assert data["total_candidates"] == 0


# ---------------------------------------------------------------------------
# Test: POST /inject with embedder unavailable
# ---------------------------------------------------------------------------
def test_inject_embedder_unavailable(client):
    """Inject returns empty results with warning when embedder is unavailable."""
    resp = client.post(
        "/inject",
        json={"text": "test text", "top_k": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["total_candidates"] == 0
    assert "warning" in data
    assert data["warning"] == "embedder_unavailable"


# ---------------------------------------------------------------------------
# Test: POST /inject basic structure
# ---------------------------------------------------------------------------
def test_inject_response_structure(client):
    """Inject endpoint returns proper response structure."""
    resp = client.post(
        "/inject",
        json={"text": "test", "top_k": 5, "time_weighted": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_candidates" in data


# ---------------------------------------------------------------------------
# Test: POST /search basic structure
# ---------------------------------------------------------------------------
def test_search_response_structure(client):
    """Search endpoint returns proper response structure."""
    resp = client.post(
        "/search",
        json={"query": "test query", "top_k": 5, "min_similarity": 0.0, "time_weighted": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_candidates" in data


# ---------------------------------------------------------------------------
# Test: End-to-end observe -> search flow
# ---------------------------------------------------------------------------
def test_observe_and_search(client):
    """End-to-end: observe a memory, then search for it.

    Since the embedder is unavailable:
    - Observe should succeed and store the session.
    - Search should return empty (no embedding to search with).

    When the embedder IS available, the observed memory would be retrievable
    via search. This test documents the current expected behavior.
    """
    # Step 1: Observe some text
    resp1 = client.post(
        "/observe",
        json={
            "text": "I implemented a FastAPI server with async endpoints",
            "session_id": "e2e_test_session",
        },
    )
    assert resp1.status_code == 200
    observe_data = resp1.json()
    assert observe_data["session_id"] == "e2e_test_session"
    assert observe_data["discarded"] is False

    # Step 2: Search for the observed text
    # With embedder unavailable, search returns empty results
    # This is the expected behavior when no embedding model is available
    resp2 = client.post(
        "/search",
        json={"query": "FastAPI server async", "top_k": 10},
    )
    assert resp2.status_code == 200
    search_data = resp2.json()
    # Search returns empty because no embedder to compute query vector
    assert search_data["results"] == []
    assert search_data["total_candidates"] == 0


# ---------------------------------------------------------------------------
# Test: Full lifecycle — inject user message + observe agent response
# ---------------------------------------------------------------------------
def test_full_lifecycle_inject_observe_search(client):
    """Full lifecycle: inject user message, observe agent response, search, verify.

    This is the core Step 10 test that exercises the complete flow:
    1. Start the memory library (via TestClient lifespan).
    2. Send a user message for injection via /inject.
    3. Send an agent response for observation via /observe.
    4. Search for the stored memory via /search.
    5. Verify memory was stored (via health endpoint and/or search results).
    """
    # Step 1: Memory library is already started by the client fixture.
    # Verify it's healthy.
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"

    # Step 2: Inject a user message (memory context injection).
    # Since embedder is unavailable, this returns empty results with a warning.
    inject_resp = client.post(
        "/inject",
        json={"text": "What is the capital of France?", "top_k": 5},
    )
    assert inject_resp.status_code == 200
    inject_data = inject_resp.json()
    assert "results" in inject_data
    assert "total_candidates" in inject_data
    assert "warning" in inject_data
    assert inject_data["warning"] == "embedder_unavailable"

    # Step 3: Observe an agent response (feed it to the memory system).
    observe_resp = client.post(
        "/observe",
        json={
            "text": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            "session_id": "lifecycle_test_session",
        },
    )
    assert observe_resp.status_code == 200
    observe_data = observe_resp.json()
    assert observe_data["session_id"] == "lifecycle_test_session"
    assert observe_data["discarded"] is False
    assert "brief" in observe_data
    assert "tags" in observe_data
    assert "importance" in observe_data
    assert "score" in observe_data

    # Step 4: Search for the stored memory.
    search_resp = client.post(
        "/search",
        json={"query": "capital France Paris", "top_k": 10},
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert "results" in search_data
    assert "total_candidates" in search_data

    # With embedder unavailable, search returns empty — this is expected.
    # The important thing is the search was exercised without errors.
    assert search_data["results"] == []
    assert search_data["total_candidates"] == 0

    # Step 5: Verify memory was stored by checking the health endpoint.
    health_resp2 = client.get("/health")
    assert health_resp2.status_code == 200
    health_data2 = health_resp2.json()
    assert health_data2["status"] == "ok"
    # At least one session should be in RAM (the observe call we just made)
    assert health_data2["sessions_in_ram"] >= 1


# ---------------------------------------------------------------------------
# Test: Multiple observe calls accumulate sessions
# ---------------------------------------------------------------------------
def test_multiple_observes(client):
    """Multiple observe calls create multiple sessions."""
    for i in range(5):
        resp = client.post(
            "/observe",
            json={
                "text": f"Test message number {i}",
                "session_id": f"multi_test_{i}",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == f"multi_test_{i}"
        assert data["discarded"] is False


# ---------------------------------------------------------------------------
# Test: Health shows sessions in RAM after observes
# ---------------------------------------------------------------------------
def test_health_shows_sessions_after_observe(client):
    """Health endpoint reflects sessions added via observe.

    Note: This test uses the same client fixture as other tests,
    so sessions from previous tests in the same module may be present.
    """
    # First, observe a session
    client.post(
        "/observe",
        json={
            "text": "Health check session",
            "session_id": "health_test_session",
        },
    )

    # Check health
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessions_in_ram"] >= 1  # At least the session we just added


# ---------------------------------------------------------------------------
# Test: POST /shutdown returns correct response
# ---------------------------------------------------------------------------
def test_shutdown_endpoint(client):
    """Shutdown endpoint returns shutting_down status."""
    resp = client.post("/shutdown")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "shutting_down"
