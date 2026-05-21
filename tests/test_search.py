"""Unit tests for the HTTP search endpoint and browser UI.

No Qdrant, no model loading — all store/embedder calls are mocked.
"""
from __future__ import annotations

import numpy as np
import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock

from ccmcp.metrics import make_observable_app


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def store():
    s = MagicMock()
    s.list_scopes.return_value = [
        {"name": "myrepo", "source_root": "/repos/myrepo", "tags": ["python"]},
        {"name": "", "source_root": "/repos/shared-lib", "tags": []},
    ]
    s.search.return_value = [
        {
            "text": "Hybrid search combines dense and sparse retrieval.",
            "source_uri": "file:///repos/hybrid_search.md",
            "section": "Overview",
            "source_type": "fs",
        }
    ]
    return s


@pytest.fixture
def embedder():
    e = MagicMock()
    dense = np.random.rand(1, 384).astype(np.float32)
    sparse = MagicMock()
    e.embed.return_value = (dense, [sparse])
    return e


async def _mock_mcp(scope, receive, send):
    """Minimal ASGI app standing in for the MCP SSE app."""
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"not found"})


@pytest.fixture
def client(store, embedder):
    app = make_observable_app(_mock_mcp, store=store, embedder=embedder)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def client_no_search():
    """App built without store/embedder — search routes should be absent."""
    app = make_observable_app(_mock_mcp)
    return TestClient(app, raise_server_exceptions=True)


# ── UI ───────────────────────────────────────────────────────────────────────

def test_ui_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ui_contains_search_input(client):
    r = client.get("/")
    assert 'id="query"' in r.text
    assert 'id="scope"' in r.text
    assert 'id="search-form"' in r.text


def test_ui_absent_without_store(client_no_search):
    r = client_no_search.get("/")
    assert r.status_code == 404


# ── /scopes ──────────────────────────────────────────────────────────────────

def test_scopes_returns_list(client, store):
    r = client.get("/scopes")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "myrepo"
    assert data[1]["source_root"] == "/repos/shared-lib"


def test_scopes_absent_without_store(client_no_search):
    r = client_no_search.get("/scopes")
    assert r.status_code == 404


def test_scopes_error_returns_500(client, store):
    store.list_scopes.side_effect = RuntimeError("db error")
    r = client.get("/scopes")
    assert r.status_code == 500


# ── /search ──────────────────────────────────────────────────────────────────

def test_search_returns_results(client, store, embedder):
    r = client.post("/search", json={"query": "hybrid search"})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "hybrid search"
    assert data["count"] == 1
    assert data["results"][0]["source_uri"] == "file:///repos/hybrid_search.md"
    embedder.embed.assert_called_once_with(["hybrid search"])


def test_search_empty_query_returns_400(client):
    r = client.post("/search", json={"query": ""})
    assert r.status_code == 400
    assert "query" in r.json()["detail"]


def test_search_missing_query_returns_400(client):
    r = client.post("/search", json={})
    assert r.status_code == 400


def test_search_invalid_json_returns_400(client):
    r = client.post("/search", content=b"not-json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_search_with_explicit_scope(client, store, embedder):
    r = client.post("/search", json={"query": "dense vectors", "scope": ["myrepo"]})
    assert r.status_code == 200
    # store.search should be called with a non-None filter
    call_kwargs = store.search.call_args
    assert call_kwargs.kwargs.get("filter") is not None or call_kwargs.args[2] is not None


def test_search_scope_wildcard_sends_no_filter(client, store):
    r = client.post("/search", json={"query": "anything", "scope": ["*"]})
    assert r.status_code == 200
    call_kwargs = store.search.call_args
    sent_filter = (
        call_kwargs.kwargs.get("filter")
        if call_kwargs.kwargs
        else call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )
    assert sent_filter is None


def test_search_limit_capped_at_50(client, store, embedder):
    r = client.post("/search", json={"query": "anything", "limit": 9999})
    assert r.status_code == 200
    _, kwargs = store.search.call_args
    assert kwargs.get("limit", 0) <= 50


def test_search_absent_without_store(client_no_search):
    r = client_no_search.post("/search", json={"query": "test"})
    assert r.status_code in (404, 405)


def test_search_no_results(client, store):
    store.search.return_value = []
    r = client.post("/search", json={"query": "xyzzy"})
    assert r.status_code == 200
    assert r.json()["count"] == 0
    assert r.json()["results"] == []


# ── /metrics still works when search is present ──────────────────────────────

def test_metrics_endpoint_unaffected(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ccmcp_" in r.text or "python" in r.text
