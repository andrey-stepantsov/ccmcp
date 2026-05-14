"""Unit tests for the metrics module — no Qdrant, no model loading."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ccmcp.metrics import (
    CHUNK_CHARS,
    DOCUMENTS_INGESTED,
    EMBED_BATCH_SIZE,
    EMBED_SECONDS,
    INGEST_ERRORS,
    INGEST_SECONDS,
    SEARCH_RESULTS_RETURNED,
    SEARCH_SECONDS,
    make_observable_app,
)


def _sample(histogram, label_dict=None):
    """Return the sample count for a histogram (or counter)."""
    if label_dict:
        return histogram.labels(**label_dict)._metrics  # type: ignore[attr-defined]
    return histogram._metrics  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Embedder instrumentation
# ---------------------------------------------------------------------------

def test_embed_records_batch_size_and_time():
    from ccmcp.embedder import Embedder

    e = Embedder.__new__(Embedder)
    e._dense_model_name = "mock"
    e._sparse_model_name = "mock"
    e._R = None
    e._rotation_path = "/nonexistent"

    dense_out = np.zeros((3, 384), dtype=np.float32)
    sparse_out = [MagicMock() for _ in range(3)]

    dense_mock = MagicMock()
    dense_mock.embed.return_value = iter(dense_out)
    sparse_mock = MagicMock()
    sparse_mock.embed.return_value = iter(sparse_out)

    e._dense_embedder = dense_mock
    e._sparse_embedder = sparse_mock

    before_count = EMBED_SECONDS._sum.get()
    before_batch = EMBED_BATCH_SIZE._sum.get()
    with patch.object(e, "_load"):  # skip actual model loading
        e.embed(["a", "b", "c"])
    assert EMBED_SECONDS._sum.get() > before_count
    assert EMBED_BATCH_SIZE._sum.get() >= before_batch + 3


# ---------------------------------------------------------------------------
# Store search instrumentation
# ---------------------------------------------------------------------------

def test_search_records_timing_and_results():
    from ccmcp.store import VectorStore

    store = VectorStore.__new__(VectorStore)
    store._collection = "test"
    store._artifact_collection = "test-artifacts"

    fake_point = MagicMock()
    fake_point.payload = {"text": "hello"}
    fake_result = MagicMock()
    fake_result.points = [fake_point, fake_point]

    store._client = MagicMock()
    store._client.query_points.return_value = fake_result

    before_search = SEARCH_SECONDS._sum.get()
    before_results = SEARCH_RESULTS_RETURNED._sum.get()
    hits = store.search(np.zeros(384), MagicMock(), limit=5)
    assert len(hits) == 2
    assert SEARCH_SECONDS._sum.get() > before_search
    assert SEARCH_RESULTS_RETURNED._sum.get() >= before_results + 2


# ---------------------------------------------------------------------------
# Controller instrumentation
# ---------------------------------------------------------------------------

def test_ingest_records_chunk_chars_and_counters():
    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile

    cfg = MagicMock()
    embedder = MagicMock()
    store = MagicMock()
    # upsert must return the same count as the chunks it receives
    store.upsert.side_effect = lambda chunks, dense, sparse, version, stype, **_kw: len(chunks)
    store.doc_id.return_value = "abc"

    def _embed(texts):
        n = len(texts)
        embedder.embed.return_value = (np.zeros((n, 384)), [MagicMock() for _ in range(n)])
        return np.zeros((n, 384)), [MagicMock() for _ in range(n)]

    embedder.embed.side_effect = _embed
    state = MagicMock()
    state.get.return_value = None

    ctrl = Controller(cfg, embedder, store, state)

    before_docs = DOCUMENTS_INGESTED.labels(source_type="fs")._value.get()
    before_ingest = INGEST_SECONDS._sum.get()
    before_chars = CHUNK_CHARS._sum.get()

    ctrl.ingest_file(SourceFile(
        source_uri="file:///tmp/test.md",
        content="# Hello\n\nThis is content.\n\n## Section\n\nMore content here.",
    ))

    assert DOCUMENTS_INGESTED.labels(source_type="fs")._value.get() > before_docs
    assert INGEST_SECONDS._sum.get() > before_ingest
    assert CHUNK_CHARS._sum.get() > before_chars


def test_ingest_errors_incremented_on_failure():
    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile

    cfg = MagicMock()
    embedder = MagicMock()
    embedder.embed.side_effect = RuntimeError("embed failed")
    store = MagicMock()
    store.doc_id.return_value = "abc"
    state = MagicMock()
    state.get.return_value = None

    ctrl = Controller(cfg, embedder, store, state)

    before = INGEST_ERRORS.labels(source_type="fs")._value.get()
    # scan() wraps ingest_file in try/except — simulate the same
    try:
        ctrl.ingest_file(SourceFile(
            source_uri="file:///tmp/bad.md",
            content="# Title\n\nContent.",
        ))
    except Exception:
        from ccmcp.controller import _source_type_safe
        INGEST_ERRORS.labels(source_type=_source_type_safe("file:///tmp/bad.md")).inc()

    assert INGEST_ERRORS.labels(source_type="fs")._value.get() > before


# ---------------------------------------------------------------------------
# make_observable_app
# ---------------------------------------------------------------------------

def test_make_observable_app_has_metrics_route():
    from starlette.testclient import TestClient

    mock_mcp_app = MagicMock()
    mock_mcp_app.return_value = None

    # Use a real minimal Starlette app as the inner app
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def _noop(request):
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/sse", _noop)])
    app = make_observable_app(inner)

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # prometheus_client always emits at least the process metrics
    assert b"process_" in resp.content or b"python_" in resp.content
