from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from ccmcp.chunker import Chunk
from ccmcp.store import VectorStore

_TEST_COLLECTION = "ccmcp-test-store"
_TEST_ARTIFACTS = "ccmcp-test-artifacts"


def _sparse(indices=(1, 2), values=(0.5, 0.3)):
    return SparseEmbedding(
        indices=np.array(indices, dtype=np.int32),
        values=np.array(values, dtype=np.float32),
    )


def _chunks(n=3, uri="file:///test.md"):
    return [Chunk(text=f"chunk {i}", section="", chunk_index=i, source_uri=uri) for i in range(n)]


def _dense(n=3, dim=384):
    return np.random.randn(n, dim).astype(np.float32)


def _mock_store():
    store = VectorStore.__new__(VectorStore)
    store._client = MagicMock()
    store._collection = _TEST_COLLECTION
    store._artifact_collection = _TEST_ARTIFACTS
    return store


# ---------------------------------------------------------------------------
# Unit tests (no Qdrant required)
# ---------------------------------------------------------------------------

def test_doc_id_is_deterministic():
    store = _mock_store()
    assert store.doc_id("file:///a.md") == store.doc_id("file:///a.md")
    assert store.doc_id("file:///a.md") != store.doc_id("file:///b.md")


def test_upsert_empty_returns_zero():
    store = _mock_store()
    result = store.upsert([], np.array([]), [], version=1, source_type="fs")
    assert result == 0
    store._client.upsert.assert_not_called()


def test_store_artifact_metadata_cannot_overwrite_system_fields():
    store = _mock_store()
    dense = np.zeros(384, dtype=np.float32)

    store.store_artifact(
        text="real text",
        dense=dense,
        sparse=_sparse(),
        session_id="s1",
        metadata={"title": "ok", "text": "INJECTED", "source_type": "hacked"},
    )

    call_args = store._client.upsert.call_args
    payload = call_args.kwargs["points"][0].payload
    assert payload["text"] == "real text"
    assert payload["source_type"] == "artifact"
    assert payload["session_id"] == "s1"


def test_store_artifact_uses_artifact_collection():
    store = _mock_store()
    store.store_artifact(
        text="note",
        dense=np.zeros(384, dtype=np.float32),
        sparse=_sparse(),
        session_id="",
        metadata={},
    )
    call_args = store._client.upsert.call_args
    assert call_args.kwargs["collection_name"] == _TEST_ARTIFACTS


def test_upsert_returns_point_count():
    store = _mock_store()
    chunks = _chunks(3)
    dense = _dense(3)
    sparse = [_sparse() for _ in range(3)]
    result = store.upsert(chunks, dense, sparse, version=1, source_type="fs")
    assert result == 3


# ---------------------------------------------------------------------------
# Integration tests (require Qdrant on localhost:6333)
# ---------------------------------------------------------------------------

@pytest.fixture
def istore():
    s = VectorStore(
        "http://localhost:6333",
        _TEST_COLLECTION,
        artifact_collection=_TEST_ARTIFACTS,
    )
    try:
        s.setup(384)
    except Exception:
        pytest.skip("Qdrant not available at localhost:6333")
    yield s
    for name in (_TEST_COLLECTION, _TEST_ARTIFACTS):
        try:
            s._client.delete_collection(name)
        except Exception:
            pass


@pytest.mark.integration
def test_integration_upsert_and_search(istore):
    chunks = _chunks(2, uri="file:///search.md")
    dense = _dense(2)
    sparse = [_sparse() for _ in range(2)]
    count = istore.upsert(chunks, dense, sparse, version=1, source_type="fs")
    assert count == 2

    results = istore.search(dense[0], sparse[0], limit=5)
    assert len(results) > 0
    assert all("text" in r for r in results)


@pytest.mark.integration
def test_integration_versioned_swap(istore):
    uri = "file:///swap.md"
    chunks_v1 = _chunks(2, uri=uri)
    d1, s1 = _dense(2), [_sparse() for _ in range(2)]
    istore.upsert(chunks_v1, d1, s1, version=1, source_type="fs")

    chunks_v2 = _chunks(3, uri=uri)
    d2, s2 = _dense(3), [_sparse() for _ in range(3)]
    count = istore.upsert(chunks_v2, d2, s2, version=2, source_type="fs")
    assert count == 3

    istore.delete_old_version(uri, old_version=1)
    results = istore.search(d2[0], s2[0], limit=10)
    versions = {r.get("version") for r in results if r.get("source_uri") == uri}
    assert versions <= {2}


@pytest.mark.integration
def test_integration_delete_doc(istore):
    uri = "file:///todel.md"
    chunks = _chunks(2, uri=uri)
    d, s = _dense(2), [_sparse() for _ in range(2)]
    istore.upsert(chunks, d, s, version=1, source_type="fs")
    istore.delete_doc(uri)
    results = istore.search(d[0], s[0], limit=10)
    assert all(r.get("source_uri") != uri for r in results)


@pytest.mark.integration
def test_integration_store_artifact(istore):
    dense = _dense(1)[0]
    point_id = istore.store_artifact(
        text="test note",
        dense=dense,
        sparse=_sparse(),
        session_id="sess1",
        metadata={"title": "My Note"},
    )
    assert point_id is not None
    assert isinstance(point_id, str)
