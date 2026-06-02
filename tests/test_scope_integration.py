"""Live-store scope isolation integration test.

Guards against drift between:
  - the path controller.py:124 stamps into `source_root` via _resolved(sp.path),
  - and the path qdrant_find queries via _build_path_index + _match_root.

If those two diverge, every search silently falls back to full-corpus — the
v0.1.0 bug. Pure-function tests in test_root_scoping.py exercise each side in
isolation; this test walks the full live pipeline (Controller.scan →
VectorStore.upsert → VectorStore.search-with-filter) against an in-memory
Qdrant and asserts both sides agree.

In-memory Qdrant via QdrantClient(location=":memory:") — no Docker required.
"""
from __future__ import annotations

import numpy as np
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from ccmcp.__main__ import _build_path_index
from ccmcp.config import Config, SourcePath
from ccmcp.controller import Controller, _resolved
from ccmcp.state import StateDB
from ccmcp.store import VectorStore


class _StubEmbedder:
    """Deterministic 384-d embedder — no model load, no network.

    Vector per text is derived from a hash of the text, so identical texts
    produce identical vectors (BM25-like exact match for the test) and distinct
    texts produce distinct vectors.
    """

    dim = 384

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[SparseEmbedding]]:
        dense = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            dense[i] = rng.standard_normal(self.dim).astype(np.float32)
        sparse = [
            SparseEmbedding(
                indices=np.array([abs(hash(t)) % 1000], dtype=np.int64),
                values=np.array([1.0], dtype=np.float32),
            )
            for t in texts
        ]
        return dense, sparse


def _make_store() -> VectorStore:
    client = QdrantClient(location=":memory:")
    store = VectorStore(client=client, collection="scope_test")
    store.setup(dense_dim=_StubEmbedder.dim)
    return store


def test_controller_stamps_what_path_index_queries(tmp_path):
    """End-to-end: source_root stamped by Controller.scan matches the keys
    _build_path_index produces. A drift between the two would silently
    re-introduce the v0.1.0 full-corpus fallback bug."""
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    (proj_a / "doc.md").write_text("# Greetings\n\nALPHA_TOKEN content from project A.")
    (proj_b / "doc.md").write_text("# Salutations\n\nBETA_TOKEN content from project B.")

    cfg = Config()
    cfg.sources.filesystem.paths = [
        SourcePath(path=str(proj_a), name="proj-a", tags=["a"]),
        SourcePath(path=str(proj_b), name="proj-b", tags=["b"]),
    ]

    store = _make_store()
    embedder = _StubEmbedder()
    state = StateDB(str(tmp_path / "state.db"))

    Controller(cfg, embedder, store, state).scan()  # type: ignore[arg-type]

    # Collect every source_root the controller stamped onto upserted points.
    points, _ = store._client.scroll(
        collection_name="scope_test", limit=100, with_payload=True
    )
    stamped_roots = {p.payload["source_root"] for p in points if p.payload}
    assert stamped_roots == {_resolved(str(proj_a)), _resolved(str(proj_b))}, \
        "Controller did not stamp both projects' source_root values"

    # The keys qdrant_find queries against — must match the stamped values byte-for-byte.
    path_index_keys = set(_build_path_index(cfg).keys())
    assert path_index_keys == stamped_roots, (
        "DRIFT! controller stamps source_root keys not present in _build_path_index "
        f"(stamped={stamped_roots}, queried={path_index_keys})"
    )


def test_filter_scoped_to_one_project_excludes_the_other(tmp_path):
    """Live-store check: a Qdrant filter built the way _roots_filter builds it,
    targeting only project A's source_root, returns zero points from project B —
    even when the query text matches project B's content."""
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    (proj_a / "alpha.md").write_text("# Alpha\n\nALPHA_TOKEN content from project A.")
    (proj_b / "beta.md").write_text("# Beta\n\nBETA_TOKEN content from project B.")

    cfg = Config()
    cfg.sources.filesystem.paths = [
        SourcePath(path=str(proj_a), name="proj-a", tags=["a"]),
        SourcePath(path=str(proj_b), name="proj-b", tags=["b"]),
    ]

    store = _make_store()
    embedder = _StubEmbedder()
    state = StateDB(str(tmp_path / "state.db"))
    Controller(cfg, embedder, store, state).scan()  # type: ignore[arg-type]

    a_root = _resolved(str(proj_a))
    flt = qdrant_models.Filter(should=[
        qdrant_models.FieldCondition(
            key="source_root",
            match=qdrant_models.MatchAny(any=[a_root]),
        )
    ])

    # Query for beta-side content while scoped to project A.
    dense, sparse = embedder.embed(["BETA_TOKEN content from project B."])
    hits = store.search(dense[0], sparse[0], limit=10, filter=flt)

    # Every hit must come from project A; no proj-b chunk may leak through.
    assert hits, "filter is over-aggressive — returned nothing from project A"
    for h in hits:
        assert h["source_root"] == a_root, (
            f"scope leak: chunk from {h['source_root']!r} "
            f"returned under filter scoped to {a_root!r}"
        )
        assert "BETA_TOKEN" not in h["text"], "proj-b text leaked through scoped filter"


def test_unscoped_search_can_see_both_projects(tmp_path):
    """Negative control: with NO filter, both projects' content is reachable —
    confirms the previous test's exclusion is actually the filter's doing, not
    a setup quirk where proj-b simply isn't ingested."""
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    (proj_a / "alpha.md").write_text("# Alpha\n\nALPHA_TOKEN content from project A.")
    (proj_b / "beta.md").write_text("# Beta\n\nBETA_TOKEN content from project B.")

    cfg = Config()
    cfg.sources.filesystem.paths = [
        SourcePath(path=str(proj_a), name="proj-a", tags=["a"]),
        SourcePath(path=str(proj_b), name="proj-b", tags=["b"]),
    ]

    store = _make_store()
    embedder = _StubEmbedder()
    state = StateDB(str(tmp_path / "state.db"))
    Controller(cfg, embedder, store, state).scan()  # type: ignore[arg-type]

    dense, sparse = embedder.embed(["any query"])
    hits = store.search(dense[0], sparse[0], limit=20, filter=None)
    roots = {h["source_root"] for h in hits}
    assert _resolved(str(proj_a)) in roots
    assert _resolved(str(proj_b)) in roots
