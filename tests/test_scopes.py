"""Tests for agent-controlled scope: qdrant_list_scopes and qdrant_find(scope=...)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from ccmcp.store import VectorStore

# ---------------------------------------------------------------------------
# VectorStore.list_scopes
# ---------------------------------------------------------------------------

def _make_store(scroll_pages: list[list[dict]]) -> VectorStore:
    """Build a VectorStore with a mocked client that returns the given scroll pages."""
    store = VectorStore.__new__(VectorStore)
    store._collection = "test"
    store._artifact_collection = "test-artifacts"

    offsets = [f"offset-{i}" for i in range(len(scroll_pages))]

    def _scroll(**kwargs):
        if not scroll_pages:
            return [], None
        page = scroll_pages.pop(0)
        next_off = offsets.pop(0) if scroll_pages else None
        points = []
        for d in page:
            p = MagicMock()
            p.payload = d
            points.append(p)
        return points, next_off

    store._client = MagicMock()
    store._client.scroll.side_effect = _scroll
    return store


def test_list_scopes_empty_collection():
    store = _make_store([[]])
    assert store.list_scopes() == []


def test_list_scopes_skips_unscoped_points():
    store = _make_store([[
        {"source_root": "", "project_name": "", "tags": []},
        {"source_root": None, "project_name": "", "tags": []},
    ]])
    assert store.list_scopes() == []


def test_list_scopes_single_project():
    store = _make_store([[
        {"source_root": "/code/api", "project_name": "api-server", "tags": ["go", "backend"]},
        {"source_root": "/code/api", "project_name": "api-server", "tags": ["go", "backend"]},
    ]])
    result = store.list_scopes()
    assert len(result) == 1
    assert result[0]["name"] == "api-server"
    assert result[0]["source_root"] == "/code/api"
    assert set(result[0]["tags"]) == {"go", "backend"}


def test_list_scopes_merges_tags_across_chunks():
    # Different chunks from the same root may have different tags listed
    store = _make_store([[
        {"source_root": "/code/api", "project_name": "api-server", "tags": ["go"]},
        {"source_root": "/code/api", "project_name": "api-server", "tags": ["backend"]},
    ]])
    result = store.list_scopes()
    assert len(result) == 1
    assert "go" in result[0]["tags"]
    assert "backend" in result[0]["tags"]


def test_list_scopes_multiple_projects():
    store = _make_store([[
        {"source_root": "/code/api", "project_name": "api-server", "tags": ["go"]},
        {"source_root": "/code/ui", "project_name": "web-frontend", "tags": ["react"]},
    ]])
    result = store.list_scopes()
    names = {s["name"] for s in result}
    assert names == {"api-server", "web-frontend"}


def test_list_scopes_multi_page_scroll():
    store = _make_store([
        [{"source_root": "/code/api", "project_name": "api-server", "tags": ["go"]}],
        [{"source_root": "/code/ui", "project_name": "web-frontend", "tags": ["react"]}],
    ])
    result = store.list_scopes()
    assert len(result) == 2


def test_list_scopes_name_falls_back_to_directory():
    store = _make_store([[
        {"source_root": "/code/my-lib", "project_name": "", "tags": ["python"]},
    ]])
    result = store.list_scopes()
    assert result[0]["name"] == "my-lib"


# ---------------------------------------------------------------------------
# _scope_filter helper
# ---------------------------------------------------------------------------

def test_scope_filter_matches_project_name_and_tags():
    from qdrant_client import models

    from ccmcp.__main__ import _scope_filter

    f = _scope_filter(["api-server", "shared-lib"])
    assert isinstance(f, models.Filter)
    # should have two conditions in should[]
    assert len(f.should) == 2
    keys = {c.key for c in f.should}
    assert keys == {"project_name", "tags"}


# ---------------------------------------------------------------------------
# Controller passes project_name through to store.upsert
# ---------------------------------------------------------------------------

def test_ingest_file_passes_project_name():
    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile

    cfg = MagicMock()
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: (
        np.zeros((len(texts), 384)), [MagicMock() for _ in texts]
    )
    store = MagicMock()
    store.upsert.side_effect = lambda chunks, *a, **kw: len(chunks)
    store.doc_id.return_value = "abc"
    state = MagicMock()
    state.get.return_value = None

    ctrl = Controller(cfg, embedder, store, state)
    ctrl.ingest_file(
        SourceFile(source_uri="file:///tmp/x.md", content="# Hello\n\nContent."),
        source_root="/tmp",
        project_name="my-project",
        tags=["python"],
    )

    call_kwargs = store.upsert.call_args.kwargs
    assert call_kwargs["project_name"] == "my-project"
    assert call_kwargs["source_root"] == "/tmp"
    assert call_kwargs["tags"] == ["python"]
