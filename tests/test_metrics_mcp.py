"""Metric tests for the new MCP-tool counters (G-mon-1 + G-mon-2).

Drives qdrant_find / qdrant_list_scopes / qdrant_store via the FastMCP
registry and asserts the right Counter labels increment.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from ccmcp.__main__ import _build_mcp
from ccmcp.config import Config, SourcePath
from ccmcp.metrics import FIND_REQUESTS, TOOL_CALLS


def _make_cfg(paths: list[SourcePath] | None = None) -> Config:
    cfg = Config()
    cfg.mcp.result_limit = 10
    if paths is not None:
        cfg.sources.filesystem.paths = paths
    return cfg


def _stub_components(paths: list[SourcePath] | None = None):
    cfg = _make_cfg(paths)
    embedder = MagicMock()
    embedder.embed.return_value = (np.zeros((1, 384), dtype=np.float32), [MagicMock()])
    store = MagicMock()
    store.search.return_value = []
    store.list_scopes.return_value = []
    store.store_artifact.return_value = "pt-1"
    return cfg, embedder, store


def _label_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


# ── G-mon-1: scope_mode classification ──────────────────────────────────────

async def test_qdrant_find_explicit_scope_increments_explicit():
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(FIND_REQUESTS, scope_mode="explicit")
    await mcp.call_tool("qdrant_find", {"query": "x", "scope": ["proj-a"]})
    assert _label_value(FIND_REQUESTS, scope_mode="explicit") == before + 1


async def test_qdrant_find_wildcard_scope_increments_wildcard():
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(FIND_REQUESTS, scope_mode="wildcard")
    await mcp.call_tool("qdrant_find", {"query": "x", "scope": ["*"]})
    assert _label_value(FIND_REQUESTS, scope_mode="wildcard") == before + 1


async def test_qdrant_find_empty_scope_increments_wildcard():
    """An empty scope list is treated as wildcard (no filter applied)."""
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(FIND_REQUESTS, scope_mode="wildcard")
    await mcp.call_tool("qdrant_find", {"query": "x", "scope": []})
    assert _label_value(FIND_REQUESTS, scope_mode="wildcard") == before + 1


async def test_qdrant_find_no_ctx_increments_auto_fallback():
    """When scope is omitted and there is no MCP context to expose roots,
    the call falls back to full-corpus search — count as auto_fallback."""
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(FIND_REQUESTS, scope_mode="auto_fallback")
    # mcp.call_tool passes through a real Context, so _roots_filter is exercised.
    # Without configured paths, _build_path_index is empty → no match → fallback.
    await mcp.call_tool("qdrant_find", {"query": "x"})
    assert _label_value(FIND_REQUESTS, scope_mode="auto_fallback") == before + 1


# ── G-mon-2: per-tool counters ──────────────────────────────────────────────

async def test_qdrant_find_increments_tool_calls():
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(TOOL_CALLS, tool="qdrant_find")
    await mcp.call_tool("qdrant_find", {"query": "x", "scope": ["*"]})
    assert _label_value(TOOL_CALLS, tool="qdrant_find") == before + 1


async def test_qdrant_list_scopes_increments_tool_calls():
    cfg, embedder, store = _stub_components()
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(TOOL_CALLS, tool="qdrant_list_scopes")
    await mcp.call_tool("qdrant_list_scopes", {})
    assert _label_value(TOOL_CALLS, tool="qdrant_list_scopes") == before + 1


async def test_qdrant_store_increments_tool_calls():
    cfg, embedder, store = _stub_components()
    store.cleanup_artifacts.side_effect = lambda *_a, **_kw: None
    mcp = _build_mcp(cfg, embedder, store)

    before = _label_value(TOOL_CALLS, tool="qdrant_store")
    await mcp.call_tool("qdrant_store", {"text": "note"})
    assert _label_value(TOOL_CALLS, tool="qdrant_store") == before + 1
