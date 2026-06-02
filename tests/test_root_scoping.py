"""Unit tests for MCP-root → configured-source matching used by qdrant_find."""
from __future__ import annotations

import os

import pytest

from ccmcp.__main__ import _host_to_container, _match_root
from ccmcp.config import SourcePath


@pytest.fixture
def path_index():
    return {
        "/code/api": SourcePath(path="/code/api", name="api", tags=["go"]),
        "/code/web": SourcePath(path="/code/web", name="web", tags=["ts"]),
        "/code": SourcePath(path="/code", name="all", tags=[]),
    }


def test_match_exact_root(path_index):
    match = _match_root("/code/api", path_index)
    assert match is not None
    assert match[0] == "/code/api"
    assert match[1].name == "api"


def test_match_subdir_uses_longest_prefix(path_index):
    """A subdir of /code/api must match /code/api (not the broader /code)."""
    match = _match_root("/code/api/handlers", path_index)
    assert match is not None
    assert match[0] == "/code/api"
    assert match[1].name == "api"


def test_match_deep_subdir(path_index):
    match = _match_root("/code/api/handlers/auth", path_index)
    assert match is not None
    assert match[0] == "/code/api"


def test_match_falls_back_to_broader_root(path_index):
    """A path under /code but not under any narrower root falls back to /code."""
    match = _match_root("/code/scripts", path_index)
    assert match is not None
    assert match[0] == "/code"


def test_no_match_returns_none(path_index):
    assert _match_root("/somewhere/else", path_index) is None


def test_prefix_must_be_path_component(path_index):
    """/codex is NOT under /code — substring match would be wrong."""
    assert _match_root("/codex/api", path_index) is None


def test_host_to_container_no_env(monkeypatch):
    monkeypatch.delenv("CCMCP_HOST_MOUNT", raising=False)
    monkeypatch.delenv("CCMCP_CONTAINER_MOUNT", raising=False)
    assert _host_to_container("/anything") == "/anything"


def test_host_to_container_rewrites_prefix(monkeypatch):
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/Users/alice/code")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/workspace")
    assert _host_to_container("/Users/alice/code/api") == "/workspace/api"
    assert _host_to_container("/Users/alice/code") == "/workspace"


def test_host_to_container_leaves_unrelated_paths(monkeypatch):
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/Users/alice/code")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/workspace")
    assert _host_to_container("/etc/hosts") == "/etc/hosts"


def test_host_to_container_substring_safety(monkeypatch):
    """/Users/alice/codex must NOT be rewritten just because /Users/alice/code is the host mount."""
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/Users/alice/code")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/workspace")
    assert _host_to_container("/Users/alice/codex") == "/Users/alice/codex"


def test_host_to_container_trailing_slash_normalized(monkeypatch):
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/host/")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/container/")
    assert _host_to_container("/host/proj") == "/container/proj"


def test_match_root_uses_os_sep(path_index):
    """The matcher uses os.sep so it works on the current platform."""
    sep = os.sep
    # On POSIX this is the same as the literal tests above; on Windows it
    # would use backslash. We only check the function doesn't crash.
    assert _match_root(f"{sep}code{sep}api", path_index) is not None or sep != "/"
