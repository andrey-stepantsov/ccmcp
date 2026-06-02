"""Unit tests for MCP-root → configured-source matching used by qdrant_find."""
from __future__ import annotations

import os

import pytest
from qdrant_client import models as qdrant_models

from ccmcp.__main__ import _host_to_container, _match_root, _roots_filter
from ccmcp.config import SourcePath


@pytest.fixture
def path_index():
    return {
        "/code/api": SourcePath(path="/code/api", name="api", tags=["go"]),
        "/code/web": SourcePath(path="/code/web", name="web", tags=["ts"]),
        "/code": SourcePath(path="/code", name="all", tags=[]),
    }


# ── Shared fakes for _roots_filter integration tests ────────────────────────
class _FakeRoot:
    def __init__(self, uri):
        self.uri = uri


class _FakeListRootsResult:
    def __init__(self, uris):
        self.roots = [_FakeRoot(u) for u in uris]


class _FakeSession:
    def __init__(self, uris):
        self._uris = uris

    async def list_roots(self):
        return _FakeListRootsResult(self._uris)


class _FakeRequestContext:
    def __init__(self, session):
        self.session = session


class _FakeCtx:
    def __init__(self, uris):
        self.request_context = _FakeRequestContext(_FakeSession(uris))


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
    monkeypatch.delenv("CCMCP_PATH_MAPS", raising=False)
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


def test_host_to_container_path_maps_multi_mount(monkeypatch):
    """CCMCP_PATH_MAPS supports multiple host→container pairs at once."""
    monkeypatch.delenv("CCMCP_HOST_MOUNT", raising=False)
    monkeypatch.delenv("CCMCP_CONTAINER_MOUNT", raising=False)
    monkeypatch.setenv(
        "CCMCP_PATH_MAPS",
        '{"/Users/alice/api": "/repos-api", "/Users/alice/web": "/repos-web"}',
    )
    assert _host_to_container("/Users/alice/api/handlers") == "/repos-api/handlers"
    assert _host_to_container("/Users/alice/web/components") == "/repos-web/components"
    # Unmapped paths stay untouched
    assert _host_to_container("/Users/alice/scratch") == "/Users/alice/scratch"


def test_host_to_container_path_maps_longest_prefix(monkeypatch):
    """When one mount is nested inside another, longest-prefix wins."""
    monkeypatch.delenv("CCMCP_HOST_MOUNT", raising=False)
    monkeypatch.delenv("CCMCP_CONTAINER_MOUNT", raising=False)
    monkeypatch.setenv(
        "CCMCP_PATH_MAPS",
        '{"/code": "/all", "/code/api": "/just-api"}',
    )
    # /code/api wins — it's a longer prefix than /code
    assert _host_to_container("/code/api/handlers") == "/just-api/handlers"
    # /code matches for paths NOT under /code/api
    assert _host_to_container("/code/scripts") == "/all/scripts"


def test_host_to_container_path_maps_and_legacy_pair_coexist(monkeypatch):
    """Legacy CCMCP_HOST_MOUNT/CONTAINER_MOUNT still works alongside CCMCP_PATH_MAPS."""
    monkeypatch.setenv("CCMCP_PATH_MAPS", '{"/code/api": "/repos-api"}')
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/code/web")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/repos-web")
    assert _host_to_container("/code/api/x") == "/repos-api/x"
    assert _host_to_container("/code/web/y") == "/repos-web/y"


def test_host_to_container_path_maps_invalid_json_falls_back(monkeypatch):
    """A malformed CCMCP_PATH_MAPS falls back to the legacy pair without crashing."""
    monkeypatch.setenv("CCMCP_PATH_MAPS", "not-json{")
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/code")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/repos")
    assert _host_to_container("/code/proj") == "/repos/proj"


def test_match_root_uses_os_sep(path_index):
    """The matcher uses os.sep so it works on the current platform."""
    sep = os.sep
    # On POSIX this is the same as the literal tests above; on Windows it
    # would use backslash. We only check the function doesn't crash.
    assert _match_root(f"{sep}code{sep}api", path_index) is not None or sep != "/"


# ── _roots_filter integration tests ─────────────────────────────────────────
#
# These exercise the end-to-end path the MCP client takes: a Context whose
# session.list_roots() returns workspace URIs, fed into _roots_filter, which
# must produce a Qdrant Filter that scopes search to ONLY the requested
# project. The regression these guard against: drift between the resolved
# path stamped into the `source_root` payload field by controller.ingest_file
# and the resolved path used here for matching would silently re-introduce
# the v0.1.0 bug where unmatched roots fell through to a full-corpus search.

@pytest.fixture
def two_project_index():
    return {
        "/code/proj-a": SourcePath(path="/code/proj-a", name="proj-a", tags=["a"]),
        "/code/proj-b": SourcePath(path="/code/proj-b", name="proj-b", tags=["b"]),
    }


def _source_root_match_values(flt: qdrant_models.Filter) -> list[str]:
    """Extract the MatchAny.any values from the source_root condition."""
    for cond in flt.should or []:
        if isinstance(cond, qdrant_models.FieldCondition) and cond.key == "source_root":
            assert isinstance(cond.match, qdrant_models.MatchAny)
            return list(cond.match.any)
    raise AssertionError("no source_root FieldCondition in filter.should")


def _tags_match_values(flt: qdrant_models.Filter) -> list[str]:
    """Extract the MatchAny.any values from the tags condition (or [])."""
    for cond in flt.should or []:
        if isinstance(cond, qdrant_models.FieldCondition) and cond.key == "tags":
            assert isinstance(cond.match, qdrant_models.MatchAny)
            return list(cond.match.any)
    return []


async def test_roots_filter_targets_only_requested_project(two_project_index):
    """When the MCP client sends file:///code/proj-a, the filter must scope
    to /code/proj-a exclusively — never leak proj-b into the result set."""
    ctx = _FakeCtx(["file:///code/proj-a"])
    flt = await _roots_filter(ctx, two_project_index)

    assert flt is not None
    values = _source_root_match_values(flt)
    assert "/code/proj-a" in values
    assert "/code/proj-b" not in values


async def test_roots_filter_subdirectory_resolves_to_parent_project(two_project_index):
    """A subdirectory of an indexed project must still resolve to that project
    via longest-prefix match — the agent often opens a subfolder, not the root."""
    ctx = _FakeCtx(["file:///code/proj-a/handlers/auth"])
    flt = await _roots_filter(ctx, two_project_index)

    assert flt is not None
    values = _source_root_match_values(flt)
    assert "/code/proj-a" in values
    assert "/code/proj-b" not in values


async def test_roots_filter_returns_none_when_no_root_matches(two_project_index):
    """A root with no configured ancestor returns None (unscoped). The v0.1.1
    logging path covers this case so the silent fallback is visible."""
    ctx = _FakeCtx(["file:///somewhere/else"])
    flt = await _roots_filter(ctx, two_project_index)
    assert flt is None


async def test_roots_filter_honours_host_to_container_remap(monkeypatch):
    """When CCMCP_HOST_MOUNT/CCMCP_CONTAINER_MOUNT are set (Docker), the
    incoming host-side URI must be remapped before matching against the
    container-side path index."""
    monkeypatch.setenv("CCMCP_HOST_MOUNT", "/Users/alice/code")
    monkeypatch.setenv("CCMCP_CONTAINER_MOUNT", "/workspace")
    container_index = {
        "/workspace/proj-a": SourcePath(
            path="/workspace/proj-a", name="proj-a", tags=[]
        ),
    }
    ctx = _FakeCtx(["file:///Users/alice/code/proj-a"])
    flt = await _roots_filter(ctx, container_index)

    assert flt is not None
    values = _source_root_match_values(flt)
    assert "/workspace/proj-a" in values


async def test_roots_filter_includes_tags_from_matched_source():
    """When the matched SourcePath has `include` tags, the filter must add a
    second FieldCondition on `tags` so cross-project tagged content surfaces."""
    index = {
        "/code/proj-a": SourcePath(
            path="/code/proj-a", name="proj-a", tags=["a"], include=["api"]
        ),
    }
    ctx = _FakeCtx(["file:///code/proj-a"])
    flt = await _roots_filter(ctx, index)

    assert flt is not None
    assert flt.should is not None
    assert len(flt.should) == 2
    assert "/code/proj-a" in _source_root_match_values(flt)
    assert _tags_match_values(flt) == ["api"]


async def test_roots_filter_empty_roots_returns_none(two_project_index):
    """An MCP session that exposes no roots returns None — there is nothing
    to scope to, and falling back to full-corpus is the correct behaviour
    only in this explicit no-roots case."""
    ctx = _FakeCtx([])
    flt = await _roots_filter(ctx, two_project_index)
    assert flt is None
