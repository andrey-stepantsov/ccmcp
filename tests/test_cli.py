"""Tests for CLI commands using Click's test runner.

These tests do NOT require a running Qdrant instance or ML models.
They verify command wiring, help output, and error handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ccmcp.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_components(tmp_path):
    """Patch _components to return lightweight mocks; no Qdrant, no model loading."""
    cfg = MagicMock()
    cfg.embedding.dense_model = "BAAI/bge-small-en-v1.5"
    cfg.embedding.sparse_model = "Qdrant/bm25"
    cfg.embedding.rotation_matrix = str(tmp_path / "rotation_matrix.npy")
    cfg.qdrant.url = "http://localhost:6333"
    cfg.qdrant.collection = "techdocs"
    cfg.qdrant.api_key = ""
    cfg.mcp.host = "127.0.0.1"
    cfg.mcp.port = 7700
    cfg.state.db_path = str(tmp_path / "state.db")
    cfg.state.artifact_ttl_days = 30

    embedder = MagicMock()
    embedder.dim = 384

    store = MagicMock()
    state = MagicMock()
    state.all.return_value = []

    with patch("ccmcp.__main__._components", return_value=(cfg, embedder, store, state)):
        yield cfg, embedder, store, state


# ---------------------------------------------------------------------------
# Help output
# ---------------------------------------------------------------------------

def test_root_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Commands:" in result.output
    for cmd in ("init", "scan", "watch", "ingest", "status", "reset", "serve", "doctor", "run"):
        assert cmd in result.output


@pytest.mark.parametrize("cmd", [
    "init", "scan", "watch", "ingest", "status", "reset", "serve", "doctor", "run",
])
def test_command_help(runner, cmd):
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_no_truncated_descriptions(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Command description lines start with two spaces followed by a command name.
    # None of them should be cut off with "..."
    for line in result.output.splitlines():
        if line.startswith("  ") and not line.startswith("   "):
            assert not line.rstrip().endswith("..."), f"Truncated description: {line!r}"


# ---------------------------------------------------------------------------
# setup command
# ---------------------------------------------------------------------------

def test_init_calls_embedder_and_store(runner, mock_components):
    cfg, embedder, store, _ = mock_components
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    embedder.setup.assert_called_once()
    store.setup.assert_called_once_with(embedder.dim)


def test_init_tolerates_existing_rotation_matrix(runner, mock_components):
    _, embedder, _, _ = mock_components
    embedder.setup.side_effect = FileExistsError("rotation_matrix.npy already exists")
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "rotation_matrix.npy already exists" in result.output


def test_init_qdrant_unreachable_shows_error(runner, mock_components):
    _, _, store, _ = mock_components
    store.setup.side_effect = Exception("Connection refused")
    result = runner.invoke(cli, ["init"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def test_status_shows_source_counts(runner, mock_components):
    _, _, store, state = mock_components
    store.collection_info.return_value = {"points_count": 42, "status": "green"}
    store.artifact_collection_info.return_value = {"points_count": 3, "status": "green"}
    state.all.return_value = [
        MagicMock(source_uri="file:///a.md"),
        MagicMock(source_uri="file:///b.md"),
        MagicMock(source_uri="https://example.com"),
    ]
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "3" in result.output  # total sources


def test_status_artifact_collection_unavailable(runner, mock_components):
    _, _, store, _ = mock_components
    store.collection_info.return_value = {"points_count": 0, "status": "green"}
    store.artifact_collection_info.side_effect = Exception("not found")
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "not available" in result.output.lower()


# ---------------------------------------------------------------------------
# reset command
# ---------------------------------------------------------------------------

def test_reset_requires_confirmation(runner, mock_components):
    _, _, store, _ = mock_components
    result = runner.invoke(cli, ["reset"], input="no\n")
    assert result.exit_code == 0
    store.drop_collections.assert_not_called()


def test_reset_confirmed_drops_collections(runner, mock_components):
    _, _, store, _ = mock_components
    result = runner.invoke(cli, ["reset"], input="RESET\n")
    assert result.exit_code == 0
    store.drop_collections.assert_called_once()


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------

def test_ingest_file(runner, mock_components, tmp_path):
    from ccmcp.controller import Controller

    test_file = tmp_path / "notes.md"
    test_file.write_text("# Hello\n\nSome content.", encoding="utf-8")

    with patch.object(Controller, "ingest_file", return_value=None) as mock_ingest:
        result = runner.invoke(cli, ["ingest", str(test_file)])
    assert result.exit_code == 0
    assert "Ingested" in result.output


def test_ingest_missing_file(runner, mock_components):
    result = runner.invoke(cli, ["ingest", "/nonexistent/file.md"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# validate: unreachable Qdrant produces a clean error message
# ---------------------------------------------------------------------------

def test_doctor_unreachable_qdrant_clean_error(runner, mock_components):
    from ccmcp.store import VectorStore

    with patch.object(VectorStore, "drop_collections", side_effect=Exception("name resolution")):
        result = runner.invoke(cli, ["doctor"])

    assert result.exit_code != 0
    output = result.output + (str(result.exception) if result.exception else "")
    # Should mention Qdrant and give actionable guidance — not a raw traceback
    assert "qdrant" in output.lower() or "Qdrant" in output
    assert "docker compose exec" in output or "docker" in output.lower()
