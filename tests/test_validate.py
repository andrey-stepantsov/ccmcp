"""Integration + slow test: runs the full validation suite end-to-end."""
from __future__ import annotations

import pytest

from ccmcp.validate import (
    _FIXTURES,
    _SCENARIOS,
    _TOKEN_V1,
    _TOKEN_V2,
    _VALIDATE_COLLECTION,
    run_validation,
)

# ---------------------------------------------------------------------------
# Unit tests (no Qdrant, no model loading)
# ---------------------------------------------------------------------------

def test_all_scenario_fixtures_exist():
    for s in _SCENARIOS:
        assert s.expected_fixture in _FIXTURES, (
            f"Scenario '{s.name}' references '{s.expected_fixture}' which is not in _FIXTURES"
        )


def test_fixtures_have_content():
    for name, content in _FIXTURES.items():
        assert content.strip(), f"Fixture '{name}' is empty"


def test_swap_tokens_are_unique():
    all_fixture_text = "\n".join(_FIXTURES.values())
    assert _TOKEN_V1 not in all_fixture_text, "v1 canary token appears in fixtures"
    assert _TOKEN_V2 not in all_fixture_text, "v2 canary token appears in fixtures"
    assert _TOKEN_V1 != _TOKEN_V2


def test_scenarios_have_unique_names():
    names = [s.name for s in _SCENARIOS]
    assert len(names) == len(set(names)), "Duplicate scenario names"


def test_validate_collection_name_does_not_clash():
    # Ensure the validate collection won't accidentally clobber the default
    assert _VALIDATE_COLLECTION != "techdocs"
    assert not _VALIDATE_COLLECTION.startswith("ccmcp-artifacts")


# ---------------------------------------------------------------------------
# Integration + slow: full pipeline (needs Qdrant + model download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _embedder(tmp_path_factory):
    from ccmcp.embedder import Embedder
    rot = str(tmp_path_factory.mktemp("rot") / "rotation.npy")
    e = Embedder(
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm25",
        rotation_matrix_path=rot,
    )
    e.setup()
    return e


@pytest.fixture(scope="module")
def _cfg():
    from ccmcp.config import Config
    return Config()


@pytest.mark.integration
@pytest.mark.slow
def test_full_validation_passes(_cfg, _embedder):
    """All scenarios must pass: retrieval quality, version swap, artifact round-trip."""
    try:
        passed, total = run_validation(_cfg, _embedder)
    except Exception as exc:
        pytest.skip(f"Qdrant not available: {exc}")
    assert passed == total, f"Validation: {passed}/{total} passed"


@pytest.mark.integration
@pytest.mark.slow
def test_all_retrieval_scenarios_pass(_cfg, _embedder):
    """Each retrieval scenario individually — helps pinpoint failures."""
    import shutil
    import tempfile
    from pathlib import Path

    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile
    from ccmcp.state import StateDB
    from ccmcp.store import VectorStore

    store = VectorStore(
        "http://localhost:6333",
        "ccmcp-validate-detail",
        artifact_collection="ccmcp-validate-detail-artifacts",
    )
    try:
        store.setup(_embedder.dim)
    except Exception:
        pytest.skip("Qdrant not available at localhost:6333")

    state_dir = tempfile.mkdtemp()
    state = StateDB(str(Path(state_dir) / "state.db"))
    ctrl = Controller(_cfg, _embedder, store, state)

    fixture_uris: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory() as doc_dir:
            for name, content in _FIXTURES.items():
                path = Path(doc_dir) / name
                path.write_text(content, encoding="utf-8")
                uri = f"file://{path}"
                fixture_uris[name] = uri
                ctrl.ingest_file(SourceFile(source_uri=uri, content=content))

        failures = []
        for scenario in _SCENARIOS:
            expected_uri = fixture_uris[scenario.expected_fixture]
            dense, sparse_list = _embedder.embed([scenario.query])
            hits = store.search(dense[0], sparse_list[0], limit=scenario.top_k)
            rank = next(
                (i + 1 for i, h in enumerate(hits) if h.get("source_uri") == expected_uri),
                None,
            )
            if rank is None:
                returned = [h.get("source_uri", "?").split("/")[-1] for h in hits]
                failures.append(
                    f"'{scenario.name}': expected {scenario.expected_fixture} in top-"
                    f"{scenario.top_k}, got {returned}"
                )
    finally:
        store.drop_collections()
        shutil.rmtree(state_dir, ignore_errors=True)

    assert not failures, "Retrieval scenario failures:\n" + "\n".join(failures)
