from pathlib import Path

import yaml

from ccmcp.config import load_config


def _write_config(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CCMCP_SOURCE_PATH", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert cfg.qdrant.url == "http://localhost:6333"
    assert cfg.qdrant.collection == "techdocs"
    assert cfg.mcp.port == 7700


def test_source_path_absent_leaves_roots_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("CCMCP_SOURCE_PATH", raising=False)
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert cfg.sources.filesystem.roots == []


def test_file_overrides_defaults(tmp_path):
    path = _write_config(tmp_path, {
        "qdrant": {"url": "http://remote:6333", "collection": "mydocs"},
        "mcp": {"port": 8800},
    })
    cfg = load_config(path)
    assert cfg.qdrant.url == "http://remote:6333"
    assert cfg.qdrant.collection == "mydocs"
    assert cfg.mcp.port == 8800


def test_env_overrides_file(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"qdrant": {"url": "http://file:6333"}})
    monkeypatch.setenv("QDRANT_URL", "http://env:6333")
    cfg = load_config(path)
    assert cfg.qdrant.url == "http://env:6333"


def test_source_path_env_sets_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos")
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert cfg.sources.filesystem.roots == ["/repos"]
    assert cfg.sources.filesystem.enabled is True


def test_source_path_not_overridden_when_roots_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos")
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"roots": ["/custom"], "enabled": True}}
    })
    cfg = load_config(path)
    assert cfg.sources.filesystem.roots == ["/custom"]


def test_filesystem_extensions_loaded(tmp_path):
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"extensions": [".md", ".py"]}}
    })
    cfg = load_config(path)
    assert ".md" in cfg.sources.filesystem.extensions
    assert ".py" in cfg.sources.filesystem.extensions


def test_state_db_path_expanded(tmp_path):
    path = _write_config(tmp_path, {"state": {"db_path": "~/mydb/state.db"}})
    cfg = load_config(path)
    assert not cfg.state.db_path.startswith("~")


def test_extraction_backend_default():
    cfg = load_config("/nonexistent/path.yaml")
    assert cfg.extraction.backend == "none"
