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


def test_source_path_absent_leaves_paths_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("CCMCP_SOURCE_PATH", raising=False)
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert cfg.sources.filesystem.paths == []


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


def test_source_path_env_sets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos")
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert len(cfg.sources.filesystem.paths) == 1
    assert cfg.sources.filesystem.paths[0].path == "/repos"
    assert cfg.sources.filesystem.enabled is True


def test_source_path_not_overridden_when_paths_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos")
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"paths": ["/custom"], "enabled": True}}
    })
    cfg = load_config(path)
    assert cfg.sources.filesystem.paths[0].path == "/custom"


def test_paths_plain_string(tmp_path):
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"paths": ["/code/myproject"]}}
    })
    cfg = load_config(path)
    sp = cfg.sources.filesystem.paths[0]
    assert sp.path == "/code/myproject"
    assert sp.name == ""
    assert sp.tags == []


def test_paths_object_with_metadata(tmp_path):
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"paths": [
            {"path": "/code/api", "name": "api-server", "tags": ["go", "backend"]}
        ]}}
    })
    cfg = load_config(path)
    sp = cfg.sources.filesystem.paths[0]
    assert sp.path == "/code/api"
    assert sp.name == "api-server"
    assert sp.tags == ["go", "backend"]


def test_paths_backwards_compat_roots_key(tmp_path):
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"roots": ["/code/legacy"]}}
    })
    cfg = load_config(path)
    assert cfg.sources.filesystem.paths[0].path == "/code/legacy"


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


def test_default_extensions_include_cc_cxx_hpp():
    cfg = load_config("/nonexistent/path.yaml")
    exts = cfg.sources.filesystem.extensions
    assert ".cc" in exts
    assert ".cxx" in exts
    assert ".hpp" in exts


def test_default_extensions_drop_json():
    """JSON inflates code indexes with generated/lockfile content — opt-in only."""
    cfg = load_config("/nonexistent/path.yaml")
    assert ".json" not in cfg.sources.filesystem.extensions


def test_default_respect_gitignore_is_true():
    cfg = load_config("/nonexistent/path.yaml")
    assert cfg.sources.filesystem.respect_gitignore is True


def test_respect_gitignore_can_be_disabled(tmp_path):
    path = _write_config(tmp_path, {
        "sources": {"filesystem": {"respect_gitignore": False}}
    })
    cfg = load_config(path)
    assert cfg.sources.filesystem.respect_gitignore is False


def test_source_path_env_derives_name(tmp_path, monkeypatch):
    """CCMCP_SOURCE_PATH must stamp a project name so scoping works under Docker."""
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos/zsdk")
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    sp = cfg.sources.filesystem.paths[0]
    assert sp.path == "/repos/zsdk"
    assert sp.name == "zsdk"


def test_source_path_env_trailing_slash(tmp_path, monkeypatch):
    monkeypatch.setenv("CCMCP_SOURCE_PATH", "/repos/zsdk/")
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert cfg.sources.filesystem.paths[0].name == "zsdk"
