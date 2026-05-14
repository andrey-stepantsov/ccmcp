"""Tests for the .ccmcp marker file parser."""
from __future__ import annotations

from pathlib import Path

from ccmcp.marker import MARKER_FILENAME, MarkerFile, load


def _write(tmp_path: Path, content: str) -> str:
    (tmp_path / MARKER_FILENAME).write_text(content, encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Absent / empty / malformed
# ---------------------------------------------------------------------------

def test_no_marker_returns_none(tmp_path):
    assert load(str(tmp_path)) is None


def test_empty_marker_returns_default(tmp_path):
    _write(tmp_path, "")
    m = load(str(tmp_path))
    assert isinstance(m, MarkerFile)
    assert m.name == tmp_path.name
    assert m.tags == []
    assert m.include == []


def test_malformed_yaml_returns_default(tmp_path):
    _write(tmp_path, ":::not yaml:::")
    m = load(str(tmp_path))
    assert isinstance(m, MarkerFile)
    assert m.tags == []


def test_non_dict_yaml_returns_default(tmp_path):
    _write(tmp_path, "- item1\n- item2\n")
    m = load(str(tmp_path))
    assert isinstance(m, MarkerFile)
    assert m.tags == []


# ---------------------------------------------------------------------------
# Full valid marker
# ---------------------------------------------------------------------------

def test_full_marker_parsed(tmp_path):
    _write(tmp_path, "name: my-app\ntags:\n  - python\n  - web\ninclude:\n  - shared-lib\n")
    m = load(str(tmp_path))
    assert m is not None
    assert m.name == "my-app"
    assert m.tags == ["python", "web"]
    assert m.include == ["shared-lib"]


def test_source_root_is_resolved_absolute(tmp_path):
    _write(tmp_path, "name: proj")
    m = load(str(tmp_path))
    assert m is not None
    assert Path(m.source_root).is_absolute()
    assert m.source_root == str(tmp_path.resolve())


def test_name_defaults_to_dir_name(tmp_path):
    _write(tmp_path, "tags:\n  - go\n")
    m = load(str(tmp_path))
    assert m is not None
    assert m.name == tmp_path.name


# ---------------------------------------------------------------------------
# Partial markers
# ---------------------------------------------------------------------------

def test_tags_only(tmp_path):
    _write(tmp_path, "tags:\n  - rust\n  - embedded\n")
    m = load(str(tmp_path))
    assert m is not None
    assert m.tags == ["rust", "embedded"]
    assert m.include == []


def test_include_only(tmp_path):
    _write(tmp_path, "include:\n  - common-lib\n")
    m = load(str(tmp_path))
    assert m is not None
    assert m.include == ["common-lib"]
    assert m.tags == []


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def test_numeric_tags_coerced_to_str(tmp_path):
    _write(tmp_path, "tags:\n  - 42\n  - 3.14\n")
    m = load(str(tmp_path))
    assert m is not None
    assert m.tags == ["42", "3.14"]
