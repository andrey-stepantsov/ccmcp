from pathlib import Path

from ccmcp.sources.filesystem import _gitignore_patterns, _matches_ignore, scan


def _make_tree(root: Path):
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# Hello")
    (root / "docs" / "notes.txt").write_text("some notes")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main(): pass")
    (root / "src" / "ignored.pyc").write_bytes(b"\x00")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "pkg.js").write_text("// vendor")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]")


def test_scan_finds_expected_extensions(tmp_path):
    _make_tree(tmp_path)
    files = scan([str(tmp_path)], [".md", ".txt", ".py"], ["node_modules", ".git", "*.pyc"])
    uris = {f.source_uri for f in files}
    assert any("readme.md" in u for u in uris)
    assert any("notes.txt" in u for u in uris)
    assert any("main.py" in u for u in uris)


def test_scan_excludes_ignore_dirs(tmp_path):
    _make_tree(tmp_path)
    files = scan([str(tmp_path)], [".js", ".md"], ["node_modules", ".git"])
    uris = {f.source_uri for f in files}
    assert not any("node_modules" in u for u in uris)
    assert not any(".git" in u for u in uris)


def test_scan_excludes_pyc(tmp_path):
    _make_tree(tmp_path)
    files = scan([str(tmp_path)], [".py", ".pyc"], ["*.pyc"])
    uris = {f.source_uri for f in files}
    assert not any(".pyc" in u for u in uris)


def test_scan_content_readable(tmp_path):
    _make_tree(tmp_path)
    files = scan([str(tmp_path)], [".md"], ["node_modules", ".git"])
    md_files = [f for f in files if f.source_uri.endswith(".md")]
    assert len(md_files) == 1
    assert "Hello" in md_files[0].content


def test_scan_multiple_roots(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.md").write_text("A")
    (root_b / "b.md").write_text("B")
    files = scan([str(root_a), str(root_b)], [".md"], [])
    uris = {f.source_uri for f in files}
    assert any("a.md" in u for u in uris)
    assert any("b.md" in u for u in uris)


def test_gitignore_respected(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.md\nbuild/\n")
    (tmp_path / "public.md").write_text("public")
    (tmp_path / "secret.md").write_text("secret")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.md").write_text("built")

    files = scan([str(tmp_path)], [".md"], [])
    uris = {f.source_uri for f in files}
    assert any("public.md" in u for u in uris)
    assert not any("secret.md" in u for u in uris)
    assert not any("build" in u for u in uris)


def test_matches_ignore_glob():
    assert _matches_ignore("file.pyc", ["*.pyc"])
    assert _matches_ignore("node_modules", ["node_modules"])
    assert not _matches_ignore("main.py", ["*.pyc"])


def test_gitignore_patterns_parsed(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text("# comment\nsecret/\n*.log\n\nbuild\n")
    patterns = _gitignore_patterns(str(tmp_path))
    assert "secret" in patterns
    assert "*.log" in patterns
    assert "build" in patterns
    assert "# comment" not in patterns
