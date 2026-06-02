from pathlib import Path

from ccmcp.sources.filesystem import GitIgnoreCascade, _matches_ignore, scan


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


def test_gitignore_cascades_to_nested_dirs(tmp_path):
    """A root .gitignore must apply to files in nested subdirectories."""
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "deeper").mkdir()
    (tmp_path / "deep" / "deeper" / "trace.log").write_text("log")
    (tmp_path / "deep" / "deeper" / "code.py").write_text("x=1")

    files = scan([str(tmp_path)], [".log", ".py"], [])
    uris = {f.source_uri for f in files}
    assert not any("trace.log" in u for u in uris)
    assert any("code.py" in u for u in uris)


def test_gitignore_anchored_pattern(tmp_path):
    """A leading-slash pattern anchors to the .gitignore's directory."""
    (tmp_path / ".gitignore").write_text("/topfile.md\n")
    (tmp_path / "topfile.md").write_text("at root")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "topfile.md").write_text("nested")

    files = scan([str(tmp_path)], [".md"], [])
    uris = {f.source_uri for f in files}
    assert not any(u.endswith("/topfile.md") and "/sub/" not in u for u in uris)
    assert any("/sub/topfile.md" in u for u in uris)


def test_gitignore_double_star(tmp_path):
    """Pattern with ** matches at any depth."""
    (tmp_path / ".gitignore").write_text("**/generated/*.py\n")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "generated").mkdir()
    (tmp_path / "a" / "generated" / "gen.py").write_text("auto")
    (tmp_path / "a" / "manual.py").write_text("hand")

    files = scan([str(tmp_path)], [".py"], [])
    uris = {f.source_uri for f in files}
    assert not any("gen.py" in u for u in uris)
    assert any("manual.py" in u for u in uris)


def test_gitignore_nested_overrides_parent(tmp_path):
    """A nested .gitignore can add additional ignores."""
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".gitignore").write_text("secret.md\n")
    (tmp_path / "sub" / "secret.md").write_text("x")
    (tmp_path / "sub" / "public.md").write_text("y")

    files = scan([str(tmp_path)], [".md"], [])
    uris = {f.source_uri for f in files}
    assert not any("secret.md" in u for u in uris)
    assert any("public.md" in u for u in uris)


def test_gitignore_opt_out(tmp_path):
    """respect_gitignore=False indexes everything regardless of .gitignore."""
    (tmp_path / ".gitignore").write_text("secret.md\n")
    (tmp_path / "secret.md").write_text("s")
    (tmp_path / "public.md").write_text("p")

    files = scan([str(tmp_path)], [".md"], [], respect_gitignore=False)
    uris = {f.source_uri for f in files}
    assert any("secret.md" in u for u in uris)
    assert any("public.md" in u for u in uris)


def test_cascade_outside_root_not_ignored(tmp_path):
    """Files outside the cascade root must never be reported ignored."""
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / ".gitignore").write_text("*.log\n")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "trace.log").write_text("log")

    cascade = GitIgnoreCascade(str(tmp_path / "root"))
    assert cascade.is_ignored(str(tmp_path / "other" / "trace.log")) is False


def test_matches_ignore_glob():
    assert _matches_ignore("file.pyc", ["*.pyc"])
    assert _matches_ignore("node_modules", ["node_modules"])
    assert not _matches_ignore("main.py", ["*.pyc"])
