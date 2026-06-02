from ccmcp.chunker import MAX_TOKENS, _tok, chunk_file

URI = "file:///test"


def test_markdown_splits_on_headings():
    content = "# Alpha\n\nFirst section.\n\n## Beta\n\nSecond section."
    chunks = chunk_file("doc.md", content, URI)
    assert len(chunks) >= 1
    texts = " ".join(c.text for c in chunks)
    assert "Alpha" in texts
    assert "Beta" in texts


def test_markdown_heading_preserved_in_sub_chunk():
    # Body large enough to force sub-chunking
    body = "word " * 600
    content = f"## Section\n\n{body}"
    chunks = chunk_file("doc.md", content, URI)
    assert any("Section" in c.text for c in chunks)


def test_markdown_section_field():
    content = "## MySection\n\nSome text here."
    chunks = chunk_file("doc.md", content, URI)
    assert any(c.section == "MySection" for c in chunks)


def test_markdown_no_chunk_exceeds_max(tmp_path):
    body = "word " * 1000
    content = f"# Big\n\n{body}"
    chunks = chunk_file("doc.md", content, URI)
    for c in chunks:
        assert _tok(c.text) <= MAX_TOKENS * 1.1  # 10% tolerance for heading prefix


def test_chunk_index_sequential():
    content = "# A\n\ntext\n\n# B\n\nmore text\n\n# C\n\neven more"
    chunks = chunk_file("doc.md", content, URI)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_python_splits_on_functions():
    content = "MODULE = True\n\ndef foo():\n    pass\n\ndef bar():\n    return 1\n"
    chunks = chunk_file("script.py", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "foo" in texts
    assert "bar" in texts


def test_python_class_split():
    content = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
    chunks = chunk_file("mod.py", content, URI)
    assert len(chunks) >= 1


def test_text_splits_on_blank_lines():
    paragraphs = ["word " * 50 for _ in range(20)]
    content = "\n\n".join(paragraphs)
    chunks = chunk_file("notes.txt", content, URI)
    assert len(chunks) > 1
    for c in chunks:
        assert _tok(c.text) <= MAX_TOKENS * 1.1


def test_small_chunks_merged():
    # 3 small paragraphs individually below MIN_TOKENS, combined below MAX_TOKENS — merge to 1
    small = "hello world. " * 4  # ~13 tokens each
    content = f"{small}\n\n{small}\n\n{small}"
    chunks = chunk_file("doc.md", content, URI)
    assert len(chunks) < 3


def test_large_chunks_not_merged():
    # Paragraphs too large to combine (each ~325 tokens, combined ~651 > MAX_TOKENS)
    big = "word " * 260
    content = f"{big}\n\n{big}"
    chunks = chunk_file("doc.md", content, URI)
    assert len(chunks) == 2


def test_go_function_split():
    content = "package main\n\nfunc Alpha() {}\n\nfunc Beta() {}\n"
    chunks = chunk_file("main.go", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "Alpha" in texts
    assert "Beta" in texts


def test_source_uri_preserved():
    chunks = chunk_file("doc.md", "# Test\n\nContent.", "file:///myfile.md")
    for c in chunks:
        assert c.source_uri == "file:///myfile.md"


def test_empty_content():
    chunks = chunk_file("doc.md", "", URI)
    assert chunks == []


def test_yaml_falls_back_to_text():
    content = "key: value\nother: thing\n"
    chunks = chunk_file("config.yaml", content, URI)
    assert len(chunks) >= 1


def test_rst_splits_on_headings():
    content = "Introduction\n============\n\nIntro text.\n\nDetails\n-------\n\nMore text."
    chunks = chunk_file("doc.rst", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "Introduction" in texts
    assert "Details" in texts


def test_rst_section_field():
    content = "MySection\n=========\n\nSome content here."
    chunks = chunk_file("doc.rst", content, URI)
    assert any(c.section == "MySection" for c in chunks)


def test_rst_no_headings_falls_back_to_text():
    content = "Just plain paragraphs.\n\nNo RST headings at all."
    chunks = chunk_file("doc.rst", content, URI)
    assert len(chunks) >= 1
    assert all(c.source_uri == URI for c in chunks)


def test_hard_split_very_long_line():
    # A single line of 1000 tokens must be split into multiple chunks
    long_line = ("word " * 1000).strip()
    chunks = chunk_file("notes.txt", long_line, URI)
    assert len(chunks) > 1
    for c in chunks:
        assert _tok(c.text) <= MAX_TOKENS * 1.1


def _cpp_body(name: str) -> str:
    """Function body large enough to exceed MIN_TOKENS so chunks don't merge."""
    return f"int {name}(int x) {{\n" + "  return x;\n" * 80 + "}\n"


def test_cpp_cc_extension_function_chunked():
    content = _cpp_body("Alpha") + "\n" + _cpp_body("Beta")
    chunks = chunk_file("file.cc", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "Alpha" in texts
    assert "Beta" in texts
    # Two large functions must split — would be one chunk if .cc fell through to text.
    assert len(chunks) >= 2


def test_cpp_cxx_extension_function_chunked():
    content = _cpp_body("Foo") + "\n" + _cpp_body("Bar")
    chunks = chunk_file("file.cxx", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "Foo" in texts
    assert "Bar" in texts
    assert len(chunks) >= 2


def test_hpp_extension_function_chunked():
    content = (
        "class Widget {\npublic:\n  void DoOne();\n  void DoTwo();\n};\n\n"
        "void Widget::DoOne() { return; }\n\n"
        "void Widget::DoTwo() { return; }\n"
    )
    chunks = chunk_file("widget.hpp", content, URI)
    texts = " ".join(c.text for c in chunks)
    assert "DoOne" in texts
    assert "DoTwo" in texts
