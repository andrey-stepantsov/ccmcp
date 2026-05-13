from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_TOKENS = 512
MIN_TOKENS = 50
CHARS_PER_TOKEN = 4  # ~4 chars per token for English/code


def _tok(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    source_uri: str


HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

_CODE_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py":  re.compile(r"^(async def |def |class )", re.MULTILINE),
    ".go":  re.compile(r"^func ", re.MULTILINE),
    ".rs":  re.compile(r"^(pub fn |fn |impl |pub impl |pub struct |struct )", re.MULTILINE),
    ".js":  re.compile(r"^(function |class |const \w+ = (?:async )?(?:function|\())", re.MULTILINE),
    ".ts":  re.compile(
        r"^(function |class |export function |export class "
        r"|const \w+ = (?:async )?(?:function|\())",
        re.MULTILINE,
    ),
    ".c":   re.compile(r"^[A-Za-z_][\w\s*]+\s+\w+\s*\([^)]*\)\s*\{", re.MULTILINE),
    ".cpp": re.compile(r"^[A-Za-z_][\w\s*:]+\s+\w+\s*\([^)]*\)\s*\{", re.MULTILINE),
    ".h":   re.compile(r"^[A-Za-z_][\w\s*:]+\s+\w+\s*\([^)]*\)\s*[{;]", re.MULTILINE),
}

_MARKDOWN_EXTS = {".md", ".rst"}
_CODE_EXTS = set(_CODE_PATTERNS)
_TEXT_EXTS = {".txt", ".yaml", ".yml", ".json"}


def chunk_file(path: str, content: str, source_uri: str) -> list[Chunk]:
    ext = Path(path).suffix.lower()
    if ext in _MARKDOWN_EXTS:
        return _chunk_markdown(content, source_uri)
    if ext in _CODE_EXTS:
        return _chunk_code(content, source_uri, ext)
    return _chunk_text(content, source_uri)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, max_tok: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if _tok(candidate) <= max_tok:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if _tok(para) > max_tok:
                chunks.extend(_hard_split(para, max_tok))
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, max_tok: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = (current + "\n" + line) if current else line
        if _tok(candidate) <= max_tok:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if _tok(line) <= max_tok:
                current = line
            else:
                # line itself is too long — split by words
                current = ""
                for word in line.split():
                    trial = (current + " " + word) if current else word
                    if _tok(trial) <= max_tok:
                        current = trial
                    else:
                        if current:
                            chunks.append(current)
                        current = word
    if current:
        chunks.append(current)
    return chunks


def _merge_small(chunks: list[str], min_tok: int) -> list[str]:
    if not chunks:
        return chunks
    result = [chunks[0]]
    for chunk in chunks[1:]:
        if _tok(result[-1]) < min_tok:
            result[-1] = result[-1] + "\n\n" + chunk
        else:
            result.append(chunk)
    return result


def _first_heading(text: str) -> str:
    m = HEADING_RE.search(text)
    return m.group(2).strip() if m else ""


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _chunk_markdown(content: str, source_uri: str) -> list[Chunk]:
    matches = list(HEADING_RE.finditer(content))
    if not matches:
        return _chunk_text(content, source_uri)

    sections: list[tuple[str, str]] = []
    preamble = content[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(0)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        sections.append((heading, body))

    raw: list[tuple[str, str]] = []  # (heading, text)
    for heading, body in sections:
        full = (heading + "\n\n" + body).strip() if heading else body
        if _tok(full) <= MAX_TOKENS:
            raw.append((heading, full))
        else:
            budget = MAX_TOKENS - (_tok(heading) + 2 if heading else 0)
            for para in _split_paragraphs(body, budget):
                text = (heading + "\n\n" + para).strip() if heading else para
                raw.append((heading, text))

    texts = _merge_small([t for _, t in raw], MIN_TOKENS)
    return [
        Chunk(text=t, section=_first_heading(t), chunk_index=i, source_uri=source_uri)
        for i, t in enumerate(texts)
    ]


def _chunk_code(content: str, source_uri: str, ext: str) -> list[Chunk]:
    pattern = _CODE_PATTERNS.get(ext)
    if pattern is None:
        return _chunk_text(content, source_uri)

    matches = list(pattern.finditer(content))
    if not matches:
        return _chunk_text(content, source_uri)

    sections: list[str] = []
    preamble = content[: matches[0].start()].strip()
    if preamble:
        sections.append(preamble)

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[m.start() : end].strip()
        if not block:
            continue
        if _tok(block) <= MAX_TOKENS:
            sections.append(block)
        else:
            sections.extend(_hard_split(block, MAX_TOKENS))

    texts = _merge_small(sections, MIN_TOKENS)
    return [
        Chunk(text=t, section="", chunk_index=i, source_uri=source_uri)
        for i, t in enumerate(texts)
    ]


def _chunk_text(content: str, source_uri: str) -> list[Chunk]:
    texts = _merge_small(_split_paragraphs(content, MAX_TOKENS), MIN_TOKENS)
    return [
        Chunk(text=t, section="", chunk_index=i, source_uri=source_uri)
        for i, t in enumerate(texts)
    ]
