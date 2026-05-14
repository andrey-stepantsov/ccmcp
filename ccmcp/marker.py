"""Parser for the per-project .ccmcp marker file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

MARKER_FILENAME = ".ccmcp"


@dataclass
class MarkerFile:
    source_root: str        # absolute, resolved path to the root directory
    name: str               # project name (defaults to directory name)
    tags: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)


def load(root: str) -> MarkerFile | None:
    """Return MarkerFile for root, or None if .ccmcp is absent.

    An empty or unparseable .ccmcp still counts as opt-in (returns a MarkerFile
    with no tags/includes); only a missing file returns None.
    """
    p = Path(root) / MARKER_FILENAME
    if not p.exists():
        return None

    resolved = str(Path(root).resolve())
    default = MarkerFile(source_root=resolved, name=Path(root).name)

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return default

    if not isinstance(data, dict):
        return default

    return MarkerFile(
        source_root=resolved,
        name=str(data.get("name", Path(root).name)),
        tags=[str(t) for t in data.get("tags", [])],
        include=[str(t) for t in data.get("include", [])],
    )
