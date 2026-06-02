from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pathspec
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from ccmcp.sources import SourceFile

log = logging.getLogger(__name__)

_GITIGNORE = ".gitignore"
# Newer pathspec (>=0.12) renames "gitwildmatch" to "gitignore"; fall back if
# we're somehow running against an older release.
try:
    pathspec.PathSpec.from_lines("gitignore", [])
    _PATHSPEC_STYLE = "gitignore"
except (LookupError, KeyError):
    _PATHSPEC_STYLE = "gitwildmatch"


def _matches_ignore(name: str, patterns: list[str]) -> bool:
    """Match a single path component against ccmcp.ignore glob patterns."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


class GitIgnoreCascade:
    """Per-root cascade of .gitignore patterns.

    Built incrementally: the constructor only records the root. Callers invoke
    `load_dir(dirpath)` for each directory they visit (in ancestor-first order)
    so the cascade is populated lazily as a tree walk descends. Because
    gitignore semantics only need ANCESTOR .gitignore files when judging a
    path, a single pruned walk that calls `load_dir` at each step produces
    correct results without re-walking the tree.

    To test a path, walks up its ancestors (root → leaf) and applies each spec
    relative to the directory that owns it. Within a single .gitignore,
    pathspec handles `!` negation correctly; cross-file negation is not
    modelled (rare in practice and the simplification is documented in the
    project notes).
    """

    def __init__(self, root: str):
        self._root = os.path.realpath(root)
        self._specs: dict[str, pathspec.PathSpec] = {}
        self._loaded_dirs: set[str] = set()

    def load_dir(self, dirpath: str) -> None:
        """Load `dirpath/.gitignore` if present and not yet loaded."""
        real = os.path.realpath(dirpath)
        if real in self._loaded_dirs:
            return
        self._loaded_dirs.add(real)
        gi = Path(dirpath) / _GITIGNORE
        if not gi.is_file():
            return
        try:
            lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return
        self._specs[real] = pathspec.PathSpec.from_lines(_PATHSPEC_STYLE, lines)

    def _load(self) -> None:
        """Backwards-compatible eager load: walk the whole tree, no pruning.

        Retained so callers that constructed a cascade via the old eager flow
        (or via tests that touch this private helper) still get identical
        behaviour. New code paths in scan()/watch() use `load_dir` instead.
        """
        for dirpath, _dirnames, _filenames in os.walk(self._root):
            self.load_dir(dirpath)

    def is_ignored(self, path: str, is_dir: bool = False) -> bool:
        abs_path = os.path.realpath(path)
        try:
            common = os.path.commonpath([abs_path, self._root])
        except ValueError:
            return False
        if common != self._root:
            return False  # outside the cascade — leave to caller

        # Walk ancestor .gitignore dirs from root to the closest one.
        ancestors: list[str] = []
        cur = abs_path if is_dir else os.path.dirname(abs_path)
        while True:
            if cur in self._specs:
                ancestors.append(cur)
            if cur == self._root or len(cur) <= len(self._root):
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        ancestors.reverse()

        for anc in ancestors:
            spec = self._specs[anc]
            rel = os.path.relpath(abs_path, anc)
            if rel in (".", ""):
                continue
            rel = rel.replace(os.sep, "/")
            if is_dir and not rel.endswith("/"):
                rel = rel + "/"
            if spec.match_file(rel):
                return True
        return False


def _excluded(
    abs_path: str,
    is_dir: bool,
    ignore_patterns: list[str],
    cascade: GitIgnoreCascade | None,
) -> bool:
    """True if this path should be skipped per ccmcp.ignore or .gitignore."""
    name = os.path.basename(abs_path.rstrip(os.sep))
    if _matches_ignore(name, ignore_patterns):
        return True
    if cascade is not None and cascade.is_ignored(abs_path, is_dir=is_dir):
        return True
    return False


def scan(
    roots: list[str],
    extensions: list[str],
    ignore: list[str],
    respect_gitignore: bool = True,
) -> list[SourceFile]:
    ext_set = {e.lower() for e in extensions}
    files: list[SourceFile] = []
    for root in roots:
        cascade = GitIgnoreCascade(root) if respect_gitignore else None
        # Single pruned walk: load each directory's .gitignore into the
        # cascade BEFORE pruning, so children can be judged against any
        # patterns the directory itself contributes; then prune. os.walk's
        # top-down semantics guarantee ancestors are visited before
        # descendants, which is all gitignore semantics require.
        for dirpath, dirnames, filenames in os.walk(root):
            if cascade is not None:
                cascade.load_dir(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if not _excluded(os.path.join(dirpath, d), True, ignore, cascade)
            ]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if _excluded(fpath, False, ignore, cascade):
                    continue
                if Path(fname).suffix.lower() not in ext_set:
                    continue
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    files.append(SourceFile(source_uri=f"file://{fpath}", content=content))
                except OSError:
                    pass
    return files


def _build_cascade_pruned(root: str, ignore_patterns: list[str]) -> GitIgnoreCascade:
    """Pre-populate a cascade by doing one pruned walk of `root`.

    Used by watch() so .git/, node_modules/, build/, etc. are skipped during
    cascade construction. The prune predicate is exactly the one used by the
    scan path, so the resulting cascade covers every directory the watcher
    will be asked to judge.
    """
    cascade = GitIgnoreCascade(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        cascade.load_dir(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if not _excluded(os.path.join(dirpath, d), True, ignore_patterns, cascade)
        ]
    return cascade


class _Handler(FileSystemEventHandler):
    def __init__(
        self,
        callback: Callable[[SourceFile], None],
        extensions: set[str],
        ignore: list[str],
        cascades: list[GitIgnoreCascade],
    ):
        self._cb = callback
        self._extensions = extensions
        self._ignore = ignore
        self._cascades = cascades

    def _cascade_for(self, abs_path: str) -> GitIgnoreCascade | None:
        for c in self._cascades:
            try:
                common = os.path.commonpath([abs_path, c._root])
            except ValueError:
                continue
            if common == c._root:
                return c
        return None

    def _ok(self, path: str) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self._extensions:
            return False
        if any(_matches_ignore(part, self._ignore) for part in p.parts):
            return False
        abs_path = os.path.realpath(path)
        cascade = self._cascade_for(abs_path)
        if cascade is not None and cascade.is_ignored(abs_path, is_dir=False):
            return False
        return True

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._ok(str(event.src_path)):
            self._emit(str(event.src_path))

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._ok(str(event.src_path)):
            self._emit(str(event.src_path))

    def _emit(self, path: str):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self._cb(SourceFile(source_uri=f"file://{path}", content=content))
        except OSError:
            pass


def watch(
    roots: list[str],
    extensions: list[str],
    ignore: list[str],
    callback: Callable[[SourceFile], None],
    poll_interval: int = 30,
    respect_gitignore: bool = True,
) -> PollingObserver:
    cascades = (
        [_build_cascade_pruned(r, ignore) for r in roots]
        if respect_gitignore
        else []
    )
    handler = _Handler(callback, {e.lower() for e in extensions}, ignore, cascades)
    # PollingObserver is used unconditionally: bind mounts in Docker (and WSL2
    # /mnt/ paths) do not propagate inotify events reliably.
    observer = PollingObserver(timeout=poll_interval)
    for root in roots:
        observer.schedule(handler, root, recursive=True)
    observer.start()
    return observer
