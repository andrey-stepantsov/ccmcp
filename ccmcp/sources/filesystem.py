from __future__ import annotations

import fnmatch
import functools
import os
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from ccmcp.sources import SourceFile


def _matches_ignore(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


@functools.lru_cache(maxsize=512)
def _gitignore_patterns(directory: str) -> list[str]:
    gi = Path(directory) / ".gitignore"
    if not gi.exists():
        return []
    patterns = []
    with open(gi, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip("/"))
    return patterns


def scan(roots: list[str], extensions: list[str], ignore: list[str]) -> list[SourceFile]:
    ext_set = {e.lower() for e in extensions}
    files: list[SourceFile] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            gi = _gitignore_patterns(dirpath)
            combined = ignore + gi
            dirnames[:] = [d for d in dirnames if not _matches_ignore(d, combined)]
            for fname in filenames:
                if _matches_ignore(fname, combined):
                    continue
                if Path(fname).suffix.lower() not in ext_set:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    files.append(SourceFile(source_uri=f"file://{fpath}", content=content))
                except OSError:
                    pass
    return files


class _Handler(FileSystemEventHandler):
    def __init__(
        self,
        callback: Callable[[SourceFile], None],
        extensions: set[str],
        ignore: list[str],
    ):
        self._cb = callback
        self._extensions = extensions
        self._ignore = ignore

    def _ok(self, path: str) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self._extensions:
            return False
        return not any(_matches_ignore(part, self._ignore) for part in p.parts)

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
) -> PollingObserver:
    handler = _Handler(callback, {e.lower() for e in extensions}, ignore)
    # PollingObserver is used unconditionally: bind mounts in Docker (and WSL2
    # /mnt/ paths) do not propagate inotify events reliably.
    observer = PollingObserver(timeout=poll_interval)
    for root in roots:
        observer.schedule(handler, root, recursive=True)
    observer.start()
    return observer
