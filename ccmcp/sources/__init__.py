from dataclasses import dataclass


@dataclass
class SourceFile:
    source_uri: str
    content: str
    etag: str | None = None
    last_modified: str | None = None
    drive_version: str | None = None
