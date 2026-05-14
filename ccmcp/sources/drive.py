from __future__ import annotations

import io
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from ccmcp.sources import SourceFile

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_EXPORTABLE = {"application/vnd.google-apps.document": "text/plain"}
_DIRECT = {"text/plain", "text/markdown"}


def _service(credentials_file: str):
    resolved = Path(credentials_file).expanduser().resolve()
    creds = service_account.Credentials.from_service_account_file(
        str(resolved), scopes=_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def fetch_all(credentials_file: str, folders: list, state) -> list[SourceFile]:
    svc = _service(credentials_file)
    results: list[SourceFile] = []
    for folder in folders:
        for item in _list_folder(svc, folder.id):
            sf = _fetch_item(svc, item, state)
            if sf:
                results.append(sf)
    return results


def _list_folder(svc, folder_id: str) -> list[dict]:
    items: list[dict] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, version)",
            pageToken=page_token,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _fetch_item(svc, item: dict, state) -> SourceFile | None:
    uri = f"drive://{item['id']}"
    record = state.get(uri)
    if record and record.drive_version == str(item.get("version", "")):
        return None

    mime = item["mimeType"]
    try:
        if mime in _EXPORTABLE:
            raw = svc.files().export(fileId=item["id"], mimeType=_EXPORTABLE[mime]).execute()
            content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        elif mime in _DIRECT:
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=item["id"]))
            done = False
            while not done:
                _, done = dl.next_chunk()
            content = buf.getvalue().decode("utf-8", errors="ignore")
        else:
            return None
    except Exception:
        return None

    return SourceFile(
        source_uri=uri,
        content=content,
        drive_version=str(item.get("version", "")),
    )
