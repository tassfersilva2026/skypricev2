# -*- coding: utf-8 -*-
# Drive -> baixa SOMENTE PDFs com createdTime >= corte (26/08/2025 14:00:00 SP)
# Evita repetição por fileId (state/processed_ids.json). Salva em inbox/ e sai.
import os, sys, io, json, re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not FOLDER_ID:
    print("Defina DRIVE_FOLDER_ID (secret) ou passe como argumento.")
    sys.exit(1)

CUTOVER_SP = os.environ.get("CUTOVER_SP", "2025-08-26T14:00:00")
TZ_NAME    = os.environ.get("TZ_NAME", "America/Sao_Paulo")

INBOX      = Path("inbox"); INBOX.mkdir(parents=True, exist_ok=True)
STATE_FILE = Path("state/processed_ids.json"); STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
try:
    creds = Credentials.from_service_account_file("sa.json", scopes=SCOPES)
except Exception as e:
    print(f"Erro carregando credenciais do Drive: {e}")
    sys.exit(1)
service = build("drive", "v3", credentials=creds, cache_discovery=False)

def to_utc_from_sp(local_str: str) -> str:
    tz = ZoneInfo(TZ_NAME)
    fmt = "%Y-%m-%dT%H:%M:%S" if "T" in local_str else "%Y-%m-%d %H:%M:%S"
    dt = datetime.strptime(local_str, fmt).replace(tzinfo=tz).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except:
            pass
    return {"processed_ids": []}

def save_state(s: Dict[str, Any]):
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def download_pdf(file_id: str, name: str):
    req = service.files().get_media(fileId=file_id)
    dest = INBOX / name
    with dest.open("wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    print(f"[ok] {dest}")

def safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name)

def run():
    cut_utc = to_utc_from_sp(CUTOVER_SP)
    q = (
        f"'{FOLDER_ID}' in parents and trashed=false "
        f"and mimeType='application/pdf' and createdTime >= '{cut_utc}'"
    )
    state = load_state()
    processed = set(state.get("processed_ids", []))
    page_token = None; found = 0; new = 0
    try:
        while True:
            resp = service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,createdTime)",
                orderBy="createdTime",
                pageSize=100,
                pageToken=page_token,
            ).execute()
            files = resp.get("files", [])
            found += len(files)
            for f in files:
                fid, name = f["id"], safe(f["name"])
                if fid in processed:
                    continue
                download_pdf(fid, name)
                processed.add(fid); new += 1
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as err:
        print(f"[erro] Falha na comunicação com Google Drive: {err}", file=sys.stderr)
    except Exception as e:
        print(f"[erro] Erro inesperado: {e}", file=sys.stderr)

    state["processed_ids"] = sorted(processed)
    save_state(state)
    print(f"[fim] encontrados={found}, novos={new}, inbox={INBOX}")

if __name__ == "__main__":
    run()
