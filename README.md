# Drive → Extractor → Parquet (GitHub Actions)

Fluxo pronto para:
- Buscar **PDFs** do Google Drive (pasta) **a partir de 26/08/2025 14:00:00 America/Sao_Paulo**, sem repetir (`state/processed_ids.json` via `fileId`).
- **Não** reprocessar PDFs repetidos no extrator (`state/OFERTASMATRIZ_STATE.json`, cache **mtime+tamanho**).
- Gerar **Parquet por rodada**: `data/OFERTAS_YYYY-MM-DD_HH:MM:SS.parquet` (hora SP).
- Gerar **XLS por rodada** (`out/OFERTAS_RUN_YYYY-MM-DD_HH-MM-SS.xlsx`) e subir como **Artifact** do run.
- Commitar somente `data/*.parquet` e `state/*.json`.

## Secrets obrigatórios
- `GDRIVE_SERVICE_ACCOUNT_JSON` → conteúdo do **sa.json** do Service Account.
- `DRIVE_FOLDER_ID` → ID da pasta do Google Drive que contém os PDFs.

## Rodando local
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Coloque seu sa.json na raiz e defina DRIVE_FOLDER_ID
export DRIVE_FOLDER_ID="SEU_FOLDER_ID"

python scripts/drive_pull.py
RUN_TS=$(TZ=America/Sao_Paulo date +'%Y-%m-%d_%H-%M-%S') PDF_DIR=inbox TZ_NAME=America/Sao_Paulo python scripts/PDF27.pyw --once
```

## Estrutura
```
scripts/drive_pull.py         # baixa PDFs sem repetir (desde o corte)
scripts/PDF27.pyw             # extrai e gera Parquet+XLS por rodada
.github/workflows/drive_to_parquet.yml
requirements.txt
data/                         # sai o Parquet por rodada
out/                          # sai o XLS por rodada (e OFERTASMATRIZ.xlsx consolidado)
state/                        # estados persistentes (evitar repetição)
```
