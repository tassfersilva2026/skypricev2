# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import time as dtime
from typing import List, Tuple

import gc
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")
alt.data_transformers.disable_max_rows()

# --------- CONFIG ---------
GLOB_PATTERN = "OFERTAS_*.parquet"
MAX_NEW_FILES            = 80
SKIP_LARGER_THAN_MB      = 600
SAMPLE_MAX_ROWS_PER_FILE = None
CONCAT_BATCH_SIZE        = 15

COLUMNS_NEEDED = [
    "IDPESQUISA","CIA","HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA","TIPO_VOO",
    "DATA_EMBARQUE","DATAHORA_BUSCA","AGENCIA_COMP","PRECO","TRECHO","ADVP","RANKING"
]

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False

# --------- PATHS ---------
def resolve_data_dir() -> Path:
    """Garante /data na RAIZ do repo. Se rodar em subpasta, ainda acha."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data",
        Path.cwd() / "data",
    ]
    # Preferir quem tem arquivos válidos
    for p in candidates:
        if p.exists() and any(p.glob(GLOB_PATTERN)):
            return p
    # Senão, quem existe
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

DATA_DIR = resolve_data_dir()

# --------- HELPERS ---------
def std_agencia(raw: str) -> str:
    ag = (raw or "").strip().upper()
    if ag == "BOOKINGCOM":            return "BOOKING.COM"
    if ag == "KIWICOM":               return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":   return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":   return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"):  return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"):   return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"):   return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"):   return "KISSANDFLY"
    if ag.startswith("ZUPPER"):       return "ZUPPER"
    if ag.startswith("MYTRIP"):       return "MYTRIP"
    if ag.startswith("GOTOGATE"):     return "GOTOGATE"
    if ag.startswith("DECOLAR"):      return "DECOLAR"
    if ag.startswith("EXPEDIA"):      return "EXPEDIA"
    if ag.startswith("GOL"):          return "GOL"
    if ag.startswith("LATAM"):        return "LATAM"
    if ag.startswith("TRIPCOM"):      return "TRIP.COM"
    if ag.startswith("VIAJANET"):     return "VIAJANET"
    if ag in ("", "NAN", "NONE", "NULL", "SKYSCANNER"): return "SEM OFERTAS"
    return ag

def std_cia(raw: str) -> str:
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    if s in {"AD","AZU"} or s.startswith("AZUL") or "AZUL" in s_simple: return "AZUL"
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple: return "GOL"
    if s in {"LA","JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple: return "LATAM"
    if s in {"AZUL","GOL","LATAM"}: return s
    return s

def advp_nearest(x) -> int:
    try: v = float(str(x).replace(",", "."))
    except Exception: v = np.nan
    if np.isnan(v): v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

def _rename_minimal(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
              5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",
              9:"PRECO",10:"TRECHO",11:"ADVP",12:"RANKING"}
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)
    return df

def _normalize_final(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df.get("HORA_BUSCA"), errors="coerce").dt.hour
    for c in ["DATA_EMBARQUE","DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    if "PRECO" in df.columns:
        df["PRECO"] = (df["PRECO"].astype(str)
                       .str.replace(r"[^\d,.-]", "", regex=True)
                       .str.replace(",", ".", regex=False))
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")
    df["AGENCIA_NORM"] = df.get("AGENCIA_COMP", pd.Series([None]*len(df))).apply(std_agencia)
    df["ADVP_CANON"]   = df.get("ADVP", pd.Series([None]*len(df))).apply(advp_nearest)
    df["CIA_NORM"]     = df.get("CIA", pd.Series([None]*len(df))).apply(std_cia)
    return df

def _discover_new_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    files = sorted((p for p in data_dir.glob(GLOB_PATTERN) if p.is_file()),
                   key=lambda p: p.stat().st_mtime)
    return files

def _files_cache_key(paths: List[Path]) -> Tuple[str, ...]:
    try:
        return tuple(f"{p.name}:{p.stat().st_mtime_ns}" for p in paths)
    except Exception:
        return tuple(p.name for p in paths)

def _mb(size_bytes: int) -> float:
    return round(size_bytes / (1024*1024), 2)

def _safe_read_parquet(p: Path, columns: List[str]) -> pd.DataFrame:
    if HAVE_PYARROW:
        try:
            table = pq.read_table(p.as_posix(), columns=[c for c in columns if c])
            if SAMPLE_MAX_ROWS_PER_FILE:
                table = table.slice(0, min(SAMPLE_MAX_ROWS_PER_FILE, table.num_rows))
            return table.to_pandas(types_mapper=pd.ArrowDtype)
        except Exception as e:
            st.warning(f"pyarrow falhou em {p.name}: {e}; tentando pandas…")
    return pd.read_parquet(p, columns=[c for c in columns if c])

@st.cache_data(show_spinner=False)
def load_base(data_dir: Path, new_paths_key: Tuple[str, ...], limit_new: int | None = MAX_NEW_FILES) -> pd.DataFrame:
    _ = new_paths_key
    files_all = _discover_new_files(data_dir)
    files = files_all[-limit_new:] if (limit_new and len(files_all) > limit_new) else files_all

    if not files:
        st.error("Nenhum `OFERTAS_*.parquet` encontrado em `data/` na raiz do repositório.")
        st.stop()

    frames: List[pd.DataFrame] = []
    processed = 0
    progress = st.progress(0.0, text="Preparando leitura…")

    for idx, p in enumerate(files, start=1):
        size_mb = _mb(p.stat().st_size)
        if SKIP_LARGER_THAN_MB and size_mb > SKIP_LARGER_THAN_MB:
            st.warning(f"Pulando {p.name} ({size_mb} MB) — acima de {SKIP_LARGER_THAN_MB} MB.")
            progress.progress(idx/len(files), text=f"Pulando {p.name} (grande demais)…")
            continue

        try:
            dfn = _safe_read_parquet(p, COLUMNS_NEEDED)
            dfn = _rename_minimal(dfn)
            frames.append(dfn)
            processed += len(dfn)
            progress.progress(
                idx/len(files),
                text=f"Lido {p.name} • {size_mb} MB • linhas acumuladas: {processed:,}".replace(",", ".")
            )
        except MemoryError:
            st.error(f"Sem memória ao ler {p.name}. Reduza MAX_NEW_FILES ou ative SAMPLE_MAX_ROWS_PER_FILE.")
            st.stop()
        except Exception as e:
            st.warning(f"Falha ao ler {p.name}: {e}")
            progress.progress(idx/len(files), text=f"Erro em {p.name} — continuando…")
            continue

        if len(frames) >= CONCAT_BATCH_SIZE:
            frames = [pd.concat(frames, ignore_index=True)]
            gc.collect()

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else (frames[0] if frames else pd.DataFrame(columns=COLUMNS_NEEDED))
    df = _normalize_final(df)
    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "IDPESQUISA" not in df.columns:
        return pd.DataFrame(columns=["IDPESQUISA","R1","R2","R3"])
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].dropna().unique()})
    for r in (1, 2, 3):
        s = (
            df[df["RANKING"] == r]
            .sort_values(["IDPESQUISA"])
            .drop_duplicates(subset=["IDPESQUISA"])
        )
        base = base.merge(
            s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA",
            how="left"
        )
    for r in (1, 2, 3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty: return "—"
    max_d = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce").max()
    if pd.isna(max_d): return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day.get("HORA_BUSCA"), errors="coerce").dt.time
    try:
        max_h = max([h for h in hh if pd.notna(h)], default=None)
    except Exception:
        max_h = None
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True) -> alt.Chart:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc:
        d = d.sort_values(x_col, ascending=False)
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:Q", title=None),
            y=alt.Y(f"{y_col}:N", sort='-x', title=None),
            tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col),
                     alt.Tooltip(f"{x_col}:Q", title=x_col, format=",.2f")]
        )
        .properties(height=320)
    )

# --------- UI ---------
st.title("Painel — Skyscanner (Agregados Leves)")
st.caption("Lendo `OFERTAS_*.parquet` da pasta **/data** na raiz do repo.")

# Diagnóstico do diretório/arquivos
all_paths = _discover_new_files(DATA_DIR)
with st.expander("Diagnóstico do diretório de dados", expanded=False):
    st.write("DATA_DIR:", str(DATA_DIR))
    st.write("Arquivos encontrados:", len(all_paths))
    st.write("Primeiros 10:", [p.name for p in all_paths[:10]])

paths_key = _files_cache_key(all_paths)
df = load_base(DATA_DIR, paths_key, limit_new=MAX_NEW_FILES)

# KPIs
last_upd = last_update_from_cols(df)
k1, k2, k3 = st.columns(3)
k1.metric("Pesquisas (ID únicos)", fmt_int(df["IDPESQUISA"].nunique()))
k2.metric("Linhas carregadas", fmt_int(len(df)))
k3.metric("Última atualização", last_upd)

st.divider()

wins = winners_by_position(df)
total_ids = len(wins)
if total_ids == 0:
    st.info("Sem dados para computar vencedores por ranking.")
else:
    cols = st.columns(3)
    for i, r in enumerate((1, 2, 3)):
        vc = wins[f"R{r}"].value_counts(dropna=False)
        share = (vc / total_ids * 100.0).rename("PCT").reset_index().rename(columns={"index": "AGENCIA"})
        top10 = share.head(10)
        with cols[i]:
            st.subheader(f"Ranking {r} — Share por Agência")
            st.altair_chart(make_bar(top10, "PCT", "AGENCIA"), use_container_width=True)

st.divider()

r1 = df[df.get("RANKING").astype("Int64") == 1]
if not r1.empty:
    cia_share = (
        r1["CIA_NORM"].fillna("N/A").value_counts(dropna=False, normalize=True) * 100.0
    ).rename("PCT").reset_index().rename(columns={"index": "CIA"})
    st.subheader("Ranking 1 — Share por CIA")
    st.altair_chart(make_bar(cia_share, "PCT", "CIA"), use_container_width=True)
else:
    st.info("Não há dados para Ranking 1.")

st.caption("Se pesar: reduza MAX_NEW_FILES (ex.: 30) ou use SAMPLE_MAX_ROWS_PER_FILE (ex.: 1_000_000).")
