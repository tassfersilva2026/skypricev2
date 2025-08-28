# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os, re, time, math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ========================== CONFIGURAÇÃO BASE DO APP ==========================
st.set_page_config(
    page_title="PD27 — Leitor de saídas (data/)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta simples (opcional no CSS)
st.markdown("""
<style>
.block-container { padding-top: 0.6rem; }
.dataframe { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================= RESOLUÇÃO DE CAMINHOS ==========================
APP_DIR = Path(__file__).resolve().parent
CWD     = Path.cwd()

# Candidatos para localizar a pasta data/ (suporta vários layouts)
CANDIDATES = [
    Path(os.environ.get("DATA_DIR")) if os.environ.get("DATA_DIR") else None,  # ENV sobrescreve
    APP_DIR / "data",         # scripts/data
    APP_DIR.parent / "data",  # raiz/data  ← mais comum
    CWD / "data",             # execução a partir da raiz
]
CANDIDATES = [p for p in CANDIDATES if p is not None]

def pick_data_dir(cands: List[Path]) -> Path:
    for p in cands:
        if p.exists() and p.is_dir():
            return p
    # fallback: cria em raiz/data
    fallback = APP_DIR.parent / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

DEFAULT_DATA_DIR = pick_data_dir(CANDIDATES)

# =============================== UI — SIDEBAR ================================
st.sidebar.header("📁 Fonte de dados (PD27)")
data_dir_input = st.sidebar.text_input(
    "Caminho da pasta de dados",
    value=str(DEFAULT_DATA_DIR.resolve()),
    help="Aponte para a pasta onde o PD27 salva as saídas (ex.: .../data). "
         "Por padrão, detectamos raiz/data mesmo com o app dentro de scripts/."
)
DATA_DIR = Path(data_dir_input).expanduser()

if not DATA_DIR.exists():
    st.sidebar.error(f"Pasta não encontrada: {DATA_DIR}")
    st.stop()

st.sidebar.write(f"Usando: `{DATA_DIR}`")

file_types = st.sidebar.multiselect(
    "Tipos de arquivo a considerar",
    options=["parquet", "csv", "pdf"],
    default=["parquet", "csv", "pdf"],
)

prefix_filter = st.sidebar.text_input(
    "Prefixo do nome do arquivo (opcional)",
    value="",
    placeholder="ex.: OFERTAS ou PD27_",
    help="Deixe vazio para pegar todos. Use para filtrar por prefixo (case-insensitive)."
)

st.sidebar.caption("Dica: defina DATA_DIR como variável de ambiente para fixar o caminho.")

# ================================ UTILIDADES =================================
def list_files(dirpath: Path, types: List[str], prefix: str = "") -> Dict[str, list]:
    prefix_norm = prefix.strip().lower()
    exts = {
        "parquet": [".parquet"],
        "csv":     [".csv"],
        "pdf":     [".pdf"],
    }
    found: Dict[str, list] = {k: [] for k in exts.keys()}
    for p in dirpath.glob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        name_ok = True
        if prefix_norm:
            name_ok = p.name.lower().startswith(prefix_norm)
        if not name_ok:
            continue
        for t in types:
            if ext in exts[t]:
                found[t].append(p)
                break
    # ordena por modificação (mais novo primeiro)
    for k in found:
        found[k].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return found

def _mtimes(flist: list[Path]) -> Tuple[int, ...]:
    return tuple(int(p.stat().st_mtime) for p in flist)

@st.cache_data(show_spinner=False)
def load_tabular(files_parquet: list[Path], files_csv: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    # Parquet
    for fp in files_parquet:
        try:
            df = pd.read_parquet(fp)
            df["__source_file__"] = fp.name
            pieces.append(df)
        except Exception as e:
            st.warning(f"Falha ao ler Parquet: {fp.name} — {e}")

    # CSV
    for fc in files_csv:
        try:
            df = pd.read_csv(fc, sep=None, engine="python")  # auto separador
            df["__source_file__"] = fc.name
            pieces.append(df)
        except Exception as e:
            st.warning(f"Falha ao ler CSV: {fc.name} — {e}")

    if not pieces:
        return pd.DataFrame()

    df_all = pd.concat(pieces, ignore_index=True)

    # Conversões leves: tenta números/datas nas colunas textuais
    for col in df_all.columns:
        if col == "__source_file__":
            continue
        if df_all[col].dtype == "object":
            # tenta numérico
            df_num = pd.to_numeric(df_all[col].astype(str).str.replace(",",".", regex=False), errors="ignore")
            if pd.api.types.is_numeric_dtype(df_num):
                df_all[col] = df_num
                continue
            # tenta datetime
            try:
                df_dt = pd.to_datetime(df_all[col], errors="raise", utc=False, dayfirst=False, infer_datetime_format=True)
                # só aceita se tiver variação (evitar converter códigos aleatórios)
                if df_dt.notna().sum() >= max(3, int(len(df_dt)*0.1)):
                    df_all[col] = df_dt
            except Exception:
                pass
    return df_all

# ============================== COLETA DE ARQUIVOS ============================
files_dict = list_files(DATA_DIR, file_types, prefix_filter)

parquet_files = files_dict.get("parquet", [])
csv_files     = files_dict.get("csv", [])
pdf_files     = files_dict.get("pdf", [])

st.write(f"**Arquivos encontrados** — parquet: {len(parquet_files)} | csv: {len(csv_files)} | pdf: {len(pdf_files)}")

# Revalida cache quando qualquer mtime muda
_ = _mtimes(parquet_files + csv_files)

df = load_tabular(parquet_files, csv_files)

# =============================== VISÃO PRINCIPAL ==============================
st.title("PD27 — Leitura de saídas da pasta data/")

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Arquivos tabulares", len(parquet_files) + len(csv_files))
with colB:
    st.metric("PDFs listados", len(pdf_files))
with colC:
    st.metric("Linhas carregadas", 0 if df.empty else len(df))

tabs = st.tabs(["📊 Dados", "📈 Gráficos rápidos", "📄 PDFs"])

# ------------------------------ TAB DADOS ------------------------------------
with tabs[0]:
    if df.empty:
        st.info("Nenhum .parquet / .csv carregado com o filtro atual. Ajuste o prefixo ou tipos na lateral.")
    else:
        # Colunas & filtros básicos
        st.subheader("Prévia dos dados")
        st.caption("Incluímos a coluna `__source_file__` para rastrear a origem de cada linha.")
        # Seleção de colunas para exibir
        cols = list(df.columns)
        show_cols = st.multiselect(
            "Colunas a exibir",
            options=cols,
            default=cols[: min(12, len(cols))]
        )
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        # Download opcional do combinado
        @st.cache_data(show_spinner=False)
        def to_parquet_bytes(df_in: pd.DataFrame) -> bytes:
            import io
            bio = io.BytesIO()
            df_in.to_parquet(bio, index=False)
            return bio.getvalue()

        st.download_button(
            "⬇️ Baixar combinado (.parquet)",
            data=to_parquet_bytes(df),
            file_name="PD27_combined.parquet",
            mime="application/octet-stream",
        )

        # Info rápida sobre arquivos lidos
        with st.expander("Arquivos lidos (tabulares)"):
            st.write(pd.DataFrame({
                "arquivo": [p.name for p in parquet_files + csv_files],
                "modificado_em": [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) for p in parquet_files + csv_files],
                "tamanho_KB": [round(p.stat().st_size/1024, 1) for p in parquet_files + csv_files],
            }))

# --------------------------- TAB GRÁFICOS RÁPIDOS ----------------------------
with tabs[1]:
    if df.empty:
        st.info("Sem dados tabulares para plotar.")
    else:
        # Heurísticas: escolhe 1 categórica e 1 numérica automaticamente
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "__source_file__"]
        cat_cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])) and c != "__source_file__"]

        st.subheader("Contagem por categoria")
        if cat_cols:
            cat = st.selectbox("Coluna categórica", options=cat_cols, index=0)
            topn = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)
            cnt = df[cat].value_counts(dropna=False).reset_index()
            cnt.columns = [cat, "contagem"]
            cnt = cnt.head(topn)

            chart = alt.Chart(cnt).mark_bar().encode(
                x=alt.X("contagem:Q", title="Contagem"),
                y=alt.Y(f"{cat}:N", sort='-x', title=cat),
                tooltip=[cat, "contagem"]
            ).properties(height=400, width=800)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Não encontrei colunas categóricas para contagem.")

        st.divider()
        st.subheader("Soma por categoria (escolha numérica)")
        if cat_cols and num_cols:
            cat2 = st.selectbox("Categoria (x)", options=cat_cols, index=0, key="cat2")
            num  = st.selectbox("Métrica numérica (y)", options=num_cols, index=0, key="num1")
            agg = df.groupby(cat2, dropna=False)[num].sum().reset_index().sort_values(num, ascending=False).head(30)
            chart2 = alt.Chart(agg).mark_bar().encode(
                x=alt.X(f"{num}:Q", title=f"Soma de {num}"),
                y=alt.Y(f"{cat2}:N", sort='-x', title=cat2),
                tooltip=[cat2, alt.Tooltip(f"{num}:Q", format=",.2f")]
            ).properties(height=400, width=800)
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.info("Para esta visão, preciso de pelo menos 1 coluna categórica e 1 numérica.")

# --------------------------------- TAB PDFs ----------------------------------
with tabs[2]:
    if not pdf_files:
        st.info("Nenhum PDF encontrado com o filtro atual.")
    else:
        st.subheader("Lista de PDFs (sem leitura de conteúdo)")
        df_pdf = pd.DataFrame({
            "arquivo": [p.name for p in pdf_files],
            "modificado_em": [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) for p in pdf_files],
            "tamanho_KB": [round(p.stat().st_size/1024, 1) for p in pdf_files],
        })
        st.dataframe(df_pdf, use_container_width=True, hide_index=True)
        st.caption("Se quiser que eu extraia texto dos PDFs, dá pra plugar pdfplumber numa próxima versão.")
