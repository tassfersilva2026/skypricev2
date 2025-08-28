# scripts/streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import streamlit as st

# =========================
# Config do aplicativo
# =========================
st.set_page_config(page_title="Painel OFERTAS (Parquet)", layout="wide")

st.title("📦 Painel — OFERTAS (Parquet)")
st.caption("Lendo todos os arquivos que começam com **OFERTAS** em `/data` (raiz do repositório).")

# =========================
# Localização da pasta data
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
DATA_DIR   = REPO_ROOT / "data"

if not DATA_DIR.exists():
    st.error(f"Pasta de dados não encontrada: {DATA_DIR}")
    st.stop()

# =========================
# Descobrir arquivos
# =========================
all_files = sorted(DATA_DIR.glob("OFERTAS*.parquet"))
if not all_files:
    st.warning("Nenhum arquivo encontrado com padrão **OFERTAS*.parquet** em /data.")
    st.stop()

# Mostrar lista de arquivos (colapsável)
with st.expander("Arquivos encontrados", expanded=False):
    for p in all_files:
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = "?"
        st.write(f"• `{p.name}`  _(modificado: {mtime})_")

# Opção para carregar apenas os N mais recentes (0 = todos)
col_lim, col_engine = st.columns([1,1])
with col_lim:
    n_latest = st.number_input("Carregar somente os N mais recentes (0 = todos)", min_value=0, value=0, step=1)
with col_engine:
    engine = st.selectbox("Engine de leitura", ["pyarrow", "auto"], index=0,
                          help="Se 'auto' der ruim, fique no 'pyarrow'.")

files_to_load = all_files if n_latest == 0 else all_files[-int(n_latest):]

# =========================
# Cache do carregamento
# =========================
@st.cache_data(show_spinner=True)
def _load_parquets(paths_with_mtime: tuple[tuple[str,int], ...], engine_choice: str) -> pd.DataFrame:
    dfs = []
    for p_str, _mt in paths_with_mtime:
        fp = Path(p_str)
        try:
            df = pd.read_parquet(fp, engine=None if engine_choice=="auto" else engine_choice)
            # Padronizações leves e opcionais:
            # Datas/Horas
            for c in ("DATAHORA_BUSCA", "DATA_HORA_BUSCA"):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            if "DATA_BUSCA" in df.columns:
                df["DATA_BUSCA"] = pd.to_datetime(df["DATA_BUSCA"], errors="coerce").dt.date
            if "HORA_BUSCA" in df.columns:
                # aceita "HH:MM:SS" como texto
                df["HORA_BUSCA"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.time

            # Numéricos comuns
            for num_col in ("ANTECEDENCIA", "ADVP", "NUMERO_VOO", "NUM_VOO", "PRECO", "VALOR_TOTAL"):
                if num_col in df.columns:
                    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

            # Trecho, se existir origem/destino e faltar "TRECHO"
            cand_origem = [c for c in df.columns if c.upper() in ("ORIGEM","DE","FROM")]
            cand_dest   = [c for c in df.columns if c.upper() in ("DESTINO","PARA","TO")]
            if "TRECHO" not in df.columns and cand_origem and cand_dest:
                o, d = cand_origem[0], cand_dest[0]
                df["TRECHO"] = (df[o].astype(str).str.upper().str.strip() + "-" +
                                df[d].astype(str).str.upper().str.strip())

            # CIA padronizada (cria alias 'CIA_PAD' se possível)
            cia_cols = [c for c in df.columns if c.upper() in ("CIA","CIA_AEREA","COMPANHIA")]
            if cia_cols and "CIA_PAD" not in df.columns:
                base = cia_cols[0]
                df["CIA_PAD"] = df[base].astype(str).str.upper().str.strip()

            # Agência padronizada (cria alias 'AGENCIA_PAD' se possível)
            ag_cols = [c for c in df.columns if c.upper().startswith("AGENCIA") or c.upper() in ("AG","AGENCY")]
            if ag_cols and "AGENCIA_PAD" not in df.columns:
                base = ag_cols[0]
                df["AGENCIA_PAD"] = (df[base].astype(str)
                                     .str.upper()
                                     .str.replace("BOOKINGCOM","BOOKING.COM", regex=False)
                                     .str.replace("KIWICOM","KIWI.COM", regex=False)
                                     .str.strip())

            dfs.append(df)
        except Exception as e:
            # Se um arquivo der erro, segue o baile e mostra aviso
            st.warning(f"Falha ao ler `{fp.name}`: {e}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

key_for_cache = tuple((str(p), p.stat().st_mtime_ns) for p in files_to_load)
df = _load_parquets(key_for_cache, engine)

if df.empty:
    st.error("Nenhuma linha carregada. Verifique se os arquivos possuem dados válidos.")
    st.stop()

st.success(f"✅ Carregado: {len(files_to_load)} arquivo(s), **{len(df):,}** linhas.")

# =========================
# Filtros rápidos (dinâmicos)
# =========================
left, mid, right = st.columns([1.2,1,1])

# Filtro por intervalo de data se tiver DATAHORA_BUSCA ou DATA_BUSCA
date_col = None
if "DATAHORA_BUSCA" in df.columns:
    date_col = "DATAHORA_BUSCA"
elif "DATA_BUSCA" in df.columns:
    date_col = "DATA_BUSCA"

if date_col:
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        min_d = df[date_col].min().date()
        max_d = df[date_col].max().date()
    else:
        # quando for apenas 'date' (no caso de DATA_BUSCA)
        min_d = pd.to_datetime(df[date_col], errors="coerce").min()
        max_d = pd.to_datetime(df[date_col], errors="coerce").max()
        min_d = min_d.date() if pd.notna(min_d) else date.today()
        max_d = max_d.date() if pd.notna(max_d) else date.today()
    with left:
        d_ini, d_fim = st.date_input("Data (intervalo)", (min_d, max_d))
    if isinstance(d_ini, tuple) or isinstance(d_ini, list):
        d_ini, d_fim = d_ini
    # Aplicar filtro de data
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df[(df[date_col].dt.date >= d_ini) & (df[date_col].dt.date <= d_fim)]
    else:
        df = df[(pd.to_datetime(df[date_col], errors="coerce").dt.date >= d_ini) &
                (pd.to_datetime(df[date_col], errors="coerce").dt.date <= d_fim)]

# Filtro por CIA se existir
cia_col = None
for c in ("CIA_PAD","CIA","CIA_AEREA","COMPANHIA"):
    if c in df.columns:
        cia_col = c
        break
if cia_col:
    with mid:
        sel_cias = st.multiselect("Filtrar CIA", sorted([x for x in df[cia_col].dropna().astype(str).unique()]))
    if sel_cias:
        df = df[df[cia_col].astype(str).isin(sel_cias)]

# Filtro por Agência se existir
ag_col = None
for c in ("AGENCIA_PAD","AGENCIA","AG","AGENCY"):
    if c in df.columns:
        ag_col = c
        break
if ag_col:
    with right:
        sel_ags = st.multiselect("Filtrar Agência", sorted([x for x in df[ag_col].dropna().astype(str).unique()]))
    if sel_ags:
        df = df[df[ag_col].astype(str).isin(sel_ags)]

# =========================
# KPIs rápidos
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Linhas", f"{len(df):,}")
k2.metric("Arquivos", f"{len(files_to_load)}")
if cia_col:
    k3.metric("Cias únicas", f"{df[cia_col].nunique()}")
if ag_col:
    k4.metric("Agências únicas", f"{df[ag_col].nunique()}")

# =========================
# Tabela
# =========================
st.subheader("Tabela consolidada")
st.dataframe(df, use_container_width=True, hide_index=True)

# =========================
# (Opcional) Top Cias por contagem
# =========================
try:
    import altair as alt
    if cia_col:
        st.subheader("Top Cias por quantidade de registros")
        top = (df.groupby(cia_col, dropna=False)
                 .size()
                 .reset_index(name="qtd")
                 .sort_values("qtd", ascending=False)
                 .head(20))
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("qtd:Q", title="Quantidade"),
            y=alt.Y(f"{cia_col}:N", sort='-x', title="CIA")
        ).properties(height=500)
        st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.info(f"Gráfico desativado (Altair indisponível ou erro: {e})")
