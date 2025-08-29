# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os, re, time, math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ========================== CONFIGURAÇÃO BASE DO APP ==========================
st.set_page_config(
    page_title="PD27 — Leitor de saídas (GitHub: data/)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 0.6rem; }
.dataframe { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
CWD     = Path.cwd()

# ========================== SYNC DO GITHUB -> .cache_data/data ===============
ALLOWED_EXTS = (".parquet", ".csv", ".pdf")

def _secrets_or_env(key: str, default: Optional[str]=None) -> Optional[str]:
    # Busca primeiro em st.secrets (Streamlit Cloud) e depois em variáveis de ambiente
    if key in st.secrets: return st.secrets[key]
    return os.environ.get(key, default)

def _gh_headers(token: Optional[str]) -> Dict[str, str]:
    hdrs = {"Accept": "application/vnd.github+json"}
    if token:
        hdrs["Authorization"] = f"token {token}"
    return hdrs

def _gh_headers_raw(token: Optional[str]) -> Dict[str, str]:
    # Para baixar conteúdo raw pelo endpoint /contents
    hdrs = {"Accept": "application/vnd.github.raw"}
    if token:
        hdrs["Authorization"] = f"token {token}"
    return hdrs

def _split_repo(repo: str) -> Tuple[str, str]:
    # "owner/name" -> ("owner","name")
    parts = repo.strip().split("/")
    if len(parts) != 2:
        raise ValueError("GH_REPO deve ser no formato 'owner/repo'.")
    return parts[0], parts[1]

def gh_sync_folder(repo: str, branch: str = "main", folder: str = "data",
                   token: Optional[str] = None) -> Path:
    """
    Lista todos os arquivos do repositório (via Git Trees API) e baixa apenas
    os que estão dentro de `folder` e terminam com ALLOWED_EXTS.
    Salva em .cache_data/<folder>/... (replicando subpastas).
    Retorna o caminho local da pasta sincronizada.
    """
    owner, name = _split_repo(repo)
    # 1) Lista todos os paths do branch (recursivo)
    tree_url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}?recursive=1"
    r = requests.get(tree_url, headers=_gh_headers(token), timeout=60)
    if r.status_code == 404:
        raise RuntimeError("Repositório/branch não encontrado. Confirme GH_REPO e GH_BRANCH.")
    r.raise_for_status()
    data = r.json()
    tree = data.get("tree", [])
    if not tree:
        raise RuntimeError("Árvore do repositório vazia ou inacessível.")

    # 2) Filtra os arquivos dentro de `folder`
    folder = folder.strip("/").strip()
    want_prefix = folder + "/"
    selected = []
    for node in tree:
        path = node.get("path", "")
        ntype = node.get("type")
        if ntype == "blob" and path.startswith(want_prefix) and path.lower().endswith(ALLOWED_EXTS):
            size = node.get("size")  # pode estar ausente
            selected.append((path, size))

    if not selected:
        raise RuntimeError(f"Nenhum arquivo suportado encontrado dentro de '{folder}/'.")

    # 3) Baixa cada arquivo via /contents (suporta repo privado)
    cache_root = Path(".cache_data")
    local_root = cache_root / folder
    local_root.mkdir(parents=True, exist_ok=True)

    with st.spinner(f"Sincronizando {len(selected)} arquivos de '{repo}/{folder}'..."):
        for rel_path, size in selected:
            dest_file = cache_root / rel_path  # preserva subpastas
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Se já existe e o tamanho bate (quando disponível), pula
            if dest_file.exists() and isinstance(size, int) and dest_file.stat().st_size == size:
                continue

            cont_url = f"https://api.github.com/repos/{owner}/{name}/contents/{rel_path}?ref={branch}"
            rr = requests.get(cont_url, headers=_gh_headers_raw(token), timeout=120)
            if rr.status_code == 404:
                # arquivo sumiu? pula
                continue
            rr.raise_for_status()
            dest_file.write_bytes(rr.content)

    return local_root

# ============================= RESOLUÇÃO DE CAMINHOS ==========================
# Preferência: se GH_REPO/GH_PATH dados -> usar sync do GitHub. Caso contrário, cair no local.
USE_GH = ("GH_REPO" in st.secrets) or ("GH_REPO" in os.environ)

DEFAULT_DATA_DIR: Path
if USE_GH:
    GH_REPO   = _secrets_or_env("GH_REPO")
    GH_TOKEN  = _secrets_or_env("GH_TOKEN")      # opcional (necessário p/ privado)
    GH_BRANCH = _secrets_or_env("GH_BRANCH", "main")
    GH_PATH   = _secrets_or_env("GH_PATH", "data")
    try:
        DEFAULT_DATA_DIR = gh_sync_folder(GH_REPO, branch=GH_BRANCH, folder=GH_PATH, token=GH_TOKEN)
        # força o app a trabalhar a partir do cache baixado
        os.environ["DATA_DIR"] = str(DEFAULT_DATA_DIR.resolve())
        st.success(f"Dados sincronizados de **{GH_REPO}/{GH_PATH}** ({GH_BRANCH}).")
    except Exception as e:
        st.error(f"Falha ao sincronizar do GitHub: {e}")
        # fallback para local
        DEFAULT_DATA_DIR = (APP_DIR.parent / "data")
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
else:
    # Sem GH_REPO definido: mantém comportamento local
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
        fallback = APP_DIR.parent / "data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    DEFAULT_DATA_DIR = pick_data_dir(CANDIDATES)

# =============================== UI — SIDEBAR ================================
st.sidebar.header("📁 Fonte de dados (PD27)")
data_dir_input = st.sidebar.text_input(
    "Caminho da pasta de dados",
    value=str(Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()),
    help=("Se GH_REPO/GH_PATH estiverem definidos, este caminho já aponta para o cache sincronizado (.cache_data/...). "
          "Você ainda pode mudar manualmente se quiser ler outra pasta local.")
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

# Botão para re-sincronizar do GitHub
if USE_GH and st.sidebar.button("🔄 Re-sincronizar do GitHub"):
    try:
        refreshed = gh_sync_folder(GH_REPO, branch=GH_BRANCH, folder=GH_PATH, token=GH_TOKEN)
        os.environ["DATA_DIR"] = str(refreshed.resolve())
        st.sidebar.success("Re-sincronizado com sucesso.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Falha ao re-sincronizar: {e}")

st.sidebar.caption("Dica: GH_REPO/GH_PATH em Secrets fazem o app puxar a pasta data direto do GitHub.")

# ================================ UTILIDADES =================================
def list_files(dirpath: Path, types: List[str], prefix: str = "") -> Dict[str, list]:
    prefix_norm = prefix.strip().lower()
    exts = {
        "parquet": [".parquet"],
        "csv":     [".csv"],
        "pdf":     [".pdf"],
    }
    found: Dict[str, list] = {k: [] for k in exts.keys()}
    for p in dirpath.rglob("*"):  # suporta subpastas dentro de data
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        name_ok = True
        if prefix_norm:
            name_ok = Path(p.name).name.lower().startswith(prefix_norm)
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
        if col == "__source_file__": continue
        if df_all[col].dtype == "object":
            # tenta numérico
            df_num = pd.to_numeric(df_all[col].astype(str).str.replace(",",".", regex=False), errors="ignore")
            if pd.api.types.is_numeric_dtype(df_num):
                df_all[col] = df_num
                continue
            # tenta datetime
            try:
                df_dt = pd.to_datetime(df_all[col], errors="raise", utc=False, dayfirst=False, infer_datetime_format=True)
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
st.title("PD27 — Leitura de saídas (GitHub → cache/data)")

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
        st.subheader("Prévia dos dados")
        st.caption("Incluímos a coluna `__source_file__` para rastrear a origem de cada linha.")
        cols = list(df.columns)
        show_cols = st.multiselect(
            "Colunas a exibir",
            options=cols,
            default=cols[: min(12, len(cols))]
        )
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

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

        with st.expander("Arquivos lidos (tabulares)"):
            st.write(pd.DataFrame({
                "arquivo": [str(p.relative_to(DATA_DIR)) for p in parquet_files + csv_files],
                "modificado_em": [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) for p in parquet_files + csv_files],
                "tamanho_KB": [round(p.stat().st_size/1024, 1) for p in parquet_files + csv_files],
            }))

# --------------------------- TAB GRÁFICOS RÁPIDOS ----------------------------
with tabs[1]:
    if df.empty:
        st.info("Sem dados tabulares para plotar.")
    else:
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
            "arquivo": [str(p.relative_to(DATA_DIR)) for p in pdf_files],
            "modificado_em": [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) for p in pdf_files],
            "tamanho_KB": [round(p.stat().st_size/1024, 1) for p in pdf_files],
        })
        st.dataframe(df_pdf, use_container_width=True, hide_index=True)
        st.caption("Se quiser que eu extraia texto dos PDFs, dá pra plugar pdfplumber numa próxima versão.")
