from __future__ import annotations
from pathlib import Path
from datetime import date, time as dtime
from typing import Callable, List, Tuple
import gc
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =================== CONFIG GERAL ===================
st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
# Somente arquivos novos (sem legado)
GLOB_PATTERN = "OFERTAS_*.parquet"
# Segurança/performance
MAX_NEW_FILES = 80 # lê no máximo N arquivos mais recentes
SKIP_LARGER_THAN_MB = 600 # pula arquivo se maior que isso (evita travar)
SAMPLE_MAX_ROWS_PER_FILE = None # ex.: 1_000_000 para amostrar; None = lê tudo
CONCAT_BATCH_SIZE = 15 # concatena em lotes para reduzir pico de memória

# Leitura enxuta: só as colunas usadas
COLUMNS_NEEDED = [
    "IDPESQUISA","CIA","HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA","TIPO_VOO",
    "DATA_EMBARQUE","DATAHORA_BUSCA","AGENCIA_COMP","PRECO","TRECHO","ADVP","RANKING"
]

# Tenta usar pyarrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False

# ============================== UTILIDADES GERAIS ==============================
def std_agencia(raw: str) -> str:
    ag = (raw or "").strip().upper()
    if ag == "BOOKINGCOM": return "BOOKING.COM"
    if ag == "KIWICOM": return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123": return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX": return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"): return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"): return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"): return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"): return "KISSANDFLY"
    if ag.startswith("ZUPPER"): return "ZUPPER"
    if ag.startswith("MYTRIP"): return "MYTRIP"
    if ag.startswith("GOTOGATE"): return "GOTOGATE"
    if ag.startswith("DECOLAR"): return "DECOLAR"
    if ag.startswith("EXPEDIA"): return "EXPEDIA"
    if ag.startswith("GOL"): return "GOL"
    if ag.startswith("LATAM"): return "LATAM"
    if ag.startswith("TRIPCOM"): return "TRIP.COM"
    if ag.startswith("VIAJANET"): return "VIAJANET"
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
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v):
        v = 1
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
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour
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
    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["ADVP_CANON"] = df["ADVP"].apply(advp_nearest)
    df["CIA_NORM"] = df.get("CIA", pd.Series([None]*len(df))).apply(std_cia)
    return df

def _discover_new_files() -> list[Path]:
    files = sorted((p for p in DATA_DIR.glob(GLOB_PATTERN) if p.is_file()), key=lambda p: p.stat().st_mtime)
    return files

def _files_cache_key(paths: list[Path]) -> tuple[str, ...]:
    try:
        return tuple(f"{p.name}:{p.stat().st_mtime_ns}" for p in paths)
    except Exception:
        return tuple(p.name for p in paths)

def _mb(size_bytes: int) -> float:
    return round(size_bytes / (1024*1024), 2)

def _safe_read_parquet(p: Path, columns: list[str]) -> pd.DataFrame:
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
def load_base(data_dir: Path, new_paths_key: tuple[str, ...], limit_new: int | None = MAX_NEW_FILES) -> pd.DataFrame:
    files_all = _discover_new_files()
    files = files_all[-limit_new:] if (limit_new and len(files_all) > limit_new) else files_all
    if not files:
        st.error("Nenhum OFERTAS_*.parquet encontrado na pasta data/.")
        st.stop()
    
    frames: list[pd.DataFrame] = []
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
            progress.progress(idx/len(files), text=f"Lido {p.name} • {size_mb} MB • linhas acumuladas: {processed:,}".replace(",", "."))
            
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
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1, 2, 3):
        s = (df[df["RANKING"] == r].sort_values(["IDPESQUISA"]).drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}), on="IDPESQUISA", how="left")
    for r in (1, 2, 3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty: return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d): return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

# ---- Estilos dos cards (Painel)
CARD_CSS = """
<style>
.cards-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 1100px) { .cards-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 700px) { .cards-grid { grid-template-columns: 1fr; } }
.card { border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card .title { font-weight:650; font-size:15px; margin-bottom:8px; }
.goldcard{background:#FFF9E5;border-color:#D4AF37;}
.silvercard{background:#F7F7FA;border-color:#C0C0C0;}
.bronzecard{background:#FFF1E8;border-color:#CD7F32;}
.row{display:flex;gap:8px;}
.item{flex:1;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-radius:10px;border:1px solid #e3e3e8;background:#fafbfc;}
.pos{font-weight:700;font-size:12px;opacity:.85;}
.pct{font-size:16px;font-weight:650;}
</style>
"""

CARDS_STACK_CSS = """
<style>
.cards-stack { display:flex; flex-direction:column; gap:10px; }
.cards-stack .card { width:100%; }
.stack-title { font-weight:800; padding:8px 10px; margin:6px 0 10px 0; border-radius:10px; border:1px solid #e9e9ee; background:#f8fafc; color:#0A2A6B; }
</style>
"""

def card_html(nome: str, p1: float, p2: float, p3: float, rank_cls: str = "") -> str:
    p1 = max(0.0, min(100.0, float(p1 or 0.0)))
    p2 = max(0.0, min(100.0, float(p2 or 0.0)))
    p3 = max(0.0, min(100.0, float(p3 or 0.0)))
    cls = f"card {rank_cls}".strip()
    return (
        f"<div class='{cls}'>"
        f"<div class='title'>{nome}</div>"
        f"<div class='row'>"
        f"<div class='item'><span class='pos'>1º</span><span class='pct'>{p1:.2f}%</span></div>"
        f"<div class='item'><span class='pos'>2º</span><span class='pct'>{p2:.2f}%</span></div>"
        f"<div class='item'><span class='pos'>3º</span><span class='pct'>{p3:.2f}%</span></div>"
        f"</div></div>"
    )

def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc:
        d = d.sort_values(x_col, ascending=False)
    
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    
    # Trecho de código corrigido e completo
    chart = alt.Chart(d).mark_bar().encode(
        x=alt.X(x_col, title=x_col),
        y=alt.Y(y_col, title=y_col, sort='-x')
    )
    return chart

# ============================== CÓDIGO DO APP ==============================

if __name__ == "__main__":
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown(CARDS_STACK_CSS, unsafe_allow_html=True)
    
    st.title("Skyscanner — Painel de Análise")
    
    files_to_load = _discover_new_files()
    if not files_to_load:
        st.error("Nenhum arquivo `OFERTAS_*.parquet` encontrado na pasta `data/`. Certifique-se de que os dados estão lá.")
        st.stop()
        
    df_base = load_base(DATA_DIR, _files_cache_key(files_to_load))

    if df_base.empty:
        st.warning("O DataFrame está vazio após a leitura dos arquivos. Verifique se os arquivos `.parquet` contêm dados.")
        st.stop()

    total_pesquisas = df_base["IDPESQUISA"].nunique()
    total_ofertas = df_base.shape[0]
    ultima_atualizacao = last_update_from_cols(df_base)

    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pesquisas Únicas", fmt_int(total_pesquisas))
    col2.metric("Total de Ofertas", fmt_int(total_ofertas))
    col3.metric("Última Atualização", ultima_atualizacao)
    
    st.markdown("---")
    
    st.subheader("Ranking de Agências")
    
    winners = winners_by_position(df_base)
    ag_ranking = (winners.melt(id_vars=["IDPESQUISA"], value_vars=["R1", "R2", "R3"], var_name="RANK", value_name="AGENCIA")
                  .groupby(["AGENCIA", "RANK"])
                  .size()
                  .unstack(fill_value=0))
    
    ag_ranking["TOTAL"] = ag_ranking.sum(axis=1)
    ag_ranking = ag_ranking.sort_values("TOTAL", ascending=False)
    
    st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)
    top_agencias = ag_ranking.head(8).index
    
    for ag in top_agencias:
        if ag == "SEM OFERTAS":
            continue
        row = ag_ranking.loc[ag]
        p1 = (row.get('R1', 0) / ag_ranking['R1'].sum()) * 100
        p2 = (row.get('R2', 0) / ag_ranking['R2'].sum()) * 100
        p3 = (row.get('R3', 0) / ag_ranking['R3'].sum()) * 100
        st.markdown(card_html(ag, p1, p2, p3), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    col_charts = st.columns(2)
    
    with col_charts[0]:
        st.subheader("Distribuição por Agência (Ranking 1)")
        df_rank1 = df_base[df_base["RANKING"] == 1]
        
        if not df_rank1.empty:
            df_ag = df_rank1["AGENCIA_NORM"].value_counts().reset_index()
            df_ag.columns = ["AGENCIA_NORM", "CONTAGEM"]
            chart_ag = make_bar(df_ag, "CONTAGEM", "AGENCIA_NORM")
            st.altair_chart(chart_ag.properties(width=500, height=400), use_container_width=True)
        else:
            st.info("Nenhuma oferta encontrada para o Ranking 1.")

    with col_charts[1]:
        st.subheader("Distribuição por Companhia Aérea")
        df_cia = df_base["CIA_NORM"].value_counts().reset_index()
        df_cia.columns = ["CIA_NORM", "CONTAGEM"]
        chart_cia = make_bar(df_cia, "CONTAGEM", "CIA_NORM")
        st.altair_chart(chart_cia.properties(width=500, height=400), use_container_width=True)
