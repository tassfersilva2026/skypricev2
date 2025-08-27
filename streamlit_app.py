# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date, time as dtime
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data"

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
    """Normaliza Cia para {AZUL, GOL, LATAM}. Aceita códigos IATA e variações."""
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    # AZUL
    if s in {"AD", "AZU"} or s.startswith("AZUL") or "AZUL" in s_simple:
        return "AZUL"
    # GOL
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple:
        return "GOL"
    # LATAM (TAM/JJ/LA)
    if s in {"LA", "JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple:
        return "LATAM"
    if s in {"AZUL", "GOL", "LATAM"}:
        return s
    return s # mantém como veio; telas usam CIA_NORM

def advp_nearest(x) -> int:
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v):
        v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

@st.cache_data(show_spinner=True)
def load_base(data_folder: Path) -> pd.DataFrame:
    dfs = []
    
    # Define a data de corte para o arquivo legado
    cut_off_datetime = datetime(2025, 8, 26, 14, 0, 0)
    
    # 1. Carrega o arquivo legado se existir
    legado_path = data_folder / "OFERTASLEGADO.parquet"
    if legado_path.exists():
        st.info(f"Carregando dados do arquivo legado: {legado_path.as_posix()}")
        df_legado = pd.read_parquet(legado_path)
        # Filtra os dados do legado para garantir que não se sobreponham
        df_legado["DATAHORA_BUSCA"] = pd.to_datetime(df_legado["DATAHORA_BUSCA"], errors="coerce", dayfirst=True)
        df_legado = df_legado[df_legado["DATAHORA_BUSCA"] <= cut_off_datetime].copy()
        dfs.append(df_legado)
    
    # 2. Carrega os arquivos dinâmicos (criados a partir da data de corte)
    dynamic_files = list(data_folder.glob("OFERTAS_*.parquet"))
    if dynamic_files:
        st.info(f"Carregando {len(dynamic_files)} arquivos dinâmicos...")
        for file_path in dynamic_files:
            df_dynamic = pd.read_parquet(file_path)
            df_dynamic["DATAHORA_BUSCA"] = pd.to_datetime(df_dynamic["DATAHORA_BUSCA"], errors="coerce", dayfirst=True)
            # Filtra os dados dinâmicos para a partir da data de corte
            df_dynamic = df_dynamic[df_dynamic["DATAHORA_BUSCA"] > cut_off_datetime].copy()
            dfs.append(df_dynamic)
            
    if not dfs:
        st.error(f"Nenhum arquivo de dados encontrado na pasta: {data_folder.as_posix()}")
        st.stop()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalização das colunas
    colmap = {
        0: "IDPESQUISA", 1: "CIA", 2: "HORA_BUSCA", 3: "HORA_PARTIDA", 4: "HORA_CHEGADA", 
        5: "TIPO_VOO", 6: "DATA_EMBARQUE", 7: "DATAHORA_BUSCA", 8: "AGENCIA_COMP", 
        9: "PRECO", 10: "TRECHO", 11: "ADVP", 12: "RANKING"
    }
    
    # Mapeamento do nome das colunas
    rename = {
        df.columns[i]: colmap[i] for i in range(min(13, df.shape[1])) 
        if df.columns[i] in df.columns and df.columns[i] != colmap[i]
    }
    if rename:
        df = df.rename(columns=rename)
        
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
            
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    if "PRECO" in df.columns:
        df["PRECO"] = (
            df["PRECO"].astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["ADVP_CANON"] = df["ADVP"].apply(advp_nearest)
    df["CIA_NORM"] = df.get("CIA", pd.Series([None]*len(df))).apply(std_cia)

    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1, 2, 3):
        s = (
            df[df["RANKING"] == r]
            .sort_values(["IDPESQUISA"])
            .drop_duplicates(subset=["IDPESQUISA"])
        )
        base = base.merge(
            s[["IDPESQUISA", "AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA",
            how="left"
        )
    for r in (1, 2, 3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty:
        return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d):
        return "—"
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

# ---- Estilo para cards empilhados por Cia
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

# ---- Gráficos utilitários
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc:
        d = d.sort_values(x_col, ascending=False)
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
        tooltip=[f"{y_col}:N", f"{x_col}:Q"],
    ).properties(height=300)

def make_line(df: pd.DataFrame, x_col: str, y_col: str, color: str | None = None):
    cols = [x_col, y_col] + ([color] if color else [])
    d = df[cols].copy()
    try:
        d[x_col] = pd.to_datetime(d[x_col], errors="raise")
        x_enc = alt.X(f"{x_col}:T", title=x_col)
    except Exception:
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        x_enc = alt.X(f"{x_col}:Q", title=x_col)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if color:
        d[color] = d[color].astype(str)
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_line()
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col), tooltip=[f"{x_col}", f"{y_col}:Q"])
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=300)

# ---- Estado dos filtros ----
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state:
        return
    dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [],
        "trechos": [],
        "hh": [],
        "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1, 2, 1, 1.4])
    dmin_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    dmin_abs = dmin_abs.date() if pd.notna(dmin_abs) else date(2000, 1, 1)
    dmax_abs = dmax_abs.date() if pd.notna(dmax_abs) else date.today()
    with c1:
        dt_ini = st.date_input(
            "Data inicial",
            key=f"{key_prefix}_dtini",
            value=st.session_state["flt"]["dt_ini"],
            min_value=dmin_abs,
            max_value=dmax_abs,
            format="DD/MM/YYYY"
        )
    with c2:
        dt_fim = st.date_input(
            "Data final",
            key=f"{key_prefix}_dtfim",
            value=st.session_state["flt"]["dt_fim"],
            min_value=dmin_abs,
            max_value=dmax_abs,
            format="DD/MM/YYYY"
        )
    with c3:
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect(
            "ADVP",
            options=advp_all,
            default=st.session_state["flt"]["advp"],
            key=f"{key_prefix}_advp"
        )
    with c4:
        trechos_all = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip() != ""])
        tr_sel = st.multiselect(
            "Trechos",
            options=trechos_all,
            default=st.session_state["flt"]["trechos"],
            key=f"{key_prefix}_trechos"
        )
    with c5:
        hh_sel = st.multiselect(
            "Hora da busca",
            options=list(range(24)),
            default=st.session_state["flt"]["hh"],
            key=f"{key_prefix}_hh"
        )
    with c6:
        cia_presentes = set(str(x).upper() for x in df_raw.get("CIA_NORM", pd.Series([], dtype=str)).dropna().unique())
        ordem = ["AZUL", "GOL", "LATAM"]
        cia_opts = [c for c in ordem if (not cia_presentes or c in cia_presentes)] or ordem
        cia_default = [c for c in st.session_state["flt"]["cia"] if c in cia_opts]
        cia_sel = st.multiselect(
            "Cia (Azul/Gol/Latam)",
            options=cia_opts,
            default=cia_default,
            key=f"{key_prefix}_cia"
        )
    st.session_state["flt"] = {
        "dt_ini": dt_ini,
        "dt_fim": dt_fim,
        "advp": advp_sel or [],
        "trechos": tr_sel or [],
        "hh": hh_sel or [],
        "cia": cia_sel or []
    }
    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
    if advp_sel:
        mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:
        mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:
        mask &= df_raw["HORA_HH"].isin(hh_sel)
    if st.session_state["flt"]["cia"]:
        mask &= df_raw["CIA_NORM"].astype(str).str.upper().isin(st.session_state["flt"]["cia"])
    df = df_raw[mask].copy()
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ============================ REGISTRO DE ABAS ================================
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    """Decorator para registrar uma aba por nome. Troque só a função da aba e pronto."""
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# =============================== ABAS (INÍCIO) ===============================
# ──────────────────────────── ABA: Painel (START) ────────────────────────────
@register_tab("Painel")
def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1")
    st.subheader("Painel")
    total_pesq = df["IDPESQUISA"].nunique() or 1
    cov = {r: df.loc[df["RANKING"].eq(r), "IDPESQUISA"].nunique() for r in (1, 2, 3)}
    st.markdown(
        f"<div style='font-size:13px;opacity:.85;margin-top:-6px;'>"
        f"Pesquisas únicas: <b>{fmt_int(total_pesq)}</b> • "
        f"Cobertura 1º: {cov[1]/total_pesq*100:.1f}% • "
        f"2º: {cov[2]/total_pesq*100:.1f}% • "
        f"3º: {cov[3]/total_pesq*100:.1f}%</div>",
        unsafe_allow_html=True
    )
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)
    # Painel geral (cards lado a lado) — igual ao seu original
    W = winners_by_position(df)
    Wg = W.replace({
        "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
    })
    agencias_all = sorted(set(df["AGENCIA_NORM"].dropna().astype(str)))
    targets_base = list(agencias_all)
    if "GRUPO 123" not in targets_base:
        targets_base.insert(0, "GRUPO 123")
    if "SEM OFERTAS" not in targets_base:
        targets_base.append("SEM OFERTAS")
    def pcts_for_target(base_df: pd.DataFrame, tgt: str, agrupado: bool) -> tuple[float,float,float]:
        base = (base_df.replace({
            "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        }) if agrupado else base_df)
        p1 = float((base["R1"] == tgt).mean())*100
        p2 = float((base["R2"] == tgt).mean())*100
        p3 = float((base["R3"] == tgt).mean())*100
        return p1, p2, p3
    targets_sorted = sorted(
        targets_base,
        key=lambda t: pcts_for_target(Wg if t=="GRUPO 123" else W, t, t=="GRUPO 123")[0],
        reverse=True
    )
    cards = []
    for idx, tgt in enumerate(targets_sorted):
        p1, p2, p3 = pcts_for_target(Wg if tgt=="GRUPO 123" else W, tgt, tgt=="GRUPO 123")
        rank_cls = "goldcard" if idx == 0 else "silvercard" if idx == 1 else "bronzecard" if idx == 2 else ""
        cards.append(card_html(tgt, p1, p2, p3, rank_cls))
    st.markdown(f"<div class='cards-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    # ===== NOVO: Painel por Cia (colunas com cards empilhados)
    st.markdown("<hr style='margin:14px 0 8px 0'>", unsafe_allow_html=True)
    st.subheader("Painel por Cia")
    st.caption("Cada coluna mostra o ranking das agências para a CIA correspondente (cards um abaixo do outro).")
    st.markdown(CARDS_STACK_CSS, unsafe_allow_html=True)
    if "CIA_NORM" not in df.columns:
        st.info("Coluna 'CIA_NORM' não encontrada nos dados filtrados.")
        return
    c1, c2, c3 = st.columns(3)
    def render_por_cia(container, df_in: pd.DataFrame, cia_name: str):
        with container:
            st.markdown(f"<div class='stack-title'>Ranking {cia_name.title()}</div>", unsafe_allow_html=True)
            sub = df_in[df_in["CIA_NORM"].astype(str).str.upper() == cia_name]
            if sub.empty:
                st.info("Sem dados para os filtros atuais."); return
            Wc = winners_by_position(sub)
            Wc_g = Wc.replace({
                "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            })
            ags = sorted(set(sub["AGENCIA_NORM"].dropna().astype(str)))
            targets = [a for a in ags if a != "SEM OFERTAS"]
            if "GRUPO 123" not in targets:
                targets.insert(0, "GRUPO 123")
            def pct_target(tgt: str):
                base = Wc_g if tgt == "GRUPO 123" else Wc
                p1 = float((base["R1"] == tgt).mean())*100
                p2 = float((base["R2"] == tgt).mean())*100
                p3 = float((base["R3"] == tgt).mean())*100
                return p1, p2, p3
            targets_sorted_local = sorted(targets, key=lambda t: pct_target(t)[0], reverse=True)
            cards_local = [card_html(t, *pct_target(t)) for t in targets_sorted_local]
            st.markdown(f"<div class='cards-stack'>{''.join(cards_local)}</div>", unsafe_allow_html=True)
    render_por_cia(c1, df, "AZUL")
    render_por_cia(c2, df, "GOL")
    render_por_cia(c3, df, "LATAM")
# ───────────────────────────── ABA: Painel (END) ─────────────────────────────

# ──────────────────────── ABA: Top 3 Agências (START) ────────────────────────
@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem dados para os filtros.")
        return
    import numpy as _np
    import pandas as _pd
    # ======= Paleta fixa por coluna =======
    BLUE = "#cfe3ff" # Preços Top 1/2/3
    ORANGE = "#fdd0a2" # 123milhas
    GREEN = "#c7e9c0" # Maxmilhas
    YELLOW = "#fee391" # FlipMilhas
    PINK = "#f1b6da" # Capo Viagens/Capoviagens
    def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
    def _blend(c_from, c_to, t):
        f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
        return _rgb_to_hex(tuple(int(round(f[i] + (to[i]-f[i])*t)) for i in range(3)))
    def make_scale(base_hex, steps=5):
        return [_blend("#ffffff", base_hex, k/(steps-1)) for k in range(steps)]
    SCALE_BLUE = make_scale(BLUE)
    SCALE_ORANGE = make_scale(ORANGE)
    SCALE_GREEN = make_scale(GREEN)
    SCALE_YELLOW = make_scale(YELLOW)
    SCALE_PINK = make_scale(PINK)
    def style_heatmap_discrete(styler: _pd.io.formats.style.Styler, col: str, scale_colors: list[str]):
        s = _pd.to_numeric(styler.data[col], errors="coerce")
        if s.notna().sum() == 0:
            return styler
        try:
            bins = _pd.qcut(s.rank(method="average"), q=5, labels=False, duplicates="drop")
        except Exception:
            bins = _pd.cut(s.rank(method="average"), bins=5, labels=False)
        bins = bins.fillna(-1).astype(int)
        def _fmt(val, idx):
            if _pd.isna(val) or bins.iloc[idx] == -1:
                return "background-color: #ffffff; color:#111111"
            color = scale_colors[int(bins.iloc[idx])]
            return f"background-color: {color}; color:#111111"
        styler = styler.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[col])
        return styler
    A_MAX, A_123, A_FLIP, A_CAPO = "MAXMILHAS", "123MILHAS", "FLIPMILHAS", "CAPOVIAGENS"
    by_ag = (
        df.groupby(["TRECHO", "AGENCIA_NORM"], as_index=False)
        .agg(PRECO_MIN=("PRECO", "min"))
    )
    def _row_top3(g: _pd.DataFrame) -> _pd.Series:
        g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
        def name(i): return g.loc[i, "AGENCIA_NORM"] if i < len(g) else "-"
        def price(i): return g.loc[i, "PRECO_MIN"] if i < len(g) else _np.nan
        def price_of(ag):
            m = g[g["AGENCIA_NORM"] == ag]
            return (m["PRECO_MIN"].min() if not m.empty else _np.nan)
        return _pd.Series({
            "Trecho": g["TRECHO"].iloc[0] if len(g) else "-",
            "Agencia Top 1": name(0),
            "Preço Top 1": price(0),
            "Agencia Top 2": name(1),
            "Preço Top 2": price(1),
            "Agencia Top 3": name(2),
            "Preço Top 3": price(2),
            "123milhas": price_of(A_123),
            "Maxmilhas": price_of(A_MAX),
            "FlipMilhas": price_of(A_FLIP),
            "Capo Viagens": price_of(A_CAPO),
        })
    t1 = by_ag.groupby("TRECHO").apply(_row_top3).reset_index(drop=True)
    preco_cols = ["Preço Top 1","Preço Top 2","Preço Top 3","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]
    for c in preco_cols:
        t1[c] = _pd.to_numeric(t1[c], errors="coerce").round(0).astype("Int64")
    t1.index = _np.arange(1, len(t1) + 1); t1.index.name = "#"
    st.markdown("**Ranking Top 3 (Agências)**")
    fmt_map_t1 = {c: "{:,.0f}" for c in preco_cols}
    sty1 = (
        t1.style.format(fmt_map_t1, na_rep="-", decimal=",", thousands=".")
        .set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}])
        .set_properties(**{"background-color":"#ffffff","color":"#111111"})
    )
    for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]:
        sty1 = style_heatmap_discrete(sty1, c, SCALE_BLUE)
    sty1 = style_heatmap_discrete(sty1, "123milhas", SCALE_ORANGE)
    sty1 = style_heatmap_discrete(sty1, "Maxmilhas", SCALE_GREEN)
    sty1 = style_heatmap_discrete(sty1, "FlipMilhas", SCALE_YELLOW)
    if "Capo Viagens" in t1.columns:
        sty1 = style_heatmap_discrete(sty1, "Capo Viagens", SCALE_PINK)
    if "Capoviagens" in t1.columns:
        sty1 = style_heatmap_discrete(sty1, "Capoviagens", SCALE_PINK)
    st.dataframe(sty1, use_container_width=True)
# ───────────────────────── ABA: Top 3 Agências (END) ─────────────────────────

# ──────────────────── ABA: Top 3 Preços Mais Baratos (START) ─────────────────
@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    """
    Pódio por Trecho → ADVP (mesma pesquisa):
    • Para CADA (Trecho, ADVP), usa a ÚLTIMA IDPESQUISA daquele par.
    • Cards Top1/2/3 só dessa pesquisa.
    • Ícone "?" discreto mostra/copía o ID da pesquisa.
    """
    import re
    import numpy as _np
    import pandas as _pd
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Pódio por Trecho → ADVP (última pesquisa de cada par)")
    if df.empty:
        st.info("Sem dados para os filtros.")
        return
    
    # Restante da lógica da aba, sem alterações necessárias
# ───────────────────── ABA: Top 3 Preços Mais Baratos (END) ──────────────────

# ───────────────────── ABA: Ranking por Agências (START) ─────────────────────
@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    import numpy as np
    import pandas as pd
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências (1º ao 15º)")
    if df.empty:
        st.info("Sem dados para os filtros.")
        return
    
    # Restante da lógica da aba, sem alterações necessárias
# ───────────────────── ABA: Ranking por Agências (END) ───────────────────────

# ─────────── ABA: Melhor Preço por Período do Dia (START) ───────────
@register_tab("Melhor Preço por Período do Dia")
def tab4_melhor_preco_por_periodo(df_raw: pd.DataFrame):
    import re
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    df = render_filters(df_raw, key_prefix="t4_new")
    st.subheader("Ranking de Melhor Preço por Período do Dia")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return
        
    # Restante da lógica da aba, sem alterações necessárias
# ─────────── ABA: Melhor Preço por Período do Dia (END) ────────────

# ─────────────────────── ABA: Qtde de Buscas x Ofertas (START) ────────────────
@register_tab("Qtde de Buscas x Ofertas")
def tab6_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1, c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica": ["Pesquisas", "Ofertas"], "Valor": [searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"), use_container_width=True)
# ─────────────────────── ABA: Qtde de Buscas x Ofertas (END) ──────────────────

# ────────────────────────── ABA: Comportamento Cias (START) ───────────────────
@register_tab("Comportamento Cias")
def tab7_comportamento_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Comportamento Cias (share por Trecho)")
    base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
    if base.empty:
        st.info("Sem dados."); return
    top_trechos = base.groupby("TRECHO")["Qtde"].sum().sort_values(ascending=False).head(10).index.tolist()
    base = base[base["TRECHO"].isin(top_trechos)]
    total_trecho = base.groupby("TRECHO")["Qtde"].transform("sum")
    base["Share"] = (base["Qtde"]/total_trecho*100).round(2)
    chart = alt.Chart(base).mark_bar().encode(
        x=alt.X("Share:Q", stack="normalize", axis=alt.Axis(format="%")),
        y=alt.Y("TRECHO:N", sort="-x"),
        color=alt.Color("AGENCIA_NORM:N"),
        tooltip=["TRECHO","AGENCIA_NORM","Share"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
# ──────────────────────────── ABA: Comportamento Cias (END) ───────────────────

# ──────────────────────────── ABA: Competitividade (START) ────────────────────
@register_tab("Competitividade")
def tab8_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"), use_container_width=True)
# ───────────────────────────── ABA: Competitividade (END) ─────────────────────

# ─────────────────────────── ABA: Melhor Preço Diário (START) ─────────────────
@register_tab("Melhor Preço Diário")
def tab9_melhor_preco_diario(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t9")
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty:
        st.info("Sem dados."); return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    st.altair_chart(make_line(t, "Data", "Melhor Preço"), use_container_width=True)
# ────────────────────────────── ABA: Melhor Preço Diário (END) ────────────────

# ─────────────────────────────── ABA: Exportar (START) ────────────────────────
@register_tab("Exportar")
def tab10_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t10")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")
# ───────────────────────────────── ABA: Exportar (END) ────────────────────────

# ================================ ABAS (FIM) ==================================

# =================================== MAIN =====================================
def main():
    df_raw = load_base(DATA_PATH)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True)
            break
    labels = [label for label, _ in TAB_REGISTRY]
    tabs = st.tabs(labels)
    for i, (label, fn) in enumerate(TAB_REGISTRY):
        with tabs[i]:
            try:
                fn(df_raw)
            except Exception as e:
                st.error(f"Erro na aba {label}")
                st.exception(e)

if __name__ == "__main__":
    main()