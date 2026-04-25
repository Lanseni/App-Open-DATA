"""
Application d'analyse DPE & Consommation Électrique
Enedis x ADEME — Open Data University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import KDTree
from pyproj import Transformer
import requests
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "DPE_enedis_1.csv")

ELECTRICITY_PRICE_2024 = 0.2516
DEFAULT_PRICE_INCREASE  = 3

DPE_COLORS = {
    "A":"#009900","B":"#55C400","C":"#AAFF00",
    "D":"#FFFF00","E":"#FFB432","F":"#FF5000","G":"#FF0000",
}
DPE_TEXT_COLORS = {
    "A":"white","B":"white","C":"black","D":"black","E":"white","F":"white","G":"white",
}
DPE_ORDER = ["A","B","C","D","E","F","G"]

USAGE_COLS = {
    "conso_chauffage_ef"      :"Chauffage",
    "conso_ecs_ef"            :"Eau Chaude",
    "conso_eclairage_ef"      :"Eclairage",
    "conso_refroidissement_ef":"Refroidissement",
    "conso_auxiliaires_ef"    :"Auxiliaires",
}
USAGE_ICONS  = {"Chauffage":"fire","Eau Chaude":"shower","Eclairage":"lightbulb",
                "Refroidissement":"snowflake","Auxiliaires":"cogs"}
USAGE_COLORS = ["#E45756","#4C78A8","#F28E2B","#76B7B2","#59A14F"]

DEPERD_COLS = {
    "deperditions_enveloppe"     :"Enveloppe totale",
    "deperditions_murs"          :"Murs",
    "deperditions_planchers_bas" :"Planchers bas",
    "deperditions_planchers_hauts":"Planchers hauts",
}

st.set_page_config(
    page_title="DPE & Conso Electrique",
    page_icon="lightning",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container{padding-top:1.2rem}
  .stMetric{background:#f5f7fa;border-radius:10px;padding:8px 12px}
  .dpe-label{display:inline-block;padding:3px 14px;border-radius:5px;
             font-weight:bold;font-size:1rem;letter-spacing:.5px}
  .section-sep{border-top:2px solid #eee;margin:16px 0}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TRANSFORMERS
# ─────────────────────────────────────────────────────────────
_l93_to_wgs84 = Transformer.from_crs("EPSG:2154","EPSG:4326", always_xy=True)
_wgs84_to_l93 = Transformer.from_crs("EPSG:4326","EPSG:2154", always_xy=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des donnees...")
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    mask = df["coordonnee_cartographique_x_ban"] > 1000
    lons, lats = _l93_to_wgs84.transform(
        df.loc[mask,"coordonnee_cartographique_x_ban"].values,
        df.loc[mask,"coordonnee_cartographique_y_ban"].values,
    )
    df.loc[mask,"lon"] = lons
    df.loc[mask,"lat"] = lats

    df["conso_dpe_kwh"]   = df["conso_5_usages_par_m2_ef"] * df["surface_habitable_logement"]
    df["conso_relle_kwh"] = df["Consommation annuelle moyenne par logement de l'adresse (MWh)"] * 1000
    df["etiquette_dpe"]   = df["etiquette_dpe"].astype(str).str.strip().str.upper()
    df = df[df["etiquette_dpe"].isin(DPE_ORDER)].copy()
    df["etiquette_dpe"]   = pd.Categorical(df["etiquette_dpe"], categories=DPE_ORDER, ordered=True)
    df["_has_l93"]        = df["coordonnee_cartographique_x_ban"] > 1000
    return df


@st.cache_resource(show_spinner=False)
def build_kdtree(df):
    valid  = df[df["_has_l93"]].copy()
    coords = valid[["coordonnee_cartographique_x_ban",
                     "coordonnee_cartographique_y_ban"]].values
    return KDTree(coords), valid.index.tolist()


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Fichier CSV introuvable : {DATA_PATH}")
    st.stop()

tree, tree_idx = build_kdtree(df)
communes_list  = sorted(df["nom_commune_ban"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def dpe_badge(label, size="1rem"):
    c = DPE_COLORS.get(str(label), "#888")
    t = DPE_TEXT_COLORS.get(str(label), "white")
    return (f'<span class="dpe-label" '
            f'style="background:{c};color:{t};font-size:{size}"> {label} </span>')


@st.cache_data(show_spinner=False, ttl=86400)
def geocode_address(address):
    """Geocode via API BAN officielle (adresse.data.gouv.fr)."""
    try:
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": address, "limit": 1},
            timeout=6,
        )
        data = r.json()
        if data.get("features"):
            feat  = data["features"][0]
            lon, lat = feat["geometry"]["coordinates"]
            props = feat["properties"]
            return lat, lon, props.get("label",""), props.get("score",0), props.get("city","")
    except Exception:
        pass
    return None, None, None, None, None


def latlon_to_l93(lat, lon):
    x, y = _wgs84_to_l93.transform(lon, lat)
    return x, y


def find_nearest_from_l93(x, y, n=10):
    dists, positions = tree.query([x, y], k=min(n, len(tree_idx)))
    if n == 1:
        dists, positions = [dists], [positions]
    indices = [tree_idx[p] for p in positions]
    result  = df.loc[indices].copy()
    result["distance_m"] = [int(d) for d in dists]
    return result


def commune_center_l93(commune):
    sub = df[df["nom_commune_ban"] == commune]
    return (sub["coordonnee_cartographique_x_ban"].mean(),
            sub["coordonnee_cartographique_y_ban"].mean())


def median_conso_by_dpe(data):
    return (
        data.groupby("etiquette_dpe", observed=True)
        .agg(conso_dpe_kwh=("conso_dpe_kwh","median"),
             conso_relle_kwh=("conso_relle_kwh","median"),
             n=("conso_relle_kwh","count"))
        .reindex(DPE_ORDER).dropna(subset=["conso_relle_kwh"])
    )


def predict_costs(kwh_year, prix, taux, dpe_cur):
    conso_med = median_conso_by_dpe(df)["conso_relle_kwh"]
    base_g    = conso_med.get("G", kwh_year) or kwh_year
    def factor(cls):
        v = conso_med.get(cls)
        return (v/base_g) if v and base_g else 1.0
    years = list(range(2025, 2036))
    s = {}
    s["Sans renovation"] = [kwh_year*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur not in ("A","B"):
        r = factor("B")/max(factor(dpe_cur),0.01)
        nk = kwh_year*r
        s["Renovation -> Classe B"] = [(kwh_year if i<2 else nk)*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur != "A":
        r = factor("A")/max(factor(dpe_cur),0.01)
        nk = kwh_year*r
        s["Renovation -> Classe A"] = [(kwh_year if i<3 else nk)*prix*(1+taux)**i for i in range(len(years))]
    return years, s


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Logo_Enedis.svg/320px-Logo_Enedis.svg.png", width=130)
    except Exception:
        pass

    st.markdown("## Votre adresse")
    address_input = st.text_input(
        "Adresse complete",
        placeholder="Ex : 12 rue Jean Jaures, Orleans",
        help="Geocodage via l'API officielle BAN (adresse.data.gouv.fr).",
    )
    geocode_btn = st.button("Geolocaliser & Analyser", type="primary", use_container_width=True)

    if geocode_btn and address_input.strip():
        with st.spinner("Geocodage en cours..."):
            lat, lon, label, score, city = geocode_address(address_input.strip())
        if lat:
            st.session_state.update({
                "geo_lat":lat,"geo_lon":lon,
                "geo_label":label,"geo_city":city,"geo_score":score,
            })
        else:
            st.error("Adresse introuvable. Verifiez la saisie.")
            for k in ["geo_lat","geo_lon","geo_label","geo_city","geo_score"]:
                st.session_state.pop(k, None)

    if "geo_label" in st.session_state:
        st.success(f"{st.session_state['geo_label']}")
        st.caption(f"Confiance geocodage : {int(st.session_state.get('geo_score',0)*100)}%")

    st.markdown("---")
    st.markdown("## Votre logement")

    default_city  = st.session_state.get("geo_city","")
    matched       = next((c for c in communes_list if default_city.lower() in c.lower()), None) if default_city else None
    default_idx   = communes_list.index(matched) if matched else (
        communes_list.index("Orleans") if "Orleans" in communes_list else 0
    )
    commune       = st.selectbox("Commune (onglets 2-3)", communes_list, index=default_idx)
    type_batiment = st.selectbox("Type de batiment", ["appartement","maison","immeuble"])
    surface       = st.slider("Surface habitable (m2)", 15, 300, 65, step=5)
    dpe_actuel    = st.select_slider("Classe DPE actuelle", options=DPE_ORDER, value="D")
    periode_construction = st.selectbox(
        "Periode de construction",
        ["Inconnue","Avant 1948","1948-1974","1975-1989",
         "1990-2000","2001-2005","2006-2012","2013-2021","Apres 2021"],
    )
    st.markdown(f"Classe : {dpe_badge(dpe_actuel)}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Hypotheses tarifaires")
    prix_kwh    = st.number_input("Prix actuel (euros/kWh)", 0.10, 1.0, ELECTRICITY_PRICE_2024, 0.01)
    taux_hausse = st.slider("Hausse annuelle (%)", 0, 10, DEFAULT_PRICE_INCREASE) / 100

    st.markdown("---")
    st.caption(f"Base : {len(df):,} logements - {len(communes_list)} communes")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("DPE & Consommation Electrique — Analyse & Prediction")
st.caption("Donnees ADEME (DPE) x Enedis (consommations reelles Linky). Saisissez votre adresse pour une analyse geolocalise.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Logements proches",
    "Analyse commune",
    "Analyse France",
    "Prediction 10 ans",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE DETAILLEE DES 10 LOGEMENTS LES PLUS PROCHES
# ══════════════════════════════════════════════════════════════
with tab1:

    # Point de reference
    has_geo = "geo_lat" in st.session_state
    if has_geo:
        user_lat  = st.session_state["geo_lat"]
        user_lon  = st.session_state["geo_lon"]
        user_x, user_y = latlon_to_l93(user_lat, user_lon)
        ref_label = st.session_state["geo_label"]
        nearest   = find_nearest_from_l93(user_x, user_y, n=10)
    else:
        cx, cy    = commune_center_l93(commune)
        user_x, user_y = cx, cy
        user_lon, user_lat = _l93_to_wgs84.transform(cx, cy)
        ref_label = f"Centre de {commune} (aucune adresse saisie)"
        nearest   = find_nearest_from_l93(cx, cy, n=10)

    if has_geo:
        st.info(f"Analyse depuis : **{ref_label}**")
    else:
        st.warning(
            "Saisissez votre adresse dans la barre laterale pour une analyse precise. "
            f"Reference actuelle : centre de **{commune}**."
        )

    if nearest.empty:
        st.error("Aucun logement trouve a proximite.")
        st.stop()

    # ──────────────────────────────────────────────────────────
    # A. KPIs
    # ──────────────────────────────────────────────────────────
    st.markdown("### Vue d'ensemble des 10 logements")

    pct_pass      = nearest["etiquette_dpe"].isin(["F","G"]).mean()*100
    dist_max      = nearest["distance_m"].max()
    dist_moy      = nearest["distance_m"].mean()
    conso_med_r   = nearest["conso_relle_kwh"].median()
    conso_med_d   = nearest["conso_dpe_kwh"].median()
    ecart_pct     = (conso_med_r-conso_med_d)/conso_med_d*100 if conso_med_d else 0
    dpe_dom       = nearest["etiquette_dpe"].astype(str).mode()[0]
    cout_med      = conso_med_r * prix_kwh

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Rayon couvert",        f"{dist_max:,.0f} m",    f"moy. {dist_moy:,.0f} m")
    k2.metric("DPE dominant",         dpe_dom)
    k3.metric("Conso reelle mediane", f"{conso_med_r:,.0f} kWh")
    k4.metric("Conso DPE mediane",    f"{conso_med_d:,.0f} kWh", f"{ecart_pct:+.1f}% vs DPE")
    k5.metric("Cout annuel median",   f"{cout_med:,.0f} euros")
    k6.metric("Passoires (F+G)",      f"{pct_pass:.0f}%")

    st.markdown('<hr style="border-top:2px solid #eee;margin:14px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # B. CARTE INTERACTIVE
    # ──────────────────────────────────────────────────────────
    st.markdown("### Carte interactive")

    center_lat = (nearest["lat"].mean() + user_lat) / 2
    center_lon = (nearest["lon"].mean() + user_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Marqueur utilisateur
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="Votre adresse",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        popup=folium.Popup(f"<b>{ref_label}</b>", max_width=260),
    ).add_to(m)

    # Cercle rayon max
    folium.Circle(
        location=[user_lat, user_lon], radius=int(dist_max),
        color="#1f77b4", fill=False, weight=1.5, dash_array="6",
        tooltip=f"Rayon : {dist_max:.0f} m",
    ).add_to(m)

    for rank, (_, row) in enumerate(nearest.iterrows(), start=1):
        if pd.isna(row.get("lat")):
            continue
        dpe  = str(row["etiquette_dpe"])
        chex = DPE_COLORS.get(dpe,"#888")
        surf = row.get("surface_habitable_logement",0)
        dkwh = row.get("conso_dpe_kwh",0)
        rkwh = row.get("conso_relle_kwh",0)
        ecrt = (rkwh-dkwh)/dkwh*100 if dkwh else 0
        dist = row.get("distance_m",0)
        cout = rkwh*prix_kwh

        # % usage chauffage
        total_use = row.get("conso_5_usages_ef",1) or 1
        ch_pct  = row.get("conso_chauffage_ef",0)/total_use*100
        ecs_pct = row.get("conso_ecs_ef",0)/total_use*100
        ecl_pct = row.get("conso_eclairage_ef",0)/total_use*100

        popup_html = f"""
        <div style='font-family:Arial;min-width:260px;font-size:12.5px;line-height:1.65'>
          <div style='background:{chex};color:{"white" if dpe in("A","B","E","F","G") else "black"};
               padding:4px 10px;border-radius:4px;font-weight:bold;font-size:14px;margin-bottom:7px'>
            #{rank} &mdash; Classe DPE {dpe}
          </div>
          <b>Adresse :</b> {row.get("adresse_ban","N/A")}<br>
          <b>Type :</b> {row.get("type_batiment","N/A")} &nbsp;|&nbsp;
          <b>Surface :</b> {surf:.0f} m2<br>
          <b>Periode :</b> {row.get("periode_construction","N/A")} &nbsp;|&nbsp;
          <b>Zone :</b> {row.get("zone_climatique","N/A")}<br>
          <b>Score DPE :</b> {row.get("score_dpe","N/A")}<br>
          <hr style='margin:5px 0'>
          <b>Consommation DPE estimee (3CL)</b><br>
          &nbsp; Total : <b>{dkwh:,.0f} kWh/an</b> ({row.get("conso_5_usages_par_m2_ef",0):.0f} kWh/m2)<br>
          &nbsp; Chauffage : {row.get("conso_chauffage_ef",0):,.0f} kWh ({ch_pct:.0f}%)<br>
          &nbsp; Eau chaude : {row.get("conso_ecs_ef",0):,.0f} kWh ({ecs_pct:.0f}%)<br>
          &nbsp; Eclairage : {row.get("conso_eclairage_ef",0):,.0f} kWh ({ecl_pct:.0f}%)<br>
          &nbsp; Refroidissement : {row.get("conso_refroidissement_ef",0):,.0f} kWh<br>
          &nbsp; Auxiliaires : {row.get("conso_auxiliaires_ef",0):,.0f} kWh<br>
          <hr style='margin:5px 0'>
          <b>Consommation reelle Enedis</b><br>
          &nbsp; <b>{rkwh:,.0f} kWh/an</b>
          (<span style='color:{"#c0392b" if ecrt>0 else "#27ae60"}'>{ecrt:+.1f}% vs DPE</span>)<br>
          &nbsp; Cout estime : <b>{cout:,.0f} euros/an</b><br>
          <hr style='margin:5px 0'>
          <b>Thermique</b><br>
          &nbsp; Deperditions enveloppe : {row.get("deperditions_enveloppe",0):,.0f} W/K<br>
          &nbsp; Deperditions murs : {row.get("deperditions_murs",0):,.0f} W/K<br>
          &nbsp; Isolation murs : {row.get("qualite_isolation_murs","N/A")}<br>
          &nbsp; Isolation enveloppe : {row.get("qualite_isolation_enveloppe","N/A")}<br>
          <hr style='margin:5px 0'>
          <b>Distance :</b> {dist:,} m
        </div>"""

        folium.CircleMarker(
            location=[row["lat"],row["lon"]],
            radius=18, color="white", weight=2,
            fill=True, fill_color=chex, fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{rank} - DPE {dpe} - {surf:.0f}m2 - {dist:,}m",
        ).add_to(m)
        folium.Marker(
            location=[row["lat"],row["lon"]],
            icon=folium.DivIcon(
                html=(f'<div style="color:{"white" if dpe in("A","B","E","F","G") else "black"};'
                      f'font-weight:bold;font-size:12px;text-align:center;line-height:22px;width:22px">{rank}</div>'),
                icon_size=(22,22), icon_anchor=(11,11),
            ),
        ).add_to(m)

    st_folium(m, height=460, use_container_width=True)

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # C. ANALYSE DES CONSOMMATIONS
    # ──────────────────────────────────────────────────────────
    st.markdown("### Analyse des consommations")

    c1, c2 = st.columns(2)

    with c1:
        df_bar = nearest[["distance_m","etiquette_dpe","conso_dpe_kwh","conso_relle_kwh"]].copy()
        df_bar = df_bar.sort_values("distance_m")
        df_bar["label"] = [f"#{i+1}" for i in range(len(df_bar))]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="DPE estime (3CL)", x=df_bar["label"], y=df_bar["conso_dpe_kwh"],
            marker_color="#4C78A8", opacity=0.85,
            hovertemplate="<b>%{x}</b><br>DPE : %{y:,.0f} kWh<extra></extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            name="Reel Enedis", x=df_bar["label"], y=df_bar["conso_relle_kwh"],
            marker_color="#F28E2B", opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Reel : %{y:,.0f} kWh<extra></extra>",
        ))
        fig_cmp.update_layout(
            barmode="group",
            title="DPE estime vs Consommation reelle par logement",
            xaxis_title="Logement (rang par distance)",
            yaxis_title="kWh/an",
            legend=dict(orientation="h", y=-0.22),
            height=370,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with c2:
        df_bar["ecart_pct"] = (df_bar["conso_relle_kwh"]-df_bar["conso_dpe_kwh"]) \
                              / df_bar["conso_dpe_kwh"].replace(0,np.nan)*100
        colors_e = ["#c0392b" if v>0 else "#27ae60" for v in df_bar["ecart_pct"]]

        fig_e = go.Figure(go.Bar(
            x=df_bar["label"], y=df_bar["ecart_pct"],
            marker_color=colors_e,
            text=[f"{v:+.1f}%" for v in df_bar["ecart_pct"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ecart : %{y:+.1f}%<extra></extra>",
        ))
        fig_e.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_e.update_layout(
            title="Ecart consommation reelle vs DPE estime (%)",
            xaxis_title="Logement",
            yaxis_title="Ecart (%)",
            height=370,
        )
        st.caption("Rouge = consommation reelle superieure au DPE | Vert = inferieure")
        st.plotly_chart(fig_e, use_container_width=True)

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # D. DETAIL PAR USAGE
    # ──────────────────────────────────────────────────────────
    st.markdown("### Detail par usage (consommations DPE estimees)")

    c3, c4 = st.columns(2)

    with c3:
        usage_rows = []
        for i, (_, row) in enumerate(nearest.iterrows(), start=1):
            for col, lbl in USAGE_COLS.items():
                v = row.get(col, 0)
                if pd.notna(v) and v > 0:
                    usage_rows.append({"Logement": f"#{i}", "Usage": lbl, "kWh/an": round(v)})
        if usage_rows:
            udf = pd.DataFrame(usage_rows)
            fig_st = px.bar(
                udf, x="Logement", y="kWh/an", color="Usage",
                color_discrete_sequence=USAGE_COLORS,
                title="Decomposition par usage (DPE estime)",
                barmode="stack",
            )
            fig_st.update_layout(height=360, legend=dict(orientation="h",y=-0.3))
            st.plotly_chart(fig_st, use_container_width=True)

    with c4:
        avg_use = {
            lbl: nearest[col].mean()
            for col,lbl in USAGE_COLS.items()
            if col in nearest.columns
        }
        avg_use = {k:v for k,v in avg_use.items() if pd.notna(v) and v>0}
        if avg_use:
            fig_pie = px.pie(
                values=list(avg_use.values()),
                names=list(avg_use.keys()),
                color_discrete_sequence=USAGE_COLORS,
                title="Repartition moyenne des usages",
                hole=0.38,
            )
            fig_pie.update_traces(texttemplate="%{label}<br>%{percent:.0%}")
            fig_pie.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # E. THERMIQUE & ISOLATION
    # ──────────────────────────────────────────────────────────
    st.markdown("### Deperditions thermiques & qualite d'isolation")

    c5, c6 = st.columns(2)

    with c5:
        dep_rows = []
        for i, (_, row) in enumerate(nearest.iterrows(), start=1):
            for col,lbl in DEPERD_COLS.items():
                v = row.get(col,0)
                if pd.notna(v) and v>0:
                    dep_rows.append({"Logement":f"#{i}","Composant":lbl,"W/K":round(v,1)})
        if dep_rows:
            ddf = pd.DataFrame(dep_rows)
            fig_dep = px.bar(
                ddf[ddf["Composant"]!="Enveloppe totale"],
                x="Logement", y="W/K", color="Composant",
                title="Deperditions par composant (W/K)",
                barmode="group",
                color_discrete_sequence=["#e15759","#f28e2b","#76b7b2"],
            )
            fig_dep.update_layout(height=340, legend=dict(orientation="h",y=-0.3))
            st.plotly_chart(fig_dep, use_container_width=True)
        else:
            st.info("Donnees de deperditions thermiques non disponibles pour ces logements.")

    with c6:
        # Radar deperditions moyennes
        dep_means = {
            lbl: nearest[col].mean()
            for col,lbl in DEPERD_COLS.items()
            if col in nearest.columns
        }
        dep_means = {k:v for k,v in dep_means.items() if pd.notna(v) and v>0}
        if len(dep_means) >= 3:
            cats   = list(dep_means.keys())
            values = list(dep_means.values())
            fig_rad = go.Figure(go.Scatterpolar(
                r=values+[values[0]], theta=cats+[cats[0]],
                fill="toself",
                fillcolor="rgba(231,76,60,0.18)",
                line_color="#e74c3c", name="Deperditions moyennes",
            ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Radar deperditions moyennes (W/K)",
                height=340,
            )
            st.plotly_chart(fig_rad, use_container_width=True)

    c7, c8 = st.columns(2)

    with c7:
        sc_df = nearest.dropna(subset=["surface_habitable_logement","conso_relle_kwh"])
        if not sc_df.empty:
            sc_df = sc_df.copy()
            sc_df["rang"] = [f"#{i+1}" for i in range(len(sc_df))]
            fig_surf = px.scatter(
                sc_df,
                x="surface_habitable_logement", y="conso_relle_kwh",
                color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                size="conso_dpe_kwh", size_max=22,
                text="rang",
                title="Surface vs Consommation reelle",
                labels={
                    "surface_habitable_logement":"Surface (m2)",
                    "conso_relle_kwh":"Conso reelle (kWh/an)",
                    "etiquette_dpe":"Classe DPE",
                },
                hover_data=["adresse_ban","periode_construction","distance_m"],
            )
            fig_surf.update_traces(textposition="top center")
            fig_surf.update_layout(height=340)
            st.plotly_chart(fig_surf, use_container_width=True)

    with c8:
        # Qualite d'isolation
        iso_counts = nearest["qualite_isolation_enveloppe"].dropna().value_counts()
        iso_counts2 = nearest["qualite_isolation_murs"].dropna().value_counts()
        if not iso_counts.empty or not iso_counts2.empty:
            fig_iso = go.Figure()
            if not iso_counts.empty:
                fig_iso.add_trace(go.Bar(
                    name="Enveloppe", x=iso_counts.index, y=iso_counts.values,
                    marker_color="#4C78A8",
                    text=iso_counts.values, textposition="outside",
                ))
            if not iso_counts2.empty:
                fig_iso.add_trace(go.Bar(
                    name="Murs", x=iso_counts2.index, y=iso_counts2.values,
                    marker_color="#F28E2B", opacity=0.8,
                    text=iso_counts2.values, textposition="outside",
                ))
            fig_iso.update_layout(
                barmode="group",
                title="Qualite d'isolation (nombre de logements)",
                yaxis_title="Nb logements",
                xaxis_title="Niveau d'isolation",
                height=340,
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_iso, use_container_width=True)

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # F. TABLEAU RECAPITULATIF COMPLET
    # ──────────────────────────────────────────────────────────
    st.markdown("### Tableau recapitulatif complet")

    recap_cols = {
        "distance_m"                  :"Dist. (m)",
        "etiquette_dpe"               :"DPE",
        "score_dpe"                   :"Score DPE",
        "type_batiment"               :"Type",
        "surface_habitable_logement"  :"Surface (m2)",
        "periode_construction"        :"Periode",
        "qualite_isolation_enveloppe" :"Iso. enveloppe",
        "qualite_isolation_murs"      :"Iso. murs",
        "zone_climatique"             :"Zone clim.",
        "conso_5_usages_par_m2_ef"   :"kWh/m2 (DPE)",
        "conso_dpe_kwh"               :"kWh/an (DPE)",
        "conso_relle_kwh"             :"kWh/an (reel)",
        "conso_chauffage_ef"          :"Chauf. (kWh)",
        "conso_ecs_ef"                :"ECS (kWh)",
        "conso_eclairage_ef"          :"Eclair. (kWh)",
        "conso_refroidissement_ef"    :"Refroid. (kWh)",
        "conso_auxiliaires_ef"        :"Auxil. (kWh)",
        "deperditions_enveloppe"      :"Dep. env. (W/K)",
        "deperditions_murs"           :"Dep. murs (W/K)",
        "deperditions_planchers_bas"  :"Dep. pl.bas (W/K)",
        "deperditions_planchers_hauts":"Dep. pl.hts (W/K)",
        "Nombre de logements"         :"Nb logements",
    }
    avail = {k:v for k,v in recap_cols.items() if k in nearest.columns}
    recap = nearest[list(avail.keys())].copy()
    recap.columns = list(avail.values())
    recap.index   = [f"#{i+1}" for i in range(len(recap))]

    num_cols = [v for k,v in avail.items()
                if k not in ("etiquette_dpe","type_batiment","periode_construction",
                             "qualite_isolation_enveloppe","qualite_isolation_murs","zone_climatique")]
    for c_ in num_cols:
        if c_ in recap.columns:
            recap[c_] = pd.to_numeric(recap[c_], errors="coerce").round(1)

    # Ajouter colonne cout reel
    if "kWh/an (reel)" in recap.columns:
        recap.insert(recap.columns.get_loc("kWh/an (reel)")+1,
                     "Cout/an (euros)", (recap["kWh/an (reel)"]*prix_kwh).round(0))

    st.dataframe(recap, use_container_width=True, height=370)

    # Bouton export
    csv_data = recap.to_csv(encoding="utf-8-sig")
    st.download_button(
        label="Telecharger le tableau (CSV)",
        data=csv_data,
        file_name="dpe_10_logements_proches.csv",
        mime="text/csv",
    )

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # G. FICHES INDIVIDUELLES
    # ──────────────────────────────────────────────────────────
    st.markdown("### Fiches individuelles detaillees")
    st.caption("Cliquez pour developper la fiche complete de chaque logement.")

    for rank, (_, row) in enumerate(nearest.iterrows(), start=1):
        dpe  = str(row.get("etiquette_dpe","?"))
        surf = row.get("surface_habitable_logement",0)
        dist = row.get("distance_m",0)
        dkwh = row.get("conso_dpe_kwh",0)
        rkwh = row.get("conso_relle_kwh",0)
        ecrt = (rkwh-dkwh)/dkwh*100 if dkwh else 0
        cout = rkwh*prix_kwh

        with st.expander(
            f"#{rank} — {row.get('adresse_ban','N/A')}  |  "
            f"DPE {dpe}  |  {surf:.0f} m2  |  {dist:,} m  |  {rkwh:,.0f} kWh/an"
        ):
            fa, fb, fc = st.columns(3)

            with fa:
                st.markdown("**Caracteristiques du logement**")
                st.markdown(f"""
| Champ | Valeur |
|---|---|
| Type batiment | {row.get("type_batiment","N/A")} |
| Surface | {surf:.0f} m2 |
| Periode construction | {row.get("periode_construction","N/A")} |
| Zone climatique | {row.get("zone_climatique","N/A")} |
| Nb logements adresse | {row.get("Nombre de logements","N/A")} |
| Score DPE | {row.get("score_dpe","N/A")} |
| Classe DPE | **{dpe}** |
| Energie chauffage | {row.get("type_energie_principale_chauffage","N/A")} |
| Energie ECS | {row.get("type_energie_principale_ecs","N/A")} |
| Date DPE | {row.get("date_etablissement_dpe","N/A")} |
""")

            with fb:
                st.markdown("**Consommations detaillees**")
                st.markdown(f"""
| Usage | kWh/an (DPE) | % total |
|---|---|---|
| Chauffage | {row.get("conso_chauffage_ef",0):,.0f} | {row.get("conso_chauffage_ef",0)/max(dkwh,1)*100:.0f}% |
| Eau chaude sanitaire | {row.get("conso_ecs_ef",0):,.0f} | {row.get("conso_ecs_ef",0)/max(dkwh,1)*100:.0f}% |
| Eclairage | {row.get("conso_eclairage_ef",0):,.0f} | {row.get("conso_eclairage_ef",0)/max(dkwh,1)*100:.0f}% |
| Refroidissement | {row.get("conso_refroidissement_ef",0):,.0f} | {row.get("conso_refroidissement_ef",0)/max(dkwh,1)*100:.0f}% |
| Auxiliaires | {row.get("conso_auxiliaires_ef",0):,.0f} | {row.get("conso_auxiliaires_ef",0)/max(dkwh,1)*100:.0f}% |
| **Total DPE** | **{dkwh:,.0f}** | 100% |
| DPE par m2 | {row.get("conso_5_usages_par_m2_ef",0):.0f} kWh/m2 | |
| **Reel Enedis** | **{rkwh:,.0f}** | |
| Ecart reel/DPE | {ecrt:+.1f}% | |
| **Cout annuel** | **{cout:,.0f} euros/an** | |
""")

                # Mini pie usage
                use_vals = {lbl: row.get(col,0) for col,lbl in USAGE_COLS.items()}
                use_vals = {k:v for k,v in use_vals.items() if pd.notna(v) and v>0}
                if use_vals:
                    fig_mp = px.pie(
                        values=list(use_vals.values()),
                        names=list(use_vals.keys()),
                        color_discrete_sequence=USAGE_COLORS,
                        hole=0.4, height=220,
                    )
                    fig_mp.update_traces(texttemplate="%{percent:.0%}", textposition="inside")
                    fig_mp.update_layout(
                        margin=dict(t=10,b=10,l=0,r=0),
                        showlegend=True,
                        legend=dict(font_size=10),
                    )
                    st.plotly_chart(fig_mp, use_container_width=True)

            with fc:
                st.markdown("**Thermique & Isolation**")
                st.markdown(f"""
| Composant | Valeur |
|---|---|
| Iso. murs | {row.get("qualite_isolation_murs","N/A")} |
| Iso. enveloppe | {row.get("qualite_isolation_enveloppe","N/A")} |
| Dep. enveloppe | {row.get("deperditions_enveloppe",0):,.0f} W/K |
| Dep. murs | {row.get("deperditions_murs",0):,.0f} W/K |
| Dep. planchers bas | {row.get("deperditions_planchers_bas",0):,.0f} W/K |
| Dep. planchers hauts | {row.get("deperditions_planchers_hauts",0):,.0f} W/K |
""")

                # Jauge score DPE
                score = row.get("score_dpe",0) or 0
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(score),
                    title={"text":f"Score DPE — Classe {dpe}", "font":{"size":13}},
                    gauge={
                        "axis":{"range":[0,700],"tickfont":{"size":10}},
                        "bar":{"color":DPE_COLORS.get(dpe,"#888")},
                        "steps":[
                            {"range":[0,70],  "color":"#009900"},
                            {"range":[70,110],"color":"#55C400"},
                            {"range":[110,180],"color":"#AAFF00"},
                            {"range":[180,250],"color":"#FFFF00"},
                            {"range":[250,330],"color":"#FFB432"},
                            {"range":[330,420],"color":"#FF5000"},
                            {"range":[420,700],"color":"#FF0000"},
                        ],
                    },
                ))
                fig_g.update_layout(height=210, margin=dict(t=40,b=10,l=15,r=15))
                st.plotly_chart(fig_g, use_container_width=True)

                st.markdown(f"**Distance :** {dist:,} m de votre adresse")

    st.markdown('<hr style="border-top:2px solid #eee;margin:16px 0">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # H. POSITIONNEMENT VS COMMUNE & FRANCE
    # ──────────────────────────────────────────────────────────
    st.markdown("### Positionnement vs commune et France")

    nat_med   = df["conso_relle_kwh"].median()
    comm_med  = df[df["nom_commune_ban"]==commune]["conso_relle_kwh"].median()
    near_med  = nearest["conso_relle_kwh"].median()

    h1, h2 = st.columns(2)

    with h1:
        cmp_df = pd.DataFrame({
            "Niveau": ["France",""+commune,"10 proches","Votre DPE ("+dpe_actuel+")"],
            "kWh/an": [
                nat_med, comm_med, near_med,
                df[df["etiquette_dpe"]==dpe_actuel]["conso_relle_kwh"].median(),
            ],
        }).dropna()
        fig_cmp3 = px.bar(
            cmp_df, x="Niveau", y="kWh/an",
            color="Niveau",
            color_discrete_sequence=["#1f77b4","#2ca02c","#9467bd","#d62728"],
            title="Conso reelle mediane — Comparaison niveaux",
            text="kWh/an",
        )
        fig_cmp3.update_traces(texttemplate="%{text:,.0f} kWh", textposition="outside")
        fig_cmp3.update_layout(showlegend=False, height=330)
        st.plotly_chart(fig_cmp3, use_container_width=True)

    with h2:
        # Distribution France avec position des 10 proches
        nat_s = df["conso_relle_kwh"].dropna().sample(min(5000,len(df)), random_state=42)
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=nat_s, name="France (echantillon)",
            opacity=0.4, marker_color="#1f77b4",
            xbins=dict(size=500), histnorm="percent",
        ))
        for v in nearest["conso_relle_kwh"].dropna():
            fig_h.add_vline(x=v, line_width=1.5, line_color="#d62728", opacity=0.6)
        fig_h.add_vline(
            x=near_med, line_width=2.5, line_color="#d62728",
            annotation_text="Mediane\nproches", annotation_position="top right",
        )
        fig_h.update_layout(
            title="Distribution nationale vs vos 10 logements proches",
            xaxis_title="kWh/an", yaxis_title="%",
            height=330,
            xaxis_range=[0, min(nat_s.quantile(0.97)*1.1, 14000)],
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE COMMUNE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Analyse de la commune : {commune}")

    comm_df = df[df["nom_commune_ban"] == commune].copy()
    if comm_df.empty:
        st.warning(f"Aucune donnee pour {commune}.")
    else:
        pct_pass = comm_df["etiquette_dpe"].isin(["F","G"]).mean()*100
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Logements", f"{len(comm_df):,}")
        k2.metric("Conso reelle mediane", f"{comm_df['conso_relle_kwh'].median():,.0f} kWh/an")
        k3.metric("Conso DPE mediane",    f"{comm_df['conso_dpe_kwh'].median():,.0f} kWh/an")
        k4.metric("Passoires (F+G)",      f"{pct_pass:.1f}%")

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            dpe_cnt = comm_df["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER,fill_value=0)
            fig_b = px.bar(x=dpe_cnt.index, y=dpe_cnt.values,
                           color=dpe_cnt.index, color_discrete_map=DPE_COLORS,
                           title=f"Distribution DPE — {commune}",
                           labels={"x":"Classe","y":"Nb logements"}, text=dpe_cnt.values)
            fig_b.update_traces(textposition="outside")
            fig_b.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_b, use_container_width=True)

        with c2:
            mc = median_conso_by_dpe(comm_df).reset_index()
            fig_g2 = go.Figure()
            fig_g2.add_trace(go.Bar(name="DPE estime",  x=mc["etiquette_dpe"], y=mc["conso_dpe_kwh"],  marker_color="#4C9BE8",opacity=0.8))
            fig_g2.add_trace(go.Bar(name="Reel Enedis", x=mc["etiquette_dpe"], y=mc["conso_relle_kwh"],marker_color="#F28B30",opacity=0.8))
            fig_g2.update_layout(barmode="group", title=f"Conso mediane par classe — {commune}",
                                  xaxis_title="DPE",yaxis_title="kWh/an",
                                  height=340,legend=dict(orientation="h",y=-0.22))
            st.plotly_chart(fig_g2, use_container_width=True)

        c3,c4 = st.columns(2)
        with c3:
            sc = comm_df.dropna(subset=["conso_dpe_kwh","conso_relle_kwh"])
            sc = sc[sc["conso_relle_kwh"]<sc["conso_relle_kwh"].quantile(0.99)]
            if not sc.empty:
                fig_sc2 = px.scatter(sc,x="conso_dpe_kwh",y="conso_relle_kwh",
                                     color="etiquette_dpe",color_discrete_map=DPE_COLORS,
                                     title=f"DPE estime vs Reel — {commune}",
                                     labels={"conso_dpe_kwh":"DPE (kWh)","conso_relle_kwh":"Reel (kWh)","etiquette_dpe":"Classe"},
                                     opacity=0.55,height=340)
                mx = max(sc["conso_dpe_kwh"].max(),sc["conso_relle_kwh"].max())
                fig_sc2.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
                                             line=dict(dash="dash",color="grey"),name="Estimation parfaite"))
                st.plotly_chart(fig_sc2, use_container_width=True)

        with c4:
            mc2 = median_conso_by_dpe(comm_df)["conso_relle_kwh"]
            gains = []
            for i in range(len(DPE_ORDER)-1):
                frm,to_ = DPE_ORDER[i+1],DPE_ORDER[i]
                if frm in mc2.index and to_ in mc2.index:
                    g = mc2[frm]-mc2[to_]
                    if g>0:
                        gains.append({"Passage":f"{frm}→{to_}","kWh/an":round(g),"euros/an":round(g*prix_kwh)})
            if gains:
                gdf = pd.DataFrame(gains)
                fig_gn2 = px.bar(gdf,x="Passage",y="kWh/an",
                                 title=f"Gains par amelioration DPE — {commune}",
                                 color="kWh/an",color_continuous_scale=["#FF5000","#55C400"],
                                 text="euros/an",height=340)
                fig_gn2.update_traces(texttemplate="%{text} euros/an",textposition="outside")
                fig_gn2.update_coloraxes(showscale=False)
                st.plotly_chart(fig_gn2, use_container_width=True)

        st.markdown("### Recapitulatif par classe DPE")
        tbl = median_conso_by_dpe(comm_df).reset_index()
        tbl["Cout median (euros/an)"] = (tbl["conso_relle_kwh"]*prix_kwh).round(0).astype("Int64")
        tbl.columns = ["Classe DPE","Conso DPE (kWh)","Conso reelle (kWh)","Nb logements","Cout median (euros/an)"]
        for c_ in ["Conso DPE (kWh)","Conso reelle (kWh)"]:
            tbl[c_] = tbl[c_].round(0).astype("Int64")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE FRANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analyse nationale — Benchmark toutes communes")

    nat = df.copy()
    ppt = nat["etiquette_dpe"].isin(["F","G"]).mean()*100
    n1,n2,n3,n4 = st.columns(4)
    n1.metric("Total logements",      f"{len(nat):,}")
    n2.metric("Conso reelle mediane", f"{nat['conso_relle_kwh'].median():,.0f} kWh/an")
    n3.metric("Conso DPE mediane",    f"{nat['conso_dpe_kwh'].median():,.0f} kWh/an")
    n4.metric("Passoires (F+G)",      f"{ppt:.1f}%")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        dpe_nat = nat["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER,fill_value=0)
        fig_pie2 = px.pie(values=dpe_nat.values,names=dpe_nat.index,
                          color=dpe_nat.index,color_discrete_map=DPE_COLORS,
                          title="Repartition nationale des classes DPE",hole=0.35)
        fig_pie2.update_traces(texttemplate="%{label}<br>%{percent:.1%}",textposition="inside")
        st.plotly_chart(fig_pie2, use_container_width=True)

    with c2:
        box_df = nat.dropna(subset=["conso_relle_kwh","etiquette_dpe"])
        box_df = box_df[box_df["conso_relle_kwh"]<box_df["conso_relle_kwh"].quantile(0.99)]
        fig_box2 = px.box(box_df,x="etiquette_dpe",y="conso_relle_kwh",
                          color="etiquette_dpe",color_discrete_map=DPE_COLORS,
                          category_orders={"etiquette_dpe":DPE_ORDER},
                          title="Distribution conso reelle par classe DPE",
                          labels={"etiquette_dpe":"Classe","conso_relle_kwh":"kWh/an"},
                          points="outliers")
        fig_box2.update_layout(showlegend=False)
        st.plotly_chart(fig_box2, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        nat_mc2 = median_conso_by_dpe(nat).reset_index()
        fig_nc2 = go.Figure()
        fig_nc2.add_trace(go.Bar(name="DPE estime",x=nat_mc2["etiquette_dpe"].astype(str),y=nat_mc2["conso_dpe_kwh"],marker_color="#4C9BE8"))
        fig_nc2.add_trace(go.Bar(name="Reel Enedis",x=nat_mc2["etiquette_dpe"].astype(str),y=nat_mc2["conso_relle_kwh"],marker_color="#F28E2B"))
        fig_nc2.update_layout(barmode="group",title="DPE estime vs Reel — mediane nationale",
                               xaxis_title="Classe DPE",yaxis_title="kWh/an",legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig_nc2, use_container_width=True)
        st.caption("Effet precarite : les menages F/G se chauffent insuffisamment par contrainte budgetaire.")

    with c4:
        user_mc2  = nat[nat["etiquette_dpe"]==dpe_actuel]["conso_relle_kwh"].median()
        comm_mc2  = df[df["nom_commune_ban"]==commune]["conso_relle_kwh"].median()
        near_mc2  = nearest["conso_relle_kwh"].median() if not nearest.empty else np.nan
        nat_mc3   = nat["conso_relle_kwh"].median()
        cmp_df2 = pd.DataFrame({
            "Niveau": ["France",commune,"10 proches","DPE "+dpe_actuel],
            "kWh/an": [nat_mc3,comm_mc2,near_mc2,user_mc2],
        }).dropna()
        fig_pos2 = px.bar(cmp_df2,x="Niveau",y="kWh/an",color="Niveau",
                          color_discrete_sequence=["#1f77b4","#2ca02c","#9467bd","#d62728"],
                          title="Votre position relative",text="kWh/an")
        fig_pos2.update_traces(texttemplate="%{text:,.0f} kWh",textposition="outside")
        fig_pos2.update_layout(showlegend=False)
        st.plotly_chart(fig_pos2, use_container_width=True)

    st.markdown("### Gains nationaux par amelioration DPE")
    nat_mc_cls = median_conso_by_dpe(nat)["conso_relle_kwh"]
    gains_nat = []
    for i in range(len(DPE_ORDER)-1):
        frm,to_ = DPE_ORDER[i+1],DPE_ORDER[i]
        if frm in nat_mc_cls.index and to_ in nat_mc_cls.index:
            g = nat_mc_cls[frm]-nat_mc_cls[to_]
            if g>0:
                gains_nat.append({
                    "Amelioration":f"{frm}→{to_}",
                    "Gain (kWh/an)":f"{g:,.0f}",
                    "Gain (euros/an)":f"{g*prix_kwh:,.0f}",
                    "Gain sur 10 ans":f"{g*prix_kwh*10*(1+taux_hausse*5):,.0f}",
                })
    if gains_nat:
        st.dataframe(pd.DataFrame(gains_nat), use_container_width=True, hide_index=True)

    prd_df = (nat.dropna(subset=["conso_relle_kwh","periode_construction"])
                 .groupby(["periode_construction","etiquette_dpe"],observed=True)
                 ["conso_relle_kwh"].median().reset_index())
    prd_order = ["Avant 1948","1948-1974","1975-1989","1990-2000",
                 "2001-2005","2006-2012","2013-2021","Apres 2021","Inconnue"]
    fig_prd2 = px.bar(prd_df[prd_df["periode_construction"].isin(prd_order)],
                      x="periode_construction",y="conso_relle_kwh",
                      color="etiquette_dpe",color_discrete_map=DPE_COLORS,
                      category_orders={"periode_construction":prd_order},
                      barmode="group",
                      title="Conso reelle mediane par periode x classe DPE",
                      labels={"conso_relle_kwh":"kWh/an","periode_construction":"Periode"})
    fig_prd2.update_layout(height=370)
    st.plotly_chart(fig_prd2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION 10 ANS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Prediction des couts electriques sur 10 ans")

    user_data2 = df[
        (df["etiquette_dpe"]==dpe_actuel) &
        df["surface_habitable_logement"].between(surface*0.7, surface*1.4)
    ]["conso_relle_kwh"].dropna()
    if user_data2.empty:
        user_data2 = df[df["etiquette_dpe"]==dpe_actuel]["conso_relle_kwh"].dropna()
    estimated_kwh = int(user_data2.median()) if not user_data2.empty else 4000

    col_p, col_c = st.columns([1,3])
    with col_p:
        st.markdown("#### Parametres")
        custom_kwh = st.number_input("Conso annuelle estimee (kWh)",500,50000,estimated_kwh,100)
        st.markdown(
            f"**Profil**<br>"
            f"Classe DPE : {dpe_badge(dpe_actuel)}<br>"
            f"Surface : **{surface} m2**<br>"
            f"Conso : **{custom_kwh:,} kWh/an**<br>"
            f"**Cout 2024 : {custom_kwh*prix_kwh:,.0f} euros/an**",
            unsafe_allow_html=True,
        )

    with col_c:
        years, scenarios = predict_costs(custom_kwh, prix_kwh, taux_hausse, dpe_actuel)
        fig_pred2 = go.Figure()
        for (name,costs),col_s in zip(scenarios.items(),["#d62728","#FFB432","#2ca02c"]):
            fig_pred2.add_trace(go.Scatter(
                x=years, y=[round(c) for c in costs],
                mode="lines+markers", name=name,
                line=dict(color=col_s,width=3), marker=dict(size=7),
                hovertemplate=f"<b>{name}</b><br>%{{x}} : %{{y:,.0f}} euros<extra></extra>",
            ))
        if len(scenarios)>1:
            fig_pred2.add_vrect(x0=2026.5,x1=2028.5,fillcolor="lightgreen",opacity=0.1,
                                annotation_text="Fenetre\nrenovation",annotation_position="top left")
        fig_pred2.update_layout(
            title=f"Couts electriques — DPE {dpe_actuel}, {surface} m2",
            xaxis_title="Annee",yaxis_title="euros/an",
            hovermode="x unified",legend=dict(orientation="h",y=1.1,x=0),height=400,
        )
        st.plotly_chart(fig_pred2, use_container_width=True)

    base_costs = list(scenarios.values())[0]
    rows2 = []
    for yr,base in zip(years,base_costs):
        row2 = {"Annee":yr,list(scenarios.keys())[0]:f"{base:,.0f}"}
        for name,costs in list(scenarios.items())[1:]:
            c2 = costs[years.index(yr)]
            row2[name] = f"{c2:,.0f}"
            row2["Economie vs sans renov. (euros)"] = f"{base-c2:,.0f}"
        rows2.append(row2)
    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    st.markdown("### Economies cumulees sur 10 ans")
    total_base = sum(base_costs)
    ecols2 = st.columns(len(scenarios))
    for i,(name,costs) in enumerate(scenarios.items()):
        saving = total_base-sum(costs)
        with ecols2[i]:
            st.metric(name, f"{sum(costs):,.0f} euros cumules",
                      delta=f"-{saving:,.0f} euros" if saving>0 else "Reference",
                      delta_color="inverse" if saving>0 else "off")

    if len(scenarios)>1:
        cum_rows = []
        for name,costs in list(scenarios.items())[1:]:
            cum = 0
            for yr,(base,alt) in zip(years,zip(base_costs,costs)):
                cum += base-alt
                cum_rows.append({"Annee":yr,"Scenario":name,"Economie cumulee (euros)":round(cum)})
        fig_cum2 = px.area(pd.DataFrame(cum_rows),x="Annee",y="Economie cumulee (euros)",
                           color="Scenario",color_discrete_sequence=["#FFB432","#2ca02c"],
                           title="Economies cumulees vs sans renovation",markers=True)
        fig_cum2.update_layout(height=300,hovermode="x unified")
        st.plotly_chart(fig_cum2, use_container_width=True)

st.markdown("---")
st.caption(
    "Sources : ADEME (DPE logements existants post-2021) x Enedis (conso residentielle par adresse). "
    "Geocodage via API BAN — adresse.data.gouv.fr. "
    "Predictions basees sur des hypotheses tarifaires simplifiees."
)
