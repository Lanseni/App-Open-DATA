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

def fmt(n, unit=""):
    """Format number with space as thousands separator (French style)."""
    try:
        s = f"{int(round(n)):,}".replace(",", " ")
        return f"{s} {unit}".strip() if unit else s
    except Exception:
        return str(n)



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
    """Prediction basee sur les medianes de consommation reelle Enedis par classe DPE."""
    conso_med = median_conso_by_dpe(df)["conso_relle_kwh"]
    base_g    = conso_med.get("G", kwh_year) or kwh_year
    def factor(cls):
        v = conso_med.get(cls)
        return (v/base_g) if v and base_g else 1.0
    years = list(range(2025, 2036))
    s = {}
    s["Sans renovation"] = [kwh_year*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur not in ("A","B"):
        r  = factor("B")/max(factor(dpe_cur),0.01)
        nk = kwh_year*r
        s["Renovation -> Classe B"] = [(kwh_year if i<2 else nk)*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur != "A":
        r  = factor("A")/max(factor(dpe_cur),0.01)
        nk = kwh_year*r
        s["Renovation -> Classe A"] = [(kwh_year if i<3 else nk)*prix*(1+taux)**i for i in range(len(years))]
    return years, s


def predict_costs_3cl(kwh_year, prix, taux, dpe_cur):
    """
    Prediction basee sur le standard DPE 3CL (methode officielle de calcul thermique).
    Les ratios de reduction correspondent au potentiel technique reel de la renovation.
    Source : medianes kWh/m2/an calcules sur le dataset ADEME (methode 3CL).
    """
    DPE_KWH_M2 = {
        "A": 25.3, "B": 41.4, "C": 69.1,
        "D": 96.6, "E": 132.0, "F": 172.0, "G": 242.0,
    }
    cur_m2 = DPE_KWH_M2.get(dpe_cur, 100.0)
    years  = list(range(2025, 2036))
    s = {}
    s["Sans renovation"] = [kwh_year*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur not in ("A", "B"):
        nk_b = kwh_year * (DPE_KWH_M2["B"] / cur_m2)
        s["Renovation -> Classe B"] = [(kwh_year if i<2 else nk_b)*prix*(1+taux)**i for i in range(len(years))]
    if dpe_cur != "A":
        nk_a = kwh_year * (DPE_KWH_M2["A"] / cur_m2)
        s["Renovation -> Classe A"] = [(kwh_year if i<3 else nk_a)*prix*(1+taux)**i for i in range(len(years))]
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
    commune = st.selectbox(
        "Commune (onglets 2-3)",
        communes_list,
        index=default_idx,
        help=(
            "Le dataset couvre ~400 communes. "
            "Les grandes villes (Paris, Lyon, Marseille, Nice, Strasbourg...) "
            "ne sont pas disponibles car Enedis n y publie pas de donnees "
            "a la maille adresse (moins de 10 logements par adresse)."
        ),
    )
    # Avertissement si commune tapee mais non trouvee
    if default_city and not matched:
        st.warning(
            f"**{default_city}** n est pas dans le dataset Enedis. "
            "Les grandes metropoles (Nice, Paris, Lyon, Marseille...) sont absentes "
            "car les donnees Enedis ne sont publiees que pour les adresses "
            "avec au moins 10 logements. La commune la plus proche est selectionnee."
        )
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
# TAB 4 — PREDICTION 10 ANS (ENEDIS + 3CL)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Prediction des couts electriques sur 10 ans")

    # ── Estimation de la consommation actuelle ──
    user_data2 = df[
        (df["etiquette_dpe"] == dpe_actuel) &
        df["surface_habitable_logement"].between(surface * 0.7, surface * 1.4)
    ]["conso_relle_kwh"].dropna()
    if user_data2.empty:
        user_data2 = df[df["etiquette_dpe"] == dpe_actuel]["conso_relle_kwh"].dropna()
    estimated_kwh = int(user_data2.median()) if not user_data2.empty else 4000

    # ── Paramètres communs aux deux modeles ──
    col_params, col_right = st.columns([1, 3])
    with col_params:
        st.markdown("#### Parametres")
        custom_kwh = st.number_input(
            "Conso annuelle actuelle (kWh)",
            min_value=500, max_value=50000,
            value=estimated_kwh, step=100,
            help="Votre consommation reelle (ou estimee depuis le dataset Enedis).",
        )
        st.markdown(
            f"**Profil**<br>"
            f"Classe DPE : {dpe_badge(dpe_actuel)}<br>"
            f"Surface : **{surface} m2**<br>"
            f"Conso : **{custom_kwh:,} kWh/an**<br>"
            f"**Cout actuel : {custom_kwh * prix_kwh:.0f} euros/an**",
            unsafe_allow_html=True,
        )

        # Potentiel 3CL affiché dans la sidebar du tab
        DPE_KWH_M2 = {"A": 25.3, "B": 41.4, "C": 69.1, "D": 96.6, "E": 132.0, "F": 172.0, "G": 242.0}
        cur_m2 = DPE_KWH_M2.get(dpe_actuel, 100.0)
        ratio_a = DPE_KWH_M2["A"] / cur_m2
        ratio_b = DPE_KWH_M2.get("B", cur_m2) / cur_m2
        st.markdown("---")
        st.markdown("**Potentiel technique (3CL)**")
        if dpe_actuel not in ("A", "B"):
            st.metric(
                f"Apres renovation → B",
                f"{custom_kwh * ratio_b:,.0f} kWh/an",
                delta=f"-{(1 - ratio_b) * 100:.0f}% de conso",
                delta_color="inverse",
            )
        if dpe_actuel != "A":
            st.metric(
                f"Apres renovation → A",
                f"{custom_kwh * ratio_a:,.0f} kWh/an",
                delta=f"-{(1 - ratio_a) * 100:.0f}% de conso",
                delta_color="inverse",
            )

    with col_right:
        # ── Explication des deux approches ──
        exp1, exp2 = st.columns(2)
        with exp1:
            st.info(
                "**Modele Enedis** — Basé sur les consommations réelles mesurées "
                "par les compteurs Linky. Intègre l'effet précarité (ménages F/G "
                "sous-chauffés). Reflète ce qui se passe vraiment."
            )
        with exp2:
            st.success(
                "**Modele 3CL** — Basé sur le standard DPE officiel (kWh/m²/an). "
                "Mesure le potentiel physique de la rénovation. "
                "Donne les économies maximales atteignables."
            )

    st.markdown("---")

    # ══════════════════════════════
    # MODELE 1 — ENEDIS (reel)
    # ══════════════════════════════
    st.markdown("### Modele Enedis — Consommations reelles Linky")

    years, scenarios = predict_costs(custom_kwh, prix_kwh, taux_hausse, dpe_actuel)

    col_e1, col_e2 = st.columns([3, 2])
    with col_e1:
        fig_pred_e = go.Figure()
        for (name, costs), col_s in zip(scenarios.items(), ["#d62728", "#FFB432", "#2ca02c"]):
            fig_pred_e.add_trace(go.Scatter(
                x=years, y=[round(c) for c in costs],
                mode="lines+markers", name=name,
                line=dict(color=col_s, width=3), marker=dict(size=7),
                hovertemplate=f"<b>{name}</b><br>%{{x}} : %{{y:.0f}} euros<extra></extra>",
            ))
        if len(scenarios) > 1:
            fig_pred_e.add_vrect(
                x0=2026.5, x1=2028.5, fillcolor="lightgreen", opacity=0.1,
                annotation_text="Fenetre renovation", annotation_position="top left",
            )
        fig_pred_e.update_layout(
            title=f"Enedis — DPE {dpe_actuel}, {surface} m2",
            xaxis_title="Annee", yaxis_title="euros/an",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0),
            height=360,
        )
        st.plotly_chart(fig_pred_e, use_container_width=True, key="chart_pred_enedis")

    with col_e2:
        # Economies cumulees Enedis
        base_e = list(scenarios.values())[0]
        total_e = sum(base_e)
        st.markdown("**Economies cumulees (11 ans)**")
        for name, costs in list(scenarios.items())[1:]:
            saving = total_e - sum(costs)
            st.metric(name.replace("Renovation -> ", "→ "),
                      f"{sum(costs):.0f} euros",
                      delta=f"-{saving:.0f} euros economies" if saving > 0 else "Reference",
                      delta_color="inverse" if saving > 0 else "off")

    # Tableau annuel Enedis
    with st.expander("Voir le tableau annuel detaille — Modele Enedis"):
        base_costs = list(scenarios.values())[0]
        rows2 = []
        for yr, base in zip(years, base_costs):
            row2 = {"Annee": yr, list(scenarios.keys())[0]: f"{base:.0f}"}
            for name, costs in list(scenarios.items())[1:]:
                c2 = costs[years.index(yr)]
                row2[name] = f"{c2:.0f}"
                row2["Economie (euros)"] = f"{base - c2:.0f}"
            rows2.append(row2)
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    if len(scenarios) > 1:
        cum_rows = []
        for name, costs in list(scenarios.items())[1:]:
            cum = 0
            for yr, (base, alt) in zip(years, zip(base_costs, costs)):
                cum += base - alt
                cum_rows.append({"Annee": yr, "Scenario": name, "Economie cumulee (euros)": round(cum)})
        fig_cum_e = px.area(
            pd.DataFrame(cum_rows), x="Annee", y="Economie cumulee (euros)", color="Scenario",
            color_discrete_sequence=["#FFB432", "#2ca02c"],
            title="Economies cumulees — Modele Enedis",
            markers=True,
        )
        fig_cum_e.update_layout(height=260, hovermode="x unified")
        st.plotly_chart(fig_cum_e, use_container_width=True, key="chart_cum_enedis")

    st.markdown("---")

    # ══════════════════════════════
    # MODELE 2 — 3CL (potentiel technique)
    # ══════════════════════════════
    st.markdown("### Modele 3CL — Potentiel technique de renovation")
    st.caption(
        "Les ratios sont calcules depuis les kWh/m2/an de la methode officielle de calcul thermique DPE 3CL. "
        "Ils representent le potentiel maximal atteignable independamment des comportements."
    )

    years3, scenarios3 = predict_costs_3cl(custom_kwh, prix_kwh, taux_hausse, dpe_actuel)

    col_3a, col_3b = st.columns([3, 2])
    with col_3a:
        fig_pred_3cl = go.Figure()
        for (name, costs), col_s in zip(scenarios3.items(), ["#d62728", "#FFB432", "#2ca02c"]):
            fig_pred_3cl.add_trace(go.Scatter(
                x=years3, y=[round(c) for c in costs],
                mode="lines+markers", name=name,
                line=dict(color=col_s, width=3), marker=dict(size=7),
                hovertemplate=f"<b>{name}</b><br>%{{x}} : %{{y:.0f}} euros<extra></extra>",
            ))
        if len(scenarios3) > 1:
            fig_pred_3cl.add_vrect(
                x0=2026.5, x1=2028.5, fillcolor="lightgreen", opacity=0.1,
                annotation_text="Fenetre renovation", annotation_position="top left",
            )
        fig_pred_3cl.update_layout(
            title=f"3CL — DPE {dpe_actuel}, {surface} m2",
            xaxis_title="Annee", yaxis_title="euros/an",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0),
            height=360,
        )
        st.plotly_chart(fig_pred_3cl, use_container_width=True, key="chart_pred_3cl")

    with col_3b:
        base3 = list(scenarios3.values())[0]
        total3 = sum(base3)
        st.markdown("**Economies cumulees (11 ans)**")
        for name, costs in list(scenarios3.items())[1:]:
            saving3 = total3 - sum(costs)
            st.metric(
                name.replace("Renovation -> ", "→ "),
                f"{sum(costs):.0f} euros",
                delta=f"-{saving3:.0f} euros economies" if saving3 > 0 else "Reference",
                delta_color="inverse" if saving3 > 0 else "off",
            )

    # Tableau annuel 3CL
    with st.expander("Voir le tableau annuel detaille — Modele 3CL"):
        rows3 = []
        for yr, base in zip(years3, base3):
            row3 = {"Annee": yr, list(scenarios3.keys())[0]: f"{base:.0f} euros"}
            for name, costs in list(scenarios3.items())[1:]:
                c3_ = costs[years3.index(yr)]
                row3[name] = f"{c3_:.0f} euros"
                row3["Economie (euros)"] = f"{base - c3_:.0f}"
            rows3.append(row3)
        st.dataframe(pd.DataFrame(rows3), use_container_width=True, hide_index=True)

    if len(scenarios3) > 1:
        cum_rows3 = []
        for name, costs in list(scenarios3.items())[1:]:
            cum3 = 0
            for yr, (base, alt) in zip(years3, zip(base3, costs)):
                cum3 += base - alt
                cum_rows3.append({"Annee": yr, "Scenario": name, "Economie cumulee (euros)": round(cum3)})
        fig_cum_3cl = px.area(
            pd.DataFrame(cum_rows3), x="Annee", y="Economie cumulee (euros)", color="Scenario",
            color_discrete_sequence=["#FFB432", "#2ca02c"],
            title="Economies cumulees — Modele 3CL",
            markers=True,
        )
        fig_cum_3cl.update_layout(height=260, hovermode="x unified")
        st.plotly_chart(fig_cum_3cl, use_container_width=True, key="chart_cum_3cl")

    st.markdown("---")

    # ══════════════════════════════
    # COMPARAISON ENEDIS vs 3CL
    # ══════════════════════════════
    st.markdown("### Comparaison Enedis vs 3CL — Economies totales sur 11 ans")

    cmp_rows = []
    for (name_e, costs_e), (name_3, costs_3) in zip(
        list(scenarios.items())[1:], list(scenarios3.items())[1:]
    ):
        eco_e = sum(base_costs) - sum(costs_e)
        eco_3 = sum(base3) - sum(costs_3)
        label = name_e.replace("Renovation -> ", "")
        cmp_rows.append({
            "Scenario": label,
            "Enedis — reel (euros)": round(eco_e),
            "3CL — potentiel max (euros)": round(eco_3),
            "Ecart (euros)": round(eco_3 - eco_e),
        })

    if cmp_rows:
        cdf = pd.DataFrame(cmp_rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)

        fig_cmp_both = go.Figure()
        colors_both = {"Enedis — reel (euros)": "#F28E2B", "3CL — potentiel max (euros)": "#2ca02c"}
        for col_name, bar_color in colors_both.items():
            fig_cmp_both.add_trace(go.Bar(
                name=col_name,
                x=cdf["Scenario"],
                y=cdf[col_name],
                marker_color=bar_color,
                text=cdf[col_name].apply(lambda v: f"{v:.0f} euros"),
                textposition="outside",
            ))
        fig_cmp_both.update_layout(
            barmode="group",
            title="Economies totales sur 11 ans : Modele Enedis vs Modele 3CL",
            yaxis_title="Economies cumulees (euros)",
            legend=dict(orientation="h", y=-0.25),
            height=380,
        )
        st.plotly_chart(fig_cmp_both, use_container_width=True, key="chart_cmp_both")
        st.caption(
            "Le modele Enedis reflète les comportements reels (avec effet precarite energetique). "
            "Le modele 3CL indique le potentiel maximal atteignable par la renovation thermique. "
            "La verite se situe generalement entre les deux."
        )

        # Tableau kWh/m2 par classe
        st.markdown("**Reference — kWh/m2/an par classe DPE (standard 3CL)**")
        ref_df = pd.DataFrame([
            {"Classe": cls, "kWh/m2/an (3CL)": v,
             "Reduction vs G": f"{(1 - v/242)*100:.0f}%",
             "Ratio vs classe actuelle": f"{v/cur_m2:.2f}"}
            for cls, v in DPE_KWH_M2.items()
        ])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.caption(
    "Sources : ADEME (DPE logements existants post-2021) x Enedis (conso residentielle par adresse). "
    "Geocodage via API BAN — adresse.data.gouv.fr. "
    "Predictions basees sur des hypotheses tarifaires simplifiees."
)
