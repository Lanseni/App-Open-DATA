"""
Application d'analyse DPE & Consommation Électrique
Enedis × ADEME — Open Data University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import KDTree
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Mettre à jour ce chemin si votre CSV est ailleurs
DATA_PATH = os.path.join(os.path.dirname(__file__), "DPE_enedis_1.csv")

ELECTRICITY_PRICE_2024 = 0.2516  # €/kWh — TRV Bleu EDF 2024
DEFAULT_PRICE_INCREASE  = 3       # % annuel hypothèse

DPE_COLORS = {
    "A": "#009900", "B": "#55C400", "C": "#AAFF00",
    "D": "#FFFF00", "E": "#FFB432", "F": "#FF5000", "G": "#FF0000",
}
DPE_TEXT_COLORS = {
    "A": "white", "B": "white", "C": "black",
    "D": "black",  "E": "white", "F": "white", "G": "white",
}
DPE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

st.set_page_config(
    page_title="DPE & Conso Électrique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stMetric { background:#f5f7fa; border-radius:10px; padding:8px 12px; }
  .dpe-label {
    display:inline-block; padding:2px 10px; border-radius:4px;
    font-weight:bold; font-size:1.1rem;
  }
  h1 { font-size:1.8rem !important; }
  .section-title { font-size:1.1rem; font-weight:600; color:#333; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="📂 Chargement des données…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # ── Conversion Lambert 93 → WGS84 ──────────────────────
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    mask_l93 = df["coordonnee_cartographique_x_ban"] > 1000
    x = df.loc[mask_l93, "coordonnee_cartographique_x_ban"].values
    y = df.loc[mask_l93, "coordonnee_cartographique_y_ban"].values
    lons, lats = transformer.transform(x, y)
    df.loc[mask_l93, "lon"] = lons
    df.loc[mask_l93, "lat"] = lats

    # ── Colonnes calculées ──────────────────────────────────
    df["conso_dpe_kwh"]    = df["conso_5_usages_par_m2_ef"] * df["surface_habitable_logement"]
    df["conso_relle_kwh"]  = df["Consommation annuelle moyenne par logement de l'adresse (MWh)"] * 1000
    df["etiquette_dpe"]    = df["etiquette_dpe"].astype(str).str.strip().str.upper()
    df = df[df["etiquette_dpe"].isin(DPE_ORDER)].copy()
    df["etiquette_dpe"]    = pd.Categorical(df["etiquette_dpe"], categories=DPE_ORDER, ordered=True)

    # ── KDTree sur coord Lambert93 (pour distance rapide) ──
    valid_coords = df[["coordonnee_cartographique_x_ban",
                        "coordonnee_cartographique_y_ban"]].dropna()
    df["_valid_coord"] = df.index.isin(valid_coords.index)
    return df


@st.cache_resource(show_spinner=False)
def build_kdtree(df):
    valid = df[df["_valid_coord"]]
    coords = valid[["coordonnee_cartographique_x_ban",
                     "coordonnee_cartographique_y_ban"]].values
    tree  = KDTree(coords)
    idx   = valid.index.tolist()
    return tree, idx


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ Fichier CSV introuvable : `{DATA_PATH}`")
    st.info("Placez `DPE_enedis_1.csv` dans le même dossier que `app.py`, ou modifiez `DATA_PATH`.")
    st.stop()

tree, tree_idx = build_kdtree(df)
communes_list  = sorted(df["nom_commune_ban"].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def dpe_badge(label: str) -> str:
    c = DPE_COLORS.get(label, "#888")
    t = DPE_TEXT_COLORS.get(label, "white")
    return f'<span class="dpe-label" style="background:{c};color:{t}">  {label}  </span>'


def get_commune_center(commune: str):
    sub = df[df["nom_commune_ban"] == commune]
    return (
        sub["coordonnee_cartographique_x_ban"].mean(),
        sub["coordonnee_cartographique_y_ban"].mean(),
    )


def find_nearest(commune: str, n: int = 10) -> pd.DataFrame:
    cx, cy = get_commune_center(commune)
    dists, positions = tree.query([cx, cy], k=min(n, len(tree_idx)))
    if n == 1:
        dists, positions = [dists], [positions]
    indices = [tree_idx[p] for p in positions]
    result = df.loc[indices].copy()
    result["distance_m"] = [int(d) for d in dists]
    return result


def median_conso_by_dpe(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("etiquette_dpe", observed=True)
        .agg(
            conso_dpe_kwh=("conso_dpe_kwh", "median"),
            conso_relle_kwh=("conso_relle_kwh", "median"),
            n=("conso_relle_kwh", "count"),
        )
        .reindex(DPE_ORDER)
        .dropna(subset=["conso_relle_kwh"])
    )


def predict_costs(kwh_year: float, prix: float, taux: float, dpe_actuel: str):
    """Retourne (années, dict scénarios) pour les 10 prochaines années."""
    # Facteurs de réduction par classe (relatif au max G)
    conso_dpe_medians = median_conso_by_dpe(df)["conso_relle_kwh"]
    base = conso_dpe_medians.get("G", kwh_year)

    def factor(cls):
        v = conso_dpe_medians.get(cls)
        return (v / base) if v and base else 1.0

    years = list(range(2025, 2036))
    scenarios = {}

    # Scénario 1 — Sans rénovation
    s1 = [kwh_year * prix * (1 + taux) ** i for i in range(len(years))]
    scenarios["🔴 Sans rénovation"] = s1

    # Scénario 2 — Rénovation vers classe B (si pas déjà A/B)
    if dpe_actuel not in ("A", "B"):
        ratio = factor("B") / factor(dpe_actuel)
        new_kwh = kwh_year * ratio
        s2 = [
            (kwh_year if i < 2 else new_kwh) * prix * (1 + taux) ** i
            for i in range(len(years))
        ]
        scenarios["🟡 Rénovation → Classe B"] = s2

    # Scénario 3 — Rénovation vers classe A (si pas déjà A)
    if dpe_actuel != "A":
        ratio = factor("A") / factor(dpe_actuel)
        new_kwh = kwh_year * ratio
        s3 = [
            (kwh_year if i < 3 else new_kwh) * prix * (1 + taux) ** i
            for i in range(len(years))
        ]
        scenarios["🟢 Rénovation → Classe A"] = s3

    return years, scenarios


# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Logo_Enedis.svg/320px-Logo_Enedis.svg.png",
        width=140,
    )
    st.markdown("## 🏠 Votre logement")

    default_commune = "Orléans" if "Orléans" in communes_list else communes_list[0]
    commune = st.selectbox("📍 Commune", communes_list,
                           index=communes_list.index(default_commune))

    type_batiment = st.selectbox(
        "🏗️ Type de bâtiment",
        ["appartement", "maison", "immeuble"],
    )

    surface = st.slider("📐 Surface habitable (m²)", 15, 300, 65, step=5)

    dpe_actuel = st.select_slider(
        "🏷️ Classe DPE actuelle",
        options=DPE_ORDER, value="D",
    )
    st.markdown(f"**Classe sélectionnée :** {dpe_badge(dpe_actuel)}",
                unsafe_allow_html=True)

    periode_construction = st.selectbox(
        "📅 Période de construction",
        ["Inconnue", "Avant 1948", "1948-1974", "1975-1989",
         "1990-2000", "2001-2005", "2006-2012", "2013-2021", "Après 2021"],
    )

    st.markdown("---")
    st.markdown("### 💰 Hypothèses tarifaires")
    prix_kwh   = st.number_input("Prix actuel (€/kWh)", 0.10, 1.0,
                                  ELECTRICITY_PRICE_2024, 0.01, format="%.4f")
    taux_hausse = st.slider("Hausse annuelle du prix électricité (%)",
                             0, 10, DEFAULT_PRICE_INCREASE) / 100

    st.markdown("---")
    st.markdown(f"**Données :** {len(df):,} logements · {communes_list.__len__()} communes")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.title("⚡ DPE & Consommation Électrique — Analyse & Prédiction")
st.caption(
    "Données ADEME (DPE) croisées avec les consommations réelles Enedis. "
    "Sélectionnez votre commune et les caractéristiques de votre logement dans la barre latérale."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Logements proches",
    "🏘️ Analyse commune",
    "🇫🇷 Analyse France",
    "📈 Prédiction 10 ans",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — CARTE : 10 LOGEMENTS PROCHES
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"🗺️ Les 10 adresses les plus proches de {commune}")
    st.caption(
        "Le centre de la commune est utilisé comme point de référence. "
        "Cliquez sur un marqueur pour obtenir le détail."
    )

    nearest = find_nearest(commune, n=10)

    if nearest.empty:
        st.warning("Aucune coordonnée disponible pour cette commune.")
    else:
        center_lat = nearest["lat"].mean()
        center_lon = nearest["lon"].mean()

        # ── Carte Folium ───────────────────────────────────────
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                       tiles="CartoDB positron")

        # Marqueur "votre position estimée"
        folium.Marker(
            location=[center_lat, center_lon],
            tooltip="Votre position estimée (centre commune)",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)

        for rank, (_, row) in enumerate(nearest.iterrows(), start=1):
            if pd.isna(row.get("lat")):
                continue
            dpe = str(row.get("etiquette_dpe", "?"))
            color_hex = DPE_COLORS.get(dpe, "#888888")
            real_kwh  = row.get("conso_relle_kwh", np.nan)
            dpe_kwh   = row.get("conso_dpe_kwh", np.nan)
            ecart_pct = ((real_kwh - dpe_kwh) / dpe_kwh * 100) if dpe_kwh else float("nan")

            popup_html = f"""
            <div style='font-family:Arial;min-width:220px;font-size:13px'>
              <h4 style='margin:0 0 6px;color:{color_hex}'>#{rank} — DPE {dpe}</h4>
              <b>Adresse :</b> {row.get('adresse_ban','N/A')}<br>
              <b>Type :</b> {row.get('type_batiment','N/A')}<br>
              <b>Surface :</b> {row.get('surface_habitable_logement',0):.0f} m²<br>
              <b>Période :</b> {row.get('periode_construction','N/A')}<br>
              <hr style='margin:4px 0'>
              <b>Conso DPE estimée :</b> {dpe_kwh:,.0f} kWh/an<br>
              <b>Conso réelle Enedis :</b> {real_kwh:,.0f} kWh/an<br>
              <b>Écart DPE/réel :</b> {ecart_pct:+.1f}%<br>
              <hr style='margin:4px 0'>
              <b>Distance :</b> {row.get('distance_m',0):,} m
            </div>
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=16,
                color="white",
                weight=2,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f"#{rank} · DPE {dpe} · {row.get('surface_habitable_logement',0):.0f}m²",
            ).add_to(m)

            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="color:white;font-weight:bold;font-size:11px;'
                         f'text-align:center;line-height:20px">{rank}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(m)

        col_map, col_table = st.columns([3, 2])

        with col_map:
            st_folium(m, height=460, use_container_width=True)

        with col_table:
            # Résumé statistique
            avg_real = nearest["conso_relle_kwh"].mean()
            avg_dpe  = nearest["conso_dpe_kwh"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Conso réelle médiane", f"{nearest['conso_relle_kwh'].median():,.0f} kWh")
            c2.metric("Conso DPE médiane",    f"{nearest['conso_dpe_kwh'].median():,.0f} kWh",
                      delta=f"{((avg_real-avg_dpe)/avg_dpe*100):+.1f}% réel vs DPE" if avg_dpe else None)

            # Tableau détail
            st.markdown("**Détail des 10 adresses**")
            show_cols = {
                "etiquette_dpe": "DPE",
                "surface_habitable_logement": "Surface (m²)",
                "type_batiment": "Type",
                "periode_construction": "Période",
                "conso_dpe_kwh": "DPE estimé (kWh)",
                "conso_relle_kwh": "Réel Enedis (kWh)",
                "distance_m": "Distance (m)",
            }
            disp = nearest[[c for c in show_cols if c in nearest.columns]].copy()
            disp.columns = [show_cols[c] for c in disp.columns]
            for col in ["Surface (m²)", "DPE estimé (kWh)", "Réel Enedis (kWh)", "Distance (m)"]:
                if col in disp.columns:
                    disp[col] = disp[col].round(0).astype("Int64")
            st.dataframe(disp, use_container_width=True, hide_index=True,
                         height=320)

            # Mini boxplot DPE des 10 proches
            dpe_counts_near = nearest["etiquette_dpe"].astype(str).value_counts()
            fig_mini = px.bar(
                x=dpe_counts_near.index, y=dpe_counts_near.values,
                color=dpe_counts_near.index,
                color_discrete_map=DPE_COLORS,
                labels={"x": "Classe DPE", "y": "Nb adresses"},
                title="Classes DPE des 10 adresses proches",
                height=200,
            )
            fig_mini.update_layout(showlegend=False, margin=dict(t=30,b=20))
            st.plotly_chart(fig_mini, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE COMMUNE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🏘️ Analyse de la commune : {commune}")

    comm_df = df[df["nom_commune_ban"] == commune].copy()

    if comm_df.empty:
        st.warning(f"Pas de données pour {commune}.")
    else:
        # KPIs
        pct_passoires = (comm_df["etiquette_dpe"].isin(["F","G"])).mean() * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Logements dans la base",    f"{len(comm_df):,}")
        k2.metric("Conso réelle médiane",       f"{comm_df['conso_relle_kwh'].median():,.0f} kWh/an")
        k3.metric("Conso DPE médiane",          f"{comm_df['conso_dpe_kwh'].median():,.0f} kWh/an")
        k4.metric("Passoires (F+G)",            f"{pct_passoires:.1f}%")

        st.markdown("---")

        c1, c2 = st.columns(2)

        # Distribution des classes DPE
        with c1:
            dpe_cnt = comm_df["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER, fill_value=0)
            fig_bar = px.bar(
                x=dpe_cnt.index, y=dpe_cnt.values,
                color=dpe_cnt.index, color_discrete_map=DPE_COLORS,
                title=f"Distribution DPE — {commune}",
                labels={"x": "Classe", "y": "Nb logements"},
                text=dpe_cnt.values,
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Conso estimée vs réelle par classe DPE
        with c2:
            conso_cls = median_conso_by_dpe(comm_df).reset_index()
            fig_grp = go.Figure()
            fig_grp.add_trace(go.Bar(
                name="DPE estimé", x=conso_cls["etiquette_dpe"],
                y=conso_cls["conso_dpe_kwh"], marker_color="#4C9BE8", opacity=0.8,
            ))
            fig_grp.add_trace(go.Bar(
                name="Réel Enedis", x=conso_cls["etiquette_dpe"],
                y=conso_cls["conso_relle_kwh"], marker_color="#F28B30", opacity=0.8,
            ))
            fig_grp.update_layout(
                barmode="group",
                title=f"Consommation médiane par classe DPE — {commune}",
                xaxis_title="Classe DPE", yaxis_title="kWh/an",
                height=350, legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_grp, use_container_width=True)

        c3, c4 = st.columns(2)

        # Nuage DPE estimé vs réel
        with c3:
            sc = comm_df.dropna(subset=["conso_dpe_kwh", "conso_relle_kwh"])
            sc = sc[sc["conso_relle_kwh"] < sc["conso_relle_kwh"].quantile(0.99)]
            if not sc.empty:
                fig_sc = px.scatter(
                    sc, x="conso_dpe_kwh", y="conso_relle_kwh",
                    color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                    labels={
                        "conso_dpe_kwh": "Conso DPE (kWh)",
                        "conso_relle_kwh": "Conso réelle (kWh)",
                        "etiquette_dpe": "Classe",
                    },
                    title=f"DPE estimé vs Réel — {commune}",
                    opacity=0.6, height=340,
                )
                max_v = max(sc["conso_dpe_kwh"].max(), sc["conso_relle_kwh"].max())
                fig_sc.add_trace(go.Scatter(
                    x=[0, max_v], y=[0, max_v], mode="lines",
                    line=dict(dash="dash", color="grey"), name="Parfaite prédiction",
                ))
                st.plotly_chart(fig_sc, use_container_width=True)

        # Gains par amélioration de classe
        with c4:
            mc = median_conso_by_dpe(comm_df)["conso_relle_kwh"]
            gains = []
            for i in range(len(DPE_ORDER) - 1):
                frm, to_ = DPE_ORDER[i + 1], DPE_ORDER[i]
                if frm in mc.index and to_ in mc.index:
                    g = mc[frm] - mc[to_]
                    if g > 0:
                        gains.append({
                            "Passage": f"{frm} → {to_}",
                            "kWh/an économisés": round(g),
                            "€/an (au prix actuel)": round(g * prix_kwh),
                        })
            if gains:
                gdf = pd.DataFrame(gains)
                fig_g = px.bar(
                    gdf, x="Passage", y="kWh/an économisés",
                    title=f"Gain de conso par amélioration de classe — {commune}",
                    color="kWh/an économisés",
                    color_continuous_scale=["#FF5000", "#55C400"],
                    text="€/an (au prix actuel)",
                    height=340,
                )
                fig_g.update_traces(texttemplate="%{text} €/an", textposition="outside")
                fig_g.update_coloraxes(showscale=False)
                st.plotly_chart(fig_g, use_container_width=True)

        # Tableau récap
        st.markdown("### 📋 Récapitulatif par classe DPE")
        tbl = median_conso_by_dpe(comm_df).reset_index()
        tbl["Coût médian réel (€/an)"] = (tbl["conso_relle_kwh"] * prix_kwh).round(0).astype(int)
        tbl = tbl.rename(columns={
            "etiquette_dpe": "Classe DPE",
            "conso_dpe_kwh": "Conso DPE médiane (kWh)",
            "conso_relle_kwh": "Conso réelle médiane (kWh)",
            "n": "Nb logements",
        })
        tbl["Conso DPE médiane (kWh)"]    = tbl["Conso DPE médiane (kWh)"].round(0).astype("Int64")
        tbl["Conso réelle médiane (kWh)"] = tbl["Conso réelle médiane (kWh)"].round(0).astype("Int64")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE FRANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🇫🇷 Analyse nationale — Benchmark toutes communes")

    nat = df.copy()

    pct_pass_nat = nat["etiquette_dpe"].isin(["F", "G"]).mean() * 100
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total logements",      f"{len(nat):,}")
    n2.metric("Conso réelle médiane", f"{nat['conso_relle_kwh'].median():,.0f} kWh/an")
    n3.metric("Conso DPE médiane",    f"{nat['conso_dpe_kwh'].median():,.0f} kWh/an")
    n4.metric("Passoires (F+G)",      f"{pct_pass_nat:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)

    # Pie distribution DPE
    with c1:
        dpe_nat = nat["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER, fill_value=0)
        fig_pie = px.pie(
            values=dpe_nat.values, names=dpe_nat.index,
            color=dpe_nat.index, color_discrete_map=DPE_COLORS,
            title="Répartition nationale des classes DPE",
            hole=0.35,
        )
        fig_pie.update_traces(textposition="inside",
                               texttemplate="%{label}<br>%{percent:.1%}")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Boxplot conso réelle par classe
    with c2:
        box_df = nat.dropna(subset=["conso_relle_kwh", "etiquette_dpe"])
        p99 = box_df["conso_relle_kwh"].quantile(0.99)
        box_df = box_df[box_df["conso_relle_kwh"] < p99]
        fig_box = px.box(
            box_df, x="etiquette_dpe", y="conso_relle_kwh",
            color="etiquette_dpe", color_discrete_map=DPE_COLORS,
            category_orders={"etiquette_dpe": DPE_ORDER},
            title="Distribution de la consommation réelle par classe DPE",
            labels={"etiquette_dpe": "Classe DPE", "conso_relle_kwh": "kWh/an"},
            points="outliers",
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2)

    # Écart DPE estimé vs réel
    with c3:
        nat_mc = median_conso_by_dpe(nat).reset_index()
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="DPE estimé (méthode 3CL)",
            x=nat_mc["etiquette_dpe"].astype(str),
            y=nat_mc["conso_dpe_kwh"], marker_color="#4C9BE8",
        ))
        fig_cmp.add_trace(go.Bar(
            name="Réel Enedis",
            x=nat_mc["etiquette_dpe"].astype(str),
            y=nat_mc["conso_relle_kwh"], marker_color="#F28B30",
        ))
        fig_cmp.update_layout(
            barmode="group",
            title="DPE estimé vs Consommation réelle (médiane nationale)",
            xaxis_title="Classe DPE", yaxis_title="kWh/an",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption(
            "⚠️ La consommation réelle (Enedis) ne suit pas toujours la progression DPE. "
            "Les logements F/G peuvent sous-consommer par contrainte budgétaire (effet précarité)."
        )

    # Votre position vs commune vs France
    with c4:
        user_mc   = nat[nat["etiquette_dpe"] == dpe_actuel]["conso_relle_kwh"].median()
        comm_mc   = df[df["nom_commune_ban"] == commune]["conso_relle_kwh"].median()
        nat_mc_v  = nat["conso_relle_kwh"].median()

        cmp_data = pd.DataFrame({
            "Niveau": ["🇫🇷 France entière", f"🏘️ {commune}", f"🏠 Votre profil (DPE {dpe_actuel})"],
            "Conso médiane (kWh/an)": [nat_mc_v, comm_mc, user_mc],
        }).dropna()

        fig_pos = px.bar(
            cmp_data, x="Niveau", y="Conso médiane (kWh/an)",
            color="Niveau",
            color_discrete_sequence=["#1f77b4", "#2ca02c", "#d62728"],
            title="Votre profil vs commune vs France",
            text="Conso médiane (kWh/an)",
        )
        fig_pos.update_traces(texttemplate="%{text:,.0f} kWh", textposition="outside")
        fig_pos.update_layout(showlegend=False)
        st.plotly_chart(fig_pos, use_container_width=True)

    # Tableau national des gains
    st.markdown("### 💰 Gains nationaux estimés par amélioration de classe DPE")
    nat_mc_cls = median_conso_by_dpe(nat)["conso_relle_kwh"]
    gains_nat = []
    for i in range(len(DPE_ORDER) - 1):
        frm, to_ = DPE_ORDER[i + 1], DPE_ORDER[i]
        if frm in nat_mc_cls.index and to_ in nat_mc_cls.index:
            g = nat_mc_cls[frm] - nat_mc_cls[to_]
            if g > 0:
                gains_nat.append({
                    "Amélioration": f"{frm} → {to_}",
                    "Gain médian (kWh/an)": f"{g:,.0f}",
                    "Gain estimé (€/an)": f"{g * prix_kwh:,.0f} €",
                    "Gain estimé (€/10 ans)": f"{g * prix_kwh * 10 * (1 + taux_hausse * 5):,.0f} €",
                })
    if gains_nat:
        st.dataframe(pd.DataFrame(gains_nat), use_container_width=True, hide_index=True)
    else:
        st.info("Données insuffisantes pour calculer les gains.")

    # Conso par période de construction
    st.markdown("### 🏗️ Consommation réelle selon la période de construction")
    prd_df = (
        nat.dropna(subset=["conso_relle_kwh","periode_construction"])
        .groupby(["periode_construction","etiquette_dpe"], observed=True)["conso_relle_kwh"]
        .median().reset_index()
    )
    prd_order = ["Avant 1948","1948-1974","1975-1989","1990-2000",
                 "2001-2005","2006-2012","2013-2021","Après 2021","Inconnue"]
    fig_prd = px.bar(
        prd_df[prd_df["periode_construction"].isin(prd_order)],
        x="periode_construction", y="conso_relle_kwh",
        color="etiquette_dpe", color_discrete_map=DPE_COLORS,
        category_orders={"periode_construction": prd_order},
        barmode="group",
        title="Conso réelle médiane par période de construction et classe DPE",
        labels={"conso_relle_kwh": "kWh/an", "periode_construction": "Période"},
    )
    fig_prd.update_layout(height=380)
    st.plotly_chart(fig_prd, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PRÉDICTION 10 ANS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Prédiction des coûts électriques sur 10 ans")

    # Estimation de la conso actuelle de l'utilisateur
    user_data = df[
        (df["etiquette_dpe"] == dpe_actuel) &
        df["surface_habitable_logement"].between(surface * 0.7, surface * 1.4)
    ]["conso_relle_kwh"].dropna()

    if user_data.empty:
        user_data = df[df["etiquette_dpe"] == dpe_actuel]["conso_relle_kwh"].dropna()

    estimated_kwh = int(user_data.median()) if not user_data.empty else 4000

    col_params, col_chart = st.columns([1, 3])

    with col_params:
        st.markdown("#### ⚙️ Paramètres")
        custom_kwh = st.number_input(
            "Consommation annuelle estimée (kWh)",
            min_value=500, max_value=50_000,
            value=estimated_kwh, step=100,
            help="Calculée sur les logements similaires (DPE + surface). Modifiable.",
        )

        cout_actuel = custom_kwh * prix_kwh
        st.markdown(
            f"""
            **Votre profil estimé**
            - Classe DPE : {dpe_badge(dpe_actuel)}
            - Surface : **{surface} m²**
            - Type : {type_batiment}
            - Période : {periode_construction}
            - Conso : **{custom_kwh:,} kWh/an**
            - **Coût 2024 : {cout_actuel:,.0f} €/an**
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "**Hypothèses de rénovation**  \n"
            "- Classe B : rénovation en 2027  \n"
            "- Classe A : rénovation en 2028  \n"
            "Les consommations sont interpolées depuis les médianes réelles par classe."
        )

    with col_chart:
        years, scenarios = predict_costs(custom_kwh, prix_kwh, taux_hausse, dpe_actuel)

        colors_s = ["#d62728", "#FFB432", "#2ca02c"]
        fig_pred = go.Figure()

        for (name, costs), col_s in zip(scenarios.items(), colors_s):
            fig_pred.add_trace(go.Scatter(
                x=years, y=[round(c, 0) for c in costs],
                mode="lines+markers",
                name=name,
                line=dict(color=col_s, width=3),
                marker=dict(size=7),
                hovertemplate=f"<b>{name}</b><br>%{{x}} : %{{y:,.0f}} €<extra></extra>",
            ))

        # Zone de rénovation
        if len(scenarios) > 1:
            fig_pred.add_vrect(
                x0=2026.5, x1=2028.5,
                fillcolor="lightgreen", opacity=0.1,
                annotation_text="Fenêtre de\nrénovation", annotation_position="top left",
            )

        fig_pred.update_layout(
            title=f"Évolution du coût électrique — DPE {dpe_actuel}, {surface} m², {commune}",
            xaxis_title="Année",
            yaxis_title="Coût annuel (€)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            height=400,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # Tableau des coûts annuels
    st.markdown("### 📅 Coûts annuels détaillés")

    base_costs = list(scenarios.values())[0]
    rows = []
    for yr, base in zip(years, base_costs):
        row = {"Année": yr, list(scenarios.keys())[0]: f"{base:,.0f} €"}
        for name, costs in list(scenarios.items())[1:]:
            c = costs[years.index(yr)]
            econo = base - c
            row[name] = f"{c:,.0f} €"
            row[f"Économie vs sans rénov."] = f"💚 {econo:,.0f} €"
        rows.append(row)

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Économies cumulées
    st.markdown("### 🏆 Économies cumulées sur 10 ans")
    total_base = sum(base_costs)

    ecols = st.columns(len(scenarios))
    palette = ["#d62728", "#FFB432", "#2ca02c"]
    for i, (name, costs) in enumerate(scenarios.items()):
        total = sum(costs)
        saving = total_base - total
        with ecols[i]:
            st.metric(
                name,
                f"{total:,.0f} € cumulés",
                delta=f"−{saving:,.0f} € vs sans rénov." if saving > 0 else "Scénario de référence",
                delta_color="inverse" if saving > 0 else "off",
            )

    # Graphique économies cumulées
    if len(scenarios) > 1:
        cum_data = []
        for name, costs in list(scenarios.items())[1:]:
            for yr, (base, alt) in zip(years, zip(base_costs, costs)):
                cum_data.append({"Année": yr, "Scénario": name, "Économie cumulée (€)": base - alt})

        cum_df = pd.DataFrame(cum_data)
        cum_df["Économie cumulée (€)"] = (
            cum_df.groupby("Scénario")["Économie cumulée (€)"].cumsum()
        )
        fig_cum = px.area(
            cum_df, x="Année", y="Économie cumulée (€)", color="Scénario",
            color_discrete_sequence=["#FFB432", "#2ca02c"],
            title="Économies cumulées par rapport au scénario sans rénovation",
            markers=True,
        )
        fig_cum.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(fig_cum, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Sources : ADEME (DPE Logements existants, depuis juillet 2021) × Enedis (consommations résidentielles par adresse). "
    "Les consommations réelles sont des médianes par logement et ne représentent pas votre situation individuelle. "
    "Prédictions basées sur des hypothèses tarifaires et de rénovation simplifiées."
)
