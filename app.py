"""
Application DPE & Consommation Electrique
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
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "DPE_enedis_1.csv")
ELECTRICITY_PRICE = 0.2516
DEFAULT_INCREASE   = 3

DPE_COLORS = {"A":"#009900","B":"#55C400","C":"#AAFF00",
              "D":"#FFFF00","E":"#FFB432","F":"#FF5000","G":"#FF0000"}
DPE_TEXT   = {"A":"white","B":"white","C":"black","D":"black",
              "E":"white","F":"white","G":"white"}
DPE_ORDER  = ["A","B","C","D","E","F","G"]

USAGE_COLS = {"conso_chauffage_ef":"Chauffage","conso_ecs_ef":"Eau Chaude",
              "conso_eclairage_ef":"Eclairage","conso_refroidissement_ef":"Refroidissement",
              "conso_auxiliaires_ef":"Auxiliaires"}
USAGE_PAL  = ["#E45756","#4C78A8","#F28E2B","#76B7B2","#59A14F"]

DEPERD_COLS = {"deperditions_enveloppe":"Enveloppe totale",
               "deperditions_murs":"Murs",
               "deperditions_planchers_bas":"Planchers bas",
               "deperditions_planchers_hauts":"Planchers hauts"}

DPE_KWH_M2 = {"A":25.3,"B":41.4,"C":69.1,"D":96.6,"E":132.0,"F":172.0,"G":242.0}

st.set_page_config(page_title="DPE & Conso", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container{padding-top:1.2rem}
  .stMetric{background:#f5f7fa;border-radius:10px;padding:8px 12px}
  .dpe-label{display:inline-block;padding:3px 14px;border-radius:5px;
             font-weight:bold;font-size:1rem}
  hr.sep{border-top:2px solid #eee;margin:16px 0}
</style>""", unsafe_allow_html=True)

SEP = '<hr class="sep">'

# ─────────────────────────────────────────────────────────────
# TRANSFORMERS
# ─────────────────────────────────────────────────────────────
_l93_wgs = Transformer.from_crs("EPSG:2154","EPSG:4326", always_xy=True)
_wgs_l93 = Transformer.from_crs("EPSG:4326","EPSG:2154", always_xy=True)

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des donnees...")
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    mask = df["coordonnee_cartographique_x_ban"] > 1000
    lons, lats = _l93_wgs.transform(
        df.loc[mask,"coordonnee_cartographique_x_ban"].values,
        df.loc[mask,"coordonnee_cartographique_y_ban"].values)
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
def build_tree(df):
    v = df[df["_has_l93"]].copy()
    coords = v[["coordonnee_cartographique_x_ban","coordonnee_cartographique_y_ban"]].values
    return KDTree(coords), v.index.tolist()

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Fichier CSV introuvable : {DATA_PATH}")
    st.stop()

tree, tree_idx = build_tree(df)
communes_list  = sorted(df["nom_commune_ban"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def badge(label, size="1rem"):
    c = DPE_COLORS.get(str(label),"#888")
    t = DPE_TEXT.get(str(label),"white")
    return f'<span class="dpe-label" style="background:{c};color:{t};font-size:{size}"> {label} </span>'

@st.cache_data(show_spinner=False, ttl=86400)
def geocode(address):
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/",
                         params={"q":address,"limit":1}, timeout=6)
        d = r.json()
        if d.get("features"):
            f = d["features"][0]
            lon, lat = f["geometry"]["coordinates"]
            p = f["properties"]
            return lat, lon, p.get("label",""), p.get("score",0), p.get("city","")
    except Exception:
        pass
    return None, None, None, None, None

def nearest_from_l93(x, y, n=10):
    dists, pos = tree.query([x,y], k=min(n, len(tree_idx)))
    if n == 1: dists, pos = [dists],[pos]
    res = df.loc[[tree_idx[p] for p in pos]].copy()
    res["distance_m"] = [int(d) for d in dists]
    return res

def commune_l93(commune):
    s = df[df["nom_commune_ban"]==commune]
    return s["coordonnee_cartographique_x_ban"].mean(), s["coordonnee_cartographique_y_ban"].mean()

def med_by_dpe(data):
    return (data.groupby("etiquette_dpe", observed=True)
                .agg(conso_dpe_kwh=("conso_dpe_kwh","median"),
                     conso_relle_kwh=("conso_relle_kwh","median"),
                     n=("conso_relle_kwh","count"))
                .reindex(DPE_ORDER).dropna(subset=["conso_relle_kwh"]))

def predict_enedis(kwh, prix, taux, dpe):
    med = med_by_dpe(df)["conso_relle_kwh"]
    bg  = med.get("G", kwh) or kwh
    f   = lambda c: (med.get(c)/bg) if med.get(c) and bg else 1.0
    yrs = list(range(2025,2036)); s = {}
    s["Sans renovation"] = [kwh*prix*(1+taux)**i for i in range(len(yrs))]
    if dpe not in ("A","B"):
        nk = kwh * f("B")/max(f(dpe),0.01)
        s["Renovation → Classe B"] = [(kwh if i<2 else nk)*prix*(1+taux)**i for i in range(len(yrs))]
    if dpe != "A":
        nk = kwh * f("A")/max(f(dpe),0.01)
        s["Renovation → Classe A"] = [(kwh if i<3 else nk)*prix*(1+taux)**i for i in range(len(yrs))]
    return yrs, s

def predict_3cl(kwh, prix, taux, dpe):
    cm  = DPE_KWH_M2.get(dpe, 100.0)
    yrs = list(range(2025,2036)); s = {}
    s["Sans renovation"] = [kwh*prix*(1+taux)**i for i in range(len(yrs))]
    if dpe not in ("A","B"):
        nk = kwh * DPE_KWH_M2["B"]/cm
        s["Renovation → Classe B"] = [(kwh if i<2 else nk)*prix*(1+taux)**i for i in range(len(yrs))]
    if dpe != "A":
        nk = kwh * DPE_KWH_M2["A"]/cm
        s["Renovation → Classe A"] = [(kwh if i<3 else nk)*prix*(1+taux)**i for i in range(len(yrs))]
    return yrs, s

def prediction_chart(yrs, scenarios, title, key):
    fig = go.Figure()
    for (name,costs), col_s in zip(scenarios.items(), ["#d62728","#FFB432","#2ca02c"]):
        fig.add_trace(go.Scatter(
            x=yrs, y=[round(c) for c in costs], mode="lines+markers", name=name,
            line=dict(color=col_s,width=3), marker=dict(size=7),
            hovertemplate=f"<b>{name}</b><br>%{{x}} : %{{y:.0f}} euros<extra></extra>"))
    if len(scenarios)>1:
        fig.add_vrect(x0=2026.5, x1=2028.5, fillcolor="lightgreen", opacity=0.1,
                      annotation_text="Fenetre renovation", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Annee", yaxis_title="euros/an",
                      hovermode="x unified", legend=dict(orientation="h",y=1.1,x=0), height=360)
    st.plotly_chart(fig, use_container_width=True, key=key)

def cumul_chart(yrs, scenarios, title, key):
    base = list(scenarios.values())[0]
    if len(scenarios) < 2: return
    rows = []
    for name, costs in list(scenarios.items())[1:]:
        cum = 0
        for yr,(b,a) in zip(yrs,zip(base,costs)):
            cum += b-a
            rows.append({"Annee":yr,"Scenario":name,"Economie cumulee (euros)":round(cum)})
    fig = px.area(pd.DataFrame(rows), x="Annee", y="Economie cumulee (euros)",
                  color="Scenario", color_discrete_sequence=["#FFB432","#2ca02c"],
                  title=title, markers=True)
    fig.update_layout(height=260, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key=key)

def savings_metrics(scenarios, label=""):
    base_total = sum(list(scenarios.values())[0])
    for name, costs in list(scenarios.items())[1:]:
        saving = base_total - sum(costs)
        st.metric(name.replace("Renovation → ","→ "),
                  f"{sum(costs):.0f} euros",
                  delta=f"-{saving:.0f} euros" if saving>0 else "Reference",
                  delta_color="inverse" if saving>0 else "off")

def annual_table(yrs, scenarios, key_prefix):
    base = list(scenarios.values())[0]
    rows = []
    for yr, b in zip(yrs, base):
        row = {"Annee":yr, list(scenarios.keys())[0]: f"{b:.0f}"}
        for name, costs in list(scenarios.items())[1:]:
            c = costs[yrs.index(yr)]
            row[name] = f"{c:.0f}"
            row["Economie (euros)"] = f"{b-c:.0f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Logo_Enedis.svg/320px-Logo_Enedis.svg.png", width=130)
    except Exception:
        pass

    st.markdown("## 📍 Votre adresse")
    st.caption("Entrez votre adresse pour localiser les logements similaires les plus proches.")
    addr_input = st.text_input("Adresse complete",
                               placeholder="Ex: 12 rue Jean Jaures, Orleans",
                               help="Geocodage via l'API BAN (adresse.data.gouv.fr).")
    geo_btn = st.button("Geolocaliser & Analyser", type="primary", use_container_width=True)

    if geo_btn and addr_input.strip():
        with st.spinner("Geocodage..."):
            lat, lon, lbl, score, city = geocode(addr_input.strip())
        if lat:
            st.session_state.update({"geo_lat":lat,"geo_lon":lon,
                                     "geo_label":lbl,"geo_city":city,"geo_score":score})
        else:
            st.error("Adresse introuvable.")
            for k in ["geo_lat","geo_lon","geo_label","geo_city","geo_score"]:
                st.session_state.pop(k, None)

    if "geo_label" in st.session_state:
        st.success(st.session_state["geo_label"])
        st.caption(f"Confiance : {int(st.session_state.get('geo_score',0)*100)}%")

    st.markdown("---")
    st.markdown("## 🏠 Votre logement")
    st.caption("Ces informations personnalisent l'analyse et la prediction.")

    geo_city  = st.session_state.get("geo_city","")
    matched   = next((c for c in communes_list if geo_city.lower() in c.lower()), None) if geo_city else None
    def_idx   = communes_list.index(matched) if matched else (
                communes_list.index("Orleans") if "Orleans" in communes_list else 0)
    commune   = st.selectbox("Commune (onglets 2-3)", communes_list, index=def_idx,
                             help="Grandes metropoles absentes du dataset Enedis (min 10 logements/adresse).")
    if geo_city and not matched:
        st.warning(f"**{geo_city}** absent du dataset. Grandes villes (Nice, Paris, Lyon...) non disponibles.")

    type_bat  = st.selectbox("Type de batiment", ["appartement","maison","immeuble"])
    surface   = st.slider("Surface habitable (m2)", 15, 300, 65, step=5)
    dpe_cur   = st.select_slider("Classe DPE actuelle", options=DPE_ORDER, value="D")
    periode   = st.selectbox("Periode de construction",
                             ["Inconnue","Avant 1948","1948-1974","1975-1989",
                              "1990-2000","2001-2005","2006-2012","2013-2021","Apres 2021"])
    st.markdown(f"Classe : {badge(dpe_cur)}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💶 Hypotheses tarifaires")
    st.caption("Ces valeurs servent a convertir les kWh en euros dans les previsions.")
    prix_kwh    = st.number_input("Prix (euros/kWh)", 0.10, 1.0, ELECTRICITY_PRICE, 0.01)
    taux_hausse = st.slider("Hausse annuelle (%)", 0, 10, DEFAULT_INCREASE) / 100
    st.caption(f"Base : {len(df):,} logements · {len(communes_list)} communes")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("DPE & Consommation Electrique — Analyse & Prediction")
st.caption("Comparez votre consommation electrique avec vos voisins et estimez les economies d'une renovation energetique. Donnees ADEME (DPE) x Enedis.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Logements proches",
    "🏘️ Ma commune",
    "🇫🇷 Benchmark France",
    "📈 Prevision de mes couts",
    "🔬 Analyses approfondies",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LOGEMENTS PROCHES
# ══════════════════════════════════════════════════════════════
with tab1:
    has_geo = "geo_lat" in st.session_state
    if has_geo:
        u_lat, u_lon = st.session_state["geo_lat"], st.session_state["geo_lon"]
        u_x, u_y    = _wgs_l93.transform(u_lon, u_lat)
        ref_lbl     = st.session_state["geo_label"]
        nearest     = nearest_from_l93(u_x, u_y)
        st.info(f"📍 **{ref_lbl}**")
    else:
        cx, cy = commune_l93(commune)
        u_x, u_y = cx, cy
        u_lon, u_lat = _l93_wgs.transform(cx, cy)
        ref_lbl  = f"Centre de {commune}"
        nearest  = nearest_from_l93(cx, cy)
        st.warning(f"Saisissez votre adresse pour une analyse precise. Reference : **{commune}**.")

    if nearest.empty:
        st.error("Aucun logement trouve.")
        st.stop()

    # ── KPIs ──
    st.markdown("### 🏠 Vue d'ensemble des 10 logements les plus proches de votre adresse")
    pct_pass  = nearest["etiquette_dpe"].isin(["F","G"]).mean()*100
    dist_max  = nearest["distance_m"].max()
    med_reel  = nearest["conso_relle_kwh"].median()
    med_dpe   = nearest["conso_dpe_kwh"].median()
    ecart_pct = (med_reel-med_dpe)/med_dpe*100 if med_dpe else 0
    dpe_dom   = nearest["etiquette_dpe"].astype(str).mode()[0]

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Rayon couvert",        f"{dist_max:.0f} m", f"moy {nearest['distance_m'].mean():.0f} m")
    k2.metric("DPE dominant",         dpe_dom)
    k3.metric("Conso reelle mediane", f"{med_reel:.0f} kWh")
    k4.metric("Conso DPE mediane",    f"{med_dpe:.0f} kWh", f"{ecart_pct:+.1f}% vs estime")
    k5.metric("Passoires (F+G)",      f"{pct_pass:.0f}%")
    st.markdown(SEP, unsafe_allow_html=True)

    # ── CARTE ──
    st.markdown("### 🗺️ Carte — Cliquez sur un logement pour voir ses details")
    m = folium.Map(location=[(nearest["lat"].mean()+u_lat)/2,
                              (nearest["lon"].mean()+u_lon)/2],
                   zoom_start=14, tiles="CartoDB positron")
    folium.Marker([u_lat,u_lon], tooltip="Votre adresse",
                  icon=folium.Icon(color="blue",icon="home",prefix="fa"),
                  popup=folium.Popup(f"<b>{ref_lbl}</b>",max_width=260)).add_to(m)
    folium.Circle([u_lat,u_lon], radius=int(dist_max),
                  color="#1f77b4", fill=False, weight=1.5, dash_array="6").add_to(m)

    for rank, (_, row) in enumerate(nearest.iterrows(), 1):
        if pd.isna(row.get("lat")): continue
        dpe  = str(row["etiquette_dpe"])
        chex = DPE_COLORS.get(dpe,"#888")
        tw   = "white" if dpe in("A","B","E","F","G") else "black"
        surf = row.get("surface_habitable_logement",0)
        dkwh = row.get("conso_dpe_kwh",0) or 0
        rkwh = row.get("conso_relle_kwh",0) or 0
        ecrt = (rkwh-dkwh)/dkwh*100 if dkwh else 0
        cout = rkwh*prix_kwh
        tot  = row.get("conso_5_usages_ef",1) or 1
        ch_p = row.get("conso_chauffage_ef",0)/tot*100
        ec_p = row.get("conso_ecs_ef",0)/tot*100
        el_p = row.get("conso_eclairage_ef",0)/tot*100

        popup = f"""<div style='font-family:Arial;min-width:260px;font-size:12.5px;line-height:1.65'>
          <div style='background:{chex};color:{tw};padding:4px 10px;border-radius:4px;
               font-weight:bold;font-size:14px;margin-bottom:6px'>#{rank} — DPE {dpe}</div>
          <b>Adresse :</b> {row.get('adresse_ban','N/A')}<br>
          <b>Type :</b> {row.get('type_batiment','N/A')} | <b>Surface :</b> {surf:.0f} m2<br>
          <b>Periode :</b> {row.get('periode_construction','N/A')}<br>
          <hr style='margin:5px 0'>
          <b>Conso DPE estimee (3CL)</b><br>
          Total : <b>{dkwh:.0f} kWh/an</b> ({row.get('conso_5_usages_par_m2_ef',0):.0f} kWh/m2)<br>
          Chauffage : {row.get('conso_chauffage_ef',0):.0f} kWh ({ch_p:.0f}%) |
          ECS : {row.get('conso_ecs_ef',0):.0f} kWh ({ec_p:.0f}%)<br>
          Eclairage : {row.get('conso_eclairage_ef',0):.0f} kWh ({el_p:.0f}%)<br>
          <hr style='margin:5px 0'>
          <b>Conso reelle Enedis</b><br>
          <b>{rkwh:.0f} kWh/an</b>
          (<span style='color:{"#c0392b" if ecrt>0 else "#27ae60"}'>{ecrt:+.1f}%</span>)<br>
          Cout : <b>{cout:.0f} euros/an</b><br>
          <hr style='margin:5px 0'>
          Isolation murs : {row.get('qualite_isolation_murs','N/A')}<br>
          Deperditions enveloppe : {row.get('deperditions_enveloppe',0):.0f} W/K<br>
          <b>Distance :</b> {row.get('distance_m',0):,} m
        </div>"""

        folium.CircleMarker([row["lat"],row["lon"]], radius=18, color="white", weight=2,
                            fill=True, fill_color=chex, fill_opacity=0.88,
                            popup=folium.Popup(popup,max_width=300),
                            tooltip=f"#{rank} · DPE {dpe} · {surf:.0f}m2").add_to(m)
        folium.Marker([row["lat"],row["lon"]],
                      icon=folium.DivIcon(
                          html=f'<div style="color:{tw};font-weight:bold;font-size:12px;'
                               f'text-align:center;line-height:22px;width:22px">{rank}</div>',
                          icon_size=(22,22), icon_anchor=(11,11))).add_to(m)

    st_folium(m, height=460, use_container_width=True)
    st.markdown(SEP, unsafe_allow_html=True)

    # ── CONSOMMATIONS ──
    st.markdown("### ⚡ Consommations : estimees (DPE) vs mesurees (Enedis)")
    st.caption("Le DPE estime la consommation theorique. Enedis mesure la consommation reelle. L'ecart revele le comportement reel des occupants.")
    c1, c2 = st.columns(2)
    with c1:
        nb = nearest.sort_values("distance_m").copy()
        nb["lbl"] = [f"#{i+1}" for i in range(len(nb))]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="DPE estime", x=nb["lbl"], y=nb["conso_dpe_kwh"],
                             marker_color="#4C78A8", opacity=0.85,
                             hovertemplate="<b>%{x}</b><br>DPE : %{y:.0f} kWh<extra></extra>"))
        fig.add_trace(go.Bar(name="Reel Enedis", x=nb["lbl"], y=nb["conso_relle_kwh"],
                             marker_color="#F28E2B", opacity=0.85,
                             hovertemplate="<b>%{x}</b><br>Reel : %{y:.0f} kWh<extra></extra>"))
        fig.update_layout(barmode="group", title="DPE estime vs Reel par logement",
                          xaxis_title="Logement", yaxis_title="kWh/an",
                          legend=dict(orientation="h",y=-0.22), height=360)
        st.plotly_chart(fig, use_container_width=True, key="t1_cmp")

    with c2:
        nb["ecart_pct"] = (nb["conso_relle_kwh"]-nb["conso_dpe_kwh"]) \
                         / nb["conso_dpe_kwh"].replace(0,np.nan)*100
        fig2 = go.Figure(go.Bar(
            x=nb["lbl"], y=nb["ecart_pct"],
            marker_color=["#c0392b" if v>0 else "#27ae60" for v in nb["ecart_pct"]],
            text=[f"{v:+.1f}%" for v in nb["ecart_pct"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ecart : %{y:+.1f}%<extra></extra>"))
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        fig2.update_layout(title="Ecart reel vs DPE (%)",
                           xaxis_title="Logement", yaxis_title="%", height=360)
        st.caption("🔴 Rouge : la consommation reelle depasse l'estimation du DPE (souvent logements bien chauffes). 🟢 Vert : la conso reelle est inferieure (souvent logements sous-chauffes par contrainte budgetaire).")
        st.plotly_chart(fig2, use_container_width=True, key="t1_ecart")

    st.markdown(SEP, unsafe_allow_html=True)

    # ── USAGES ──
    st.markdown("### 🔍 A quoi sert l'energie dans ces logements ?")
    st.caption("Repartition estimee par usage : chauffage, eau chaude, eclairage, etc.")
    c3, c4 = st.columns(2)
    with c3:
        urows = []
        for i,(_, row) in enumerate(nearest.iterrows(), 1):
            for col,lbl in USAGE_COLS.items():
                v = row.get(col,0)
                if pd.notna(v) and v>0:
                    urows.append({"Logement":f"#{i}","Usage":lbl,"kWh/an":round(v)})
        if urows:
            fig3 = px.bar(pd.DataFrame(urows), x="Logement", y="kWh/an", color="Usage",
                          color_discrete_sequence=USAGE_PAL,
                          title="Decomposition par usage", barmode="stack")
            fig3.update_layout(height=350, legend=dict(orientation="h",y=-0.3))
            st.plotly_chart(fig3, use_container_width=True, key="t1_stack")

    with c4:
        avg_u = {lbl: nearest[col].mean() for col,lbl in USAGE_COLS.items()
                 if col in nearest.columns}
        avg_u = {k:v for k,v in avg_u.items() if pd.notna(v) and v>0}
        if avg_u:
            fig4 = px.pie(values=list(avg_u.values()), names=list(avg_u.keys()),
                          color_discrete_sequence=USAGE_PAL,
                          title="Repartition moyenne des usages", hole=0.38)
            fig4.update_traces(texttemplate="%{label}<br>%{percent:.0%}")
            fig4.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True, key="t1_pie")

    st.markdown(SEP, unsafe_allow_html=True)

    # ── THERMIQUE ──
    st.markdown("### 🏚️ Qualite de l'isolation — Ou part la chaleur ?")
    st.caption("Les deperditions (W/K) mesurent la quantite de chaleur perdue. Plus c'est eleve, plus le logement est mal isole et consomme beaucoup.")
    c5, c6 = st.columns(2)
    with c5:
        drows = []
        for i,(_, row) in enumerate(nearest.iterrows(), 1):
            for col,lbl in DEPERD_COLS.items():
                v = row.get(col,0)
                if pd.notna(v) and v>0:
                    drows.append({"Logement":f"#{i}","Composant":lbl,"W/K":round(v,1)})
        if drows:
            ddf = pd.DataFrame(drows)
            fig5 = px.bar(ddf[ddf["Composant"]!="Enveloppe totale"],
                          x="Logement", y="W/K", color="Composant",
                          title="Deperditions par composant (W/K)", barmode="group",
                          color_discrete_sequence=["#e15759","#f28e2b","#76b7b2"])
            fig5.update_layout(height=330, legend=dict(orientation="h",y=-0.3))
            st.plotly_chart(fig5, use_container_width=True, key="t1_dep")

    with c6:
        iso1 = nearest["qualite_isolation_enveloppe"].dropna().value_counts()
        iso2 = nearest["qualite_isolation_murs"].dropna().value_counts()
        if not iso1.empty or not iso2.empty:
            fig6 = go.Figure()
            if not iso1.empty:
                fig6.add_trace(go.Bar(name="Enveloppe", x=iso1.index, y=iso1.values,
                                      marker_color="#4C78A8",
                                      text=iso1.values, textposition="outside"))
            if not iso2.empty:
                fig6.add_trace(go.Bar(name="Murs", x=iso2.index, y=iso2.values,
                                      marker_color="#F28E2B", opacity=0.8,
                                      text=iso2.values, textposition="outside"))
            fig6.update_layout(barmode="group", title="Qualite d'isolation",
                               yaxis_title="Nb logements", height=330,
                               legend=dict(orientation="h",y=-0.25))
            st.plotly_chart(fig6, use_container_width=True, key="t1_iso")

    st.markdown(SEP, unsafe_allow_html=True)

    # ── TABLEAU COMPLET ──
    st.markdown("### 📋 Tableau comparatif complet des 10 logements")
    rcols = {"distance_m":"Dist.(m)","etiquette_dpe":"DPE","score_dpe":"Score",
             "type_batiment":"Type","surface_habitable_logement":"Surface(m2)",
             "periode_construction":"Periode","qualite_isolation_enveloppe":"Iso.env",
             "qualite_isolation_murs":"Iso.murs","zone_climatique":"Zone",
             "conso_5_usages_par_m2_ef":"kWh/m2","conso_dpe_kwh":"kWh DPE",
             "conso_relle_kwh":"kWh Reel","conso_chauffage_ef":"Chauf.",
             "conso_ecs_ef":"ECS","conso_eclairage_ef":"Eclair.",
             "deperditions_enveloppe":"Dep.env(W/K)","deperditions_murs":"Dep.murs(W/K)",
             "Nombre de logements":"Nb.log"}
    avail = {k:v for k,v in rcols.items() if k in nearest.columns}
    rec   = nearest[list(avail.keys())].copy()
    rec.columns = list(avail.values())
    rec.index   = [f"#{i+1}" for i in range(len(rec))]
    if "kWh Reel" in rec.columns:
        rec.insert(rec.columns.get_loc("kWh Reel")+1,
                   "Cout/an(euros)", (rec["kWh Reel"]*prix_kwh).round(0))
    st.dataframe(rec, use_container_width=True, height=350)
    st.download_button("Telecharger CSV", rec.to_csv(encoding="utf-8-sig"),
                       "dpe_10_proches.csv", "text/csv")

    st.markdown(SEP, unsafe_allow_html=True)

    # ── FICHES INDIVIDUELLES ──
    st.markdown("### 📁 Fiches detaillees — Cliquez sur un logement pour tout savoir")
    st.caption("Chaque fiche contient : caracteristiques du batiment, detail des consommations par usage, qualite d'isolation et score DPE.")
    for rank, (_, row) in enumerate(nearest.iterrows(), 1):
        dpe  = str(row.get("etiquette_dpe","?"))
        surf = row.get("surface_habitable_logement",0)
        rkwh = row.get("conso_relle_kwh",0) or 0
        dkwh = row.get("conso_dpe_kwh",0) or 0
        dist = row.get("distance_m",0)
        ecrt = (rkwh-dkwh)/dkwh*100 if dkwh else 0
        cout = rkwh*prix_kwh
        with st.expander(f"#{rank} — {row.get('adresse_ban','N/A')}  |  "
                         f"DPE {dpe}  |  {surf:.0f} m2  |  {dist:,} m  |  {rkwh:.0f} kWh/an"):
            fa, fb, fc = st.columns(3)
            with fa:
                st.markdown("**Caracteristiques**")
                st.markdown(f"""| Champ | Valeur |
|---|---|
| Type | {row.get('type_batiment','N/A')} |
| Surface | {surf:.0f} m2 |
| Periode | {row.get('periode_construction','N/A')} |
| Zone clim. | {row.get('zone_climatique','N/A')} |
| Nb logements | {row.get('Nombre de logements','N/A')} |
| Score DPE | {row.get('score_dpe','N/A')} |
| Energie chauf. | {row.get('type_energie_principale_chauffage','N/A')} |
| Energie ECS | {row.get('type_energie_principale_ecs','N/A')} |""")
            with fb:
                st.markdown("**Consommations**")
                st.markdown(f"""| Usage | kWh/an |  % |
|---|---|---|
| Chauffage | {row.get('conso_chauffage_ef',0):.0f} | {row.get('conso_chauffage_ef',0)/max(dkwh,1)*100:.0f}% |
| ECS | {row.get('conso_ecs_ef',0):.0f} | {row.get('conso_ecs_ef',0)/max(dkwh,1)*100:.0f}% |
| Eclairage | {row.get('conso_eclairage_ef',0):.0f} | {row.get('conso_eclairage_ef',0)/max(dkwh,1)*100:.0f}% |
| Refroid. | {row.get('conso_refroidissement_ef',0):.0f} | — |
| Auxiliaires | {row.get('conso_auxiliaires_ef',0):.0f} | — |
| **Total DPE** | **{dkwh:.0f}** | |
| **Reel Enedis** | **{rkwh:.0f}** | |
| Ecart | {ecrt:+.1f}% | |
| **Cout/an** | **{cout:.0f} euros** | |""")
                use_vals = {lbl:row.get(col,0) for col,lbl in USAGE_COLS.items()}
                use_vals = {k:v for k,v in use_vals.items() if pd.notna(v) and v>0}
                if use_vals:
                    fp = px.pie(values=list(use_vals.values()), names=list(use_vals.keys()),
                                color_discrete_sequence=USAGE_PAL, hole=0.4, height=200)
                    fp.update_traces(texttemplate="%{percent:.0%}", textposition="inside")
                    fp.update_layout(margin=dict(t=10,b=5,l=0,r=0), showlegend=True,
                                     legend=dict(font_size=9))
                    st.plotly_chart(fp, use_container_width=True, key=f"pie_{rank}")
            with fc:
                st.markdown("**Isolation & Thermique**")
                st.markdown(f"""| Composant | Valeur |
|---|---|
| Iso. murs | {row.get('qualite_isolation_murs','N/A')} |
| Iso. enveloppe | {row.get('qualite_isolation_enveloppe','N/A')} |
| Dep. enveloppe | {row.get('deperditions_enveloppe',0):.0f} W/K |
| Dep. murs | {row.get('deperditions_murs',0):.0f} W/K |
| Dep. pl. bas | {row.get('deperditions_planchers_bas',0):.0f} W/K |
| Dep. pl. hauts | {row.get('deperditions_planchers_hauts',0):.0f} W/K |""")
                score = float(row.get("score_dpe",0) or 0)
                fg = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    title={"text":f"Score DPE · Classe {dpe}","font":{"size":12}},
                    gauge={"axis":{"range":[0,700]},
                           "bar":{"color":DPE_COLORS.get(dpe,"#888")},
                           "steps":[{"range":[0,70],"color":"#009900"},
                                    {"range":[70,110],"color":"#55C400"},
                                    {"range":[110,180],"color":"#AAFF00"},
                                    {"range":[180,250],"color":"#FFFF00"},
                                    {"range":[250,330],"color":"#FFB432"},
                                    {"range":[330,420],"color":"#FF5000"},
                                    {"range":[420,700],"color":"#FF0000"}]}))
                fg.update_layout(height=200, margin=dict(t=35,b=5,l=10,r=10))
                st.plotly_chart(fg, use_container_width=True, key=f"gauge_{rank}")

    st.markdown(SEP, unsafe_allow_html=True)

    # ── POSITIONNEMENT ──
    st.markdown("### 📊 Ou se situent ces logements par rapport a la France ?")
    st.caption("Comparez la consommation de vos 10 logements proches avec la moyenne de votre commune et la moyenne nationale.")
    c9, c10 = st.columns(2)
    with c9:
        nat_m  = df["conso_relle_kwh"].median()
        com_m  = df[df["nom_commune_ban"]==commune]["conso_relle_kwh"].median()
        near_m = nearest["conso_relle_kwh"].median()
        cmp = pd.DataFrame({"Niveau":["France",commune,"10 proches"],
                             "kWh/an":[nat_m,com_m,near_m]}).dropna()
        fig7 = px.bar(cmp, x="Niveau", y="kWh/an", color="Niveau",
                      color_discrete_sequence=["#1f77b4","#2ca02c","#d62728"],
                      title="Conso reelle mediane — comparaison", text="kWh/an")
        fig7.update_traces(texttemplate="%{text:.0f} kWh", textposition="outside")
        fig7.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig7, use_container_width=True, key="t1_pos")

    with c10:
        nat_s = df["conso_relle_kwh"].dropna().sample(min(5000,len(df)), random_state=42)
        fig8  = go.Figure()
        fig8.add_trace(go.Histogram(x=nat_s, name="France", opacity=0.4,
                                    marker_color="#1f77b4",
                                    xbins=dict(size=500), histnorm="percent"))
        for v in nearest["conso_relle_kwh"].dropna():
            fig8.add_vline(x=v, line_width=1.5, line_color="#d62728", opacity=0.6)
        fig8.add_vline(x=near_m, line_width=2.5, line_color="#d62728",
                       annotation_text="Mediane proches", annotation_position="top right")
        fig8.update_layout(title="Distribution nationale vs vos 10 proches",
                           xaxis_title="kWh/an", yaxis_title="%", height=320,
                           xaxis_range=[0, min(nat_s.quantile(0.97)*1.1, 14000)])
        st.plotly_chart(fig8, use_container_width=True, key="t1_hist")


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE COMMUNE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Analyse de la commune : {commune}")
    com_df = df[df["nom_commune_ban"]==commune].copy()
    if com_df.empty:
        st.warning(f"Aucune donnee pour {commune}.")
    else:
        pct_p = com_df["etiquette_dpe"].isin(["F","G"]).mean()*100
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Logements", f"{len(com_df):,}")
        m2.metric("Conso reelle mediane", f"{com_df['conso_relle_kwh'].median():.0f} kWh/an")
        m3.metric("Conso DPE mediane",    f"{com_df['conso_dpe_kwh'].median():.0f} kWh/an")
        m4.metric("Passoires (F+G)",      f"{pct_p:.1f}%")
        st.markdown("---")

        c1,c2 = st.columns(2)
        with c1:
            cnt = com_df["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER,fill_value=0)
            fig = px.bar(x=cnt.index, y=cnt.values, color=cnt.index,
                         color_discrete_map=DPE_COLORS,
                         title=f"Distribution DPE — {commune}",
                         labels={"x":"Classe","y":"Nb logements"}, text=cnt.values)
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True, key="t2_dist")
        with c2:
            mc = med_by_dpe(com_df).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(name="DPE estime", x=mc["etiquette_dpe"],
                                 y=mc["conso_dpe_kwh"], marker_color="#4C9BE8", opacity=0.8))
            fig.add_trace(go.Bar(name="Reel Enedis", x=mc["etiquette_dpe"],
                                 y=mc["conso_relle_kwh"], marker_color="#F28B30", opacity=0.8))
            fig.update_layout(barmode="group", title=f"Conso mediane par classe — {commune}",
                              xaxis_title="DPE", yaxis_title="kWh/an",
                              height=340, legend=dict(orientation="h",y=-0.22))
            st.plotly_chart(fig, use_container_width=True, key="t2_grp")

        c3,c4 = st.columns(2)
        with c3:
            sc = com_df.dropna(subset=["conso_dpe_kwh","conso_relle_kwh"])
            sc = sc[sc["conso_relle_kwh"]<sc["conso_relle_kwh"].quantile(0.99)]
            if not sc.empty:
                fig = px.scatter(sc, x="conso_dpe_kwh", y="conso_relle_kwh",
                                 color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                                 title=f"DPE estime vs Reel — {commune}",
                                 labels={"conso_dpe_kwh":"DPE (kWh)","conso_relle_kwh":"Reel (kWh)",
                                         "etiquette_dpe":"Classe"},
                                 opacity=0.55, height=340)
                mx = max(sc["conso_dpe_kwh"].max(), sc["conso_relle_kwh"].max())
                fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
                                         line=dict(dash="dash",color="grey"),
                                         name="Estimation parfaite"))
                st.plotly_chart(fig, use_container_width=True, key="t2_scatter")
        with c4:
            mc2 = med_by_dpe(com_df)["conso_relle_kwh"]
            gains = []
            for i in range(len(DPE_ORDER)-1):
                fr, to = DPE_ORDER[i+1], DPE_ORDER[i]
                if fr in mc2.index and to in mc2.index and mc2[fr]-mc2[to]>0:
                    g = mc2[fr]-mc2[to]
                    gains.append({"Passage":f"{fr}→{to}","kWh/an":round(g),
                                  "euros/an":round(g*prix_kwh)})
            if gains:
                gdf = pd.DataFrame(gains)
                fig = px.bar(gdf, x="Passage", y="kWh/an",
                             title=f"Gains par amelioration DPE — {commune}",
                             color="kWh/an",color_continuous_scale=["#FF5000","#55C400"],
                             text="euros/an", height=340)
                fig.update_traces(texttemplate="%{text} euros/an", textposition="outside")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True, key="t2_gains")

        st.markdown("### 📋 Recapitulatif par classe DPE")
        st.caption("Comparaison de la consommation estimee (DPE) et reelle (Enedis) pour chaque classe, ainsi que le cout annuel moyen.")
        tbl = med_by_dpe(com_df).reset_index()
        tbl["Cout median (euros/an)"] = (tbl["conso_relle_kwh"]*prix_kwh).round(0).astype("Int64")
        tbl.columns = ["Classe DPE","Conso DPE (kWh)","Conso reelle (kWh)","Nb log","Cout (euros/an)"]
        for c_ in ["Conso DPE (kWh)","Conso reelle (kWh)"]:
            tbl[c_] = tbl[c_].round(0).astype("Int64")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE FRANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analyse nationale — Benchmark toutes communes")
    ppt = df["etiquette_dpe"].isin(["F","G"]).mean()*100
    n1,n2,n3,n4 = st.columns(4)
    n1.metric("Total logements",      f"{len(df):,}")
    n2.metric("Conso reelle mediane", f"{df['conso_relle_kwh'].median():.0f} kWh/an")
    n3.metric("Conso DPE mediane",    f"{df['conso_dpe_kwh'].median():.0f} kWh/an")
    n4.metric("Passoires (F+G)",      f"{ppt:.1f}%")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        cnt = df["etiquette_dpe"].astype(str).value_counts().reindex(DPE_ORDER,fill_value=0)
        fig = px.pie(values=cnt.values, names=cnt.index, color=cnt.index,
                     color_discrete_map=DPE_COLORS,
                     title="Repartition nationale des classes DPE", hole=0.35)
        fig.update_traces(texttemplate="%{label}<br>%{percent:.1%}", textposition="inside")
        st.plotly_chart(fig, use_container_width=True, key="t3_pie")
    with c2:
        bdf = df.dropna(subset=["conso_relle_kwh","etiquette_dpe"])
        bdf = bdf[bdf["conso_relle_kwh"]<bdf["conso_relle_kwh"].quantile(0.99)]
        fig = px.box(bdf, x="etiquette_dpe", y="conso_relle_kwh",
                     color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                     category_orders={"etiquette_dpe":DPE_ORDER},
                     title="Distribution conso reelle par classe",
                     labels={"etiquette_dpe":"Classe","conso_relle_kwh":"kWh/an"},
                     points="outliers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="t3_box")

    c3,c4 = st.columns(2)
    with c3:
        nmc = med_by_dpe(df).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="DPE estime", x=nmc["etiquette_dpe"].astype(str),
                             y=nmc["conso_dpe_kwh"], marker_color="#4C9BE8"))
        fig.add_trace(go.Bar(name="Reel Enedis", x=nmc["etiquette_dpe"].astype(str),
                             y=nmc["conso_relle_kwh"], marker_color="#F28E2B"))
        fig.update_layout(barmode="group", title="DPE estime vs Reel — mediane nationale",
                          xaxis_title="Classe", yaxis_title="kWh/an",
                          legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig, use_container_width=True, key="t3_grp")
        st.caption("⚠️ Attention : les logements F et G consomment parfois MOINS que les D et E en reel. Ce phenomene, appele effet precarite, s'explique par le fait que les menages aux revenus modestes reduisent leur chauffage pour limiter leurs factures.")
    with c4:
        dpe_m  = df[df["etiquette_dpe"]==dpe_cur]["conso_relle_kwh"].median()
        com_m2 = df[df["nom_commune_ban"]==commune]["conso_relle_kwh"].median()
        near_m2 = nearest["conso_relle_kwh"].median() if not nearest.empty else np.nan
        nat_m2  = df["conso_relle_kwh"].median()
        cmp = pd.DataFrame({"Niveau":["France",commune,"10 proches",f"DPE {dpe_cur}"],
                             "kWh/an":[nat_m2,com_m2,near_m2,dpe_m]}).dropna()
        fig = px.bar(cmp, x="Niveau", y="kWh/an", color="Niveau",
                     color_discrete_sequence=["#1f77b4","#2ca02c","#9467bd","#d62728"],
                     title="Votre position relative", text="kWh/an")
        fig.update_traces(texttemplate="%{text:.0f} kWh", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="t3_pos")

    st.markdown("### 💰 Combien peut-on economiser en ameliorant son DPE ?")
    st.caption("Ces gains sont calcules sur les consommations reelles mesurees par Enedis, en prenant la difference de consommation mediane entre chaque classe.")
    nat_cls = med_by_dpe(df)["conso_relle_kwh"]
    gnat = []
    for i in range(len(DPE_ORDER)-1):
        fr, to = DPE_ORDER[i+1], DPE_ORDER[i]
        if fr in nat_cls.index and to in nat_cls.index and nat_cls[fr]-nat_cls[to]>0:
            g = nat_cls[fr]-nat_cls[to]
            gnat.append({"Amelioration":f"{fr}→{to}","Gain (kWh/an)":f"{g:.0f}",
                         "Gain (euros/an)":f"{g*prix_kwh:.0f}",
                         "Gain 10 ans":f"{g*prix_kwh*10*(1+taux_hausse*5):.0f}"})
    if gnat:
        st.dataframe(pd.DataFrame(gnat), use_container_width=True, hide_index=True)

    prd = (df.dropna(subset=["conso_relle_kwh","periode_construction"])
             .groupby(["periode_construction","etiquette_dpe"],observed=True)
             ["conso_relle_kwh"].median().reset_index())
    p_order = ["Avant 1948","1948-1974","1975-1989","1990-2000",
               "2001-2005","2006-2012","2013-2021","Apres 2021","Inconnue"]
    fig = px.bar(prd[prd["periode_construction"].isin(p_order)],
                 x="periode_construction", y="conso_relle_kwh",
                 color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                 category_orders={"periode_construction":p_order},
                 barmode="group", title="Conso reelle mediane par periode x classe DPE",
                 labels={"conso_relle_kwh":"kWh/an","periode_construction":"Periode"})
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True, key="t3_prd")


# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION 10 ANS (ENEDIS + 3CL)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Prediction des couts electriques sur 10 ans")

    ud = df[(df["etiquette_dpe"]==dpe_cur) &
            df["surface_habitable_logement"].between(surface*0.7,surface*1.4)
            ]["conso_relle_kwh"].dropna()
    if ud.empty:
        ud = df[df["etiquette_dpe"]==dpe_cur]["conso_relle_kwh"].dropna()
    est_kwh = int(ud.median()) if not ud.empty else 4000

    # ── Parametres (ligne du haut) ──
    pa1, pa2, pa3 = st.columns([1, 1, 2])
    with pa1:
        st.markdown("#### Parametres")
        kwh = st.number_input("Conso annuelle actuelle (kWh)",
                              min_value=500, max_value=50000,
                              value=est_kwh, step=100)
        st.markdown(
            f"**Profil**<br>Classe DPE : {badge(dpe_cur)}<br>"
            f"Surface : **{surface} m2**<br>"
            f"Conso : **{kwh:,} kWh/an**<br>"
            f"**Cout actuel : {kwh*prix_kwh:.0f} euros/an**",
            unsafe_allow_html=True)
    with pa2:
        st.markdown("#### 🎯 Economies si renovation")
        cm  = DPE_KWH_M2.get(dpe_cur,100.0)
        r_b = DPE_KWH_M2["B"]/cm
        r_a = DPE_KWH_M2["A"]/cm
        if dpe_cur not in ("A","B"):
            st.metric("Apres renovation → B", f"{kwh*r_b:.0f} kWh/an",
                      delta=f"-{(1-r_b)*100:.0f}% de conso", delta_color="inverse")
        if dpe_cur != "A":
            st.metric("Apres renovation → A", f"{kwh*r_a:.0f} kWh/an",
                      delta=f"-{(1-r_a)*100:.0f}% de conso", delta_color="inverse")
    with pa3:
        st.markdown("#### ❓ Comment sont calculees les previsions ?")
        st.info("**Modele Enedis** — Consommations reelles mesurées par les compteurs Linky. "
                "Integre l'effet precarite energetique. Reflète ce qui se passe vraiment.")
        st.success("**Modele 3CL** — Standard DPE officiel (kWh/m2/an). "
                   "Mesure le potentiel physique maximal de la renovation thermique.")

    st.markdown("---")

    # ── Calcul des deux modeles ──
    yrs_e, sc_e = predict_enedis(kwh, prix_kwh, taux_hausse, dpe_cur)
    yrs_3, sc_3 = predict_3cl(kwh, prix_kwh, taux_hausse, dpe_cur)

    # ── MODELE ENEDIS ──
    st.markdown("### 📊 Modele 1 : base sur ce que consomment vraiment les gens")
    st.caption("Ce modele calcule vos economies futures d'apres les consommations reellement mesurees par les compteurs Linky dans des logements similaires au votre.")
    ce1, ce2 = st.columns([3,2])
    with ce1:
        prediction_chart(yrs_e, sc_e, f"Enedis — DPE {dpe_cur}, {surface} m2", "t4_pred_e")
    with ce2:
        st.markdown("**Economies cumulees (11 ans)**")
        savings_metrics(sc_e)
    with st.expander("📅 Tableau des couts annuels — Modele Enedis", expanded=True):
        annual_table(yrs_e, sc_e, "enedis")
    cumul_chart(yrs_e, sc_e, "Economies cumulees — Modele Enedis", "t4_cum_e")

    st.markdown("---")

    # ── MODELE 3CL ──
    st.markdown("### 🔬 Modele 2 : base sur le potentiel technique de la renovation")
    st.caption("Ce modele calcule les economies que permettrait physiquement la renovation : si vous passez de DPE D a DPE A, votre logement consomme theoriquement 74% de moins. C'est le potentiel maximal atteignable.")
    c3a, c3b = st.columns([3,2])
    with c3a:
        prediction_chart(yrs_3, sc_3, f"3CL — DPE {dpe_cur}, {surface} m2", "t4_pred_3")
    with c3b:
        st.markdown("**Economies cumulees (11 ans)**")
        savings_metrics(sc_3)
    with st.expander("📅 Tableau des couts annuels — Modele 3CL", expanded=True):
        annual_table(yrs_3, sc_3, "3cl")
    cumul_chart(yrs_3, sc_3, "Economies cumulees — Modele 3CL", "t4_cum_3")

    st.markdown("---")

    # ── COMPARAISON ENEDIS vs 3CL ──
    st.markdown("### ⚖️ Comparaison des deux modeles — Economies totales sur 11 ans")
    st.caption("La verite se situe entre les deux modeles : le modele Enedis sous-evalue parfois les economies (effet precarite), le modele 3CL les sur-evalue (comportement ideal).")
    base_e_tot = sum(list(sc_e.values())[0])
    base_3_tot = sum(list(sc_3.values())[0])
    cmp_rows = []
    for (ne, ce), (n3, c3) in zip(list(sc_e.items())[1:], list(sc_3.items())[1:]):
        eco_e = base_e_tot - sum(ce)
        eco_3 = base_3_tot - sum(c3)
        lbl   = ne.replace("Renovation → ","")
        cmp_rows.append({"Scenario":lbl,
                         "Enedis — reel (euros)":round(eco_e),
                         "3CL — potentiel max (euros)":round(eco_3),
                         "Ecart (euros)":round(eco_3-eco_e)})
    if cmp_rows:
        cdf = pd.DataFrame(cmp_rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        fig = go.Figure()
        for col_name, col_color in [("Enedis — reel (euros)","#F28E2B"),
                                     ("3CL — potentiel max (euros)","#2ca02c")]:
            fig.add_trace(go.Bar(name=col_name, x=cdf["Scenario"], y=cdf[col_name],
                                 marker_color=col_color,
                                 text=cdf[col_name].apply(lambda v: f"{v:.0f} euros"),
                                 textposition="outside"))
        fig.update_layout(barmode="group",
                          title="Economies totales : Modele Enedis vs Modele 3CL",
                          yaxis_title="Economies (euros)",
                          legend=dict(orientation="h",y=-0.25), height=380)
        st.plotly_chart(fig, use_container_width=True, key="t4_cmp_both")


        st.markdown("**📖 Reference : consommation standard par classe DPE (methode officielle 3CL)**")
        st.caption("Par exemple, un logement classe A consomme en moyenne 25 kWh par m2 et par an, contre 242 kWh pour un logement classe G — soit 10 fois plus !")
        cm_cur = DPE_KWH_M2.get(dpe_cur, 100.0)
        ref = pd.DataFrame([{"Classe":cls,
                              "kWh/m2/an (3CL)":v,
                              "Reduction vs G":f"{(1-v/242)*100:.0f}%",
                              "Ratio vs classe actuelle":f"{v/cm_cur:.2f}"}
                             for cls,v in DPE_KWH_M2.items()])
        st.dataframe(ref, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# TAB 5 — ANALYSES APPROFONDIES
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Analyses approfondies — Reponses aux questions du projet")
    st.caption(
        "Ces analyses repondent directement aux deux questions centrales du projet Enedis x ADEME : "
        "les DPE estiment-ils bien la realite ? Et combien economise-t-on vraiment en changeant de classe ?"
    )

    DPE_ORDER_5 = ["A","B","C","D","E","F","G"]

    # ══════════════════════
    # SECTION 1 — VARIABILITE
    # ══════════════════════
    st.markdown("---")
    st.markdown("### 1️⃣  Les DPE reflètent-ils bien la realite ? Et quelle est la variabilite ?")
    st.caption(
        "Le DPE utilise une methode de calcul theorique (3CL) qui ne tient pas compte du comportement "
        "reel des occupants (temperature souhaitee, taux d'occupation, habitudes). "
        "On mesure ici l'ecart entre ce que predit le DPE et ce que mesure Enedis."
    )

    # Stats variabilite
    stats_var = df.groupby("etiquette_dpe", observed=True).agg(
        mediane_reelle=("conso_relle_kwh","median"),
        ecart_type=("conso_relle_kwh","std"),
        mediane_dpe=("conso_dpe_kwh","median"),
        count=("conso_relle_kwh","count")
    ).reindex(DPE_ORDER_5).dropna()
    stats_var["ecart_dpe_reel"] = stats_var["mediane_reelle"] - stats_var["mediane_dpe"]
    stats_var["ecart_pct"] = stats_var["ecart_dpe_reel"] / stats_var["mediane_dpe"] * 100
    stats_var["coeff_variation"] = stats_var["ecart_type"] / stats_var["mediane_reelle"] * 100

    v1, v2 = st.columns(2)
    with v1:
        # Boxplot complet
        bdf5 = df.dropna(subset=["conso_relle_kwh","etiquette_dpe"])
        bdf5 = bdf5[bdf5["conso_relle_kwh"] < bdf5["conso_relle_kwh"].quantile(0.99)]
        fig_box5 = go.Figure()
        for dpe in DPE_ORDER_5:
            vals = bdf5[bdf5["etiquette_dpe"]==dpe]["conso_relle_kwh"].dropna()
            if len(vals) > 0:
                fig_box5.add_trace(go.Box(
                    y=vals, name=dpe, marker_color=DPE_COLORS.get(dpe,"#888"),
                    boxmean=True, showlegend=False,
                ))
        fig_box5.update_layout(
            title="Distribution de la conso reelle par classe DPE",
            yaxis_title="kWh/an",
            xaxis_title="Classe DPE",
            height=400,
        )
        st.plotly_chart(fig_box5, use_container_width=True, key="t5_box")
        st.caption(
            "Chaque boite montre la dispersion des consommations reelles. "
            "La ligne centrale = mediane. La croix = moyenne. "
            "La largeur de la boite montre la variabilite due aux comportements."
        )

    with v2:
        # Ecart-type par classe (variabilite comportementale)
        fig_std = go.Figure()
        fig_std.add_trace(go.Bar(
            x=stats_var.index.astype(str),
            y=stats_var["ecart_type"].round(0),
            marker_color=[DPE_COLORS.get(d,"#888") for d in stats_var.index],
            text=stats_var["ecart_type"].round(0).astype(int),
            textposition="outside",
            hovertemplate="Classe %{x}<br>Ecart-type : %{y:.0f} kWh/an<extra></extra>",
        ))
        fig_std.update_layout(
            title="Variabilite des comportements par classe DPE",
            yaxis_title="Ecart-type (kWh/an)",
            xaxis_title="Classe DPE",
            height=400,
        )
        st.plotly_chart(fig_std, use_container_width=True, key="t5_std")
        st.caption(
            "L'ecart-type mesure la dispersion des consommations autour de la mediane. "
            "Un ecart-type de 1 885 kWh pour la classe A signifie que deux logements A peuvent "
            "consommer entre 960 et 4 730 kWh selon le comportement des occupants."
        )

    # Tableau de synthese
    st.markdown("**Tableau de synthese : DPE estime vs reel et variabilite**")
    tbl_var = stats_var.copy().reset_index()
    tbl_var.columns = ["Classe DPE","Mediane reelle (kWh)","Ecart-type (kWh)",
                        "Mediane DPE (kWh)","Nb logements",
                        "Ecart DPE-reel (kWh)","Ecart (%)","Variabilite (CV%)"]
    for c_ in ["Mediane reelle (kWh)","Ecart-type (kWh)","Mediane DPE (kWh)",
                "Ecart DPE-reel (kWh)","Nb logements"]:
        tbl_var[c_] = tbl_var[c_].round(0).astype("Int64")
    for c_ in ["Ecart (%)","Variabilite (CV%)"]:
        tbl_var[c_] = tbl_var[c_].round(1)
    st.dataframe(tbl_var, use_container_width=True, hide_index=True)
    st.caption(
        "CV% = Coefficient de variation = Ecart-type / Mediane x 100. "
        "Plus ce chiffre est eleve, plus la variabilite comportementale est forte pour cette classe."
    )

    # ══════════════════════
    # SECTION 2 — GAINS PAR USAGE
    # ══════════════════════
    st.markdown("---")
    st.markdown("### 2️⃣  Combien economise-t-on par usage en changeant de classe DPE ?")
    st.caption(
        "Au-dela du total, on decompose ici les economies par poste de consommation : "
        "chauffage, eau chaude sanitaire, eclairage, etc. "
        "Cela permet d'identifier quels travaux ont le plus d'impact."
    )

    USAGE_LABELS = {
        "conso_chauffage_ef":"🔥 Chauffage",
        "conso_ecs_ef":"🚿 Eau Chaude",
        "conso_eclairage_ef":"💡 Eclairage",
        "conso_refroidissement_ef":"❄️ Refroidissement",
        "conso_auxiliaires_ef":"⚙️ Auxiliaires",
    }
    USAGE_COLORS_5 = ["#E45756","#4C78A8","#F28E2B","#76B7B2","#59A14F"]

    # Mediane par usage x classe
    med_usage = df.groupby("etiquette_dpe", observed=True)[list(USAGE_LABELS.keys())].median()
    med_usage.columns = list(USAGE_LABELS.values())
    med_usage = med_usage.reindex(DPE_ORDER_5).dropna(how="all")

    u1, u2 = st.columns(2)
    with u1:
        # Stacked bar usage x classe
        fig_use = go.Figure()
        for col_name, col_color in zip(list(USAGE_LABELS.values()), USAGE_COLORS_5):
            if col_name in med_usage.columns:
                fig_use.add_trace(go.Bar(
                    name=col_name,
                    x=med_usage.index.astype(str),
                    y=med_usage[col_name].fillna(0).round(0),
                    marker_color=col_color,
                ))
        fig_use.update_layout(
            barmode="stack",
            title="Consommation DPE par usage et par classe",
            xaxis_title="Classe DPE", yaxis_title="kWh/an",
            legend=dict(orientation="h", y=-0.3),
            height=420,
        )
        st.plotly_chart(fig_use, use_container_width=True, key="t5_stack_use")

    with u2:
        # Gains par usage lors d'amelioration de classe
        gains_use = []
        for i in range(len(DPE_ORDER_5)-1):
            frm, to_ = DPE_ORDER_5[i+1], DPE_ORDER_5[i]
            if frm in med_usage.index and to_ in med_usage.index:
                for col_name in list(USAGE_LABELS.values()):
                    g = med_usage.loc[frm, col_name] - med_usage.loc[to_, col_name]
                    if pd.notna(g) and g > 0:
                        gains_use.append({
                            "Passage": f"{frm}→{to_}",
                            "Usage": col_name,
                            "Gain (kWh/an)": round(g),
                        })

        if gains_use:
            gdf5 = pd.DataFrame(gains_use)
            fig_gu = px.bar(
                gdf5, x="Passage", y="Gain (kWh/an)", color="Usage",
                color_discrete_sequence=USAGE_COLORS_5,
                title="Gains par usage lors d'une amelioration de classe DPE",
                barmode="stack",
            )
            fig_gu.update_layout(
                height=420,
                legend=dict(orientation="h", y=-0.3),
                xaxis_title="Amelioration DPE",
            )
            st.plotly_chart(fig_gu, use_container_width=True, key="t5_gains_use")

    # Tableau gains par usage
    st.markdown("**Gains detailles par usage (kWh/an) — mediane nationale**")
    table_rows = []
    for i in range(len(DPE_ORDER_5)-1):
        frm, to_ = DPE_ORDER_5[i+1], DPE_ORDER_5[i]
        if frm in med_usage.index and to_ in med_usage.index:
            row = {"Amelioration": f"{frm} → {to_}"}
            total_g = 0
            for col_name in list(USAGE_LABELS.values()):
                g = med_usage.loc[frm, col_name] - med_usage.loc[to_, col_name]
                val = round(g) if pd.notna(g) and g > 0 else 0
                row[col_name] = val
                total_g += val
            row["Total (kWh/an)"] = total_g
            row["Total (euros/an)"] = round(total_g * prix_kwh)
            table_rows.append(row)
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Le chauffage est de loin le poste qui beneficie le plus de la renovation energetique. "
            "Passer de G a F permet d'economiser en moyenne 1 868 kWh/an rien que sur le chauffage."
        )

    # ══════════════════════
    # SECTION 3 — IMPACT CARACTERISTIQUES
    # ══════════════════════
    st.markdown("---")
    st.markdown("### 3️⃣  L'impact des caracteristiques du batiment sur la consommation")
    st.caption(
        "A classe DPE identique, la consommation reelle peut varier fortement selon "
        "la periode de construction, la qualite de l'isolation et la zone climatique."
    )

    c1_5, c2_5 = st.columns(2)

    with c1_5:
        # Conso reelle x periode x DPE
        prd5 = (df.dropna(subset=["conso_relle_kwh","periode_construction","etiquette_dpe"])
                  .groupby(["periode_construction","etiquette_dpe"], observed=True)
                  ["conso_relle_kwh"].median().reset_index())
        p_order5 = ["Avant 1948","1948-1974","1975-1989","1990-2000",
                    "2001-2005","2006-2012","2013-2021"]
        prd5 = prd5[prd5["periode_construction"].isin(p_order5)]
        fig_prd5 = px.bar(
            prd5, x="periode_construction", y="conso_relle_kwh",
            color="etiquette_dpe", color_discrete_map=DPE_COLORS,
            category_orders={"periode_construction": p_order5,
                             "etiquette_dpe": DPE_ORDER_5},
            barmode="group",
            title="Conso reelle selon la periode de construction et la classe DPE",
            labels={"conso_relle_kwh":"kWh/an","periode_construction":"Periode"},
        )
        fig_prd5.update_layout(height=400, legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig_prd5, use_container_width=True, key="t5_prd")
        st.caption("Les logements anciens (avant 1948) en classe D consomment souvent autant "
                   "que des logements recents en classe F, a cause de l'inertie thermique differente.")

    with c2_5:
        # Conso reelle x qualite isolation x DPE
        iso_data5 = (df.dropna(subset=["conso_relle_kwh","qualite_isolation_enveloppe","etiquette_dpe"])
                       .groupby(["qualite_isolation_enveloppe","etiquette_dpe"], observed=True)
                       ["conso_relle_kwh"].median().reset_index())
        if not iso_data5.empty:
            fig_iso5 = px.bar(
                iso_data5, x="qualite_isolation_enveloppe", y="conso_relle_kwh",
                color="etiquette_dpe", color_discrete_map=DPE_COLORS,
                barmode="group",
                title="Conso reelle selon la qualite d'isolation et la classe DPE",
                labels={"conso_relle_kwh":"kWh/an",
                        "qualite_isolation_enveloppe":"Qualite isolation"},
            )
            fig_iso5.update_layout(height=400, legend=dict(orientation="h",y=-0.35))
            st.plotly_chart(fig_iso5, use_container_width=True, key="t5_iso")
            st.caption("A meme classe DPE, une mauvaise isolation peut doubler la consommation reelle. "
                       "L'isolation est le facteur le plus determinant apres la classe DPE elle-meme.")

    # Zone climatique x DPE
    zone5 = (df.dropna(subset=["conso_relle_kwh","zone_climatique","etiquette_dpe"])
               .groupby(["zone_climatique","etiquette_dpe"], observed=True)
               ["conso_relle_kwh"].median().reset_index())
    if not zone5.empty and len(zone5["zone_climatique"].unique()) > 1:
        fig_zone5 = px.bar(
            zone5, x="zone_climatique", y="conso_relle_kwh",
            color="etiquette_dpe", color_discrete_map=DPE_COLORS,
            barmode="group",
            title="Conso reelle selon la zone climatique et la classe DPE",
            labels={"conso_relle_kwh":"kWh/an","zone_climatique":"Zone climatique"},
        )
        fig_zone5.update_layout(height=360, legend=dict(orientation="h",y=-0.3))
        st.plotly_chart(fig_zone5, use_container_width=True, key="t5_zone")
        st.caption("Les zones climatiques H1 (nord/montagne) consomment davantage que H3 (mediterranee). "
                   "La classe DPE ne capture qu'imparfaitement ces differences geographiques.")

    # ══════════════════════
    # SYNTHESE FINALE
    # ══════════════════════
    st.markdown("---")
    st.markdown("### ✅ Synthese — Ce que les donnees nous apprennent")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.success(
            "**Le DPE sous-estime les classes basses**\n\n"
            "Les logements F et G consomment en realite MOINS que predit par le DPE. "
            "Raison : les menages a faibles revenus se chauffent insuffisamment (effet precarite)."
        )
    with s2:
        st.info(
            "**La variabilite comportementale est enorme**\n\n"
            "L'ecart-type est de 1 500 a 1 900 kWh/an selon les classes. "
            "Deux logements DPE identiques peuvent consommer 2 a 3 fois differemment "
            "selon le comportement des occupants."
        )
    with s3:
        st.warning(
            "**Le chauffage concentre 70-80% des gains**\n\n"
            "Quand on ameliore son DPE, l'essentiel des economies vient du chauffage. "
            "Les autres postes (ECS, eclairage) representent des gains marginaux en comparaison."
        )


st.markdown("---")
st.caption("Sources : ADEME (DPE post-2021) x Enedis (conso residentielle). "
           "Geocodage : API BAN — adresse.data.gouv.fr.")
