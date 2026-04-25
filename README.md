# ⚡ DPE & Consommation Électrique — Application Streamlit

Application d'analyse interactive croisant les données DPE (ADEME) avec les consommations électriques réelles (Enedis).

---

## 🚀 Installation & Lancement

### 1. Pré-requis
- Python 3.9+
- Le fichier `DPE_enedis_1.csv` dans le même dossier que `app.py`

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 📐 Architecture de l'application

```
dpe_app/
├── app.py              ← Application principale
├── requirements.txt    ← Dépendances Python
├── README.md           ← Ce fichier
└── DPE_enedis_1.csv    ← Données (à placer ici)
```

---

## 🗂️ Fonctionnalités

### Barre latérale (inputs utilisateur)
- Commune (recherche parmi toutes les communes disponibles)
- Type de bâtiment, surface, classe DPE actuelle, période de construction
- Prix de l'électricité et hypothèse de hausse annuelle

### 🗺️ Tab 1 — Logements proches
- Carte interactive Folium des **10 adresses géographiquement les plus proches**
- Marqueurs colorés par classe DPE (vert→rouge)
- Popup détaillé : surface, conso DPE estimée, conso réelle Enedis, écart
- Tableau comparatif et statistiques résumées

### 🏘️ Tab 2 — Analyse Commune
- KPIs : nombre de logements, consommations médianes, % passoires thermiques
- Distribution des classes DPE dans la commune
- Comparaison DPE estimé vs réel par classe
- Nuage de points DPE estimé / réel
- Gains en kWh/an et €/an par amélioration de classe

### 🇫🇷 Tab 3 — Analyse France
- Vue nationale : distribution DPE, boxplots, comparaisons
- **Position de votre profil vs commune vs France**
- Tableau des gains nationaux par amélioration de classe
- Consommation par période de construction

### 📈 Tab 4 — Prédiction 10 ans
- **3 scénarios** : sans rénovation / rénovation classe B / rénovation classe A
- Graphique d'évolution des coûts annuels (2025–2035)
- Tableau annuel détaillé avec économies
- Graphique des économies cumulées

---

## 📊 Données

| Source | Description |
|--------|-------------|
| ADEME (DPE) | Consommations estimées par la méthode 3CL, classe DPE A→G |
| Enedis | Consommations électriques réelles mesurées par compteurs Linky |
| Fusion | ~77 800 logements, majoritairement des appartements à chauffage électrique |

---

## ⚙️ Notes techniques

- **Coordonnées** : converties de Lambert 93 (EPSG:2154) vers WGS84 (EPSG:4326) via `pyproj`
- **Recherche des voisins** : KDTree scipy sur les coordonnées Lambert 93
- **Prédiction tarifaire** : modèle exponentiel simple (`prix × (1 + taux)^n`)
- **Réduction de conso par rénovation** : interpolée depuis les médianes réelles par classe DPE dans les données
