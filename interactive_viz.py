"""
JSC370 Final — HW5 Interactive Visualizations
3 Plotly figures saved as embeddable HTML + standalone viz.html page
"""

import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs")
SITE = os.path.join(BASE, "docs")  # GitHub Pages output
os.makedirs(SITE, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT, "clean_data.csv"))
with open(os.path.join(OUT, "metrics.json")) as f:
    metrics = json.load(f)

# Cluster labels (interpretive names based on composition)
cluster_labels = {0: "Agrarian", 1: "Mixed / Transitional", 2: "Urban Service"}
df["Cluster Type"] = df["cluster"].map(cluster_labels)
df["Income Group"] = df["high_income"].map({1: "High Income", 0: "Non-High Income"})
df["Log GDP per Capita"] = df["log_gdp"].round(2)
df["GDP per Capita (USD)"] = df["gdp_per_capita"].round(0).astype(int)

PALETTE = {
    "Agrarian":              "#e07b39",
    "Mixed / Transitional":  "#4c8bb5",
    "Urban Service":         "#2ca02c",
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Bubble Map — GDP per capita by country, colored by cluster type
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 1: Bubble Map...")

fig1 = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude" if "longitude" in df.columns else None,
    locations="iso3",
    locationmode="ISO-3",
    color="Cluster Type",
    color_discrete_map=PALETTE,
    size="gdp_per_capita",
    size_max=40,
    hover_name="name",
    hover_data={
        "GDP per Capita (USD)": True,
        "agriculture_share": ":.1f",
        "urbanization": ":.1f",
        "trade_share": ":.1f",
        "Cluster Type": True,
        "iso3": False,
        "latitude": False,
        "gdp_per_capita": False,
    },
    projection="natural earth",
    title="",
    labels={
        "agriculture_share": "Agriculture Share (%)",
        "urbanization": "Urbanization (%)",
        "trade_share": "Trade Share (% GDP)",
    }
)
fig1.update_layout(
    geo=dict(showframe=False, showcoastlines=True, coastlinecolor="lightgray",
             showland=True, landcolor="#f5f5f0", showocean=True, oceancolor="#e8f4f8",
             showcountries=True, countrycolor="white"),
    legend=dict(title="Cluster Type", x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(l=0, r=0, t=10, b=0),
    height=500,
    font=dict(family="Arial", size=12),
)

fig1_html = fig1.to_html(full_html=False, include_plotlyjs=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Interactive Scatter — Agriculture Share vs Log GDP, by cluster
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 2: Agriculture vs GDP Scatter...")

fig2 = px.scatter(
    df,
    x="agriculture_share",
    y="Log GDP per Capita",
    color="Cluster Type",
    color_discrete_map=PALETTE,
    symbol="Income Group",
    symbol_map={"High Income": "circle", "Non-High Income": "diamond"},
    size="urbanization",
    size_max=18,
    hover_name="name",
    hover_data={
        "GDP per Capita (USD)": True,
        "agriculture_share": ":.1f",
        "urbanization": ":.1f",
        "trade_share": ":.1f",
        "region": True,
        "Log GDP per Capita": False,
        "Cluster Type": False,
        "Income Group": False,
    },
    labels={
        "agriculture_share": "Agriculture, Forestry & Fishing (% of GDP)",
        "Log GDP per Capita": "Log GDP per Capita (USD)",
        "region": "Region",
        "urbanization": "Urbanization (%)",
        "trade_share": "Trade Share (% GDP)",
    },
    title="",
    opacity=0.82,
)

# Add OLS trendline per cluster manually
for cluster, grp in df.groupby("Cluster Type"):
    x = grp["agriculture_share"].values
    y = grp["Log GDP per Capita"].values
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        fig2.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines",
            line=dict(color=PALETTE[cluster], width=1.5, dash="dot"),
            showlegend=False,
        ))

fig2.update_layout(
    legend=dict(title="", x=0.99, y=0.99, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#ebebeb", title_font=dict(size=13)),
    yaxis=dict(showgrid=True, gridcolor="#ebebeb", title_font=dict(size=13)),
    margin=dict(l=60, r=20, t=20, b=60),
    height=480,
    font=dict(family="Arial", size=12),
)
fig2.update_xaxes(zeroline=False)
fig2.update_yaxes(zeroline=False)

fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Overlaid Histograms — Log GDP per Capita by Income Group
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 3: GDP Distribution Histogram...")

hi  = df[df["Income Group"] == "High Income"]["Log GDP per Capita"]
non = df[df["Income Group"] == "Non-High Income"]["Log GDP per Capita"]

fig3 = go.Figure()

fig3.add_trace(go.Histogram(
    x=non,
    name="Non-High Income",
    nbinsx=25,
    opacity=0.7,
    marker_color="#d73027",
    hovertemplate=(
        "Income Group: Non-High Income<br>"
        "Log GDP bin: %{x:.2f}<br>"
        "Count: %{y}<extra></extra>"
    ),
))

fig3.add_trace(go.Histogram(
    x=hi,
    name="High Income",
    nbinsx=25,
    opacity=0.7,
    marker_color="#2166ac",
    hovertemplate=(
        "Income Group: High Income<br>"
        "Log GDP bin: %{x:.2f}<br>"
        "Count: %{y}<extra></extra>"
    ),
))

fig3.update_layout(
    barmode="overlay",
    legend=dict(title="Income Group", x=0.99, y=0.99, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(title="Log GDP per Capita (USD)", showgrid=True, gridcolor="#ebebeb",
               title_font=dict(size=13)),
    yaxis=dict(title="Number of Countries", showgrid=True, gridcolor="#ebebeb",
               title_font=dict(size=13)),
    margin=dict(l=60, r=20, t=20, b=60),
    height=480,
    font=dict(family="Arial", size=12),
)

fig3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble viz.html
# ─────────────────────────────────────────────────────────────────────────────
print("Assembling viz.html...")

VIZ_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interactive Visualizations — From Geography to Development</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      background: #fafafa;
      color: #222;
      line-height: 1.6;
    }}
    h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: #1a1a2e;
      border-bottom: 2px solid #4c8bb5;
      padding-bottom: 0.5rem;
      margin-bottom: 0.4rem;
    }}
    .subtitle {{
      color: #555;
      font-size: 0.95rem;
      margin-bottom: 2rem;
    }}
    .nav {{
      display: flex;
      gap: 1rem;
      margin-bottom: 2.5rem;
      flex-wrap: wrap;
    }}
    .nav a {{
      text-decoration: none;
      color: #4c8bb5;
      font-size: 0.9rem;
      padding: 0.3rem 0;
      border-bottom: 1px solid transparent;
    }}
    .nav a:hover {{ border-bottom-color: #4c8bb5; }}
    .viz-block {{
      background: #fff;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 1.5rem 1.5rem 1rem;
      margin-bottom: 2.5rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .fig-caption {{
      font-size: 0.88rem;
      color: #444;
      margin-top: 0.8rem;
      padding-top: 0.8rem;
      border-top: 1px solid #f0f0f0;
    }}
    .fig-caption strong {{ color: #222; }}
    .fig-number {{
      display: inline-block;
      background: #4c8bb5;
      color: white;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.15rem 0.6rem;
      border-radius: 3px;
      margin-right: 0.5rem;
    }}
    h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: #1a1a2e;
      margin-bottom: 0.3rem;
    }}
    .description {{
      font-size: 0.93rem;
      color: #555;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>

<h1>From Geography to Development</h1>
<p class="subtitle">
  JSC370 Final Project · Interactive Visualizations (Homework 5) ·
  <a href="index.html">Back to main page</a>
</p>

<nav class="nav">
  <a href="#fig1">Figure 1: World Map</a>
  <a href="#fig2">Figure 2: Agriculture vs GDP</a>
  <a href="#fig3">Figure 3: GDP Distribution</a>
</nav>

<!-- Figure 1 -->
<div class="viz-block" id="fig1">
  <h2>Figure 1 — Country Economic Clusters and GDP per Capita</h2>
  <p class="description">
    Hover over each country to see its GDP per capita, agriculture share, urbanization rate,
    and trade openness. Circle size reflects GDP per capita; color indicates the K-means cluster
    type (K = 3 chosen by silhouette score). Drag to pan, scroll to zoom.
  </p>
  {fig1_html}
  <p class="fig-caption">
    <span class="fig-number">Figure 1</span>
    <strong>Global distribution of economic cluster types and national income (2019).</strong>
    Each bubble represents one country; size is proportional to GDP per capita (USD) and colour
    indicates K-means cluster assignment derived from standardised agriculture share, urbanisation
    rate, and trade openness. Agrarian economies (orange) dominate Sub-Saharan Africa and South
    Asia; Urban Service economies (green) are concentrated in Europe and North America. The map
    reveals a clear spatial gradient in economic structure that mirrors the latitude–income
    association documented in the regression analysis.
  </p>
</div>

<!-- Figure 2 -->
<div class="viz-block" id="fig2">
  <h2>Figure 2 — Agriculture Dependence, Urbanisation, and National Income</h2>
  <p class="description">
    Each point is one country. Point size reflects urbanization rate; shape indicates income
    classification (circle = high income, diamond = non-high income). Dotted lines show
    within-cluster OLS trends. Hover for full country profile.
  </p>
  {fig2_html}
  <p class="fig-caption">
    <span class="fig-number">Figure 2</span>
    <strong>Agriculture share (% of GDP) versus log GDP per capita across 173 countries (2019).</strong>
    A strong negative association (Pearson r = −0.79, p &lt; 0.001) is evident across all cluster
    types: countries with larger agricultural sectors have substantially lower incomes. Point size
    encodes urbanisation rate, highlighting that high-income countries (circles) tend to combine
    low agriculture dependence with high urbanisation. Dotted trend lines fitted separately per
    cluster confirm that the negative slope holds within each structural group, though the
    Agrarian cluster (orange) shows the steepest gradient.
  </p>
</div>

<!-- Figure 3 -->
<div class="viz-block" id="fig3">
  <h2>Figure 3 — Distribution of Log GDP per Capita by Income Group</h2>
  <p class="description">
    Overlaid histograms comparing the log GDP per capita distributions of high-income
    (blue) and non-high-income (red) countries. Hover over any bar to see the exact
    bin range and country count. Click legend items to show/hide a group.
  </p>
  {fig3_html}
  <p class="fig-caption">
    <span class="fig-number">Figure 3</span>
    <strong>Distribution of log GDP per capita for high-income vs. non-high-income countries (2019, N = 173).</strong>
    The two distributions are almost entirely non-overlapping: non-high-income countries
    (red) cluster between log GDP ≈ 6–9, while high-income countries (blue) concentrate
    above log GDP ≈ 9.5. The clear separation confirms that log GDP per capita is a
    strongly bimodal variable once countries are split by the World Bank threshold
    ($12,535), and motivates using income classification as the ML target rather than
    a continuous regression outcome.
  </p>
</div>

</body>
</html>
"""

with open(os.path.join(SITE, "viz.html"), "w", encoding="utf-8") as f:
    f.write(VIZ_HTML)

print(f"Saved: docs/viz.html")

# ─────────────────────────────────────────────────────────────────────────────
# Also write index.html (GitHub Pages main page)
# ─────────────────────────────────────────────────────────────────────────────
print("Writing index.html...")

rf_auc   = metrics["random_forest"]["auc"]
rf_f1    = metrics["random_forest"]["f1"]
rf_acc   = metrics["random_forest"]["accuracy"]
lr_auc   = metrics["logistic"]["auc"]
xgb_auc  = metrics["xgboost"]["auc"]
n_hi     = int(df["high_income"].sum())
n_total  = len(df)

INDEX_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>From Geography to Development — JSC370 Final Project</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, Arial, sans-serif; background: #f7f8fa; color: #1a1a2e; line-height: 1.65; }}
    header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); color: #fff; padding: 3.5rem 1.5rem 3rem; text-align: center; }}
    header .tag {{ display: inline-block; background: rgba(255,255,255,0.15); border-radius: 20px; font-size: 0.8rem; letter-spacing: 0.08em; padding: 0.25rem 0.9rem; margin-bottom: 1.2rem; text-transform: uppercase; }}
    header h1 {{ font-size: clamp(1.6rem, 4vw, 2.5rem); font-weight: 700; line-height: 1.2; margin-bottom: 0.8rem; }}
    header h1 span {{ color: #74c2ff; }}
    header p.sub {{ font-size: 1rem; opacity: 0.8; max-width: 600px; margin: 0 auto 2rem; }}
    .header-links {{ display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap; }}
    .btn {{ display: inline-block; padding: 0.55rem 1.3rem; border-radius: 6px; font-size: 0.88rem; font-weight: 600; text-decoration: none; transition: opacity 0.15s; }}
    .btn:hover {{ opacity: 0.85; }}
    .btn-primary {{ background: #74c2ff; color: #0f3460; }}
    .btn-outline {{ background: transparent; color: #fff; border: 1px solid rgba(255,255,255,0.5); }}
    main {{ max-width: 900px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }}
    section {{ margin-bottom: 3rem; }}
    h2 {{ font-size: 1.2rem; font-weight: 700; color: #0f3460; border-left: 4px solid #74c2ff; padding-left: 0.8rem; margin-bottom: 1rem; }}
    p {{ margin-bottom: 0.9rem; color: #333; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.2rem 0; }}
    .stat-card {{ background: #fff; border-radius: 8px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .stat-card .value {{ font-size: 1.8rem; font-weight: 700; color: #0f3460; line-height: 1; margin-bottom: 0.3rem; }}
    .stat-card .label {{ font-size: 0.82rem; color: #666; }}
    .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1.2rem 0; }}
    .model-card {{ background: #fff; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-top: 3px solid #ccc; }}
    .model-card.best {{ border-top-color: #74c476; }}
    .model-card h3 {{ font-size: 0.95rem; font-weight: 700; margin-bottom: 0.6rem; }}
    .model-card .metric-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: #555; padding: 0.2rem 0; border-bottom: 1px solid #f0f0f0; }}
    .model-card .metric-row:last-child {{ border-bottom: none; }}
    .model-card .metric-row strong {{ color: #1a1a2e; }}
    .timeline {{ list-style: none; padding: 0; }}
    .timeline li {{ display: flex; gap: 1rem; padding: 0.8rem 0; border-bottom: 1px solid #eee; font-size: 0.93rem; color: #444; }}
    .timeline li:last-child {{ border-bottom: none; }}
    .timeline .step {{ flex-shrink: 0; width: 1.8rem; height: 1.8rem; background: #0f3460; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; }}
    .findings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }}
    .finding {{ background: #fff; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .finding .icon {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
    .finding h3 {{ font-size: 0.95rem; font-weight: 700; margin-bottom: 0.4rem; }}
    .finding p {{ font-size: 0.88rem; color: #555; margin: 0; }}
    footer {{ text-align: center; padding: 1.5rem; font-size: 0.82rem; color: #888; border-top: 1px solid #e0e0e0; }}
    footer a {{ color: #4c8bb5; text-decoration: none; }}
  </style>
</head>
<body>
<header>
  <div class="tag">JSC370 Final Project · April 2026</div>
  <h1>From <span>Geography</span> to Development</h1>
  <p class="sub">Does economic structure mediate the latitude-income association? A cross-country analysis of {n_total} nations with machine learning classification.</p>
  <div class="header-links">
    <a href="report.html" class="btn btn-primary">Read Full Report</a>
    <a href="viz.html"    class="btn btn-outline">Interactive Visualizations</a>
    <a href="report.pdf"  class="btn btn-outline">Download PDF</a>
    <a href="https://github.com/UTSGJohnsonSong/JSC370-finalproject" class="btn btn-outline">GitHub Repo</a>
  </div>
</header>
<main>
  <section>
    <h2>Project Overview</h2>
    <p>Countries at higher latitudes are systematically richer than tropical nations. But is this a direct geographic effect, or does it operate through the economic structures that geography shapes? This project uses 2019 cross-country data for <strong>{n_total} nations</strong>, obtained via the REST Countries and World Bank Development Indicators APIs, to disentangle these pathways through exploratory analysis, OLS regression, and machine learning.</p>
    <div class="stats">
      <div class="stat-card"><div class="value">{n_total}</div><div class="label">Countries in sample</div></div>
      <div class="stat-card"><div class="value">{n_hi}</div><div class="label">High-income countries</div></div>
      <div class="stat-card"><div class="value">11</div><div class="label">Features selected (LASSO)</div></div>
      <div class="stat-card"><div class="value">{rf_auc}</div><div class="label">Best AUC-ROC (Random Forest)</div></div>
      <div class="stat-card"><div class="value">3</div><div class="label">Economic clusters (K-Means)</div></div>
    </div>
  </section>
  <section>
    <h2>Key Findings</h2>
    <div class="findings">
      <div class="finding"><div class="icon">&#127758;</div><h3>Geography shapes structure</h3><p>Distance from the equator correlates with lower agriculture dependence (r = -0.39) and higher urbanisation (r = 0.38), confirming geographic-structural linkages.</p></div>
      <div class="finding"><div class="icon">&#128201;</div><h3>Structure mediates ~51% of income gap</h3><p>The latitude coefficient drops from 0.045 (geography only) to 0.022 after adding structure variables - a 51% attenuation - but remains statistically significant.</p></div>
      <div class="finding"><div class="icon">&#127806;</div><h3>Agriculture is the dominant predictor</h3><p>Agriculture share (r = -0.79 with log GDP) is the top feature in all three ML models. Urbanisation ranks second (r = 0.72).</p></div>
      <div class="finding"><div class="icon">&#129302;</div><h3>ML classifies with high accuracy</h3><p>Random Forest achieves AUC = {rf_auc} and F1 = {rf_f1} on the held-out test set. All three classifiers exceed AUC = 0.92.</p></div>
    </div>
  </section>
  <section>
    <h2>Machine Learning Results</h2>
    <p>Three classifiers trained on a stratified 80/20 split (N_train=138, N_test=35) to predict high-income status (GDP per capita > $12,535). Hyperparameters selected by 5-fold CV.</p>
    <div class="model-grid">
      <div class="model-card"><h3>Logistic Regression (baseline)</h3><div class="metric-row"><span>Accuracy</span><strong>{metrics['logistic']['accuracy']*100:.1f}%</strong></div><div class="metric-row"><span>F1-Score</span><strong>{metrics['logistic']['f1']}</strong></div><div class="metric-row"><span>AUC-ROC</span><strong>{lr_auc}</strong></div></div>
      <div class="model-card best"><h3>Random Forest &#9733; Best</h3><div class="metric-row"><span>Accuracy</span><strong>{rf_acc*100:.1f}%</strong></div><div class="metric-row"><span>F1-Score</span><strong>{rf_f1}</strong></div><div class="metric-row"><span>AUC-ROC</span><strong>{rf_auc}</strong></div></div>
      <div class="model-card"><h3>XGBoost</h3><div class="metric-row"><span>Accuracy</span><strong>{metrics['xgboost']['accuracy']*100:.1f}%</strong></div><div class="metric-row"><span>F1-Score</span><strong>{metrics['xgboost']['f1']}</strong></div><div class="metric-row"><span>AUC-ROC</span><strong>{xgb_auc}</strong></div></div>
    </div>
  </section>
  <section>
    <h2>Data Sources</h2>
    <p><strong>REST Countries API:</strong> restcountries.com/v3.1/all - geographic metadata for all sovereign nations.<br>
    <strong>World Bank Development Indicators:</strong> api.worldbank.org/v2 - GDP per capita, trade share, urbanisation, agriculture share, population (2019).</p>
  </section>
</main>
<footer>JSC370 Final Project &middot; Zekun Song &middot; University of Toronto &middot; April 2026 &middot; <a href="https://github.com/UTSGJohnsonSong/JSC370-finalproject">GitHub</a></footer>
</body>
</html>"""

with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
    f.write(INDEX_HTML)

print(f"Saved: docs/index.html")
print("=== ALL DOCS COMPLETE ===")
