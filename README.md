# From Geography to Development

JSC370 Final Project — Zekun Song, University of Toronto, April 2026

## Research Question

Does economic structure mediate the latitude–income association across countries?

## Data Sources

- **REST Countries API** (`restcountries.com/v3.1/all`) — geographic metadata for all sovereign nations
- **World Bank Development Indicators** (`api.worldbank.org/v2`) — GDP per capita, trade share, urbanization, agriculture share, population (2019)

## Key Results

- **173 countries** in the analytic sample (2019)
- **51% attenuation** of the latitude coefficient (M1→M5 OLS mediation)
- **Random Forest AUC = 0.958** on the held-out test set (best of 3 classifiers)
- **K = 3 clusters** identified by silhouette score: Agrarian, Mixed/Transitional, Urban Service

## Project Structure

```
ml_pipeline.py        — data collection, feature engineering, LASSO, OLS, ML, K-Means
interactive_viz.py    — Plotly interactive visualizations (HW5)
final_report.qmd      — Quarto report source
outputs/              — pre-computed model outputs (JSON/CSV)
data/                 — clean dataset
docs/                 — GitHub Pages site (index, report, viz)
```

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the full pipeline:
```bash
python ml_pipeline.py
```

Generate interactive visualizations:
```bash
python interactive_viz.py
```

Render the report (requires Quarto):
```bash
quarto render final_report.qmd
```

## Website

Live site: [UTSGJohnsonSong.github.io/JSC370-finalproject](https://UTSGJohnsonSong.github.io/JSC370-finalproject)
