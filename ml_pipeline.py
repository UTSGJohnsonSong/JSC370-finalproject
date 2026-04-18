"""
JSC370 Final Project — data pipeline: collection, feature engineering, LASSO selection
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV

BASE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(BASE, "outputs")
DATA  = os.path.join(BASE, "data")
CACHE = os.path.join(BASE, ".cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# ── 1. Data Acquisition ───────────────────────────────────────────────────────
def fetch_rest_countries():
    cache_path = os.path.join(CACHE, "rest_countries.json")
    if os.path.exists(cache_path):
        print("  [cache] REST Countries")
        with open(cache_path) as f:
            return json.load(f)
    print("  [fetch] REST Countries API...")
    r = requests.get("https://restcountries.com/v3.1/all?fields=cca3,name,latlng,landlocked,region,subregion", timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

def fetch_world_bank():
    cache_path = os.path.join(CACHE, "world_bank.csv")
    if os.path.exists(cache_path):
        print("  [cache] World Bank")
        return pd.read_csv(cache_path)
    print("  [fetch] World Bank API...")
    indicators = {
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "NE.TRD.GNFS.ZS": "trade_share",
        "SP.URB.TOTL.IN.ZS": "urbanization",
        "NV.AGR.TOTL.ZS": "agriculture_share",
        "SP.POP.TOTL": "population",
    }
    frames = []
    for code, name in indicators.items():
        url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
               f"?date=2019&format=json&per_page=300")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            continue
        rows = [{"iso3": e["countryiso3code"], name: e["value"]}
                for e in payload[1] if e["countryiso3code"]]
        frames.append(pd.DataFrame(rows).set_index("iso3"))
    df = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "iso3"})
    df.to_csv(cache_path, index=False)
    return df

print("\n=== Stage 1: Data Acquisition ===")
rc_raw = fetch_rest_countries()
wb_raw = fetch_world_bank()

rc_rows = []
for c in rc_raw:
    iso3 = c.get("cca3", "")
    if len(iso3) != 3:
        continue
    latlng = c.get("latlng", [None, None])
    rc_rows.append({
        "iso3": iso3,
        "name": c.get("name", {}).get("common", ""),
        "latitude":  latlng[0] if len(latlng) > 0 else None,
        "longitude": latlng[1] if len(latlng) > 1 else None,
        "landlocked": 1 if c.get("landlocked", False) else 0,
        "region": c.get("region", ""),
        "subregion": c.get("subregion", ""),
    })
rc = pd.DataFrame(rc_rows).drop_duplicates("iso3")

NON_COUNTRY = {"WLD","HIC","MIC","LMC","UMC","LIC","EAS","ECS","LCN","MEA","NAC","SAS","SSF","EMU"}
wb = wb_raw.copy()
wb = wb[wb["iso3"].str.match(r"^[A-Z]{3}$", na=False)]
wb = wb[~wb["iso3"].isin(NON_COUNTRY)]
wb = wb.drop_duplicates("iso3")

df = rc.merge(wb, on="iso3", how="inner")
df = df.dropna(subset=["latitude"])
for col in ["gdp_per_capita","trade_share","urbanization","agriculture_share","population"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[df["gdp_per_capita"] > 0]
df = df[df["urbanization"].between(0, 100)]
df = df[df["agriculture_share"].between(0, 100)]
df = df[df["trade_share"] >= 0]
df = df.dropna(subset=["gdp_per_capita","trade_share","urbanization","agriculture_share","population"])
print(f"  clean sample: N = {len(df)}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n=== Stage 2: Feature Engineering ===")
df["dist_equator"]      = df["latitude"].abs()
df["tropical"]          = (df["dist_equator"] < 23.5).astype(int)
df["dist_equator_sq"]   = df["dist_equator"] ** 2
df["log_gdp"]           = np.log(df["gdp_per_capita"])
df["log_population"]    = np.log(df["population"])
df["log_trade"]         = np.log(df["trade_share"] + 1)
df["agri_x_landlocked"] = df["agriculture_share"] * df["landlocked"]
df["trade_x_urban"]     = df["trade_share"] * df["urbanization"]
le = LabelEncoder()
df["region_encoded"]    = le.fit_transform(df["region"].fillna("Unknown"))
df["high_income"]       = (df["gdp_per_capita"] > 12535).astype(int)
print(f"  high income: {df['high_income'].sum()} / {len(df)}")

# ── 3. LASSO Variable Selection ───────────────────────────────────────────────
print("\n=== Stage 3: LASSO ===")
FEATURES = ["dist_equator","dist_equator_sq","landlocked","tropical",
            "agriculture_share","trade_share","urbanization","log_trade",
            "log_population","agri_x_landlocked","trade_x_urban","region_encoded"]

X_all    = df[FEATURES].values
y_reg    = df["log_gdp"].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

alphas = np.logspace(-4, 1, 50)
lasso  = LassoCV(cv=5, alphas=alphas, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_reg)

selected_features = [f for f, s in zip(FEATURES, lasso.coef_ != 0) if s]
print(f"  alpha={lasso.alpha_:.4f}, selected {len(selected_features)} features: {selected_features}")

df.to_csv(os.path.join(OUT, "clean_data.csv"), index=False)
df.to_csv(os.path.join(DATA, "clean_data.csv"), index=False)
with open(os.path.join(OUT, "lasso_selected.json"), "w") as f:
    json.dump(selected_features, f, indent=2)
print("  saved outputs/clean_data.csv, outputs/lasso_selected.json")
