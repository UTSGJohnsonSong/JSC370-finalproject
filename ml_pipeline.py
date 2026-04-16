"""
JSC370 Final Project — data collection and cleaning
"""

import requests
import pandas as pd
import numpy as np
import json
import os

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

# parse REST Countries
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

# clean World Bank
NON_COUNTRY = {"WLD","HIC","MIC","LMC","UMC","LIC","EAS","ECS","LCN","MEA","NAC","SAS","SSF","EMU"}
wb = wb_raw.copy()
wb = wb[wb["iso3"].str.match(r"^[A-Z]{3}$", na=False)]
wb = wb[~wb["iso3"].isin(NON_COUNTRY)]
wb = wb.drop_duplicates("iso3")

# merge and clean
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

df.to_csv(os.path.join(DATA, "clean_data.csv"), index=False)
print("  saved data/clean_data.csv")
