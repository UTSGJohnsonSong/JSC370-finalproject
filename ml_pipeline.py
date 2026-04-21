"""
JSC370 Final Project — pipeline: data, features, LASSO, OLS mediation models
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
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

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
print(f"  alpha={lasso.alpha_:.4f}, selected {len(selected_features)}: {selected_features}")

with open(os.path.join(OUT, "lasso_selected.json"), "w") as f:
    json.dump(selected_features, f, indent=2)

# ── 3.5. OLS Mediation Models (M1-M5, HC3 robust SE) ─────────────────────────
print("\n=== Stage 3.5: OLS Mediation Models ===")
y_ols = df["log_gdp"]
models_spec = {
    "M1": ["dist_equator"],
    "M2": ["dist_equator", "landlocked"],
    "M3": ["dist_equator", "landlocked", "agriculture_share"],
    "M4": ["dist_equator", "landlocked", "agriculture_share", "urbanization"],
    "M5": ["dist_equator", "landlocked", "agriculture_share", "urbanization", "trade_share"],
}

ols_results = {}
for mname, cols in models_spec.items():
    X_ols = sm.add_constant(df[cols])
    fit   = sm.OLS(y_ols, X_ols).fit(cov_type="HC3")
    coef  = float(fit.params["dist_equator"])
    se    = float(fit.bse["dist_equator"])
    pval  = float(fit.pvalues["dist_equator"])
    ols_results[mname] = {
        "n": int(fit.nobs),
        "r2": round(float(fit.rsquared), 4),
        "adj_r2": round(float(fit.rsquared_adj), 4),
        "coef_dist_equator": round(coef, 4),
        "se_dist_equator":   round(se,   4),
        "pval_dist_equator": round(pval, 4),
        "controls": cols[1:],
    }
    print(f"  {mname}: coef={coef:.4f}  SE={se:.4f}  p={pval:.4f}  R2={fit.rsquared:.3f}")

coef_m1 = ols_results["M1"]["coef_dist_equator"]
coef_m5 = ols_results["M5"]["coef_dist_equator"]
ols_results["attenuation_pct"] = round((coef_m1 - coef_m5) / coef_m1 * 100, 1)
print(f"  attenuation M1->M5: {ols_results['attenuation_pct']}%")

df.to_csv(os.path.join(OUT, "clean_data.csv"), index=False)
df.to_csv(os.path.join(DATA, "clean_data.csv"), index=False)
with open(os.path.join(OUT, "ols_results.json"), "w") as f:
    json.dump(ols_results, f, indent=2)
print("  saved ols_results.json")

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
print("\n=== Stage 4: Train/Test Split ===")
X = df[selected_features].values
y = df["high_income"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler_cls = StandardScaler()
X_train_s = scaler_cls.fit_transform(X_train)
X_test_s  = scaler_cls.transform(X_test)

split_info = {
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test":  int(len(X_test)),
    "high_income_train_pct": float(y_train.mean() * 100),
    "high_income_test_pct":  float(y_test.mean() * 100),
}
print(f"  Train: N={split_info['n_train']}  Test: N={split_info['n_test']}")
with open(os.path.join(OUT, "train_test_split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

# ── 5. ML Models ──────────────────────────────────────────────────────────────
print("\n=== Stage 5: ML Models ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_out = {}
conf_matrices = {}
importances = {f: {} for f in selected_features}

def eval_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    result = {
        "accuracy":  round(float(accuracy_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_te, y_prob)), 4),
    }
    conf_matrices[name] = confusion_matrix(y_te, y_pred).tolist()
    print(f"  {name}: Acc={result['accuracy']} F1={result['f1']} AUC={result['auc']}")
    return result

# Logistic Regression
print("  [A] Logistic Regression...")
lr = LogisticRegressionCV(cv=cv, max_iter=2000, random_state=42, scoring="roc_auc")
res_lr = eval_model("logistic", lr, X_train_s, X_test_s, y_train, y_test)
res_lr["best_params"] = {"C": float(lr.C_[0])}
metrics_out["logistic"] = res_lr
imp_lr = np.abs(lr.coef_[0])
imp_lr = imp_lr / imp_lr.sum() if imp_lr.sum() > 0 else imp_lr
for f, v in zip(selected_features, imp_lr):
    importances[f]["lr"] = round(float(v), 4)

# Random Forest
print("  [B] Random Forest...")
rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
rf_grid = GridSearchCV(rf_base,
    param_grid={"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 2, 5]},
    cv=cv, scoring="roc_auc", n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
res_rf = eval_model("random_forest", rf_best, X_train, X_test, y_train, y_test)
res_rf["best_params"] = rf_grid.best_params_
metrics_out["random_forest"] = res_rf
imp_rf = rf_best.feature_importances_
for f, v in zip(selected_features, imp_rf):
    importances[f]["rf"] = round(float(v), 4)

# XGBoost
print("  [C] XGBoost...")
xgb_base = XGBClassifier(n_estimators=200, random_state=42,
                          eval_metric="logloss", verbosity=0)
xgb_grid = GridSearchCV(xgb_base,
    param_grid={"max_depth": [3, 4, 5],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.7, 1.0]},
    cv=cv, scoring="roc_auc", n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
res_xgb = eval_model("xgboost", xgb_best, X_train, X_test, y_train, y_test)
res_xgb["best_params"] = xgb_grid.best_params_
metrics_out["xgboost"] = res_xgb
imp_xgb = xgb_best.feature_importances_
for f, v in zip(selected_features, imp_xgb):
    importances[f]["xgb"] = round(float(v), 4)

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
with open(os.path.join(OUT, "confusion_matrices.json"), "w") as f:
    json.dump(conf_matrices, f, indent=2)

imp_df = pd.DataFrame([
    {"feature": feat,
     "lr_importance":  importances[feat].get("lr", 0),
     "rf_importance":  importances[feat].get("rf", 0),
     "xgb_importance": importances[feat].get("xgb", 0)}
    for feat in selected_features
])
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
print("  saved metrics.json, confusion_matrices.json, feature_importance.csv")
