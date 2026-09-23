"""
train_model.py
----------------
Trains an XGBoost binary classifier to estimate shot make probability
("expected FG%") from spatial + situational features. Evaluates with
ROC-AUC, log loss, Brier score, and calibration. Exports:
  - models/xgb_shot_model.json   (trained booster)
  - models/feature_importance.csv
  - models/metrics.json
  - data/predictions.csv          (held-out predictions, for dashboard)
"""
import json

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from features import DROP_FOR_MODEL

BASE = "/sessions/compassionate-funny-sagan/mnt/outputs/nba_shot_model"

raw = pd.read_csv(f"{BASE}/data/shots_raw.csv")
feats = pd.read_csv(f"{BASE}/data/shots_features.csv")

import re


def _sanitize(name: str) -> str:
    return re.sub(r"[\[\]<>() /]+", "_", name).strip("_")


raw_feature_cols = [c for c in feats.columns if c not in DROP_FOR_MODEL]
feature_cols = [_sanitize(c) for c in raw_feature_cols]
feats = feats.rename(columns=dict(zip(raw_feature_cols, feature_cols)))
X = feats[feature_cols]
y = feats["shot_made"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, feats.index, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 8,
    "seed": 42,
}

evals_result = {}
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=25,
    evals_result=evals_result,
    verbose_eval=False,
)

pred_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
pred_all = booster.predict(xgb.DMatrix(X, feature_names=feature_cols), iteration_range=(0, booster.best_iteration + 1))

auc = roc_auc_score(y_test, pred_test)
ll = log_loss(y_test, pred_test)
brier = brier_score_loss(y_test, pred_test)
baseline_ll = log_loss(y_test, np.full_like(pred_test, y_train.mean()))

frac_pos, mean_pred = calibration_curve(y_test, pred_test, n_bins=10, strategy="quantile")

metrics = {
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "best_iteration": int(booster.best_iteration),
    "roc_auc": round(float(auc), 4),
    "log_loss": round(float(ll), 4),
    "baseline_log_loss": round(float(baseline_ll), 4),
    "brier_score": round(float(brier), 4),
    "base_rate_fg_pct": round(float(y_train.mean()), 4),
    "calibration": {
        "predicted_mean_by_bin": [round(v, 3) for v in mean_pred.tolist()],
        "actual_frac_by_bin": [round(v, 3) for v in frac_pos.tolist()],
    },
}

with open(f"{BASE}/models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

booster.save_model(f"{BASE}/models/xgb_shot_model.json")

importance = booster.get_score(importance_type="gain")
imp_df = (
    pd.DataFrame({"feature": list(importance.keys()), "gain": list(importance.values())})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
imp_df.to_csv(f"{BASE}/models/feature_importance.csv", index=False)

# Full predictions joined back with metadata for the dashboard
out = raw.copy()
out["predicted_prob"] = pred_all
out["is_test_set"] = out.index.isin(idx_test)
out.to_csv(f"{BASE}/data/predictions.csv", index=False)

print(json.dumps(metrics, indent=2))
print("\nTop 10 features by gain:")
print(imp_df.head(10).to_string(index=False))
