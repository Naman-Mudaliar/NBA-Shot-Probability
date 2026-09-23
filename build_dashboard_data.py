"""
build_dashboard_data.py
------------------------
Packages model predictions + metrics into a single JSON payload consumed by
dashboard.html. Also computes player-level "shot quality" reports:
actual FG% vs. model-expected FG% (a simple shot-making-skill signal).
"""
import json

import numpy as np
import pandas as pd

BASE = "/sessions/compassionate-funny-sagan/mnt/outputs/nba_shot_model"

preds = pd.read_csv(f"{BASE}/data/predictions.csv")
metrics = json.load(open(f"{BASE}/models/metrics.json"))
importance = pd.read_csv(f"{BASE}/models/feature_importance.csv")

# ---- player report ----
player_report = (
    preds.groupby(["player_id", "player_name", "team", "position"])
    .agg(
        attempts=("shot_made", "size"),
        makes=("shot_made", "sum"),
        actual_fg_pct=("shot_made", "mean"),
        expected_fg_pct=("predicted_prob", "mean"),
        avg_distance=("shot_distance", "mean"),
        three_pt_rate=("is_three", "mean"),
    )
    .reset_index()
)
player_report = player_report[player_report["attempts"] >= 80].copy()
player_report["fg_pct_over_expected"] = (
    player_report["actual_fg_pct"] - player_report["expected_fg_pct"]
)
player_report["makes_over_expected"] = (
    player_report["fg_pct_over_expected"] * player_report["attempts"]
)
for c in ["actual_fg_pct", "expected_fg_pct", "avg_distance", "three_pt_rate",
          "fg_pct_over_expected", "makes_over_expected"]:
    player_report[c] = player_report[c].round(3)
player_report = player_report.sort_values("makes_over_expected", ascending=False)

# ---- zone report ----
zone_report = (
    preds.groupby("shot_zone")
    .agg(
        attempts=("shot_made", "size"),
        actual_fg_pct=("shot_made", "mean"),
        expected_fg_pct=("predicted_prob", "mean"),
    )
    .reset_index()
)
for c in ["actual_fg_pct", "expected_fg_pct"]:
    zone_report[c] = zone_report[c].round(3)

# ---- shot sample for scatter/court chart (cap for browser perf) ----
sample = preds.sample(n=min(9000, len(preds)), random_state=1).copy()
shot_cols = [
    "player_name", "team", "period", "shot_clock", "loc_x", "loc_y",
    "shot_distance", "shot_zone", "is_three", "dribbles", "defender_distance",
    "catch_and_shoot", "predicted_prob", "shot_made",
]
sample = sample[shot_cols].round({"shot_clock": 1, "loc_x": 1, "loc_y": 1,
                                   "shot_distance": 1, "defender_distance": 1,
                                   "predicted_prob": 3})

payload = {
    "metrics": metrics,
    "feature_importance": importance.head(15).round(2).to_dict(orient="records"),
    "player_report": player_report.to_dict(orient="records"),
    "zone_report": zone_report.to_dict(orient="records"),
    "shots": sample.to_dict(orient="records"),
    "players": sorted(preds["player_name"].unique().tolist()),
    "teams": sorted(preds["team"].unique().tolist()),
    "zones": sorted(preds["shot_zone"].unique().tolist()),
}

out_path = f"{BASE}/dashboard/dashboard_data.json"
with open(out_path, "w") as f:
    json.dump(payload, f)

print(f"Wrote dashboard payload: {out_path}")
print(f"Shots sample: {len(sample)}, Players: {len(player_report)}")
import os
print(f"Size: {os.path.getsize(out_path)/1024:.0f} KB")
