# NBA Shot Probability Model

End-to-end pipeline that predicts shot make probability ("expected FG%") from
spatial and situational context, plus an interactive dashboard for exploring
model outputs and player shot-making performance.

**Stack:** Python, pandas, XGBoost, scikit-learn

## Pipeline

```
generate_data.py        -> data/shots_raw.csv        (raw shot events)
features.py              -> data/shots_features.csv   (engineered features)
train_model.py            -> models/xgb_shot_model.json, models/metrics.json,
                             models/feature_importance.csv, data/predictions.csv
build_dashboard_data.py   -> dashboard/dashboard_data.json (player/zone reports)
```

Run the full pipeline:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Then rebuild the self-contained dashboard (embeds the JSON payload inline so
it opens with no server, right from the file system):

```bash
python - <<'EOF'
data = open("dashboard/dashboard_data.json").read()
tmpl = open("dashboard/dashboard_template.html").read()
open("dashboard/dashboard.html", "w").write(tmpl.replace("__DATA_JSON__", data))
EOF
```

Open `dashboard/dashboard.html` in any browser — no server required.

## Data

This sandbox has no outbound network access, so `generate_data.py` produces a
**statistically calibrated synthetic dataset** (30,000 shot events, 60
players) instead of pulling live data. Make probabilities are seeded from
published NBA league-average FG% by zone (restricted area ~64%, corner 3
~39%, above-the-break 3 ~35%, etc.) and perturbed by defender distance, shot
clock, dribbles/touch time, and a latent per-player skill offset — so
downstream model behavior (feature importances, calibration, AUC) matches
what you'd see on a real shot log.

To run on real data, swap `generate_data.py` for `nba_api` calls
(`shotchartdetail` endpoint) — the feature engineering, training, and
dashboard code are unchanged.

## Features

Spatial: shot distance, angle from hoop, quadrant, distance bucket, zone.
Situational: shot clock (+ pressure buckets), dribbles, touch time,
catch-and-shoot flag, defender distance (+ tight/open flags), clutch flag
(Q4/OT, close game, final 5 min), period, home/away, score margin, fatigue
proxy.

## Model

Binary XGBoost classifier (`binary:logistic`), early-stopped on held-out
log loss. Evaluated with ROC-AUC, log loss (vs. base-rate baseline), Brier
score, and a 10-bin calibration curve. Typical run: **AUC ≈ 0.66, log loss
≈ 0.64** — in line with published expected-FG% models, since shot outcomes
are inherently noisy (defense, contest, luck) even with strong shot-quality
features. Restricted-area zone and shot distance dominate feature
importance, as expected.

## Dashboard

Single-file HTML dashboard (Plotly.js via CDN, everything else vanilla JS,
no backend):

- Interactive shot chart on a half-court diagram, colored by predicted make
  probability, filterable by player / zone / period / result
- Model calibration curve and feature-importance chart
- Zone report: actual vs. expected FG% by shot zone
- Player report: actual FG% vs. expected FG%, sortable, surfacing shot-making
  skill (makes over expectation) — who's outperforming their shot quality
