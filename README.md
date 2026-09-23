# NBA Shot Probability (xFG%)

A **shooter-blind** expected-FG% model trained on every regular-season NBA shot from
2013-14 to 2025-26 (2.73M shots, stats.nba.com), plus a self-contained dashboard.

xFG% answers: *how likely is an average NBA player to make this shot, against this
defense?* The model sees only the shot (location, shot type, game situation) and the
opponent's defense from earlier games — never who is shooting. So a player's
**actual FG% − xFG%** is their shot-making skill (or luck), and **average xFG%** is the
quality of the shots they take.

## Results (held-out, 2022-23 → 2025-26)

| Model | Log loss | AUC |
|---|---|---|
| **Weekly-updated (walk-forward)** | **0.6386** | **0.657** |
| Frozen after training | 0.6402 | 0.652 |
| League FG% by zone | 0.6595 | 0.634 |

Beats both baselines in every held-out season; calibration within 0.6 pts in every bin.
Top shot-makers over expected (2024-25 & 2025-26): Jokić (+10.7 pts), Durant, Gilgeous-Alexander.

## How it works

```
fetch_data.py         nba_api shot charts, per team per season -> data/raw/ (checkpointed)
features.py           shot-only features + past-only opponent zone defense
split.py              train 2013-14..2020-21 | early-stop 2021-22 | walk-forward 2022-23..2025-26
train_model.py        XGBoost offline fit, then weekly predict-then-learn; frozen + zone baselines
build_dashboard_data  last 2 walk-forward seasons -> dashboard/dashboard.html (self-contained)
```

Every xFG% shown was predicted **before** the model trained on that shot.

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py --synthetic   # fast loop (~30 s, no network): generate -> features -> train -> dashboard -> tests
python fetch_data.py                 # real data, ~25 min first time; resumable, skips completed seasons
python run_pipeline.py               # real run: features -> train -> dashboard -> tests
```

Open `dashboard/dashboard.html` in a browser (needs internet once for the Plotly CDN).
Deep links: `dashboard.html?player=Stephen%20Curry&theme=dark`.

## Dashboard

- **Training data & model** — shots per season by role in training, 13-season league
  trends (FG%, 3PA rate, shot mix by zone), week-by-week held-out accuracy vs the frozen
  and zone baselines, calibration, feature importance, per-season metrics.
- **Shot maps** (filter by season, team, player, opponent, shot type, result; color by
  xFG%, actual FG% or actual − xFG) — 14-zone map, probability hex map, individual shots,
  and where the selection beats the model zone by zone.
- **xFG outliers** — shot-making leaderboard (sortable, by zone group, min attempts),
  shot quality vs shot-making quadrant, luck check with a ±2σ funnel, best/toughest shot
  diets, toughest makes and easiest misses, team offense and defense.

## Leakage safeguards

Enforced by `tests/test_leakage.py` on every fast-loop run: no shooter columns in the
model inputs, predictions invariant to shuffling shooter identity, defense features
unaffected by scrambling future results, chronological split, every walk-forward
prediction made before training on that shot (plus a label-flip test), and planted
synthetic shooter skill recovered from actual − xFG.

See `CLAUDE.md` for the development workflow and stats.nba.com gotchas.
