# NBA Shot Probability (xFG%)

Shooter-blind expected-FG% model on real NBA shot data (nba_api `ShotChartDetail`),
trained on early seasons and updated walk-forward (weekly) through recent ones.

## Workflow — synthetic first, real second
- **Fast loop, after every edit:** `python run_pipeline.py --synthetic`
  (generate → features → train → pytest; seconds, no network). Must be green before anything else.
- **Slow loop:** `python fetch_data.py [--seasons 2025-26 ...]` pulls real data. Only run it when
  the synthetic loop is green, and **never to "verify" a code change** — it is slow, rate-limited,
  and burns NBA API goodwill. `run_pipeline.py` never fetches.
- Work in phases; **commit after each phase passes its gate** so we can bisect back to a known-good phase.

## Leakage rules (enforced by `tests/test_leakage.py` — never weaken these tests)
1. **Shooter-blind.** xFG% = probability an *average* player makes this shot vs this defense.
   No player ID/name, position, shooter's own team, or shooter form/history as model inputs
   (`config.BLOCKED_FEATURES` / `BLOCKED_SUBSTRINGS`). Team-level opponent defense is allowed.
   Tested by: column check, shooter-permutation invariance, and planted-skill recovery
   (player actual − xFG must correlate with synthetic hidden skill, ρ > 0.6).
2. **Defense features use only games strictly before the shot's game.** Tested by scrambling
   future outcomes and asserting past features are unchanged.
3. **Split is chronological by season.**
4. **Every walk-forward prediction comes from a model that has not trained on that shot.**
   Tested via logged `train_max_date < chunk_min_date` and a label-flip invariance test.

## Defaults (`config.py`)
- Real seasons 2013-14 → 2025-26; `TRAIN_FRAC = 0.7` → fit on first 9 seasons, the last of those
  (2021-22) is the early-stopping season; 2022-23 → 2025-26 are walk-forward/online.
- Synthetic: 6 seasons (2018-19 → 2023-24), same schema as real (`config.SHOT_SCHEMA`).

## Layout
- `config.py` — **all paths**, seasons, split, schema, blocked features. No hardcoded paths elsewhere.
  `config.outputs(synthetic)` gives per-mode artifact paths (synthetic never clobbers real).
- `schema.py` — raw schema validation/coercion (shared by fetch + tests).
- `generate_data.py` — synthetic data in real schema, with planted player skill / team defense.
- `fetch_data.py` — checkpointed real pull: per-season `.tmp` → validate → atomic rename,
  `data/raw/manifest.json` (rows + sha256 + validator version) decides what to skip.
  stats.nba.com gotchas (both observed, both silent):
  - python-requests / `nba_api` HTTP calls **hang to timeout** (CDN client fingerprinting);
    `curl` with the same headers (incl. `x-nba-stats-origin`/`x-nba-stats-token`) works → we shell out to curl.
  - League-wide `TeamID=0` shotchartdetail **truncates at 102,400 rows** (half a season, HTTP 200).
    → fetch per team (30 calls/season, ~105 s) and validate by game count (1230; 1059 for 2019-20 incl. bubble seeding games,
    1080 for 2020-21) + last game in April. Bump `VALIDATOR_VERSION` when validation tightens.
- `features.py`, `split.py`, `train_model.py` — model pipeline.
- `dashboard/`, `build_dashboard_data.py` — legacy, to be rebuilt on the last 2 seasons later.
