"""
config.py
---------
Single source of truth for paths, season split, and the raw shot schema.
Every other module imports from here -- no hardcoded paths anywhere else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # per-season nba_api pulls + manifest
SYNTHETIC_DIR = DATA_DIR / "synthetic"     # fast-loop generated data
MODELS_DIR = ROOT / "models"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

REAL_SHOTS = DATA_DIR / "shots_raw.parquet"
SYNTHETIC_SHOTS = SYNTHETIC_DIR / "shots_raw.parquet"
SYNTHETIC_TRUTH = SYNTHETIC_DIR / "players_truth.parquet"
REAL_SAMPLE = FIXTURES_DIR / "real_sample.parquet"



def outputs(synthetic: bool) -> dict:
    """Per-mode artifact paths, so synthetic runs never clobber real ones."""
    data = SYNTHETIC_DIR if synthetic else DATA_DIR
    models = MODELS_DIR / "synthetic" if synthetic else MODELS_DIR
    return {
        "shots": SYNTHETIC_SHOTS if synthetic else REAL_SHOTS,
        "features": data / "shots_features.parquet",
        "predictions": data / "predictions.parquet",
        "models": models,
        "dashboard_data": data / "dashboard_data.json",
        # synthetic build lives with the synthetic data so dashboard/ only ever holds the real one
        "dashboard_html": (SYNTHETIC_DIR / "dashboard_synthetic.html") if synthetic
                          else (DASHBOARD_DIR / "dashboard.html"),
    }


# ---- seasons / split ----
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]  # 2013-14 .. 2025-26
SYNTHETIC_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2018, 2024)]  # 6 seasons, fast
TRAIN_FRAC = 0.7        # first 70% of seasons train (last of those = early-stopping season)
ONLINE_CHUNK = "W"      # walk-forward update cadence over held-out seasons
DASHBOARD_SEASONS = 2   # dashboard shows the last N walk-forward seasons

DASHBOARD_DIR = ROOT / "dashboard"
DASHBOARD_TEMPLATE = DASHBOARD_DIR / "dashboard_template.html"

# ---- raw schema: exactly what nba_api ShotChartDetail returns, plus SEASON ----
# kind: "int" | "str"
SHOT_SCHEMA = {
    "GRID_TYPE": "str",
    "GAME_ID": "str",
    "GAME_EVENT_ID": "int",
    "PLAYER_ID": "int",
    "PLAYER_NAME": "str",
    "TEAM_ID": "int",
    "TEAM_NAME": "str",
    "PERIOD": "int",
    "MINUTES_REMAINING": "int",
    "SECONDS_REMAINING": "int",
    "EVENT_TYPE": "str",
    "ACTION_TYPE": "str",
    "SHOT_TYPE": "str",
    "SHOT_ZONE_BASIC": "str",
    "SHOT_ZONE_AREA": "str",
    "SHOT_ZONE_RANGE": "str",
    "SHOT_DISTANCE": "int",
    "LOC_X": "int",          # tenths of a foot, hoop at (0, 0)
    "LOC_Y": "int",
    "SHOT_ATTEMPTED_FLAG": "int",
    "SHOT_MADE_FLAG": "int",
    "GAME_DATE": "str",      # YYYYMMDD
    "HTM": "str",
    "VTM": "str",
    "SEASON": "str",         # added by fetch/generate, e.g. "2023-24"
}

# ---- leakage guard: never allowed as (or as the source of) model inputs ----
# The xFG% model is shooter-blind: probability an average player makes this shot.
BLOCKED_FEATURES = {
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "team",
    "position", "player_id", "player_name",
    "GAME_ID", "GAME_EVENT_ID", "SHOT_MADE_FLAG", "EVENT_TYPE",
}
# any feature whose name contains one of these is treated as shooter history
BLOCKED_SUBSTRINGS = ("player", "shooter")
