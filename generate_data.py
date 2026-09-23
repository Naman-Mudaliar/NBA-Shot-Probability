"""
generate_data.py
----------------
Fast-loop synthetic shot log, in exactly the nba_api ShotChartDetail schema
(config.SHOT_SCHEMA), spanning config.SYNTHETIC_SEASONS with real-looking
GAME_DATEs. Used by `run_pipeline.py --synthetic` to catch wiring bugs and
run the leakage tests in seconds without touching the NBA API.

Planted ground truth (for tests only, never model inputs):
  - per-player shooting skill      -> data/synthetic/players_truth.parquet
  - per-team defensive effect      (opponent adjustment on make probability)
  - league-wide 3pt drift by season
"""
import numpy as np
import pandas as pd

import config

SHOTS_PER_TEAM_GAME = 28
GAMES_PER_SEASON = 320
PLAYERS_PER_TEAM = 5

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# zone: (SHOT_ZONE_BASIC, min_ft, max_ft, base_fg, is_three, weight)
ZONES = [
    ("Restricted Area",        0.0,  4.0, 0.60, False, 0.30),
    ("In The Paint (Non-RA)",  4.0, 14.0, 0.42, False, 0.13),
    ("Mid-Range",             10.0, 22.0, 0.41, False, 0.14),
    ("Left Corner 3",         22.0, 23.5, 0.39, True,  0.05),
    ("Right Corner 3",        22.0, 23.5, 0.39, True,  0.05),
    ("Above the Break 3",     23.8, 30.0, 0.355, True, 0.32),
    ("Backcourt",             40.0, 70.0, 0.03, True,  0.01),
]

# zone -> [(ACTION_TYPE, weight, make-prob effect)]
ACTIONS = {
    "Restricted Area": [
        ("Layup Shot", 0.25, 0.0), ("Driving Layup Shot", 0.25, -0.05),
        ("Dunk Shot", 0.12, 0.28), ("Driving Dunk Shot", 0.08, 0.25),
        ("Alley Oop Dunk Shot", 0.05, 0.25), ("Tip Layup Shot", 0.07, -0.08),
        ("Cutting Layup Shot", 0.08, 0.08), ("Putback Layup Shot", 0.10, 0.02),
    ],
    "In The Paint (Non-RA)": [
        ("Floating Jump shot", 0.30, 0.0), ("Driving Floating Jump Shot", 0.20, -0.02),
        ("Hook Shot", 0.25, 0.04), ("Turnaround Jump Shot", 0.15, -0.02),
        ("Jump Shot", 0.10, 0.0),
    ],
    "Mid-Range": [
        ("Jump Shot", 0.35, 0.03), ("Pullup Jump shot", 0.35, -0.02),
        ("Turnaround Jump Shot", 0.10, -0.03), ("Fadeaway Jump Shot", 0.10, -0.05),
        ("Step Back Jump shot", 0.10, -0.03),
    ],
    "three": [
        ("Jump Shot", 0.60, 0.03), ("Pullup Jump shot", 0.25, -0.03),
        ("Step Back Jump shot", 0.15, -0.04),
    ],
    "Backcourt": [("Jump Shot", 1.0, 0.0)],
}


def _zone_area(angle_deg: np.ndarray, zone: np.ndarray) -> np.ndarray:
    area = np.select(
        [angle_deg < -54, angle_deg < -18, angle_deg <= 18, angle_deg <= 54],
        ["Left Side(L)", "Left Side Center(LC)", "Center(C)", "Right Side Center(RC)"],
        default="Right Side(R)",
    )
    area = np.where(zone == "Restricted Area", "Center(C)", area)
    return np.where(zone == "Backcourt", "Back Court(BC)", area)


def _zone_range(dist: np.ndarray, zone: np.ndarray) -> np.ndarray:
    rng = np.select(
        [dist < 8, dist < 16, dist < 24],
        ["Less Than 8 ft.", "8-16 ft.", "16-24 ft."],
        default="24+ ft.",
    )
    return np.where(zone == "Backcourt", "Back Court Shot", rng)


def generate(seasons=None, seed: int = 42, games_per_season: int = GAMES_PER_SEASON):
    """Return (shots_df in SHOT_SCHEMA, players_truth_df)."""
    seasons = seasons or config.SYNTHETIC_SEASONS
    rng = np.random.default_rng(seed)

    team_ids = np.arange(1610612737, 1610612737 + len(TEAMS))
    team_def = rng.normal(0.0, 0.03, len(TEAMS))  # + = leakier defense

    n_players = len(TEAMS) * PLAYERS_PER_TEAM
    players = pd.DataFrame({
        "PLAYER_ID": np.arange(200001, 200001 + n_players),
        "PLAYER_NAME": [f"Player {i:03d}" for i in range(n_players)],
        "team_idx": np.repeat(np.arange(len(TEAMS)), PLAYERS_PER_TEAM),
        "skill": rng.normal(0.0, 0.045, n_players).clip(-0.10, 0.12),
    })

    zone_w = np.array([z[5] for z in ZONES])
    zone_w = zone_w / zone_w.sum()

    frames = []
    for s_idx, season in enumerate(seasons):
        start_year = int(season[:4])
        season_start = pd.Timestamp(f"{start_year}-10-20")
        day_offsets = np.sort(rng.integers(0, 170, games_per_season))
        dates = season_start + pd.to_timedelta(day_offsets, unit="D")
        home = rng.integers(0, len(TEAMS), games_per_season)
        away = (home + rng.integers(1, len(TEAMS), games_per_season)) % len(TEAMS)

        # one row per shot: each game has SHOTS_PER_TEAM_GAME shots per side
        n = games_per_season * 2 * SHOTS_PER_TEAM_GAME
        g = np.repeat(np.arange(games_per_season), 2 * SHOTS_PER_TEAM_GAME)
        side_home = np.tile(np.repeat([1, 0], SHOTS_PER_TEAM_GAME), games_per_season)
        off_team = np.where(side_home == 1, home[g], away[g])
        def_team = np.where(side_home == 1, away[g], home[g])
        shooter = off_team * PLAYERS_PER_TEAM + rng.integers(0, PLAYERS_PER_TEAM, n)

        zone_idx = rng.choice(len(ZONES), size=n, p=zone_w)
        zname = np.array([ZONES[i][0] for i in zone_idx])
        dmin = np.array([ZONES[i][1] for i in zone_idx])
        dmax = np.array([ZONES[i][2] for i in zone_idx])
        base = np.array([ZONES[i][3] for i in zone_idx])
        is_three = np.array([ZONES[i][4] for i in zone_idx])
        dist = rng.uniform(dmin, dmax)

        angle = rng.uniform(-75, 75, n)
        angle = np.where(zname == "Left Corner 3", rng.uniform(-88, -80, n), angle)
        angle = np.where(zname == "Right Corner 3", rng.uniform(80, 88, n), angle)
        loc_x = np.round(dist * np.sin(np.deg2rad(angle)) * 10).astype(int)
        loc_y = np.round(dist * np.cos(np.deg2rad(angle)) * 10).astype(int)

        action = np.empty(n, dtype=object)
        action_eff = np.zeros(n)
        for z in np.unique(zname):
            key = "three" if z in ("Left Corner 3", "Right Corner 3", "Above the Break 3") else z
            opts = ACTIONS[key]
            mask = zname == z
            w = np.array([o[1] for o in opts])
            pick = rng.choice(len(opts), size=mask.sum(), p=w / w.sum())
            action[mask] = [opts[i][0] for i in pick]
            action_eff[mask] = [opts[i][2] for i in pick]

        period = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.245, 0.245, 0.245, 0.245, 0.02])
        mins = rng.integers(0, 12, n)
        mins = np.where(period == 5, rng.integers(0, 5, n), mins)
        secs = rng.integers(0, 60, n)

        p = (
            base
            + action_eff
            + players["skill"].to_numpy()[shooter]
            + team_def[def_team]
            + np.where(is_three, 0.004 * s_idx, 0.0)       # league 3pt drift
            + np.where(side_home == 1, 0.005, 0.0)         # small home edge
            - np.where((mins == 0) & (secs < 3), 0.05, 0.0)  # end-of-quarter rush
            + rng.normal(0, 0.01, n)
        )
        p = np.clip(p, 0.02, 0.95)
        made = (rng.random(n) < p).astype(int)

        yy = str(start_year)[-2:]
        game_ids = np.array([f"002{yy}{i + 1:05d}" for i in range(games_per_season)])
        frames.append(pd.DataFrame({
            "GRID_TYPE": "Shot Chart Detail",
            "GAME_ID": game_ids[g],
            "GAME_EVENT_ID": np.tile(np.arange(1, 2 * SHOTS_PER_TEAM_GAME + 1), games_per_season) * 7,
            "PLAYER_ID": players["PLAYER_ID"].to_numpy()[shooter],
            "PLAYER_NAME": players["PLAYER_NAME"].to_numpy()[shooter],
            "TEAM_ID": team_ids[off_team],
            "TEAM_NAME": np.array([f"Team {t}" for t in TEAMS])[off_team],
            "PERIOD": period,
            "MINUTES_REMAINING": mins,
            "SECONDS_REMAINING": secs,
            "EVENT_TYPE": np.where(made == 1, "Made Shot", "Missed Shot"),
            "ACTION_TYPE": action.astype(str),
            "SHOT_TYPE": np.where(is_three, "3PT Field Goal", "2PT Field Goal"),
            "SHOT_ZONE_BASIC": zname,
            "SHOT_ZONE_AREA": _zone_area(angle, zname),
            "SHOT_ZONE_RANGE": _zone_range(dist, zname),
            "SHOT_DISTANCE": np.round(dist).astype(int),
            "LOC_X": loc_x,
            "LOC_Y": loc_y,
            "SHOT_ATTEMPTED_FLAG": 1,
            "SHOT_MADE_FLAG": made,
            "GAME_DATE": dates[g].strftime("%Y%m%d"),
            "HTM": np.array(TEAMS)[home[g]],
            "VTM": np.array(TEAMS)[away[g]],
            "SEASON": season,
        }))

    shots = pd.concat(frames, ignore_index=True)[list(config.SHOT_SCHEMA)]
    truth = players[["PLAYER_ID", "skill"]].copy()
    return shots, truth


if __name__ == "__main__":
    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    shots, truth = generate()
    shots.to_parquet(config.SYNTHETIC_SHOTS, index=False)
    truth.to_parquet(config.SYNTHETIC_TRUTH, index=False)
    print(f"Wrote {len(shots):,} synthetic shots over {shots['SEASON'].nunique()} seasons "
          f"-> {config.SYNTHETIC_SHOTS.relative_to(config.ROOT)}")
    print(shots.groupby("SEASON")["SHOT_MADE_FLAG"].agg(["size", "mean"]).round(3).to_string())
