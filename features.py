"""
features.py
-----------
Raw ShotChartDetail rows -> shooter-blind feature matrix for the xFG% model.

Model inputs describe only the SHOT and the DEFENSE, never the shooter:
  location   distance, angle, zone / side one-hots, 3pt, corner 3
  shot type  action family (dunk, layup, floater, pullup, step-back, ...) + drive/cut flags
  situation  period, OT, seconds left, end-of-period, home/away
  defense    opponent's FG% allowed in this zone group, season-to-date, using only
             games strictly before this shot's date, shrunk toward the league rate
  era        league FG% in this zone group over the trailing window (past dates only)

Shooter identity columns are carried through as metadata for reporting, but are
never in MODEL_FEATURES (enforced here and in tests/test_leakage.py).
"""
import numpy as np
import pandas as pd

import config

ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3",
         "Right Corner 3", "Above the Break 3", "Backcourt"]
AREAS = ["Center(C)", "Left Side(L)", "Right Side(R)", "Left Side Center(LC)",
         "Right Side Center(RC)", "Back Court(BC)"]
ACTION_FAMILIES = ["dunk", "putback", "layup", "hook", "floater", "jump", "pullup",
                   "stepback", "fadeaway", "turnaround", "other"]
ZONE_GROUP = {
    "Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
    "Left Corner 3": "corner3", "Right Corner 3": "corner3",
    "Above the Break 3": "atb3", "Backcourt": "atb3",
}

DEF_PRIOR_SHOTS = 150     # pseudo-count shrinking team defense toward the league rate
LEAGUE_WINDOW_DAYS = 60   # trailing distinct game-dates for league zone rates
LEAGUE_FALLBACK = 0.46    # only used before any history exists

META_COLS = ["GAME_ID", "GAME_EVENT_ID", "GAME_DATE", "SEASON", "PLAYER_ID", "PLAYER_NAME",
             "TEAM_ID", "SHOT_ZONE_BASIC", "SHOT_TYPE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]

MODEL_FEATURES = (
    ["shot_distance", "loc_dist_ft", "angle", "abs_angle", "is_three", "corner_three"]
    + [f"zone_{z}" for z in ZONES]
    + [f"area_{a}" for a in AREAS]
    + [f"action_{a}" for a in ACTION_FAMILIES]
    + ["is_driving", "is_running", "is_cutting", "is_alley_oop"]
    + ["period", "is_ot", "seconds_left", "last_24s", "last_3s", "home"]
    + ["league_zone_fg", "opp_def_fg", "opp_def_delta", "opp_def_attempts"]
)


def _check_blind(cols):
    bad = [c for c in cols if c in config.BLOCKED_FEATURES
           or any(s in c.lower() for s in config.BLOCKED_SUBSTRINGS)]
    assert not bad, f"shooter-identifying columns in model features: {bad}"


_check_blind(MODEL_FEATURES)


def action_family(action: pd.Series) -> pd.Series:
    a = action.str.lower()
    rules = [
        ("dunk", a.str.contains("dunk")),
        ("putback", a.str.contains("tip|putback")),
        ("hook", a.str.contains("hook")),
        ("floater", a.str.contains("float")),
        ("layup", a.str.contains("layup|finger roll")),
        ("stepback", a.str.contains("step back|step-back")),
        ("fadeaway", a.str.contains("fadeaway")),
        ("turnaround", a.str.contains("turnaround")),
        ("pullup", a.str.contains("pullup|pull-up")),
        ("jump", a.str.contains("jump shot|jumper|bank shot")),
    ]
    return pd.Series(np.select([m for _, m in rules], [n for n, _ in rules], default="other"),
                     index=action.index)


def team_abbr(df: pd.DataFrame) -> pd.Series:
    """Shooter team's abbreviation per row: for each (TEAM_ID, SEASON), the HTM/VTM
    code that appears in every one of its games."""
    both = pd.concat([
        df[["TEAM_ID", "SEASON", "HTM"]].rename(columns={"HTM": "abbr"}),
        df[["TEAM_ID", "SEASON", "VTM"]].rename(columns={"VTM": "abbr"}),
    ])
    mode = (both.groupby(["TEAM_ID", "SEASON"])["abbr"]
            .agg(lambda s: s.value_counts().idxmax()).rename("team_abbr"))
    return df[["TEAM_ID", "SEASON"]].join(mode, on=["TEAM_ID", "SEASON"])["team_abbr"]


def _past_only_rates(df: pd.DataFrame) -> pd.DataFrame:
    """League zone-group rate (trailing window) and opponent defense (season-to-date),
    both computed from dates strictly before each shot's date."""
    date = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d")

    # league: per (zone group, date) totals -> trailing window, shifted one date
    lg = (df.assign(date=date).groupby(["zone_group", "date"])["SHOT_MADE_FLAG"]
          .agg(makes="sum", att="size").reset_index().sort_values(["zone_group", "date"]))
    roll = lg.groupby("zone_group")[["makes", "att"]].transform(
        lambda s: s.shift(1).rolling(LEAGUE_WINDOW_DAYS, min_periods=1).sum())
    lg["league_zone_fg"] = (roll["makes"] / roll["att"]).fillna(LEAGUE_FALLBACK)

    # opponent defense: per (season, opp, zone group, date) -> season-to-date cumsum, shifted
    od = (df.assign(date=date).groupby(["SEASON", "opp_abbr", "zone_group", "date"])["SHOT_MADE_FLAG"]
          .agg(makes="sum", att="size").reset_index()
          .sort_values(["SEASON", "opp_abbr", "zone_group", "date"]))
    g = od.groupby(["SEASON", "opp_abbr", "zone_group"])
    od["def_makes"] = g["makes"].cumsum() - od["makes"]
    od["def_att"] = g["att"].cumsum() - od["att"]

    out = (df[["SEASON", "opp_abbr", "zone_group"]].assign(date=date)
           .merge(lg[["zone_group", "date", "league_zone_fg"]], on=["zone_group", "date"], how="left")
           .merge(od[["SEASON", "opp_abbr", "zone_group", "date", "def_makes", "def_att"]],
                  on=["SEASON", "opp_abbr", "zone_group", "date"], how="left"))
    out["opp_def_fg"] = ((out["def_makes"] + DEF_PRIOR_SHOTS * out["league_zone_fg"])
                         / (out["def_att"] + DEF_PRIOR_SHOTS))
    out["opp_def_delta"] = out["opp_def_fg"] - out["league_zone_fg"]
    out["opp_def_attempts"] = out["def_att"]
    out.index = df.index
    return out[["league_zone_fg", "opp_def_fg", "opp_def_delta", "opp_def_attempts"]]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return META_COLS + MODEL_FEATURES, one row per input row, in input order."""
    df = raw.reset_index(drop=True).copy()
    out = df[META_COLS].copy()

    # --- location ---
    x, y = df["LOC_X"] / 10.0, df["LOC_Y"] / 10.0
    out["shot_distance"] = df["SHOT_DISTANCE"].astype(float)
    out["loc_dist_ft"] = np.hypot(x, y)
    out["angle"] = np.degrees(np.arctan2(x, y.clip(lower=0.1)))
    out["abs_angle"] = out["angle"].abs()
    out["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)
    out["corner_three"] = df["SHOT_ZONE_BASIC"].str.contains("Corner 3").astype(int)
    for z in ZONES:
        out[f"zone_{z}"] = (df["SHOT_ZONE_BASIC"] == z).astype(int)
    for a in AREAS:
        out[f"area_{a}"] = (df["SHOT_ZONE_AREA"] == a).astype(int)

    # --- shot type ---
    fam = action_family(df["ACTION_TYPE"])
    for a in ACTION_FAMILIES:
        out[f"action_{a}"] = (fam == a).astype(int)
    act = df["ACTION_TYPE"].str.lower()
    out["is_driving"] = act.str.contains("driving").astype(int)
    out["is_running"] = act.str.contains("running").astype(int)
    out["is_cutting"] = act.str.contains("cutting").astype(int)
    out["is_alley_oop"] = act.str.contains("alley oop").astype(int)

    # --- situation ---
    out["period"] = df["PERIOD"].clip(upper=5)
    out["is_ot"] = (df["PERIOD"] > 4).astype(int)
    out["seconds_left"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    out["last_24s"] = (out["seconds_left"] <= 24).astype(int)
    out["last_3s"] = (out["seconds_left"] <= 3).astype(int)
    team = team_abbr(df)
    out["home"] = (team == df["HTM"]).astype(int)

    # --- defense + era (past-only) ---
    df["opp_abbr"] = np.where(out["home"] == 1, df["VTM"], df["HTM"])
    df["zone_group"] = df["SHOT_ZONE_BASIC"].map(ZONE_GROUP).fillna("atb3")
    out = pd.concat([out, _past_only_rates(df)], axis=1)

    _check_blind(MODEL_FEATURES)
    return out[META_COLS + MODEL_FEATURES]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    paths = config.outputs(args.synthetic)
    raw = pd.read_parquet(paths["shots"])
    feats = build_features(raw)
    feats.to_parquet(paths["features"], index=False)
    print(f"Wrote {len(feats):,} rows x {len(MODEL_FEATURES)} model features "
          f"-> {paths['features'].relative_to(config.ROOT)}")
