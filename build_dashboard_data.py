"""
build_dashboard_data.py
-----------------------
Packages walk-forward predictions + training history into one JSON payload for
dashboard/dashboard_template.html, then embeds it into a self-contained HTML file.

Only the last config.DASHBOARD_SEASONS walk-forward seasons are shown, and every
shot's xFG% is the model's prediction made BEFORE it trained on that shot.

Payload sections
  meta         seasons, split, lookups (players, teams, zones, actions, dates)
  training     coverage by season, 13-season league trends, weekly learning curve,
               calibration, feature importance, per-season metrics
  zones        14-zone court grid (1 ft cells -> zone index) for the zone map
  shots        every dashboard shot, columnar typed arrays, gzip + base64
  leaderboards player / team offense / team defense, per season scope, overall + by zone
  outliers     toughest makes and easiest misses (all shots and excluding heaves)

  python build_dashboard_data.py [--synthetic]
"""
import base64
import gzip
import json

import numpy as np
import pandas as pd

import config
from features import action_family, team_abbr
from split import season_split

ZONE14 = [
    ("Restricted Area", "Center(C)", "Restricted Area"),
    ("In The Paint (Non-RA)", "Left Side(L)", "Paint Left"),
    ("In The Paint (Non-RA)", "Center(C)", "Paint Center"),
    ("In The Paint (Non-RA)", "Right Side(R)", "Paint Right"),
    ("Mid-Range", "Left Side(L)", "Mid Left Baseline"),
    ("Mid-Range", "Left Side Center(LC)", "Mid Left Wing"),
    ("Mid-Range", "Center(C)", "Mid Center"),
    ("Mid-Range", "Right Side Center(RC)", "Mid Right Wing"),
    ("Mid-Range", "Right Side(R)", "Mid Right Baseline"),
    ("Left Corner 3", "Left Side(L)", "Left Corner 3"),
    ("Right Corner 3", "Right Side(R)", "Right Corner 3"),
    ("Above the Break 3", "Left Side Center(LC)", "Left Wing 3"),
    ("Above the Break 3", "Center(C)", "Top of Key 3"),
    ("Above the Break 3", "Right Side Center(RC)", "Right Wing 3"),
]
ZONE_NAMES = [z[2] for z in ZONE14] + ["Backcourt"]
ZONE_GROUPS = {  # coarse groups for leaderboard splits
    "Rim": ["Restricted Area"],
    "Paint": ["Paint Left", "Paint Center", "Paint Right"],
    "Mid-Range": ["Mid Left Baseline", "Mid Left Wing", "Mid Center", "Mid Right Wing", "Mid Right Baseline"],
    "Corner 3": ["Left Corner 3", "Right Corner 3"],
    "Above-Break 3": ["Left Wing 3", "Top of Key 3", "Right Wing 3", "Backcourt"],
}
GRID_X = (-25, 25)       # feet, 1 ft cells
GRID_Y = (-5, 40)
HEAVE_FT = 35            # heave = 35+ ft, or HEAVE_BUZZER_FT+ ft with <= 3 s left in the period;
HEAVE_BUZZER_FT = 28     # outlier lists are also offered without heaves
N_OUTLIERS = 50
MIN_ATTEMPTS_DEFAULT = 300
MIN_ATTEMPTS_ZONE_SPLIT = 100   # per-zone splits only for players with real volume (payload size)


# ---------------- helpers ----------------
_AREA_NEIGHBOUR = {"Left Side Center(LC)": "Left Side(L)", "Right Side Center(RC)": "Right Side(R)",
                   "Left Side(L)": "Left Side Center(LC)", "Right Side(R)": "Right Side Center(RC)"}


def zone_index(basic: pd.Series, area: pd.Series) -> np.ndarray:
    """(SHOT_ZONE_BASIC, SHOT_ZONE_AREA) -> index into ZONE_NAMES. Pairs outside the
    standard 14 fall back to the neighbouring area of the same basic zone."""
    lookup = {(b, a): i for i, (b, a, _) in enumerate(ZONE14)}
    first_of_basic = {}
    for i, (b, _, _) in enumerate(ZONE14):
        first_of_basic.setdefault(b, i)
    backcourt = len(ZONE14)
    pairs = pd.DataFrame({"b": basic.to_numpy(), "a": area.to_numpy()})
    mapping = {}
    for b, a in pairs.drop_duplicates().itertuples(index=False):
        if (b, a) in lookup:
            mapping[(b, a)] = lookup[(b, a)]
        elif b == "Backcourt" or a == "Back Court(BC)":
            mapping[(b, a)] = backcourt
        else:
            mapping[(b, a)] = lookup.get((b, _AREA_NEIGHBOUR.get(a)), first_of_basic.get(b, backcourt))
    idx = pd.MultiIndex.from_frame(pairs).map(mapping)
    return np.asarray(idx, dtype=np.uint8)


def _pack(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr, dtype=dtype).tobytes(), 6)).decode()


def unpack(s: str, dtype) -> np.ndarray:
    return np.frombuffer(gzip.decompress(base64.b64decode(s)), dtype=dtype)


def _rnd(df: pd.DataFrame, n=4) -> list:
    return json.loads(df.round(n).to_json(orient="records"))


def load_dashboard_shots(synthetic: bool):
    """Walk-forward predictions for the dashboard seasons, joined back to raw shot context."""
    paths = config.outputs(synthetic)
    preds = pd.read_parquet(paths["predictions"])
    online = sorted(preds["SEASON"].unique())
    seasons = online[-config.DASHBOARD_SEASONS:]
    preds = preds[preds["SEASON"].isin(seasons)]

    raw_cols = ["GAME_ID", "GAME_EVENT_ID", "SEASON", "TEAM_ID", "PERIOD", "MINUTES_REMAINING",
                "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_ZONE_AREA", "SHOT_DISTANCE", "HTM", "VTM"]
    raw = pd.read_parquet(paths["shots"], columns=raw_cols)
    raw = raw[raw["SEASON"].isin(seasons)]
    raw["team"] = team_abbr(raw)
    raw["opp"] = np.where(raw["team"] == raw["HTM"], raw["VTM"], raw["HTM"])
    shots = preds.merge(raw.drop(columns=["SEASON", "TEAM_ID"]), on=["GAME_ID", "GAME_EVENT_ID"],
                        how="left", validate="one_to_one")
    assert shots["team"].notna().all(), "predictions missing raw context"
    shots["zone"] = zone_index(shots["SHOT_ZONE_BASIC"], shots["SHOT_ZONE_AREA"])
    shots["zone_name"] = np.asarray(ZONE_NAMES)[shots["zone"]]
    shots["family"] = action_family(shots["ACTION_TYPE"])
    return shots.reset_index(drop=True), seasons, online


# ---------------- sections ----------------
def training_section(synthetic: bool, online_seasons) -> dict:
    paths = config.outputs(synthetic)
    raw = pd.read_parquet(paths["shots"], columns=["SEASON", "GAME_ID", "SHOT_MADE_FLAG", "SHOT_TYPE",
                                                   "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA"])
    seasons = sorted(raw["SEASON"].unique())
    train, stop, online = season_split(seasons, config.TRAIN_FRAC)
    role = {**{s: "train" for s in train}, stop: "early-stop", **{s: "walk-forward" for s in online}}
    dash = online_seasons[-config.DASHBOARD_SEASONS:]

    manifest = {}
    if not synthetic and (config.RAW_DIR / "manifest.json").exists():
        manifest = json.loads((config.RAW_DIR / "manifest.json").read_text())

    raw["zone_name"] = np.asarray(ZONE_NAMES)[zone_index(raw["SHOT_ZONE_BASIC"], raw["SHOT_ZONE_AREA"])]
    group_of = {z: g for g, zs in ZONE_GROUPS.items() for z in zs}
    raw["zone_group"] = raw["zone_name"].map(group_of)
    by_season = raw.groupby("SEASON")
    coverage = pd.DataFrame({
        "season": seasons,
        "shots": by_season.size().to_numpy(),
        "games": by_season["GAME_ID"].nunique().to_numpy(),
        "fg_pct": by_season["SHOT_MADE_FLAG"].mean().to_numpy(),
        "three_rate": by_season["SHOT_TYPE"].apply(lambda s: (s == "3PT Field Goal").mean()).to_numpy(),
        "dropped_no_location": [manifest.get(s, {}).get("dropped_no_location", 0) for s in seasons],
        "role": [role[s] for s in seasons],
        "on_dashboard": [s in dash for s in seasons],
    })
    zone_mix = (raw.groupby(["SEASON", "zone_group"]).agg(share=("SHOT_MADE_FLAG", "size"),
                                                         fg_pct=("SHOT_MADE_FLAG", "mean"))
                .reset_index())
    zone_mix["share"] = zone_mix["share"] / zone_mix.groupby("SEASON")["share"].transform("sum")

    log = pd.read_csv(paths["models"] / "online_log.csv", dtype={"chunk_min_date": str, "chunk_max_date": str})
    curve = log[["chunk", "season", "n", "chunk_min_date", "xfg_log_loss", "xfg_frozen_log_loss",
                 "xfg_zone_log_loss"]].copy()
    for c in ["xfg_log_loss", "xfg_frozen_log_loss", "xfg_zone_log_loss"]:
        # shot-weighted 4-week rolling mean, so tiny weeks don't dominate
        curve[c + "_roll"] = ((curve[c] * curve["n"]).rolling(4, min_periods=1).sum()
                              / curve["n"].rolling(4, min_periods=1).sum())

    metrics = json.loads((paths["models"] / "metrics.json").read_text())
    imp = pd.read_csv(paths["models"] / "feature_importance.csv").head(20)
    return {
        "coverage": _rnd(coverage),
        "zone_mix": _rnd(zone_mix),
        "zone_groups": list(ZONE_GROUPS),
        "learning_curve": _rnd(curve, 5),
        "metrics": metrics,
        "feature_importance": _rnd(imp, 2),
        "split": {"train": train, "stop": stop, "online": online, "dashboard": dash},
    }


def zone_grid(shots: pd.DataFrame) -> dict:
    """1-ft grid over the half court; each cell gets the zone most shots in it belong to,
    empty cells take their nearest labelled neighbour. Drawn as a heatmap in the browser."""
    xs = np.arange(*GRID_X)
    ys = np.arange(*GRID_Y)
    cx = np.floor(shots["LOC_X"] / 10.0).astype(int)
    cy = np.floor(shots["LOC_Y"] / 10.0).astype(int)
    inside = cx.between(GRID_X[0], GRID_X[1] - 1) & cy.between(GRID_Y[0], GRID_Y[1] - 1)
    cells = (pd.DataFrame({"cx": cx[inside], "cy": cy[inside], "zone": shots.loc[inside, "zone"]})
             .groupby(["cx", "cy"])["zone"].agg(lambda s: s.value_counts().idxmax()))
    grid = np.full((len(ys), len(xs)), -1, dtype=int)
    for (x, y), z in cells.items():
        grid[y - GRID_Y[0], x - GRID_X[0]] = z
    known = np.argwhere(grid >= 0)
    for iy, ix in np.argwhere(grid < 0):
        j = np.argmin((known[:, 0] - iy) ** 2 + (known[:, 1] - ix) ** 2)
        grid[iy, ix] = grid[tuple(known[j])]
    return {"x": xs.tolist(), "y": ys.tolist(), "cells": grid.tolist(), "names": ZONE_NAMES}


def shots_section(shots: pd.DataFrame, lookups: dict) -> dict:
    def codes(col, values):
        pos = {v: i for i, v in enumerate(values)}
        return shots[col].map(pos).to_numpy()

    clock = (shots["PERIOD"].clip(upper=7) * 1024
             + shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"]).to_numpy()
    cols = {
        "x": (shots["LOC_X"].to_numpy(), "<i2"),
        "y": (shots["LOC_Y"].to_numpy(), "<i2"),
        "xfg": (np.round(shots["xfg"].to_numpy() * 1000), "<u2"),       # permille
        "player": (codes("PLAYER_NAME", lookups["players"]), "<u2"),
        "date": (codes("GAME_DATE", lookups["dates"]), "<u2"),
        "clock": (clock, "<u2"),                                        # period*1024 + secs left
        "made": (shots["SHOT_MADE_FLAG"].to_numpy(), "u1"),
        "zone": (shots["zone"].to_numpy(), "u1"),
        "action": (codes("ACTION_TYPE", lookups["actions"]), "u1"),
        "family": (codes("family", lookups["families"]), "u1"),
        "team": (codes("team", lookups["teams"]), "u1"),
        "opp": (codes("opp", lookups["teams"]), "u1"),
        "season": (codes("SEASON", lookups["seasons"]), "u1"),
        "dist": (shots["SHOT_DISTANCE"].clip(upper=255).to_numpy(), "u1"),
    }
    return {"n": int(len(shots)),
            "columns": {k: {"dtype": dt, "data": _pack(a, dt)} for k, (a, dt) in cols.items()}}


def shot_value(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["SHOT_TYPE"] == "3PT Field Goal", 3, 2)


def _board(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Actual vs expected per group, in makes (xFG) and in points (xPts = xFG x shot value).

    Luck: each shot is a Bernoulli(xFG) trial, so under "no skill beyond the model" the
    over-expected total has variance sum(p(1-p)) for makes and sum(v^2 p(1-p)) for points.
    z = over-expected / sqrt(variance); sd_* is that sqrt, for the ±2σ funnel."""
    v = shot_value(df)
    p = df["xfg"].to_numpy()
    g = df.assign(var=p * (1 - p), pts=df["SHOT_MADE_FLAG"] * v, xpts=p * v,
                  var_pts=v ** 2 * p * (1 - p)).groupby(keys)
    out = g.agg(attempts=("SHOT_MADE_FLAG", "size"), makes=("SHOT_MADE_FLAG", "sum"),
                xmakes=("xfg", "sum"), var=("var", "sum"),
                pts=("pts", "sum"), xpts=("xpts", "sum"), var_pts=("var_pts", "sum"),
                three_rate=("SHOT_TYPE", lambda s: (s == "3PT Field Goal").mean()),
                avg_dist=("SHOT_DISTANCE", "mean")).reset_index()
    out["fg_pct"] = out["makes"] / out["attempts"]
    out["xfg_pct"] = out["xmakes"] / out["attempts"]
    out["fg_oe"] = out["fg_pct"] - out["xfg_pct"]
    out["makes_oe"] = out["makes"] - out["xmakes"]
    out["sd_makes"] = np.sqrt(out["var"])
    out["z"] = out["makes_oe"] / out["sd_makes"]
    out["pps"] = out["pts"] / out["attempts"]          # points per shot
    out["xpps"] = out["xpts"] / out["attempts"]        # expected points per shot (shot quality in points)
    out["pps_oe"] = out["pps"] - out["xpps"]
    out["pts_oe"] = out["pts"] - out["xpts"]
    out["sd_pts"] = np.sqrt(out["var_pts"])
    out["z_pts"] = out["pts_oe"] / out["sd_pts"]
    return out.drop(columns=["var", "var_pts", "xmakes", "xpts"])


def leaderboards(shots: pd.DataFrame, seasons) -> dict:
    group_of = {z: g for g, zs in ZONE_GROUPS.items() for z in zs}
    shots = shots.assign(zone_group=shots["zone_name"].map(group_of))
    scopes = {"All": shots, **{s: shots[shots["SEASON"] == s] for s in seasons}}
    player_team = (shots.groupby(["PLAYER_NAME", "team"]).size().reset_index(name="n")
                   .sort_values("n").groupby("PLAYER_NAME").tail(1).set_index("PLAYER_NAME")["team"])
    boards = {}
    for scope, df in scopes.items():
        players = _board(df, ["PLAYER_ID", "PLAYER_NAME"]).sort_values("makes_oe", ascending=False)
        players["team"] = players["PLAYER_NAME"].map(player_team)
        regulars = players.loc[players["attempts"] >= MIN_ATTEMPTS_ZONE_SPLIT, "PLAYER_NAME"]
        by_zone = _board(df[df["PLAYER_NAME"].isin(regulars)], ["PLAYER_NAME", "zone_group"])
        boards[scope] = {
            "players": _rnd(players),
            "players_by_zone": _rnd(by_zone),
            "team_offense": _rnd(_board(df, ["team"]).sort_values("makes_oe", ascending=False)),
            "team_defense": _rnd(_board(df, ["opp"]).rename(columns={"opp": "team"})
                                 .sort_values("makes_oe")),  # fewest makes allowed over expected first
            "league_zones": _rnd(_board(df, ["zone_name"])),
        }
    return {"scopes": list(scopes), "boards": boards, "min_attempts_default": MIN_ATTEMPTS_DEFAULT}


def outliers(shots: pd.DataFrame) -> dict:
    cols = ["PLAYER_NAME", "team", "opp", "GAME_DATE", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
            "ACTION_TYPE", "zone_name", "SHOT_DISTANCE", "xfg", "value", "xpts", "pts_oe", "SHOT_MADE_FLAG"]
    shots = shots.assign(value=shot_value(shots))
    shots["xpts"] = shots["xfg"] * shots["value"]
    shots["pts_oe"] = shots["SHOT_MADE_FLAG"] * shots["value"] - shots["xpts"]
    made, missed = shots[shots["SHOT_MADE_FLAG"] == 1], shots[shots["SHOT_MADE_FLAG"] == 0]
    buzzer = shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"] <= 3
    heave = (shots["SHOT_DISTANCE"] >= HEAVE_FT) | (buzzer & (shots["SHOT_DISTANCE"] >= HEAVE_BUZZER_FT))
    near = lambda d: d[~heave.loc[d.index]]
    return {
        "heave_ft": HEAVE_FT,
        # xFG: least likely makes / most likely misses
        "toughest_makes": _rnd(made.nsmallest(N_OUTLIERS, "xfg")[cols]),
        "toughest_makes_no_heaves": _rnd(near(made).nsmallest(N_OUTLIERS, "xfg")[cols]),
        "easiest_misses": _rnd(missed.nlargest(N_OUTLIERS, "xfg")[cols]),
        # xPts: most points gained over expected on one make / most expected points thrown away on one miss
        "biggest_gains": _rnd(made.nlargest(N_OUTLIERS, "pts_oe")[cols]),
        "biggest_gains_no_heaves": _rnd(near(made).nlargest(N_OUTLIERS, "pts_oe")[cols]),
        "biggest_losses": _rnd(missed.nsmallest(N_OUTLIERS, "pts_oe")[cols]),
    }


def build_payload(synthetic: bool) -> dict:
    shots, seasons, online = load_dashboard_shots(synthetic)
    lookups = {
        "players": sorted(shots["PLAYER_NAME"].unique()),
        "teams": sorted(set(shots["team"]) | set(shots["opp"])),
        "zones": ZONE_NAMES,
        "actions": sorted(shots["ACTION_TYPE"].unique()),
        "families": sorted(shots["family"].unique()),
        "dates": sorted(shots["GAME_DATE"].unique()),
        "seasons": seasons,
    }
    return {
        "meta": {"synthetic": synthetic, "seasons": seasons, "lookups": lookups,
                 "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")},
        "training": training_section(synthetic, online),
        "zones": zone_grid(shots),
        "shots": shots_section(shots, lookups),
        "leaderboards": leaderboards(shots, seasons),
        "outliers": outliers(shots),
    }


PLACEHOLDER = "__DASHBOARD_DATA__"


def embed(payload_json: str, out_path) -> bool:
    """Inject the payload into the template. Returns False if the template has no placeholder."""
    tmpl = config.DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in tmpl:
        return False
    # the payload sits inside a <script type="application/json"> tag; keep "</" from closing it
    out_path.write_text(tmpl.replace(PLACEHOLDER, payload_json.replace("</", "<\\/")), encoding="utf-8")
    return True


def main(synthetic: bool) -> None:
    paths = config.outputs(synthetic)
    payload = build_payload(synthetic)
    text = json.dumps(payload, separators=(",", ":"))
    paths["dashboard_data"].write_text(text, encoding="utf-8")
    print(f"dashboard payload: {payload['shots']['n']:,} shots, seasons {payload['meta']['seasons']}, "
          f"{len(text) / 1e6:.1f} MB -> {paths['dashboard_data'].relative_to(config.ROOT)}")
    if embed(text, paths["dashboard_html"]):
        print(f"embedded -> {paths['dashboard_html'].relative_to(config.ROOT)}")
    else:
        print("template has no data placeholder yet; skipped embed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
