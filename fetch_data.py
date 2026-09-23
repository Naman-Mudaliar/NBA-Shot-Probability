"""
fetch_data.py
-------------
Pulls real regular-season shot logs (every FGA, league-wide) from stats.nba.com
(the shotchartdetail endpoint), one season at a time, and checkpoints aggressively:

  - each season is written to shots_{season}.parquet.tmp, validated against
    config.SHOT_SCHEMA (+ row-count / duplicate / season sanity checks), then
    atomically renamed into place;
  - data/raw/manifest.json records rows + sha256 per season; a season is skipped
    only if its file exists AND matches the manifest, so a killed run never
    leaves a half-written file that a later run trusts;
  - stray .tmp files are deleted at startup.

Afterwards all seasons are combined into data/shots_raw.parquet and a 2k-row
sample is written to tests/fixtures/real_sample.parquet for the schema tests.

  python fetch_data.py                      # all config.SEASONS
  python fetch_data.py --seasons 2025-26    # subset
  python fetch_data.py --combine-only       # rebuild combined file from checkpoints

Slow loop only: run when `python run_pipeline.py --synthetic` is green.

Requests go through the system `curl`, not python-requests/nba_api: stats.nba.com's
CDN fingerprints clients and silently stalls python-requests (observed: read
timeouts), while curl with the same headers gets 200s in ~1s.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from urllib.parse import urlencode

import pandas as pd

import config
from schema import coerce, schema_errors

MANIFEST = config.RAW_DIR / "manifest.json"
# Bump when validation gets stricter: older manifest entries are then refetched.
# v2: league-wide calls silently cap at 102,400 rows (observed 2025-26: 573 of 1230
#     games), so fetch per team and validate completeness by game count.
VALIDATOR_VERSION = 2
REGULAR_SEASON_GAMES = {"2019-20": 1059, "2020-21": 1080}  # COVID seasons (2019-20 incl. 88 bubble seeding games); else 1230
MIN_GAME_FRAC = 0.99
# a handful of shots per season come back with no location at all (2013-14: 46 of
# 204k); a location model can't score them, so they are dropped and counted. More
# than this fraction means something is actually wrong.
MAX_NO_LOCATION_FRAC = 0.005
LOCATION_COLS = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_ZONE_BASIC"]
SLEEP_S = 1.5
RETRIES = 3
TIMEOUT_S = 180
URL = "https://stats.nba.com/stats/shotchartdetail"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    # without these two, stats.nba.com silently drops the connection (observed: hangs to timeout)
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}


# ---------------- checkpoint helpers ----------------
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, MANIFEST)


def season_path(season: str):
    return config.RAW_DIR / f"shots_{season}.parquet"


def is_complete(season: str, manifest: dict) -> bool:
    path, entry = season_path(season), manifest.get(season)
    if not path.exists() or not entry or entry.get("validator") != VALIDATOR_VERSION:
        return False
    return entry.get("sha256") == _sha256(path)


def clean_stray_tmp() -> None:
    for tmp in config.RAW_DIR.rglob("*.tmp"):
        print(f"removing stray partial file {tmp.name}")
        tmp.unlink()


def atomic_write_parquet(df: pd.DataFrame, path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------- API ----------------
def _params(season: str, team_id: int) -> dict:
    blank = ["AheadBehind", "ClutchTime", "ContextFilter", "DateFrom", "DateTo", "EndPeriod",
             "EndRange", "GameID", "GameSegment", "Location", "Outcome", "PlayerPosition",
             "PointDiff", "Position", "RangeType", "RookieYear", "SeasonSegment", "StartPeriod",
             "StartRange", "VsConference", "VsDivision"]
    return {**{k: "" for k in blank},
            "ContextMeasure": "FGA", "LastNGames": 0, "LeagueID": "00", "Month": 0,
            "OpponentTeamID": 0, "Period": 0, "PlayerID": 0, "Season": season,
            "SeasonType": "Regular Season", "TeamID": team_id}


def _get(season: str, team_id: int) -> pd.DataFrame:
    cmd = ["curl", "-s", "--compressed", "-m", str(TIMEOUT_S), "-w", "\n%{http_code}"]
    for k, v in HEADERS.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd.append(f"{URL}?{urlencode(_params(season, team_id))}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    body, _, code = res.stdout.rpartition("\n")
    if res.returncode != 0:
        raise RuntimeError(f"curl exit {res.returncode} (28 = timeout)")
    if code != "200" or not body.strip():
        raise RuntimeError(f"HTTP {code}, {len(body)} bytes")
    rs = json.loads(body)["resultSets"][0]   # Shot_Chart_Detail
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def _call(season: str, team_id: int = 0) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            return _get(season, team_id)
        except Exception as e:  # timeout / empty body / throttling -- retry with backoff
            last_err = e
            wait = SLEEP_S * 4 ** attempt
            print(f"  {season} team={team_id}: attempt {attempt} failed ({e}); retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{season} team={team_id}: all {RETRIES} attempts failed") from last_err


def fetch_season(season: str) -> pd.DataFrame:
    """One call per team. (League-wide calls truncate at 102,400 rows.)"""
    from nba_api.stats.static import teams  # offline team list, no network

    team_dir = config.RAW_DIR / "teams" / season
    team_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for t in teams.get_teams():
        # raw per-team responses are cached as-is (atomic), so any failure after
        # download -- coercion, validation, a crash -- never costs another API call
        cache = team_dir / f"{t['abbreviation']}.parquet"
        if cache.exists():
            parts.append(pd.read_parquet(cache))
            continue
        df = _call(season, team_id=t["id"])
        atomic_write_parquet(df, cache)
        parts.append(df)
        print(f"    {t['abbreviation']}: {len(df):,}", flush=True)
        time.sleep(SLEEP_S)
    return pd.concat(parts, ignore_index=True)


def expected_games(season: str) -> int:
    return REGULAR_SEASON_GAMES.get(season, 1230)


def completeness_errors(df: pd.DataFrame, season: str) -> list[str]:
    errors = []
    n_games, want = df["GAME_ID"].nunique(), expected_games(season)
    if n_games < MIN_GAME_FRAC * want:
        errors.append(f"only {n_games} of {want} regular-season games (truncated response?)")
    end_year = int(season[:4]) + 1
    if df["GAME_DATE"].max() < f"{end_year}0401":
        errors.append(f"last game {df['GAME_DATE'].max()} is before April {end_year}")
    teams_per_game = df.groupby("GAME_ID")["TEAM_ID"].nunique()
    if (teams_per_game < 2).mean() > 0.01:
        errors.append(f"{int((teams_per_game < 2).sum())} games with shots from only one team")
    return errors


def validate_season(df: pd.DataFrame, season: str) -> list[str]:
    errors = schema_errors(df)
    if errors:
        return errors
    errors += completeness_errors(df, season)
    dups = df.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum()
    if dups:
        errors.append(f"{dups} duplicate (GAME_ID, GAME_EVENT_ID) rows")
    yy = season[2:4]
    wrong = (df["GAME_ID"].str[3:5] != yy).sum()
    if wrong:
        errors.append(f"{wrong} rows whose GAME_ID is not from season {season}")
    if (df["SHOT_ATTEMPTED_FLAG"] != 1).any():
        errors.append("rows with SHOT_ATTEMPTED_FLAG != 1")
    return errors


def pull(seasons: list[str]) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    clean_stray_tmp()
    manifest = load_manifest()
    for i, season in enumerate(seasons):
        if is_complete(season, manifest):
            print(f"{season}: checkpoint OK ({manifest[season]['rows']:,} rows), skipping")
            continue
        print(f"{season}: fetching ...", flush=True)
        t0 = time.time()
        df = fetch_season(season)
        df["SEASON"] = season
        no_loc = df[LOCATION_COLS].isna().any(axis=1)
        if no_loc.mean() > MAX_NO_LOCATION_FRAC:
            raise SystemExit(f"{season}: {int(no_loc.sum())} shots ({no_loc.mean():.2%}) have no location")
        df = df[~no_loc]
        df = coerce(df[[c for c in config.SHOT_SCHEMA if c in df.columns]])
        # each shot is returned once (by the shooter's team); dedupe defensively
        df = df.drop_duplicates(["GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)
        errors = validate_season(df, season)
        if errors:
            raise SystemExit(f"{season}: validation failed, not saving:\n  " + "\n  ".join(errors))

        path = season_path(season)
        atomic_write_parquet(df, path)
        manifest[season] = {
            "rows": int(len(df)),
            "fg_pct": round(float(df["SHOT_MADE_FLAG"].mean()), 4),
            "date_min": df["GAME_DATE"].min(),
            "date_max": df["GAME_DATE"].max(),
            "n_games": int(df["GAME_ID"].nunique()),
            "dropped_no_location": int(no_loc.sum()),
            "sha256": _sha256(path),
            "validator": VALIDATOR_VERSION,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_manifest(manifest)
        print(f"{season}: saved {len(df):,} rows, FG% {manifest[season]['fg_pct']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)


FIXTURE_CATEGORICALS = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_TYPE",
                        "EVENT_TYPE", "ACTION_TYPE", "SEASON"]


def fixture_sample(df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Random rows plus one row per value of each categorical, so rare real labels
    (e.g. 17 'Backcourt' shots a season) are always present for the schema tests."""
    keep = [df.sample(n=min(n, len(df)), random_state=0)]
    keep += [df.groupby(c, group_keys=False).head(1) for c in FIXTURE_CATEGORICALS]
    return pd.concat(keep).drop_duplicates(["GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)


def combine() -> None:
    manifest = load_manifest()
    done = [s for s in config.SEASONS if is_complete(s, manifest)]
    if not done:
        raise SystemExit("no complete seasons to combine")
    df = pd.concat([pd.read_parquet(season_path(s)) for s in done], ignore_index=True)
    atomic_write_parquet(df, config.REAL_SHOTS)
    config.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(fixture_sample(df), config.REAL_SAMPLE)

    missing = [s for s in config.SEASONS if s not in done]
    print(f"\nCombined {len(done)} seasons, {len(df):,} shots -> {config.REAL_SHOTS.relative_to(config.ROOT)}")
    if missing:
        print(f"Not yet fetched: {missing}")
    summary = pd.DataFrame(manifest).T.loc[done, ["rows", "n_games", "fg_pct", "date_min", "date_max"]]
    print(summary.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=config.SEASONS)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()
    bad = [s for s in args.seasons if s not in config.SEASONS]
    if bad:
        raise SystemExit(f"unknown seasons {bad}; expected from {config.SEASONS}")
    if not args.combine_only:
        pull(args.seasons)
    combine()
