"""
features.py
------------
Transforms raw shot events into an analysis-ready feature matrix.

Engineers spatial features (angle, quadrant, normalized court position),
situational features (clutch flag, shot-clock pressure bucket, fatigue proxy),
and target-safe categorical encodings for zone/position/team.
"""
import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["shot_zone", "position", "quadrant", "clock_bucket"]
DROP_FOR_MODEL = [
    "game_id", "player_id", "player_name", "team", "opponent",
    "true_prob", "shot_made", "loc_x", "loc_y",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- spatial ---
    df["angle_from_hoop"] = np.degrees(np.arctan2(df["loc_x"], df["loc_y"].clip(lower=0.1)))
    df["quadrant"] = np.select(
        [
            (df["loc_x"] >= 0) & (df["angle_from_hoop"].abs() <= 15),
            df["loc_x"] > 0,
            df["loc_x"] < 0,
        ],
        ["Center", "Right", "Left"],
        default="Center",
    )
    df["dist_bucket"] = pd.cut(
        df["shot_distance"],
        bins=[-0.1, 4, 8, 16, 22, 23.9, 30, 100],
        labels=["0-4ft", "4-8ft", "8-16ft", "16-22ft", "Long2", "3pt", "Deep3"],
    )

    # --- situational ---
    df["seconds_left_period"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
    df["clutch"] = (
        (df["period"] >= 4)
        & (df["seconds_left_period"] <= 300)
        & (df["score_margin"].abs() <= 5)
    ).astype(int)
    df["clock_bucket"] = pd.cut(
        df["shot_clock"], bins=[-0.1, 4, 7, 14, 24],
        labels=["Desperation(<4s)", "Late(4-7s)", "Mid(7-14s)", "Early(14-24s)"],
    )
    df["late_clock_pressure"] = (df["shot_clock"] < 7).astype(int)
    df["off_dribble"] = (df["catch_and_shoot"] == 0).astype(int)
    df["heavily_guarded"] = (df["defender_distance"] < 3).astype(int)
    df["wide_open"] = (df["defender_distance"] > 6).astype(int)
    df["high_volume_touch"] = (df["dribbles"] >= 4).astype(int)
    df["fatigue_proxy"] = df["period"].clip(upper=4) * (12 - df["minutes_remaining"]).clip(lower=0) / 48

    # --- one-hot encode low-cardinality categoricals ---
    encoded = pd.get_dummies(
        df[CATEGORICAL_COLS + ["dist_bucket"]].astype(str),
        prefix=CATEGORICAL_COLS + ["dist"],
    )

    numeric_cols = [
        "shot_distance", "angle_from_hoop", "shot_clock", "dribbles", "touch_time",
        "defender_distance", "is_three", "catch_and_shoot", "off_dribble",
        "heavily_guarded", "wide_open", "high_volume_touch", "late_clock_pressure",
        "clutch", "fatigue_proxy", "home", "score_margin", "period",
    ]

    feature_df = pd.concat([df[numeric_cols], encoded], axis=1)
    meta_df = df[[c for c in DROP_FOR_MODEL if c in df.columns]]
    full_df = pd.concat([meta_df, feature_df], axis=1)
    return full_df


if __name__ == "__main__":
    raw = pd.read_csv("/sessions/compassionate-funny-sagan/mnt/outputs/nba_shot_model/data/shots_raw.csv")
    feats = engineer_features(raw)
    out_path = "/sessions/compassionate-funny-sagan/mnt/outputs/nba_shot_model/data/shots_features.csv"
    feats.to_csv(out_path, index=False)
    print(f"Wrote {feats.shape[0]:,} rows x {feats.shape[1]} cols to {out_path}")
    print("Feature columns:", [c for c in feats.columns if c not in DROP_FOR_MODEL])
