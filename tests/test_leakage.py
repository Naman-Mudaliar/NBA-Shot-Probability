"""
Executable leakage rules. A leak never crashes -- it just makes the numbers
suspiciously good -- so every rule here is enforced by a test that runs in
`python run_pipeline.py --synthetic`.

  1. Shooter-blind: no shooter identity / position / history in model inputs,
     and the model must leave planted shooter skill in (actual - xFG).
  2. Defense features use only games strictly before the shot's game.
  3. Split is chronological.
  4. Every walk-forward prediction comes from a model that had not yet
     trained on that shot.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

import config
from generate_data import generate


def _defines(module: str, name: str) -> bool:
    """True if module.py defines `name` -- checked by source so we never import
    a not-yet-rebuilt module (the legacy scripts run at import time)."""
    path = config.ROOT / f"{module}.py"
    return path.exists() and (f"def {name}" in path.read_text() or f"{name} =" in path.read_text())


PHASE2 = _defines("features", "MODEL_FEATURES") and _defines("split", "season_split")
PHASE3 = PHASE2 and _defines("train_model", "walk_forward")
features = importlib.import_module("features") if PHASE2 else None
split = importlib.import_module("split") if PHASE2 else None
train_model = importlib.import_module("train_model") if PHASE3 else None
phase2 = pytest.mark.skipif(not PHASE2, reason="features/split not rebuilt yet (Phase 2)")
phase3 = pytest.mark.skipif(not PHASE3, reason="walk-forward training not built yet (Phase 3)")

# deterministic, small, fast settings for tests that retrain
TEST_PARAMS = {"nthread": 1, "seed": 0}


@pytest.fixture(scope="module")
def small_raw():
    shots, truth = generate(seasons=config.SYNTHETIC_SEASONS[:4], games_per_season=60, seed=7)
    return shots, truth


# ---------------- Rule 1: shooter-blind ----------------
@phase2
def test_no_shooter_columns():
    feats = set(features.MODEL_FEATURES)
    assert not feats & config.BLOCKED_FEATURES, feats & config.BLOCKED_FEATURES
    for f in feats:
        assert not any(s in f.lower() for s in config.BLOCKED_SUBSTRINGS), f


@phase2
def test_model_features_do_not_vary_by_shooter(small_raw):
    """Swap every shooter identity: model inputs must be byte-identical."""
    shots, _ = small_raw
    swapped = shots.copy()
    rng = np.random.default_rng(0)
    swapped["PLAYER_ID"] = rng.permutation(swapped["PLAYER_ID"].to_numpy())
    swapped["PLAYER_NAME"] = "anon"
    a = features.build_features(shots)[features.MODEL_FEATURES]
    b = features.build_features(swapped)[features.MODEL_FEATURES]
    pd.testing.assert_frame_equal(a, b)


@phase3
def test_shooter_blind_recovers_skill():
    """On the synthetic run, player (actual - xFG) must track planted skill.
    If shooter identity leaks into the model, this gap collapses toward zero."""
    out = config.outputs(synthetic=True)
    if not out["predictions"].exists():
        pytest.fail("run `python run_pipeline.py --synthetic` first")
    preds = pd.read_parquet(out["predictions"])
    truth = pd.read_parquet(config.SYNTHETIC_TRUTH)
    by_player = (
        preds.assign(resid=preds["SHOT_MADE_FLAG"] - preds["xfg"])
        .groupby("PLAYER_ID")["resid"].mean()
        .rename("fg_over_expected").reset_index()
        .merge(truth, on="PLAYER_ID")
    )
    rho = by_player["fg_over_expected"].corr(by_player["skill"], method="spearman")
    assert rho > 0.6, f"shooter skill not recoverable from residuals (rho={rho:.2f})"


# ---------------- Rule 2: defense uses only the past ----------------
@phase2
def test_defense_uses_only_past(small_raw):
    """Scramble every result on/after date D: features for shots on/before D must not move."""
    shots, _ = small_raw
    dates = np.sort(shots["GAME_DATE"].unique())
    d = dates[len(dates) // 2]
    scrambled = shots.copy()
    future = scrambled["GAME_DATE"] >= d
    rng = np.random.default_rng(1)
    scrambled.loc[future, "SHOT_MADE_FLAG"] = rng.integers(0, 2, future.sum())

    a = features.build_features(shots)
    b = features.build_features(scrambled)
    keep_a = a["GAME_DATE"] <= d
    keep_b = b["GAME_DATE"] <= d
    pd.testing.assert_frame_equal(
        a.loc[keep_a, features.MODEL_FEATURES].reset_index(drop=True),
        b.loc[keep_b, features.MODEL_FEATURES].reset_index(drop=True),
    )


# ---------------- Rule 3: chronological split ----------------
@phase2
def test_split_is_chronological():
    train, stop, online = split.season_split(config.SEASONS, config.TRAIN_FRAC)
    assert train and online
    assert max(train) < stop < min(online)
    assert set(train) | {stop} | set(online) == set(config.SEASONS)
    n_fit = len(train) + 1
    assert 0.6 <= n_fit / len(config.SEASONS) <= 0.8


# ---------------- Rule 4: predict before train ----------------
def _walk(raw, flip_chunk=None):
    feats = features.build_features(raw)
    seasons = sorted(raw["SEASON"].unique())
    train, stop, online = split.season_split(seasons, 0.5)
    booster = train_model.fit_offline(feats, train, stop, extra_params=TEST_PARAMS)
    return train_model.walk_forward(booster, feats, online, extra_params=TEST_PARAMS,
                                    flip_chunk=flip_chunk)


@phase3
def test_predictions_precede_training(small_raw):
    preds, log = _walk(small_raw[0])
    assert (log["train_max_date"] < log["chunk_min_date"]).all()
    merged = preds.merge(log[["chunk", "train_max_date"]], on="chunk")
    assert (merged["GAME_DATE"] > merged["train_max_date"]).all()


@phase3
def test_label_flip_invariance(small_raw):
    """Flipping chunk k's outcomes must not change predictions for chunks <= k."""
    base_preds, log = _walk(small_raw[0])
    k = int(log["chunk"].median())
    flip_preds, _ = _walk(small_raw[0], flip_chunk=k)
    a = base_preds[base_preds["chunk"] <= k]["xfg"].to_numpy()
    b = flip_preds[flip_preds["chunk"] <= k]["xfg"].to_numpy()
    np.testing.assert_array_equal(a, b)
    # sanity: the flip does change what comes after
    a2 = base_preds[base_preds["chunk"] > k]["xfg"].to_numpy()
    b2 = flip_preds[flip_preds["chunk"] > k]["xfg"].to_numpy()
    assert not np.array_equal(a2, b2)
