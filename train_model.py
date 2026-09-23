"""
train_model.py
--------------
Shooter-blind xFG% model: offline fit on early seasons, then walk-forward
(weekly) through the held-out seasons -- predict a chunk, log it, THEN learn
from it.

  1. Offline: XGBoost binary:logistic on train seasons, early-stopped on the
     stop season (never on online seasons); then refit on train + stop with the
     chosen number of rounds so the online phase starts current.
  2. Walk-forward: for each weekly chunk of the online seasons, predict with the
     current booster, then continue boosting a few low-eta rounds on a sliding
     window of recent chunks.
  3. Baselines scored on the same chunks: frozen offline model, league FG% by zone.

Outputs (config.outputs(...)):
  models/booster_offline.json, models/booster_final.json, models/online_log.csv,
  models/metrics.json, models/feature_importance.csv, data/predictions.parquet
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config
from features import MODEL_FEATURES
from split import season_split

OFFLINE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "seed": 42,
}
MAX_ROUNDS = 1500
EARLY_STOP = 40

# online updates: shallow, heavily regularized trees at a small step so weekly
# refreshes track drift without fitting a week's noise (deeper trees lost to the
# frozen model on synthetic data)
ONLINE_PARAMS = {"eta": 0.02, "max_depth": 3, "min_child_weight": 200}
ROUNDS_PER_CHUNK = 3
WINDOW_CHUNKS = 8         # sliding window of recent weekly chunks to learn from


def _dmatrix(df: pd.DataFrame, label=None) -> xgb.DMatrix:
    return xgb.DMatrix(df[MODEL_FEATURES].astype(float), label=label, feature_names=MODEL_FEATURES)


def _scores(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
    }


def fit_offline(feats: pd.DataFrame, train_seasons, stop_season, extra_params=None):
    params = {**OFFLINE_PARAMS, **(extra_params or {})}
    tr = feats[feats["SEASON"].isin(train_seasons)]
    st = feats[feats["SEASON"] == stop_season]
    dtr, dst = _dmatrix(tr, tr["SHOT_MADE_FLAG"]), _dmatrix(st, st["SHOT_MADE_FLAG"])
    probe = xgb.train(params, dtr, MAX_ROUNDS, evals=[(dst, "stop")],
                      early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    n_rounds = probe.best_iteration + 1

    fit = pd.concat([tr, st])
    booster = xgb.train(params, _dmatrix(fit, fit["SHOT_MADE_FLAG"]), n_rounds)
    booster.set_attr(
        stop_season_log_loss=str(probe.best_score),
        n_rounds=str(n_rounds),
        train_max_date=str(fit["GAME_DATE"].max()),
    )
    return booster


def assign_chunks(df: pd.DataFrame) -> pd.Series:
    """Weekly chunk index over the (sorted) online rows."""
    week = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d").dt.to_period(config.ONLINE_CHUNK)
    return pd.Series(pd.factorize(week, sort=True)[0], index=df.index)


def walk_forward(booster, feats: pd.DataFrame, online_seasons, extra_params=None,
                 flip_chunk=None, zone_rates=None, return_booster=False):
    """Predict-then-learn over weekly chunks. Returns (predictions, per-chunk log).

    flip_chunk: test hook -- invert that chunk's labels before learning from it.
    zone_rates: league FG% by SHOT_ZONE_BASIC from fit seasons (baseline).
    """
    params = {**OFFLINE_PARAMS, **ONLINE_PARAMS, **(extra_params or {})}
    frozen = booster.copy()
    live = booster.copy()
    train_max_date = booster.attr("train_max_date")

    online = (feats[feats["SEASON"].isin(online_seasons)]
              .sort_values(["GAME_DATE", "GAME_ID", "GAME_EVENT_ID"], kind="mergesort")
              .reset_index(drop=True))
    online["chunk"] = assign_chunks(online)

    preds, log, window = [], [], []
    for k, chunk in online.groupby("chunk", sort=True):
        # ---- predict (model has seen nothing on/after this chunk) ----
        d = _dmatrix(chunk)
        out = chunk.drop(columns=MODEL_FEATURES).copy()
        out["xfg"] = live.predict(d)
        out["xfg_frozen"] = frozen.predict(d)
        if zone_rates is not None:
            out["xfg_zone"] = chunk["SHOT_ZONE_BASIC"].map(zone_rates).fillna(zone_rates.mean()).to_numpy()
        preds.append(out)

        y = chunk["SHOT_MADE_FLAG"].to_numpy()
        row = {"chunk": int(k), "season": chunk["SEASON"].iloc[0], "n": len(chunk),
               "chunk_min_date": chunk["GAME_DATE"].min(), "chunk_max_date": chunk["GAME_DATE"].max(),
               "train_max_date": train_max_date}
        for name in ["xfg", "xfg_frozen", "xfg_zone"]:
            if name in out:
                row.update({f"{name}_{m}": v for m, v in _scores(y, out[name]).items()})
        log.append(row)

        # ---- then learn from it ----
        labels = 1 - y if k == flip_chunk else y
        window.append(chunk.assign(SHOT_MADE_FLAG=labels))
        window = window[-WINDOW_CHUNKS:]
        win = pd.concat(window)
        live = xgb.train(params, _dmatrix(win, win["SHOT_MADE_FLAG"]), ROUNDS_PER_CHUNK, xgb_model=live)
        train_max_date = max(train_max_date, win["GAME_DATE"].max())

    result = (pd.concat(preds, ignore_index=True), pd.DataFrame(log))
    return (*result, live) if return_booster else result


def summarize(preds: pd.DataFrame, fit_rate: float, booster_offline) -> dict:
    per_season = {}
    for season, g in preds.groupby("SEASON"):
        y = g["SHOT_MADE_FLAG"]
        per_season[season] = {
            "n": int(len(g)),
            "actual_fg": round(float(y.mean()), 4),
            "mean_xfg": round(float(g["xfg"].mean()), 4),
            **{name: {m: round(v, 4) for m, v in _scores(y, g[name]).items()}
               for name in ["xfg", "xfg_frozen", "xfg_zone"]},
            "base_rate_log_loss": round(_scores(y, np.full(len(y), fit_rate))["log_loss"], 4),
        }
    frac, mean_pred = calibration_curve(preds["SHOT_MADE_FLAG"], preds["xfg"], n_bins=10, strategy="quantile")
    return {
        "offline": {
            "n_rounds": int(booster_offline.attr("n_rounds")),
            "stop_season_log_loss": round(float(booster_offline.attr("stop_season_log_loss")), 4),
            "train_max_date": booster_offline.attr("train_max_date"),
        },
        "online_overall": {name: {m: round(v, 4) for m, v in _scores(preds["SHOT_MADE_FLAG"], preds[name]).items()}
                           for name in ["xfg", "xfg_frozen", "xfg_zone"]},
        "online_by_season": per_season,
        "calibration": {
            "predicted_mean_by_bin": [round(v, 3) for v in mean_pred],
            "actual_frac_by_bin": [round(v, 3) for v in frac],
            "max_abs_gap": round(float(np.max(np.abs(frac - mean_pred))), 4),
        },
    }


def main(synthetic: bool) -> None:
    paths = config.outputs(synthetic)
    paths["models"].mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(paths["features"])
    seasons = sorted(feats["SEASON"].unique())
    train, stop, online = season_split(seasons, config.TRAIN_FRAC)
    print(f"fit: {train[0]}..{train[-1]} | stop: {stop} | online: {online[0]}..{online[-1]}")

    fit_rows = feats[feats["SEASON"].isin(train + [stop])]
    zone_rates = fit_rows.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"].mean()

    booster = fit_offline(feats, train, stop)
    booster.save_model(paths["models"] / "booster_offline.json")
    print(f"offline: {booster.attr('n_rounds')} rounds, stop-season log loss {float(booster.attr('stop_season_log_loss')):.4f}")

    preds, log, final = walk_forward(booster, feats, online, zone_rates=zone_rates, return_booster=True)
    final.save_model(paths["models"] / "booster_final.json")
    metrics = summarize(preds, float(fit_rows["SHOT_MADE_FLAG"].mean()), booster)

    imp = booster.get_score(importance_type="gain")
    (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
     .sort_values("gain", ascending=False)
     .to_csv(paths["models"] / "feature_importance.csv", index=False))
    log.to_csv(paths["models"] / "online_log.csv", index=False)
    (paths["models"] / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_parquet(paths["predictions"], index=False)

    print(pd.DataFrame({s: {"logloss online": v["xfg"]["log_loss"], "frozen": v["xfg_frozen"]["log_loss"],
                            "zone": v["xfg_zone"]["log_loss"], "auc": v["xfg"]["auc"],
                            "actual": v["actual_fg"], "xfg": v["mean_xfg"]}
                        for s, v in metrics["online_by_season"].items()}).T.to_string())
    print(f"calibration max gap: {metrics['calibration']['max_abs_gap']:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
