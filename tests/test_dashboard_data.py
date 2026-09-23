"""Dashboard payload integrity, on the synthetic run (needs `run_pipeline.py --synthetic` outputs)."""
import numpy as np
import pandas as pd
import pytest

import config
import build_dashboard_data as bd


@pytest.fixture(scope="module")
def built():
    out = config.outputs(synthetic=True)
    if not out["predictions"].exists():
        pytest.fail("run `python run_pipeline.py --synthetic` first")
    payload = bd.build_payload(synthetic=True)
    preds = pd.read_parquet(out["predictions"])
    return payload, preds


def test_only_dashboard_seasons(built):
    payload, preds = built
    online = sorted(preds["SEASON"].unique())
    assert payload["meta"]["seasons"] == online[-config.DASHBOARD_SEASONS:]
    seasons = bd.unpack(payload["shots"]["columns"]["season"]["data"], "u1")
    assert set(np.unique(seasons)) <= set(range(len(payload["meta"]["seasons"])))


def test_every_shot_is_a_walk_forward_prediction(built):
    """Shots shown == the held-out predictions for those seasons, one-to-one, same xFG%."""
    payload, preds = built
    dash = preds[preds["SEASON"].isin(payload["meta"]["seasons"])]
    assert payload["shots"]["n"] == len(dash)
    xfg = bd.unpack(payload["shots"]["columns"]["xfg"]["data"], "<u2") / 1000
    made = bd.unpack(payload["shots"]["columns"]["made"]["data"], "u1")
    np.testing.assert_allclose(np.sort(xfg), np.sort(np.round(dash["xfg"].to_numpy() * 1000) / 1000))
    assert made.sum() == dash["SHOT_MADE_FLAG"].sum()


def test_packed_columns_roundtrip(built):
    payload, _ = built
    n = payload["shots"]["n"]
    lk = payload["meta"]["lookups"]
    for name, col in payload["shots"]["columns"].items():
        arr = bd.unpack(col["data"], col["dtype"])
        assert len(arr) == n, name
    player = bd.unpack(payload["shots"]["columns"]["player"]["data"], "<u2")
    zone = bd.unpack(payload["shots"]["columns"]["zone"]["data"], "u1")
    assert player.max() < len(lk["players"]) and zone.max() < len(lk["zones"])


def test_leaderboards_add_up(built):
    payload, preds = built
    dash = preds[preds["SEASON"].isin(payload["meta"]["seasons"])]
    players = pd.DataFrame(payload["leaderboards"]["boards"]["All"]["players"])
    assert players["attempts"].sum() == len(dash)
    assert players["makes"].sum() == dash["SHOT_MADE_FLAG"].sum()
    assert players["makes_oe"].sum() == pytest.approx((dash["SHOT_MADE_FLAG"] - dash["xfg"]).sum(), abs=0.5)
    offense = pd.DataFrame(payload["leaderboards"]["boards"]["All"]["team_offense"])
    defense = pd.DataFrame(payload["leaderboards"]["boards"]["All"]["team_defense"])
    assert offense["attempts"].sum() == defense["attempts"].sum() == len(dash)


def test_top_shot_maker_has_high_planted_skill(built):
    payload, _ = built
    truth = pd.read_parquet(config.SYNTHETIC_TRUTH)
    players = pd.DataFrame(payload["leaderboards"]["boards"]["All"]["players"])
    top = players.nlargest(5, "makes_oe").merge(truth, on="PLAYER_ID")
    assert (top["skill"] > truth["skill"].quantile(0.75)).mean() >= 0.6, top


def test_no_empty_sections(built):
    payload, _ = built
    t = payload["training"]
    for key in ["coverage", "zone_mix", "learning_curve", "feature_importance"]:
        assert len(t[key]) > 0, key
    assert len(payload["zones"]["cells"]) == len(payload["zones"]["y"])
    assert min(min(r) for r in payload["zones"]["cells"]) >= 0
    for scope in payload["leaderboards"]["scopes"]:
        b = payload["leaderboards"]["boards"][scope]
        for key in ["players", "players_by_zone", "team_offense", "team_defense", "league_zones"]:
            assert len(b[key]) > 0, (scope, key)
    for key in ["toughest_makes", "toughest_makes_no_heaves", "easiest_misses"]:
        assert len(payload["outliers"][key]) == bd.N_OUTLIERS, key
    assert all(r["SHOT_DISTANCE"] < bd.HEAVE_FT for r in payload["outliers"]["toughest_makes_no_heaves"])
