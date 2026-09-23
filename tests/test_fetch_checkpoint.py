"""Checkpoint/resume logic of fetch_data.py, tested offline (API call is stubbed)."""
import pytest

import config
import fetch_data
from generate_data import generate

SEASON = config.SEASONS[-1]


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fetch_data, "completeness_errors", lambda df, season: [])
    monkeypatch.setattr(fetch_data, "SLEEP_S", 0)
    calls = []

    def fake_fetch(season):
        calls.append(season)
        shots, _ = generate(seasons=[season], games_per_season=10)
        return shots.drop(columns="SEASON")

    monkeypatch.setattr(fetch_data, "fetch_season", fake_fetch)
    return tmp_path, calls


def test_pull_writes_checkpoint_and_skips_on_rerun(raw_dir):
    path, calls = raw_dir
    fetch_data.pull([SEASON])
    assert (path / f"shots_{SEASON}.parquet").exists()
    assert fetch_data.load_manifest()[SEASON]["rows"] > 0
    fetch_data.pull([SEASON])
    assert calls == [SEASON], "complete season should be skipped on rerun"


def test_corrupt_checkpoint_is_refetched(raw_dir):
    path, calls = raw_dir
    fetch_data.pull([SEASON])
    (path / f"shots_{SEASON}.parquet").write_bytes(b"truncated")
    fetch_data.pull([SEASON])
    assert calls == [SEASON, SEASON], "file not matching manifest sha must be refetched"


def test_file_without_manifest_entry_is_refetched(raw_dir):
    path, calls = raw_dir
    (path / f"shots_{SEASON}.parquet").write_bytes(b"left over from a killed run")
    fetch_data.pull([SEASON])
    assert calls == [SEASON]


def test_stray_tmp_removed(raw_dir):
    path, _ = raw_dir
    stray = path / f"shots_{SEASON}.parquet.tmp"
    stray.write_bytes(b"partial")
    fetch_data.pull([SEASON])
    assert not stray.exists()


def test_stale_validator_is_refetched(raw_dir, monkeypatch):
    path, calls = raw_dir
    fetch_data.pull([SEASON])
    monkeypatch.setattr(fetch_data, "VALIDATOR_VERSION", fetch_data.VALIDATOR_VERSION + 1)
    fetch_data.pull([SEASON])
    assert calls == [SEASON, SEASON], "entries validated by an older validator must be refetched"


def test_truncated_season_detected():
    """The observed failure: a capped league-wide response covering only half the season."""
    shots, _ = generate(seasons=[SEASON], games_per_season=40)
    shots = shots.drop(columns="SEASON")
    errs = fetch_data.completeness_errors(shots, SEASON)
    assert any("regular-season games" in e for e in errs)


def test_invalid_season_is_not_saved(raw_dir, monkeypatch):
    path, _ = raw_dir
    monkeypatch.setattr(fetch_data, "completeness_errors", lambda df, season: ["truncated"])
    with pytest.raises(SystemExit):
        fetch_data.pull([SEASON])
    assert not (path / f"shots_{SEASON}.parquet").exists()
    assert SEASON not in fetch_data.load_manifest()


def test_team_responses_cached(tmp_path, monkeypatch):
    """A failure after download (e.g. coercion) must not cost another API call on rerun."""
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "SLEEP_S", 0)
    calls = []
    shots, _ = generate(seasons=[SEASON], games_per_season=5)

    def fake_call(season, team_id=0):
        calls.append(team_id)
        return shots.drop(columns="SEASON").head(3)

    monkeypatch.setattr(fetch_data, "_call", fake_call)
    fetch_data.fetch_season(SEASON)
    n = len(calls)
    assert n == 30
    fetch_data.fetch_season(SEASON)
    assert len(calls) == n, "cached team responses must not be refetched"
