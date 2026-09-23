"""Synthetic data must look exactly like real nba_api data, or the fast loop lies."""
import pandas as pd
import pytest

import config
from generate_data import generate
from schema import schema_errors


def test_synthetic_matches_shot_schema():
    shots, _ = generate(seasons=config.SYNTHETIC_SEASONS[:2], games_per_season=20)
    assert schema_errors(shots) == []


@pytest.mark.skipif(not config.REAL_SAMPLE.exists(), reason="no real sample yet (Phase 1)")
def test_real_sample_matches_shot_schema():
    real = pd.read_parquet(config.REAL_SAMPLE)
    assert schema_errors(real) == []


@pytest.mark.skipif(not config.REAL_SAMPLE.exists(), reason="no real sample yet (Phase 1)")
def test_synthetic_categories_exist_in_real():
    """Every zone/area label the generator emits must be a label real data uses."""
    real = pd.read_parquet(config.REAL_SAMPLE)
    shots, _ = generate(seasons=config.SYNTHETIC_SEASONS[:1], games_per_season=40)
    for col in ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_TYPE", "EVENT_TYPE"]:
        unknown = set(shots[col]) - set(real[col])
        assert not unknown, f"{col}: synthetic labels not seen in real data: {unknown}"
