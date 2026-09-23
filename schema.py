"""Raw shot schema validation, shared by fetch_data.py and the tests."""
import pandas as pd
from pandas.api import types as ptypes

from config import SHOT_SCHEMA

_CHECKS = {"int": ptypes.is_integer_dtype, "str": lambda s: ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)}


def schema_errors(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema problems (empty == valid)."""
    errors = []
    missing = [c for c in SHOT_SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in SHOT_SCHEMA]
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    for col, kind in SHOT_SCHEMA.items():
        if col in df.columns and not _CHECKS[kind](df[col]):
            errors.append(f"{col}: expected {kind}, got {df[col].dtype}")
    if "GAME_DATE" in df.columns and len(df):
        bad = ~df["GAME_DATE"].astype(str).str.fullmatch(r"\d{8}")
        if bad.any():
            errors.append(f"GAME_DATE not YYYYMMDD in {int(bad.sum())} rows")
    if len(df) == 0:
        errors.append("no rows")
    return errors


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a raw nba_api frame to SHOT_SCHEMA dtypes (API sometimes returns floats/objects)."""
    df = df.copy()
    for col, kind in SHOT_SCHEMA.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype("int64") if kind == "int" else df[col].astype(str)
    return df
