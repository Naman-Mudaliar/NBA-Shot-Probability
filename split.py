"""Chronological season split: fit seasons (last one = early-stopping) vs walk-forward seasons."""
import math


def season_split(seasons, train_frac):
    """Return (train_seasons, stop_season, online_seasons).

    The first round(len * train_frac) seasons are fit seasons; the last of those is
    held out as the early-stopping season. Everything after is walk-forward/online.
    """
    seasons = sorted(seasons)
    n_fit = min(max(2, math.floor(len(seasons) * train_frac + 0.5)), len(seasons) - 1)
    fit, online = seasons[:n_fit], seasons[n_fit:]
    return fit[:-1], fit[-1], online
