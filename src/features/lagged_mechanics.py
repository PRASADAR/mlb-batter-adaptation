"""Mechanical histories known before each delivered pitch in a game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prospective import KEY, CONTACT, WHIFF, TAKE


def _prior_group_history(frame: pd.DataFrame, name: str,
                         keys: list[str]) -> None:
    """Add strictly lagged counts, outcome rates, and swing deviations."""
    group = frame.groupby(keys, sort=False, dropna=False)
    frame[f"prior_{name}_pitch_n"] = group.cumcount().astype("int16")
    totals = {}
    for source, label in [("tracked_now", "tracked_n"),
                          ("dev_clipped", "dev_sum"),
                          ("labeled_now", "labeled_n"),
                          ("contact_now", "contact_n"),
                          ("swing_now", "swing_n")]:
        totals[label] = group[source].cumsum() - frame[source]
        frame[f"prior_{name}_{label}"] = totals[label]
    n = totals["tracked_n"]
    frame[f"prior_{name}_dev_mean"] = np.divide(
        totals["dev_sum"], n, out=np.zeros(len(frame), dtype=float),
        where=n.gt(0))
    frame[f"prior_{name}_contact_rate"] = (
        (totals["contact_n"] + 1) / (totals["labeled_n"] + 2)
    ).astype("float32")
    frame[f"prior_{name}_swing_rate"] = (
        (totals["swing_n"] + 1) / (totals["labeled_n"] + 2)
    ).astype("float32")
    visible = frame.adj_distortion.clip(-frame.attrs["mechanics_clip_sd"],
                                         frame.attrs["mechanics_clip_sd"])
    last = visible.groupby([frame[k] for k in keys], sort=False,
                           dropna=False).ffill()
    frame[f"prior_{name}_dev_last"] = last.groupby(
        [frame[k] for k in keys], sort=False, dropna=False
    ).shift(1).fillna(0).astype("float32")


def add_lagged_mechanics(raw: pd.DataFrame, swings: pd.DataFrame,
                         clip: float = 4.0) -> pd.DataFrame:
    """Join tracked mechanics and construct histories that exclude this pitch.

    ``type`` histories pool the same pitch type across every pitcher in the
    game. ``pair_type`` histories retain the current pitcher, which permits an
    exact subtraction for the stricter other-pitcher history. ``pair``
    histories summarize general familiarity with the current pitcher.
    """
    required = set(KEY + ["game_date", "batter", "pitcher", "pitch_type",
                          "description"])
    if required.difference(raw) or set(KEY + ["adj_distortion"]).difference(swings):
        raise ValueError("Lagged mechanics inputs are incomplete")
    if raw.duplicated(KEY).any() or swings.duplicated(KEY).any():
        raise AssertionError("Delivered and tracked pitches need unique keys")
    d = raw.merge(swings[KEY + ["adj_distortion"]], on=KEY, how="left",
                  validate="one_to_one")
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.sort_values(["game_date", "game_pk", "at_bat_number",
                       "pitch_number"], kind="stable").reset_index(drop=True)
    d.attrs["mechanics_clip_sd"] = float(clip)
    d["tracked_now"] = d.adj_distortion.notna().astype("int8")
    d["dev_clipped"] = d.adj_distortion.clip(-clip, clip).fillna(0).astype("float32")
    d["contact_now"] = d.description.isin(CONTACT).astype("int8")
    d["swing_now"] = d.description.isin(CONTACT | WHIFF).astype("int8")
    d["labeled_now"] = d.description.isin(CONTACT | WHIFF | TAKE).astype("int8")
    groups = {
        "pair": ["game_pk", "batter", "pitcher"],
        "type": ["game_pk", "batter", "pitch_type"],
        "pair_type": ["game_pk", "batter", "pitcher", "pitch_type"],
    }
    for name, keys in groups.items():
        _prior_group_history(d, name, keys)

    # The difference is exact because type and pair-type histories are both
    # computed on delivered pitches before the outcome row.
    for suffix in ["pitch_n", "tracked_n", "dev_sum", "labeled_n",
                   "contact_n", "swing_n"]:
        d[f"prior_other_type_{suffix}"] = (
            d[f"prior_type_{suffix}"] - d[f"prior_pair_type_{suffix}"])
    other_n = d.prior_other_type_tracked_n
    d["prior_other_type_dev_mean"] = np.divide(
        d.prior_other_type_dev_sum, other_n,
        out=np.zeros(len(d), dtype=float), where=other_n.gt(0))
    d["prior_other_type_contact_rate"] = (
        (d.prior_other_type_contact_n + 1) /
        (d.prior_other_type_labeled_n + 2)
    ).astype("float32")
    d["prior_other_type_swing_rate"] = (
        (d.prior_other_type_swing_n + 1) /
        (d.prior_other_type_labeled_n + 2)
    ).astype("float32")

    return d.drop(columns=["tracked_now", "dev_clipped", "contact_now",
                           "swing_now", "labeled_now"])
