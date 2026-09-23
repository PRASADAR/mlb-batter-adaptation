"""The outcome pitch must not contribute to its own mechanical history."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.lagged_mechanics import add_lagged_mechanics


def fixture():
    d = pd.DataFrame({"game_pk": [1, 1, 1, 1, 2],
        "game_date": ["2025-06-01"] * 4 + ["2025-06-02"],
        "at_bat_number": [1, 2, 3, 4, 1], "pitch_number": [1] * 5,
        "batter": [10] * 5, "pitcher": [101, 101, 101, 102, 101],
        "pitch_type": ["FF", "SL", "FF", "FF", "FF"],
        "description": ["foul", "ball", "swinging_strike", "ball", "foul"]})
    swings = d.loc[[0, 2, 4], ["game_pk", "at_bat_number", "pitch_number"]].copy()
    swings["adj_distortion"] = [1.0, 3.0, 9.0]
    return d, swings


def test_lagged_mechanics_pools_type_across_pitchers_and_resets_by_game():
    raw, swings = fixture()
    d = add_lagged_mechanics(raw, swings)
    assert d.loc[0, "prior_pair_tracked_n"] == 0
    assert d.loc[2, "prior_pair_tracked_n"] == 1
    assert d.loc[2, "prior_type_tracked_n"] == 1
    assert d.loc[2, "prior_pair_type_tracked_n"] == 1
    assert d.loc[2, "prior_pair_dev_mean"] == pytest.approx(1)
    assert d.loc[2, "prior_pair_dev_last"] == pytest.approx(1)
    assert d.loc[3, "prior_pair_tracked_n"] == 0
    assert d.loc[3, "prior_type_pitch_n"] == 2
    assert d.loc[3, "prior_type_tracked_n"] == 2
    assert d.loc[3, "prior_type_dev_mean"] == pytest.approx(2)
    assert d.loc[3, "prior_pair_type_tracked_n"] == 0
    assert d.loc[3, "prior_other_type_tracked_n"] == 2
    assert d.loc[3, "prior_other_type_dev_mean"] == pytest.approx(2)
    assert d.loc[4, "prior_pair_tracked_n"] == 0
    assert d.loc[4, "prior_type_tracked_n"] == 0
    assert d.loc[4, "prior_other_type_tracked_n"] == 0
    assert d.loc[2, "prior_pair_contact_rate"] == pytest.approx(2 / 4)
    assert d.loc[2, "prior_type_contact_rate"] == pytest.approx(2 / 3)
    changed = swings.copy()
    changed.loc[changed.at_bat_number.eq(3), "adj_distortion"] = -3
    other = add_lagged_mechanics(raw, changed)
    assert other.loc[:2, "prior_pair_dev_mean"].equals(
        d.loc[:2, "prior_pair_dev_mean"])
    assert other.loc[3, "prior_type_dev_mean"] == pytest.approx(-1)
