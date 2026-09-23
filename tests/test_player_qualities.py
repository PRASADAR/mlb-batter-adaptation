"""Player-quality aggregation uses explicit pitch and PA denominators."""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.player_qualities import aggregate_player_qualities


def test_player_quality_denominators_and_statcast_barrel_definition():
    frame = pd.DataFrame({
        "batter": [1, 1, 1, 1, 2],
        "description": ["swinging_strike", "foul", "hit_into_play",
                        "ball", "hit_into_play"],
        "zone": [11, 5, 12, 13, 5],
        "events": [None, None, "single", "walk", "field_out"],
        "launch_speed": [None, None, 101.0, None, 88.0],
        "launch_speed_angle": [None, None, 6, None, 3],
        "estimated_woba_using_speedangle": [None, None, .62, None, .25],
        "woba_value": [None, None, .9, .7, 0],
        "woba_denom": [None, None, 1, 1, 1],
        "bat_speed": [72.0, 73.0, 74.0, None, 70.0],
    })
    result = aggregate_player_qualities(frame).set_index("player_id")
    hitter = result.loc[1]
    assert hitter.pa == 2
    assert hitter.swings == 3
    assert hitter.chase_rate == pytest.approx(2 / 3)
    assert hitter.zone_swing_rate == pytest.approx(1)
    assert hitter.whiff_rate == pytest.approx(1 / 3)
    assert hitter.walk_rate == pytest.approx(1 / 2)
    assert hitter.hard_hit_rate == pytest.approx(1)
    assert hitter.barrel_rate == pytest.approx(1)
    assert hitter.woba == pytest.approx(.8)
    assert hitter.mean_bat_speed == pytest.approx(73)
