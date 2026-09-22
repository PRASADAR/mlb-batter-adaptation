"""Scientific invariants for chronological all-pitch exposure construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.features.sequences import KEY, model_frame, prepare_pitches


def pitch(game=100, pa=1, number=1, batter=10, pitcher=101,
          pitch_type="FF", description="ball", **overrides):
    row = {
        "game_pk": game, "at_bat_number": pa, "pitch_number": number,
        "batter": batter, "pitcher": pitcher, "pitch_type": pitch_type,
        "description": description, "game_type": "R", "game_date": "2025-06-01",
        "release_speed": 95.0, "pfx_x": -0.5, "pfx_z": 1.5,
        "release_pos_x": -2.0, "release_pos_z": 6.0,
        "plate_x": 0.1, "plate_z": 2.5, "sz_bot": 1.5, "sz_top": 3.5,
        "vx0": 2.0, "vy0": -100.0, "vz0": -5.0,
        "ax": 0.0, "ay": 0.0, "az": 0.0,
        "balls": 0, "strikes": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def all_pitch_history():
    return pd.DataFrame([
        pitch(number=1, description="ball"),
        pitch(number=2, pitch_type="SL", description="swinging_strike", release_speed=85.0),
        pitch(number=3, description="foul"),
        pitch(number=4, description="called_strike"),
        pitch(pa=2, batter=20),
        pitch(pa=3, number=1),
        pitch(pa=3, number=2, pitch_type="SL", description="hit_into_play", release_speed=85.0),
        pitch(pa=4, pitcher=202),
        pitch(game=101, game_date="2025-06-02"),
    ])


def test_shuffled_input_has_identical_order_and_histories(all_pitch_history):
    ordered = prepare_pitches(all_pitch_history)
    shuffled = prepare_pitches(all_pitch_history.sample(frac=1, random_state=17))
    assert_frame_equal(ordered, shuffled)
    assert ordered[KEY].to_records(index=False).tolist() == sorted(
        ordered[KEY].to_records(index=False).tolist()
    )


def test_lags_never_cross_plate_appearances(all_pitch_history):
    d = prepare_pitches(all_pitch_history)
    starts = d.pitch_number.eq(1)
    assert d.loc[starts, "lag1_pitch_type"].isna().all()
    assert d.loc[starts, "lag2_pitch_type"].isna().all()
    assert d.loc[starts, "lag1_release_speed"].isna().all()
    assert d.loc[starts, "lag1_whiff"].eq(0).all()
    assert d.loc[starts, "pa_prior_pitches"].eq(0).all()
    assert d.loc[2, "lag1_pitch_type"] == "SL"
    assert d.loc[2, "lag2_pitch_type"] == "FF"
    assert d.loc[2, "lag1_whiff"] == 1
    assert d.loc[2, "delta_release_speed"] == 10


def test_exact_exposures_exclude_current_include_takes_and_reset(all_pitch_history):
    d = prepare_pitches(all_pitch_history)
    np.testing.assert_array_equal(d.exposure_pa, [0, 0, 1, 2, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(d.exposure_pitcher, [0, 0, 1, 2, 0, 3, 1, 0, 0])
    np.testing.assert_array_equal(d.exposure_type, [0, 0, 1, 2, 0, 3, 1, 4, 0])
    np.testing.assert_array_equal(d.exposure_pitcher_all, [0, 1, 2, 3, 0, 4, 5, 0, 0])
    np.testing.assert_array_equal(d.irrelevant_exposure, [0, 1, 1, 1, 0, 1, 4, 2, 0])
    # Pitch one is a take with no bat metrics; it still informs the later swing.
    assert not d.loc[0, "is_swing"]
    assert pd.isna(d.loc[0, "bat_speed"])
    assert d.loc[2, "is_swing"] and d.loc[2, "exposure_pa"] == 1
    assert d.loc[5, "exposure_sequence"] == 1  # second START>FF for this hitter/game
    assert d.loc[7, "exposure_sequence"] == 2


def test_automatic_count_events_do_not_create_physical_exposure_or_memory_decay():
    raw = pd.DataFrame([
        pitch(number=1, description="ball", balls=0, strikes=0),
        pitch(number=2, description="automatic_ball", balls=1, strikes=0),
        pitch(number=3, description="automatic_strike", balls=2, strikes=0),
        pitch(number=4, description="foul", balls=2, strikes=1),
    ])
    d = prepare_pitches(raw)
    # Export pitch_number preserves count-event positions; physical chronology
    # skips the penalties and never fabricates replacement pitch observations.
    np.testing.assert_array_equal(d.pitch_number, [1, 4])
    assert d.loc[1, "lag1_pitch_type"] == "FF"
    assert d.loc[1, "lag1_release_speed"] == 95
    assert pd.isna(d.loc[1, "lag2_pitch_type"])
    assert d.loc[1, "pa_prior_pitches"] == 1
    assert d.loc[1, "exposure_pa"] == 1
    assert d.loc[1, "exposure_pitcher"] == 1
    assert d.loc[1, "exposure_pitcher_all"] == 1
    assert d.loc[1, "similar_pitch_gap"] == 1
    assert d.loc[1, "kernel_exposure"] == pytest.approx(2**(-1/20))
    np.testing.assert_array_equal(d.future_exposure, [1, 0])
    # Count comes from the current exported state, including both penalties.
    assert d.loc[1, "balls"] == 2
    assert d.loc[1, "strikes"] == 1
    assert d.loc[1, "count"] == "2-1"


def test_future_mutation_does_not_change_past_predictors(all_pitch_history):
    original = prepare_pitches(all_pitch_history)
    changed_raw = all_pitch_history.copy()
    changed_raw.loc[7, ["pitch_type", "release_speed", "description", "pfx_x"]] = [
        "CU", 72.0, "swinging_strike", 2.0
    ]
    changed = prepare_pitches(changed_raw)
    # The explicitly labelled placebo is allowed to see the future; every other
    # value on earlier pitches must be identical, including kernels and lags.
    past_columns = original.columns.difference(["future_exposure"])
    assert_frame_equal(original.loc[:6, past_columns], changed.loc[:6, past_columns])


def test_appending_future_pitches_changes_only_future_placebo_for_past_rows():
    old_raw = pd.DataFrame([pitch(number=1), pitch(number=2)])
    new_raw = pd.concat([old_raw, pd.DataFrame([pitch(number=3)])], ignore_index=True)
    old, new = prepare_pitches(old_raw), prepare_pitches(new_raw)
    columns = old.columns.difference(["future_exposure"])
    assert_frame_equal(old[columns], new.loc[:1, columns])
    np.testing.assert_array_equal(old.future_exposure, [1, 0])
    np.testing.assert_array_equal(new.future_exposure, [2, 1, 0])


def test_shape_kernel_has_physical_scale_strict_past_and_documented_decay():
    d = prepare_pitches(pd.DataFrame([
        pitch(number=1),
        pitch(number=2, description="called_strike"),
        pitch(number=3, description="foul"),
        pitch(number=4, release_speed=115.0),
    ]))
    for h in [0.5, 1.0, 2.0]:
        for half in [5.0, 20.0]:
            key = f"kernel_h{h:g}_m{half:g}"
            decay = 2 ** (-1 / half)
            expected = [0.0, decay, decay + decay**2,
                        np.exp(-16 / h) * (decay + decay**2 + decay**3)]
            np.testing.assert_allclose(d[key], expected, rtol=1e-12, atol=1e-14)
    assert pd.isna(d.loc[0, "similar_pitch_gap"])
    np.testing.assert_array_equal(d.loc[1:2, "similar_pitch_gap"], [1, 1])
    assert pd.isna(d.loc[3, "similar_pitch_gap"])
    np.testing.assert_array_equal(d.kernel_exposure, d.kernel_h1_m20)


def test_other_batters_and_games_do_not_contribute_to_shape_memory():
    d = prepare_pitches(pd.DataFrame([
        pitch(pa=1, batter=10), pitch(pa=2, batter=20),
        pitch(pa=3, batter=10), pitch(game=101, batter=10),
    ]))
    np.testing.assert_allclose(d.kernel_exposure, [0, 0, 2**(-1/20), 0])
    # Memory lag is this batter's pitch exposures, not all game pitches or seconds.
    assert d.loc[2, "similar_pitch_gap"] == 1


def test_missing_shape_is_unknown_exposure_not_zero_and_cannot_inform_next_pitch():
    d = prepare_pitches(pd.DataFrame([
        pitch(number=1), pitch(number=2, pfx_x=np.nan), pitch(number=3),
    ]))
    assert not d.loc[1, "shape_valid"]
    assert pd.isna(d.loc[1, "kernel_exposure"])
    # Pitch two still advances exposure-time decay but adds no known similarity.
    np.testing.assert_allclose(d.loc[2, "kernel_exposure"], 2**(-2/20))
    assert d.loc[2, "similar_pitch_gap"] == 2
    assert d.loc[2, "exposure_pitcher"] == 2


def test_duplicate_identifiers_are_rejected(all_pitch_history):
    duplicate = pd.concat([all_pitch_history, all_pitch_history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate pitch identifiers"):
        prepare_pitches(duplicate)


def test_nonregular_games_and_missing_identifiers_are_excluded(all_pitch_history):
    extra = pd.DataFrame([pitch(game=102, game_type="S"), pitch(game=103, batter=np.nan)])
    result = prepare_pitches(pd.concat([all_pitch_history, extra], ignore_index=True))
    assert len(result) == len(all_pitch_history)


def test_trajectory_uses_same_physical_plane_across_2026_change():
    # Identical constant-velocity flight observed at two plate reference planes.
    # x(t)=x50+vx*t and z(t)=z50+vz*t, with t=(50-y)/100.
    x50, z50 = -0.5, 5.0
    rows = []
    for game, year, y_ref in [(100, 2025, 17/12), (101, 2026, 8.5/12)]:
        t_ref = (50-y_ref)/100
        rows.append(pitch(game=game, game_date=f"{year}-06-01",
                          plate_x=x50+2*t_ref, plate_z=z50-5*t_ref))
    d = prepare_pitches(pd.DataFrame(rows))
    for y in [23.8, 5.0]:
        expected_time = (50-y)/100
        np.testing.assert_allclose(d[f"traj_x_{y}"], x50+2*expected_time)
        np.testing.assert_allclose(d[f"traj_z_{y}"], z50-5*expected_time)
    np.testing.assert_allclose(d.vaa, np.degrees(np.arctan2(-5, 100)))
    np.testing.assert_allclose(d.haa, np.degrees(np.arctan2(2, 100)))
    expected_front_time = (50-17/12)/100
    np.testing.assert_allclose(d.plate_x, x50+2*expected_front_time)
    np.testing.assert_allclose(d.plate_z, z50-5*expected_front_time)


def test_physically_unreachable_or_backward_pitch_has_no_invented_trajectory():
    d = prepare_pitches(pd.DataFrame([
        pitch(number=1, vy0=-10.0, ay=30.0),
        pitch(number=2, vy0=100.0),
    ]))
    for field in ["vaa", "haa", "traj_x_23.8", "traj_z_23.8", "traj_x_5.0", "traj_z_5.0"]:
        assert d[field].isna().all(), field


def test_future_exposure_and_outcomes_are_not_expected_swing_inputs(all_pitch_history):
    d = prepare_pitches(all_pitch_history)
    current, _ = model_frame(d)
    with_history, _ = model_frame(d, history=True)
    forbidden = {"future_exposure", "is_whiff", "is_contact", "is_swing", "description",
                 "bat_speed", "swing_length", "miss_distance", "launch_speed"}
    assert not forbidden.intersection(current.columns)
    assert not forbidden.intersection(with_history.columns)
    assert not any(str(c).startswith(("lag", "exposure", "kernel")) for c in current)
    assert "lag1_whiff" in with_history
