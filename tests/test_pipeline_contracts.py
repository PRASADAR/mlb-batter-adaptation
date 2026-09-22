"""Small, model-free checks of temporal and inferential pipeline contracts.

The expensive CatBoost fits are replaced by a training-only mean predictor.
These tests check data routing and calculations, not predictive performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.models.expected as expected
from src.features.sequences import prepare_pitches
from src.models.hierarchical import fit_hierarchy, summarize_posterior


def baseline_sample():
    rng = np.random.default_rng(20260921)
    n = 1000
    pieces = []
    for block, date in enumerate(["2024-06-01", "2024-08-01", "2025-06-01", "2026-06-01"]):
        pieces.append(pd.DataFrame({
            "game_date": pd.to_datetime([date] * n),
            "season": int(date[:4]), "game_pk": block * 100 + np.arange(n) // 20,
            "is_swing": True,
            "bat_speed": 72 + rng.normal(0, 3, n),
            "swing_length": 7 + rng.normal(0, 0.4, n),
            "attack_direction": np.where(np.arange(n) % 2, 179.0, -179.0),
        }))
    return pd.concat(pieces, ignore_index=True)


@pytest.fixture
def cheap_config():
    return {"seed": 17, "catboost_iterations": 2, "max_training_swings": 100000}


@pytest.fixture
def mean_model(monkeypatch):
    calls = []

    def fit_predict(train, test, target, history=False, **kwargs):
        targets = [target] if isinstance(target, str) else list(target)
        calls.append({"train": train.copy(), "test": test.copy(),
                      "targets": targets, "history": history})
        pred = np.broadcast_to(train[targets].mean().to_numpy(), (len(test), len(targets))).copy()
        metadata = {"training_n": len(train), "training_end": str(train.game_date.max().date())}
        return pred, object(), metadata

    monkeypatch.setattr(expected, "fit_predict", fit_predict)
    return calls


def test_baseline_partitions_training_calibration_development_and_holdout(mean_model, cheap_config):
    data = baseline_sample()
    result, metrics, metadata, _ = expected.swing_baseline(data, ["bat_speed", "swing_length"], cheap_config)
    assert len(mean_model) == 2
    assert [c["history"] for c in mean_model] == [False, True]
    for call in mean_model:
        assert call["train"].game_date.lt("2024-07-01").all()
        assert call["test"].game_date.ge("2024-07-01").all()
        assert set(call["train"].game_pk).isdisjoint(call["test"].game_pk)
    assert metadata["calibration_n"] == 1000
    assert len(result) == 3000
    assert set(result.season) == {2024, 2025, 2026}
    assert set(metrics.season) == {2024, 2025, 2026}
    calibration = result.loc[result.season.eq(2024), "distortion"]
    assert calibration.mean() == pytest.approx(0, abs=1e-12)
    assert calibration.std(ddof=0) == pytest.approx(1)


def test_holdout_outcomes_cannot_change_baseline_or_covariance(mean_model, cheap_config):
    data = baseline_sample()
    first, _, metadata, _ = expected.swing_baseline(data, ["bat_speed", "swing_length"], cheap_config)
    altered = data.copy()
    altered.loc[altered.season.eq(2026), "bat_speed"] += 10
    second, _, changed_metadata, _ = expected.swing_baseline(altered, ["bat_speed", "swing_length"], cheap_config)
    for key in ["residual_scale", "covariance", "calibration_distance_mean", "calibration_distance_sd"]:
        np.testing.assert_allclose(metadata[key], changed_metadata[key])
    columns = ["distortion", "expected_bat_speed", "expected_swing_length", "resid_bat_speed"]
    pd.testing.assert_frame_equal(first.loc[first.season.lt(2026), columns],
                                  second.loc[second.season.lt(2026), columns])
    np.testing.assert_array_equal(first.expected_bat_speed, second.expected_bat_speed)


def test_attack_direction_wraps_at_180_degrees(mean_model, cheap_config):
    data = baseline_sample()
    result, _, _, _ = expected.swing_baseline(data, ["bat_speed", "attack_direction"], cheap_config)
    for call in mean_model:
        assert "attack_direction" not in call["targets"]
        assert call["targets"][-2:] == ["direction_sin", "direction_cos"]
    # Opposite representations near +/-180 describe nearly the same direction.
    assert result.resid_attack_direction.abs().max() == pytest.approx(1)
    assert result.expected_attack_direction.abs().eq(180).all()


def test_nuisance_folds_hold_out_games_and_2026_uses_only_2025(mean_model, cheap_config):
    rng = np.random.default_rng(42)
    data = baseline_sample().loc[lambda x: x.season.ge(2025)].copy().reset_index(drop=True)
    data["distortion"] = rng.normal(size=len(data))
    data["log_exposure_pitcher"] = rng.uniform(size=len(data))
    # A future variable may be a deliberately marked falsification target.
    data["log_future_exposure"] = rng.uniform(size=len(data))
    targets = ["distortion", "log_exposure_pitcher", "log_future_exposure"]
    result, metadata = expected.nuisance_residuals(data, targets, cheap_config)
    assert len(mean_model) == 4
    validation_games = []
    for call in mean_model[:3]:
        assert call["train"].season.eq(2025).all()
        assert call["test"].season.eq(2025).all()
        assert set(call["train"].game_pk).isdisjoint(call["test"].game_pk)
        validation_games.extend(call["test"].game_pk.unique().tolist())
    assert len(validation_games) == len(set(validation_games))
    assert set(validation_games) == set(data.loc[data.season.eq(2025), "game_pk"])
    final = mean_model[-1]
    assert final["train"].season.eq(2025).all()
    assert final["test"].season.eq(2026).all()
    assert metadata["training_end"] < "2026-01-01"
    assert result[[f"adj_{t}" for t in targets]].notna().all().all()


def test_actual_fit_predict_never_passes_current_outcomes_or_future_to_estimator(monkeypatch):
    captured = []

    class RecordingRegressor:
        def __init__(self, **kwargs):
            self.width = 1
            self.parameters = kwargs

        def get_params(self):
            return self.parameters

        def fit(self, frame, response):
            captured.append(frame.copy())
            self.width = 1 if np.ndim(response) == 1 else response.shape[1]

        def predict(self, frame):
            captured.append(frame.copy())
            return np.zeros((len(frame), self.width))

    monkeypatch.setattr(expected, "CatBoostRegressor", RecordingRegressor)
    raw = pd.DataFrame({
        "game_pk": [100, 100, 101], "at_bat_number": [1, 1, 1],
        "pitch_number": [1, 2, 1], "batter": [10, 10, 10], "pitcher": [20, 20, 20],
        "pitch_type": ["FF", "SL", "FF"], "description": ["foul", "swinging_strike", "foul"],
        "game_date": ["2024-06-01", "2024-06-01", "2026-06-01"],
        "bat_speed": [70.0, 75.0, 71.0], "swing_length": [7.0, 7.5, 7.1],
    })
    data = prepare_pitches(raw)
    expected.fit_predict(data.iloc[:2], data.iloc[2:], "bat_speed", history=True, iterations=1)
    forbidden = {"bat_speed", "swing_length", "description", "is_whiff", "is_contact",
                 "is_swing", "future_exposure", "miss_distance", "launch_speed"}
    assert len(captured) == 2
    for frame in captured:
        assert not forbidden.intersection(frame.columns)
        assert "lag1_whiff" in frame.columns


def slope_sample():
    # Each game has centered exposures, a game intercept, and a small game slope
    # perturbation. Symmetric perturbations leave the exact overall slope known.
    rows = []
    for game, slope_noise in enumerate(np.linspace(-0.2, 0.2, 30)):
        for x in [-1.0, 0.0, 1.0]:
            rows.append({"batter": 10, "game_pk": game,
                         "adj_log_exposure_pitcher": x,
                         "adj_distortion": 7 + game / 10 - 0.4*x + slope_noise*x})
    return pd.DataFrame(rows)


def test_player_slope_sign_cluster_uncertainty_and_intercept_invariance():
    data = slope_sample()
    fitted = expected.player_slopes(data)
    assert len(fitted) == 1
    assert fitted.loc[0, "estimate"] == pytest.approx(0.4)
    assert fitted.loc[0, "se"] > 0
    assert fitted.loc[0, "games"] == 30
    assert fitted.loc[0, "information"] == pytest.approx(60)
    shifted = data.copy()
    shifted["adj_log_exposure_pitcher"] += 20
    shifted["adj_distortion"] -= 13
    refit = expected.player_slopes(shifted)
    np.testing.assert_allclose(fitted[["estimate", "se", "information"]],
                               refit[["estimate", "se", "information"]])


def test_duplicate_observations_within_same_games_do_not_fake_independent_precision():
    data = slope_sample()
    base = expected.player_slopes(data).iloc[0]
    repeated = expected.player_slopes(pd.concat([data, data], ignore_index=True)).iloc[0]
    assert repeated.estimate == pytest.approx(base.estimate)
    assert repeated.games == base.games
    # Only the finite-sample correction changes, not sqrt(2) extra precision.
    assert repeated.se / base.se == pytest.approx(1, rel=0.01)


def test_uninformative_players_do_not_get_spurious_finite_slopes():
    data = slope_sample()
    assert expected.player_slopes(data.loc[data.game_pk.lt(7)]).empty
    constant = data.copy()
    constant["adj_log_exposure_pitcher"] = 0
    assert expected.player_slopes(constant).empty


def test_posterior_summary_accepts_pipeline_integer_name_and_count_mappings():
    posterior = fit_hierarchy([0.1, 0.0, -0.1], [0.04, 0.1, 0.2],
                              player_ids=np.array([10, 20, 30], dtype=np.int64), draws=200, seed=17)
    summary = summarize_posterior(posterior, names={10: "A Batter", 20: "B Batter"},
                                  counts={10: 120, 20: 80, 30: 45}).set_index("player_id")
    assert summary.loc[10, "player_name"] == "A Batter"
    assert summary.loc[30, "player_name"] == "30"
    assert summary.loc[20, "n_exposures"] == 80
    assert summary["rank_population_size"].eq(3).all()


def test_full_comparisons_preserve_joint_order_and_rank_mass():
    import runpy
    from pathlib import Path
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/11_pairwise_comparisons.py'))
    p={'lambda':np.array([[3.,2.,1.],[1.,3.,2.],[3.,1.,2.],[2.,1.,3.]])}
    pair,ranks=api['full_comparison_tables'](p)
    np.testing.assert_allclose(pair,[[0,.75,.5],[.25,0,.5],[.5,.5,0]])
    np.testing.assert_allclose(ranks,[[.5,.25,.25],[.25,.25,.5],[.25,.5,.25]])
