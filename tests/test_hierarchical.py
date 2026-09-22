"""Numerical posterior and identification-guardrail checks."""

import numpy as np
import pytest

from src.evaluation.recovery import clustered_exposure_estimates, run_simulations, simulate_swings
from src.models.hierarchical import compare_hitters, fit_hierarchy, measurement_error_persistence, pairwise_probability, summarize_posterior


def test_shapes_reproducibility_tail_and_grid_convergence():
    b, se = np.array([-.15, .08, .25, .02]), np.array([.03, .05, .08, .1])
    a = fit_hierarchy(b, se, draws=3000, seed=8)
    again = fit_hierarchy(b, se, draws=3000, seed=8)
    assert a["lambda"].shape == (3000, 4)
    assert a["observed_predictive"].shape == (3000, 4)
    np.testing.assert_array_equal(a["lambda"], again["lambda"])
    assert a["diagnostics"]["tail_mass_last_1pct"] < 1e-7
    assert a["diagnostics"]["max_mean_change_at_double_resolution"] < 1e-5
    assert np.all(a["tau"] > 0)  # the quadrature must not invent an atom at zero


def test_near_uninformative_likelihood_recovers_prior():
    posterior = fit_hierarchy(np.zeros(12), np.full(12, 1000.), draws=30000, seed=11)
    assert abs(posterior["mu"].mean()) < .006
    assert abs(posterior["mu"].std() - .2) < .006
    assert abs(posterior["tau"].mean() - .15 * np.sqrt(2 / np.pi)) < .003
    assert abs(posterior["lambda"].var() - (.2**2 + .15**2)) < .003


def test_precise_players_are_recovered_and_pairwise_order_is_coherent():
    posterior = fit_hierarchy([-.2, 0, .2], [.001] * 3, player_ids=[11, 22, 33], draws=4000)
    np.testing.assert_allclose(posterior["lambda"].mean(axis=0), [-.2, 0, .2], atol=.001)
    assert pairwise_probability(posterior, 33, 11) == 1
    assert pairwise_probability(posterior, 11, 33) == 0
    assert pairwise_probability(posterior, 11, 11) == 0
    summary = summarize_posterior(posterior, names={33: "Third"}, counts=[10, 20, 30])
    assert summary.iloc[0].player_id == 33
    assert summary.iloc[0].rank_median == 1
    assert summary.iloc[0].player_name == "Third"
    comparison = compare_hitters(posterior, 33, 11)
    assert comparison["cri90"][0] > .39


def test_low_information_shrinks_and_has_more_uncertainty():
    posterior = fit_hierarchy([-.1, .1, .2, .2], [.01, .01, .01, .3], draws=10000)
    mean = posterior["lambda"].mean(axis=0)
    sd = posterior["lambda"].std(axis=0)
    assert mean[3] < mean[2]
    assert sd[3] > 4 * sd[2]


@pytest.mark.parametrize("b,se", [([0], [.1]), ([0, 1], [0, .1]), ([np.nan, 1], [.1, .1]), ([0, 1], [.1])])
def test_invalid_inputs_are_rejected(b, se):
    with pytest.raises(ValueError):
        fit_hierarchy(b, se)


def test_simulation_zero_learning_does_not_select_extreme_initial_error():
    simulations = [simulate_swings(players=20, zero_learning=True, seed=seed) for seed in range(10)]
    estimates = np.concatenate([clustered_exposure_estimates(sim["data"]).estimate.to_numpy() for sim in simulations])
    # Independent raw-data worlds; an extreme-first-error design would create
    # systematic correction. This tests the generator and first-stage slope.
    assert abs(estimates.mean()) < .02


def test_simulation_smoke_produces_recovery_and_calibration():
    result = run_simulations(replicates=2, players=8, draws=500, seed=32)
    assert len(result["replicates"]) == 4
    assert result["summary"]["validation_run"] is False
    assert result["example"]["posterior"]["lambda"].shape == (500, 8)
    assert result["calibration"].n_comparisons.sum() == 2 * 28
    assert result["calibration"].mean_probability.between(0, 1).all()


def test_persistence_matches_closed_form_with_equal_known_errors():
    # With common known SEs and an interior covariance, the latent covariance
    # MLE is the empirical covariance minus the known measurement variances.
    rng = np.random.default_rng(261)
    observations = rng.multivariate_normal([.03, -.02], [[.04, .021], [.021, .06]], size=300)
    ses = np.array([.04, .06])
    covariance = np.cov(observations, rowvar=False, ddof=0) - np.diag(ses**2)
    expected = covariance[0, 1] / np.sqrt(np.prod(np.diag(covariance)))
    result = measurement_error_persistence(observations[:, 0], np.full(300, ses[0]),
                                           observations[:, 1], np.full(300, ses[1]), bootstrap=0)
    assert result["latent_correlation"] == pytest.approx(expected, abs=2e-6)
    np.testing.assert_allclose([result["latent_sd_a"], result["latent_sd_b"]], np.sqrt(np.diag(covariance)), atol=2e-6)
    assert result["optimization_diagnostics"]["best_converged"]
    assert not result["weak_identification"]
    assert result["optimization_diagnostics"]["starts_within_1e_minus_6_of_best"] >= 15


def test_persistence_near_zero_heterogeneity_is_flagged_as_unidentified():
    # Extremely small scatter compared with the known first-stage error puts
    # the latent variances at zero. Even correlated observations cannot identify
    # a latent rho here; returning the optimizer's arbitrary rho without this
    # guardrail would misleadingly suggest a meaningful persistence estimate.
    x = .03 + .002 * np.sin(np.arange(80))
    y = .04 + .002 * np.cos(np.arange(80))
    result = measurement_error_persistence(x, np.full(80, .1), y, np.full(80, .12), bootstrap=12, seed=18)
    assert result["weak_identification"]
    assert result["weak_identification_flags"]["point_signal_sd_below_10pct_median_se"]
    assert result["rho_likelihood_profile"]["all_grid_values_supported"]
    assert result["rho_likelihood_profile"]["nll_range"] < 1e-4
    assert result["bootstrap_diagnostics"]["signal_sd_below_10pct_median_se_count"] == 12
    assert "must not be treated as a reliable" in result["rho_interpretation"]
    # Multiple informative-scale variance starts are required even when moment
    # estimates vanish; varying rho at one nearly-zero SD start is insufficient.
    starts = np.asarray(result["optimization_diagnostics"]["relative_sd_starts"])
    assert np.max(starts) >= 1
    assert result["optimization_diagnostics"]["converged_starts"] >= 15
