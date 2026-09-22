"""Parameter recovery and comparison calibration in explicitly simulated worlds.

These are methodological checks, not MLB results. The raw-data simulations
include unequal numbers of independent groups, heteroscedastic error,
within-group dependence, changing pitch difficulty, and missing outcomes.
Missingness is ignorable given the included covariates; MNAR or unmeasured
pitcher responses are not validated by these experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.hierarchical import fit_hierarchy


def simulate_swings(players=30, zero_learning=False, seed=20260921):
    """Generate raw swings without selecting on unusually large first error.

    lambda is outcome units per one unit log(1 + prior exposures). Independent
    player truths follow Normal(.055,.09), or exactly zero in the null world.
    All groups are sampled before generating response noise. Return complete
    rows with an observed flag, preserving a direct missingness audit.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros(players) if zero_learning else rng.normal(0.055, 0.09, players)
    rows = []
    for player in range(players):
        # Log-uniform cluster totals make information differ substantially.
        n_groups = int(np.round(np.exp(rng.uniform(np.log(14), np.log(70)))))
        alpha = rng.normal(0.8, 0.2)
        player_noise = rng.uniform(0.30, 0.50)
        for group in range(n_groups):
            n = int(rng.integers(3, 10))
            exposures = np.log1p(np.arange(n))
            pitcher_quality = rng.normal()
            current_pitch = rng.normal(size=n) + 0.22 * exposures + 0.15 * pitcher_quality
            common_noise = rng.normal(0, 0.3)
            sd = player_noise * (1 + 0.2 * exposures)
            y = alpha - truth[player] * exposures + 0.35 * current_pitch + 0.2 * pitcher_quality + common_noise + rng.normal(0, sd)
            observed_probability = np.clip(0.90 - 0.035 * exposures - 0.025 * np.abs(current_pitch), 0.55, 0.97)
            observed = rng.random(n) < observed_probability
            for j in range(n):
                rows.append((player, group, exposures[j], current_pitch[j], pitcher_quality, y[j], bool(observed[j])))
    return {
        "data": pd.DataFrame(rows, columns=["player_id", "group", "exposure", "current_pitch", "pitcher_quality", "outcome", "observed"]),
        "truth": truth,
        "true_mu": 0.0 if zero_learning else 0.055,
        "true_tau": 0.0 if zero_learning else 0.09,
    }


def clustered_exposure_estimates(data):
    """OLS slopes with CR1 cluster-robust SEs, independently by player.

    The cluster finite-sample factor is G/(G-1) * (N-1)/(N-K). A normal
    hierarchy still approximates the slope likelihood; simulation diagnoses
    finite-cluster behavior rather than assuming nominal coverage.
    """
    output = []
    for player, frame in data[data["observed"]].groupby("player_id", sort=True):
        x = np.column_stack([np.ones(len(frame)), frame[["exposure", "current_pitch", "pitcher_quality"]].to_numpy()])
        y = frame["outcome"].to_numpy()
        inv = np.linalg.inv(x.T @ x)
        beta = inv @ x.T @ y
        residual = y - x @ beta
        codes, labels = pd.factorize(frame["group"])
        scores = np.zeros((len(labels), x.shape[1]))
        np.add.at(scores, codes, x * residual[:, None])
        g, n, k = len(labels), len(frame), x.shape[1]
        covariance = inv @ (scores.T @ scores) @ inv * g / (g - 1) * (n - 1) / (n - k)
        output.append({"player_id": player, "estimate": -float(beta[1]), "se": float(np.sqrt(covariance[1, 1])), "n_observed": n, "n_clusters": g})
    return pd.DataFrame(output)


def _calibration_table(records, replicates, rng):
    frame = pd.DataFrame(records)
    frame["bin"] = np.minimum((10 * frame["probability"]).astype(int), 9)
    output = []
    # Resample independent experiments, preserving dependence between pairs
    # involving the same hitter and between all pairs in one experiment.
    bootstrap_indices = rng.integers(0, replicates, size=(400, replicates))
    for label, bin_frame in frame.groupby("bin", sort=True):
        aggregate = bin_frame.groupby("replicate").agg(n=("truth", "size"), yes=("truth", "sum"), probability=("probability", "sum")).reindex(range(replicates), fill_value=0)
        n, yes, probability = [aggregate[c].to_numpy() for c in ("n", "yes", "probability")]
        denom = n[bootstrap_indices].sum(axis=1)
        valid = denom > 0
        rates = yes[bootstrap_indices].sum(axis=1)[valid] / denom[valid]
        errors = (yes - probability)[bootstrap_indices].sum(axis=1)[valid] / denom[valid]
        output.append({
            "bin_low": label / 10, "bin_high": (label + 1) / 10,
            "mean_probability": float(bin_frame.probability.mean()),
            "empirical_frequency": float(bin_frame.truth.mean()),
            "n_comparisons": len(bin_frame), "n_independent_experiments": int((n > 0).sum()),
            "frequency_ci90_low": float(np.quantile(rates, 0.05)), "frequency_ci90_high": float(np.quantile(rates, 0.95)),
            "calibration_error": float(bin_frame.truth.mean() - bin_frame.probability.mean()),
            "calibration_error_ci90_low": float(np.quantile(errors, 0.05)), "calibration_error_ci90_high": float(np.quantile(errors, 0.95)),
        })
    return pd.DataFrame(output)


def run_simulations(replicates=60, players=30, draws=1000, seed=20260921):
    """Run heterogeneous and zero-learning worlds, each with >=50 experiments.

    Returns tables plus actual example posterior draws for recovery/shrinkage
    plots. Pass replicates<50 only for a smoke test; summary labels this clearly.
    Pairwise calibration is against known synthetic truth. It is *not* an
    empirical claim of calibrated MLB player comparisons.
    """
    if replicates < 1 or players < 4:
        raise ValueError("replicates >=1 and players >=4 are required")
    rng = np.random.default_rng(seed)
    rows, pair_records = [], []
    example = None
    for null in (False, True):
        for replicate in range(replicates):
            simulation = simulate_swings(players, null, int(rng.integers(0, 2**32 - 1)))
            first_stage = clustered_exposure_estimates(simulation["data"])
            posterior = fit_hierarchy(first_stage["estimate"], first_stage["se"], player_ids=first_stage.player_id,
                                      draws=draws, grid_size=1201, seed=int(rng.integers(0, 2**32 - 1)))
            truth = simulation["truth"]
            lam = posterior["lambda"]
            low90, low50, high50, high90 = np.quantile(lam, [0.05, 0.25, 0.75, 0.95], axis=0)
            mu_low, mu_high = np.quantile(posterior["mu"], [0.05, 0.95])
            mean = lam.mean(axis=0)
            slope = first_stage.estimate.to_numpy()
            se = first_stage.se.to_numpy()
            rows.append({
                "scenario": "zero_learning" if null else "heterogeneous", "replicate": replicate,
                "n_players": players, "coverage90": float(np.mean((low90 <= truth) & (truth <= high90))),
                "coverage50": float(np.mean((low50 <= truth) & (truth <= high50))),
                "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                "unpooled_rmse": float(np.sqrt(np.mean((slope - truth) ** 2))),
                "mean_bias": float(np.mean(mean - truth)),
                "rank_spearman": np.nan if null else float(spearmanr(mean, truth).statistic),
                "mu_mean": float(posterior["mu"].mean()), "mu90_low": float(mu_low), "mu90_high": float(mu_high),
                "tau_mean": float(posterior["tau"].mean()), "true_mu": simulation["true_mu"], "true_tau": simulation["true_tau"],
                "mu_interval_excludes_zero": bool(mu_low > 0 or mu_high < 0),
                "first_stage_coverage90": float(np.mean(np.abs(slope - truth) <= 1.6448536269514722 * se)),
                "missing_fraction": float(1 - simulation["data"].observed.mean()),
                "median_observed": float(first_stage.n_observed.median()), "median_clusters": float(first_stage.n_clusters.median()),
            })
            if not null:
                ii, jj = np.triu_indices(players, 1)
                probabilities = (lam[:, ii] > lam[:, jj]).mean(axis=0)
                pair_records.extend({"replicate": replicate, "probability": float(p), "truth": int(truth[i] > truth[j])}
                                    for i, j, p in zip(ii, jj, probabilities))
                if example is None:
                    example = {"truth": truth, "estimates": slope, "ses": se, "counts": first_stage.n_observed.to_numpy(),
                               "posterior": posterior, "raw_data": simulation["data"], "first_stage": first_stage}
    results = pd.DataFrame(rows)
    calibration = _calibration_table(pair_records, replicates, rng)
    summary = {"seed": int(seed), "replicates_per_scenario": int(replicates), "players_per_replicate": int(players),
               "validation_run": bool(replicates >= 50), "calibration_target": "known synthetic truth, not MLB temporal outcomes",
               "missingness_assumption": "MAR given included exposure and pitch difficulty; not an MNAR sensitivity analysis"}
    for scenario, frame in results.groupby("scenario"):
        summary[scenario] = {
            "mean_coverage90": float(frame.coverage90.mean()),
            "coverage90_mcse_across_experiments": float(frame.coverage90.std(ddof=1) / np.sqrt(replicates)) if replicates > 1 else None,
            "mean_coverage50": float(frame.coverage50.mean()), "mean_rmse": float(frame.rmse.mean()),
            "mean_unpooled_rmse": float(frame.unpooled_rmse.mean()), "mean_bias": float(frame.mean_bias.mean()),
            "mean_mu": float(frame.mu_mean.mean()), "mean_tau": float(frame.tau_mean.mean()),
            "mu_excludes_zero_fraction": float(frame.mu_interval_excludes_zero.mean()),
            "mu_excludes_zero_count": int(frame.mu_interval_excludes_zero.sum()),
            "mean_first_stage_coverage90": float(frame.first_stage_coverage90.mean()),
        }
    return {"replicates": results, "calibration": calibration, "example": example, "summary": summary}
