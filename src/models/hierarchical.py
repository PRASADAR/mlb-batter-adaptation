"""Measurement-error partial pooling of conditional exposure slopes.

The input estimate is the *negative* exposure slope: positive lambda means a
decline in the supplied distortion outcome per unit of the supplied exposure.
It is neither a percentage correction nor an identified causal learning rate.

The likelihood is b_i | lambda_i ~ Normal(lambda_i, se_i**2), independently
across hitters. First-stage SEs are treated as known. Hyperparameter uncertainty
is integrated, but this does not account for estimated first-stage SEs,
cross-hitter dependence, or uncertainty in the expected-swing fit. A normal
first-stage approximation needs enough independent clusters to be credible.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _inputs(estimates, ses):
    b = np.asarray(estimates, dtype=float)
    s = np.asarray(ses, dtype=float)
    if b.ndim != 1 or s.shape != b.shape or b.size < 2:
        raise ValueError("estimates and ses must be equal 1D arrays with >=2 players")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("estimates must be finite and standard errors strictly positive")
    return b, s


def _tau_grid(b, s, mu_scale, tau_scale, grid_size, upper):
    # A squared grid resolves the small-tau boundary without a large array.
    tau = upper * np.linspace(0.0, 1.0, grid_size) ** 2
    variance = s[None, :] ** 2 + tau[:, None] ** 2
    precision = 1.0 / mu_scale**2 + (1.0 / variance).sum(axis=1)
    mu_var = 1.0 / precision
    weighted_b = (b[None, :] / variance).sum(axis=1)
    mu_mean = mu_var * weighted_b
    # Determinant lemma / completing the square integrates mu analytically.
    log_density = -0.5 * (
        np.log(variance).sum(axis=1)
        + np.log(mu_scale**2)
        + np.log(precision)
        + (b[None, :] ** 2 / variance).sum(axis=1)
        - weighted_b**2 / precision
        + (tau / tau_scale) ** 2
    )
    density = np.exp(log_density - np.max(log_density))
    dx = np.diff(tau)
    integration_weights = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    node_weights = density * integration_weights
    node_weights /= node_weights.sum()
    return tau, density, node_weights, mu_mean, mu_var


def _sample_piecewise_density(rng, grid, density, draws):
    """Sample the linearly interpolated density without an atom at tau=0."""
    dx = np.diff(grid)
    areas = (density[:-1] + density[1:]) * dx / 2
    cdf = np.r_[0.0, np.cumsum(areas)]
    u = rng.random(draws) * cdf[-1]
    index = np.minimum(np.searchsorted(cdf, u, side="right") - 1, len(dx) - 1)
    local_area = u - cdf[index]
    f0 = density[index]
    slope = (density[index + 1] - f0) / dx[index]
    root = np.sqrt(np.maximum(0.0, f0**2 + 2 * slope * local_area))
    offset = np.divide(2 * local_area, f0 + root, out=np.zeros_like(u), where=(f0 + root) > 0)
    return grid[index] + np.clip(offset, 0.0, dx[index])


def fit_hierarchy(
    estimates,
    ses,
    player_ids=None,
    draws=4000,
    seed=20260921,
    mu_scale=0.2,
    tau_scale=0.15,
    grid_size=2401,
):
    """Fit normal measurement-error hierarchy by deterministic 1D quadrature.

    Priors: mu ~ Normal(0, mu_scale), tau ~ HalfNormal(tau_scale).
    Lambda_i ~ Normal(mu, tau). Both prior scales use outcome/exposure units.
    Independent posterior draws propagate mu and tau uncertainty. The grid
    includes zero, expands until its tail is negligible, and is checked at
    double resolution. R-hat, chain ESS, and divergences do not apply: no MCMC
    is used. Monte Carlo standard error still applies to summaries of draws.
    """
    b, s = _inputs(estimates, ses)
    if draws < 2 or int(draws) != draws or grid_size < 101 or int(grid_size) != grid_size:
        raise ValueError("draws must be an integer >=2; grid_size an integer >=101")
    if not np.isfinite(mu_scale + tau_scale) or mu_scale <= 0 or tau_scale <= 0:
        raise ValueError("prior scales must be positive and finite")
    ids = np.asarray(player_ids if player_ids is not None else np.arange(len(b)))
    if ids.ndim != 1 or len(ids) != len(b) or len(set(ids.tolist())) != len(ids):
        raise ValueError("player_ids must contain one unique identifier per estimate")
    upper = max(4 * tau_scale, 2 * float(np.std(b)), 0.05)
    for expansion in range(12):
        grid, density, weights, mu_mean, mu_var = _tau_grid(b, s, mu_scale, tau_scale, grid_size, upper)
        if density[-1] < 1e-10 and weights[int(0.99 * grid_size) :].sum() < 1e-7:
            break
        upper *= 2
    else:
        raise RuntimeError("tau integration range failed to bound the posterior tail")
    # A vanishing numerical mean alone cannot diagnose an unresolved mode at
    # zero. Refine explicitly when the first two nodes hold appreciable mass.
    refinements = 0
    while weights[:2].sum() > 0.025 and refinements < 5:
        grid_size = 2 * grid_size - 1
        grid, density, weights, mu_mean, mu_var = _tau_grid(b, s, mu_scale, tau_scale, grid_size, upper)
        refinements += 1
    if weights[:2].sum() > 0.025:
        raise RuntimeError("tau posterior is unresolved at zero; increase grid_size")
    refined = _tau_grid(b, s, mu_scale, tau_scale, 2 * grid_size - 1, upper)
    fine_grid, _, fine_weights, fine_mu, _ = refined
    tau_mean = float(weights @ grid)
    mean_mu = float(weights @ mu_mean)
    quadrature_delta = max(abs(tau_mean - fine_weights @ fine_grid), abs(mean_mu - fine_weights @ fine_mu))
    if quadrature_delta > 1e-4:
        raise RuntimeError("quadrature grid too coarse; increase grid_size")
    rng = np.random.default_rng(seed)
    tau_draws = _sample_piecewise_density(rng, grid, density, int(draws))
    obs_var = s[None, :] ** 2 + tau_draws[:, None] ** 2
    conditional_mu_var = 1.0 / (1.0 / mu_scale**2 + (1.0 / obs_var).sum(axis=1))
    conditional_mu_mean = conditional_mu_var * (b[None, :] / obs_var).sum(axis=1)
    mu_draws = rng.normal(conditional_mu_mean, np.sqrt(conditional_mu_var))
    shrink = tau_draws[:, None] ** 2 / obs_var
    lambda_mean = shrink * b[None, :] + (1.0 - shrink) * mu_draws[:, None]
    lambda_sd = np.sqrt(shrink) * s[None, :]
    lambda_draws = rng.normal(lambda_mean, lambda_sd)
    return {
        "lambda": lambda_draws,
        "mu": mu_draws,
        "tau": tau_draws,
        "player_ids": ids,
        "observed_estimates": b,
        "observed_ses": s,
        "observed_predictive": rng.normal(lambda_draws, s[None, :]),
        "diagnostics": {
            "method": "analytic mu integration; trapezoid quadrature in tau; independent conditional draws",
            "grid_size": int(grid_size),
            "grid_spacing_min": float(np.diff(grid).min()),
            "grid_spacing_max": float(np.diff(grid).max()),
            "boundary_refinements": refinements,
            "tau_upper": float(upper),
            "upper_expansions": int(expansion),
            "tail_mass_last_1pct": float(weights[int(0.99 * grid_size) :].sum()),
            "endpoint_density_relative_to_mode": float(density[-1]),
            "max_mean_change_at_double_resolution": float(quadrature_delta),
            "quadrature_tau_mean": tau_mean,
            "quadrature_mu_mean": mean_mu,
            "independent_draws": int(draws),
            "mu_mean_mcse": float(np.std(mu_draws, ddof=1) / np.sqrt(draws)),
            "mu_prior_sd": float(mu_scale),
            "tau_halfnormal_scale": float(tau_scale),
            "mcmc_diagnostics": "not applicable: no Markov chains",
            "inference_scope": "conditional on first-stage slope estimates, their SEs, and independence approximation; observational association",
        },
    }


def _metadata(values, ids, default):
    if values is None:
        return [default(x) for x in ids]
    if isinstance(values, Mapping):
        return [values.get(x, default(x)) for x in ids]
    values = list(values)
    if len(values) != len(ids):
        raise ValueError("metadata must have one value per player")
    return values


def summarize_posterior(posterior, names=None, counts=None):
    """Uncertainty-aware player summaries; rank one is the largest lambda.

    The league mean is the model population mu. The league median is the
    draw-specific median of the included player population, not all MLB hitters.
    Top fractions are defined with ceil(N * fraction), documented in output.
    """
    lam = np.asarray(posterior["lambda"])
    mu = np.asarray(posterior["mu"])
    ids = np.asarray(posterior["player_ids"])
    if lam.ndim != 2 or lam.shape[1] != len(ids) or mu.shape != (len(lam),):
        raise ValueError("incompatible posterior dimensions")
    n = len(ids)
    ranks = np.argsort(np.argsort(-lam, axis=1), axis=1) + 1
    quantiles = np.quantile(lam, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    rank_quantiles = np.quantile(ranks, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return pd.DataFrame({
        "player_id": ids,
        "player_name": _metadata(names, ids, str),
        "n_exposures": _metadata(counts, ids, lambda _: np.nan),
        "posterior_mean": lam.mean(axis=0),
        "posterior_median": quantiles[2],
        "posterior_sd": lam.std(axis=0, ddof=1),
        "cri90_low": quantiles[0],
        "cri50_low": quantiles[1],
        "cri50_high": quantiles[3],
        "cri90_high": quantiles[4],
        "prob_above_league_mean": (lam > mu[:, None]).mean(axis=0),
        "prob_above_league_median": (lam > np.median(lam, axis=1)[:, None]).mean(axis=0),
        "prob_positive": (lam > 0).mean(axis=0),
        "prob_top_quartile": (ranks <= int(np.ceil(n / 4))).mean(axis=0),
        "prob_top_decile": (ranks <= int(np.ceil(n / 10))).mean(axis=0),
        "prob_bottom_decile": (ranks > n - int(np.ceil(n / 10))).mean(axis=0),
        "rank_median": rank_quantiles[2],
        "rank_cri50_low": rank_quantiles[1],
        "rank_cri50_high": rank_quantiles[3],
        "rank_cri90_low": rank_quantiles[0],
        "rank_cri90_high": rank_quantiles[4],
        "rank_population_size": n,
    }).sort_values("posterior_mean", ascending=False, ignore_index=True)


def _player_index(posterior, player):
    match = np.flatnonzero(np.asarray(posterior["player_ids"]) == player)
    if len(match) != 1:
        raise KeyError(f"unknown or ambiguous player identifier: {player!r}")
    return int(match[0])


def pairwise_probability(posterior, player_a, player_b):
    """Strict P(lambda_a > lambda_b | data); a self-comparison is zero."""
    a, b = _player_index(posterior, player_a), _player_index(posterior, player_b)
    diff = posterior["lambda"][:, a] - posterior["lambda"][:, b]
    return float(np.mean(diff > 0))


def compare_hitters(posterior, player_a, player_b):
    """Return paired posterior comparison; plot difference_draws as desired."""
    a, b = _player_index(posterior, player_a), _player_index(posterior, player_b)
    diff = posterior["lambda"][:, a] - posterior["lambda"][:, b]
    q = np.quantile(diff, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "player_a": player_a,
        "player_b": player_b,
        "probability_a_greater": pairwise_probability(posterior, player_a, player_b),
        "median_difference": float(q[2]),
        "cri50": [float(q[1]), float(q[3])],
        "cri90": [float(q[0]), float(q[4])],
        "difference_draws": diff,
    }


def measurement_error_persistence(estimates_a, ses_a, estimates_b, ses_b, bootstrap=100, seed=20260921):
    """Bivariate latent-normal persistence with known, independent error SEs.

    Jointly estimates means, between-player SDs, and latent rho by maximum
    likelihood. Means are profiled analytically; scaling and multistart
    optimization leave the likelihood unchanged. Bootstrap resamples *players*
    (not pitches). With weak heterogeneity, a numerical rho and bootstrap
    quantiles are not evidence that correlation is identified. The fixed-rho
    likelihood profile is a diagnostic, not a boundary-valid confidence set.
    Period estimation errors must be independent.
    """
    a, sa = _inputs(estimates_a, ses_a)
    b, sb = _inputs(estimates_b, ses_b)
    if a.shape != b.shape or len(a) < 10:
        raise ValueError("at least 10 matched players are required")
    if int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")

    def fit(x, sx, y, sy, profile=False):
        scales = np.array([np.median(sx), np.median(sy)])
        x, sx, y, sy = x / scales[0], sx / scales[0], y / scales[1], sy / scales[1]
        # Preserve the original physical SD and Fisher-z optimization bounds.
        bounds = [(-10 - np.log(scale), 2 - np.log(scale)) for scale in scales] + [(-3.8, 3.8)]

        def objective(theta):
            ta, tb, rho = np.exp(theta[0]), np.exp(theta[1]), np.tanh(theta[2])
            va, vb, cov = ta**2 + sx**2, tb**2 + sy**2, rho * ta * tb
            det = va * vb - cov**2
            waa, wbb, wab = vb / det, va / det, -cov / det
            precision = np.array([[waa.sum(), wab.sum()], [wab.sum(), wbb.sum()]])
            means = np.linalg.solve(precision, [(waa * x + wab * y).sum(), (wab * x + wbb * y).sum()])
            ma, mb = means
            da, db = x - ma, y - mb
            ua, ub = waa * da + wab * db, wab * da + wbb * db
            nll = 0.5 * np.sum(np.log(det) + da * ua + db * ub)
            # Envelope theorem: the derivative through the optimal means is zero.
            ga = 0.5 * np.sum(waa - ua**2)
            gb = 0.5 * np.sum(wbb - ub**2)
            gc = np.sum(wab - ua * ub)
            gradient = np.array([2 * ta**2 * ga + cov * gc, 2 * tb**2 * gb + cov * gc,
                                 (1 - rho**2) * ta * tb * gc])
            return float(nll), gradient

        moment = np.sqrt(np.maximum(.1**2, [np.var(x) - np.mean(sx**2), np.var(y) - np.mean(sy**2)]))
        # The variance score vanishes near zero in log-SD coordinates. Starting
        # every fit at .001 in physical units can falsely look converged there.
        sd_starts = np.unique(np.array([moment, [.05, .05], [.25, .25], [1., 1.], [.25, 1.], [1., .25]]), axis=0)
        options = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000, "maxls": 50}
        solutions = []
        for sd_start in sd_starts:
            for rho_start in (-0.6, 0.0, 0.6):
                start = np.r_[np.log(sd_start), np.arctanh(rho_start)]
                result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
                if np.isfinite(result.fun):
                    solutions.append(result)
        if not solutions:
            return None
        # Keep the best finite likelihood, reporting its convergence separately;
        # a worse 'success' flag must never silently replace a better likelihood.
        best = min(solutions, key=lambda r: r.fun)
        profile_result = None
        if profile:
            rho_grid = np.unique(np.r_[np.linspace(-np.tanh(3.8), np.tanh(3.8), 41), np.tanh(best.x[2])])
            profile_fits = []
            for rho in rho_grid:
                z = np.arctanh(rho)

                def fixed_objective(log_sds):
                    value, gradient = objective(np.r_[log_sds, z])
                    return value, gradient[:2]

                candidates = [minimize(fixed_objective, start, jac=True, method="L-BFGS-B", bounds=bounds[:2], options=options)
                              for start in [best.x[:2], np.log([.25, .25]), np.log([1., 1.]), np.log([.25, 1.]), np.log([1., .25])]]
                selected = min((r for r in candidates if np.isfinite(r.fun)), key=lambda r: r.fun)
                profile_fits.append(selected)
                # A diagnostic grid also serves as a numerical cross-check.
                if selected.fun < best.fun - 1e-8:
                    result = minimize(objective, np.r_[selected.x, z], jac=True, method="L-BFGS-B", bounds=bounds, options=options)
                    solutions.append(result)
                    if np.isfinite(result.fun) and result.fun < best.fun:
                        best = result
            profile_nll = np.array([r.fun for r in profile_fits])
            delta = np.maximum(0., profile_nll - best.fun)
            cutoff = 1.352771727
            support_indices = np.flatnonzero(delta <= cutoff)
            support = rho_grid[support_indices]
            support_range = [float(support.min()), float(support.max())] if len(support) else []
            if len(support):
                first, last = support_indices[0], support_indices[-1]
                # Interpolate crossings so a warning does not depend on which
                # side of a coarse rho grid point its threshold happens to lie.
                if first > 0:
                    support_range[0] = float(rho_grid[first - 1] + (cutoff - delta[first - 1]) *
                                             (rho_grid[first] - rho_grid[first - 1]) / (delta[first] - delta[first - 1]))
                if last < len(rho_grid) - 1:
                    support_range[1] = float(rho_grid[last] + (cutoff - delta[last]) *
                                             (rho_grid[last + 1] - rho_grid[last]) / (delta[last + 1] - delta[last]))
            profile_result = {
                "rho_grid": rho_grid.tolist(), "negative_log_likelihood_difference": delta.tolist(),
                "nll_range": float(np.ptp(profile_nll)),
                "likelihood_support_nll_cutoff": cutoff,
                "likelihood_support_range": support_range,
                "likelihood_support_width": float(np.ptp(support_range)) if len(support) else 0.,
                "support_endpoint_method": "linear interpolation between fixed-rho grid points; outermost support if disconnected",
                "all_grid_values_supported": bool(np.all(delta <= cutoff)),
                "converged_grid_fits": int(sum(r.success for r in profile_fits)),
                "scope": "Fixed-rho profile diagnostic; usual chi-square confidence calibration is not assumed near zero latent variance.",
            }
        estimate = np.r_[np.tanh(best.x[2]), np.exp(best.x[:2]) * scales]
        diagnostics = {
            "method": "median-SE scaling; analytically profiled means; analytic-gradient multistart L-BFGS-B",
            "finite_starts": len(solutions), "converged_starts": int(sum(r.success for r in solutions)),
            "best_converged": bool(best.success), "best_message": str(best.message),
            "best_iterations": int(best.nit), "best_gradient_max_abs": float(np.max(np.abs(best.jac))),
            "negative_log_likelihood": float(best.fun + len(x) * np.log(scales).sum()),
            "log_likelihood_constant_omitted": "n * log(2*pi)",
            "multistart_nll_range": float(np.ptp([r.fun for r in solutions])),
            "starts_within_1e_minus_6_of_best": int(sum(r.fun - best.fun <= 1e-6 for r in solutions)),
            "median_se_scaling": scales.tolist(), "relative_sd_starts": sd_starts.tolist(),
            "rho_starts": [-.6, 0., .6], "optimizer_options": options,
            "physical_sd_bounds": [float(np.exp(-10)), float(np.exp(2))],
            "rho_bounds": [-float(np.tanh(3.8)), float(np.tanh(3.8))],
        }
        return estimate, diagnostics, profile_result

    fitted = fit(a, sa, b, sb, profile=True)
    if fitted is None:
        raise RuntimeError("persistence likelihood failed to optimize")
    estimate, diagnostics, profile = fitted
    rng = np.random.default_rng(seed)
    boot = []
    boot_diagnostics = []
    boot_ratios = []
    for _ in range(int(bootstrap)):
        index = rng.integers(0, len(a), len(a))
        fitted = fit(a[index], sa[index], b[index], sb[index])
        if fitted is not None:
            boot.append(fitted[0])
            boot_diagnostics.append(fitted[1])
            boot_ratios.append(fitted[0][1:] / [np.median(sa[index]), np.median(sb[index])])
    boot = np.asarray(boot).reshape(-1, 3)
    rho_interval = np.quantile(boot[:, 0], [0.05, 0.95]).tolist() if len(boot) >= 10 else [np.nan, np.nan]
    median_se = np.array([np.median(sa), np.median(sb)])
    ratios = estimate[1:] / median_se
    boot_ratios = np.asarray(boot_ratios).reshape(-1, 2)
    near_variance_boundary = np.any(boot_ratios < .1, axis=1)
    variance_boundary_fraction = float(near_variance_boundary.mean()) if len(boot) else None
    flags = {
        "point_signal_sd_below_10pct_median_se": bool(np.any(ratios < .1)),
        "bootstrap_low_signal_fraction_above_20pct": bool(len(boot) >= 10 and variance_boundary_fraction > .2),
        "bootstrap_rho90_width_at_least_1_5": bool(len(boot) >= 10 and np.ptp(rho_interval) >= 1.5),
        "profile_support_width_at_least_1_5": bool(profile["likelihood_support_width"] >= 1.5),
        "all_profile_rhos_supported": profile["all_grid_values_supported"],
        "best_optimizer_not_converged": not diagnostics["best_converged"],
    }
    weak = any(flags.values())
    return {
        "n_players": len(a), "raw_correlation": float(np.corrcoef(a, b)[0, 1]),
        "latent_correlation": float(estimate[0]), "latent_sd_a": float(estimate[1]), "latent_sd_b": float(estimate[2]),
        "bootstrap_ci90": rho_interval, "bootstrap_successes": len(boot), "bootstrap_requested": int(bootstrap),
        "bootstrap_draws": boot, "boundary_fit": bool(abs(estimate[0]) > 0.98 or min(estimate[1:]) < 0.001),
        "weak_identification": weak, "weak_identification_flags": flags,
        "median_first_stage_se": {"a": float(median_se[0]), "b": float(median_se[1])},
        "latent_sd_to_median_se_ratio": {"a": float(ratios[0]), "b": float(ratios[1])},
        "weak_identification_ratio_threshold": .1,
        "optimization_diagnostics": diagnostics, "rho_likelihood_profile": profile,
        "bootstrap_diagnostics": {
            "seed": int(seed), "best_converged_count": int(sum(d["best_converged"] for d in boot_diagnostics)),
            "signal_sd_below_10pct_median_se_count": int(near_variance_boundary.sum()),
            "signal_sd_below_10pct_median_se_fraction": variance_boundary_fraction,
            "sd_near_optimizer_lower_bound_count": int(np.any(boot[:, 1:] <= 1.01 * np.exp(-10), axis=1).sum()),
            "abs_rho_above_0_98_count": int((np.abs(boot[:, 0]) > .98).sum()),
            "multistart_nll_range_max": max((d["multistart_nll_range"] for d in boot_diagnostics), default=None),
        },
        "rho_interpretation": ("Weakly identified: numerical rho and bootstrap quantiles must not be treated as a reliable persistence estimate."
                               if weak else "No configured weak-identification flag; inference remains conditional on the measurement-error model."),
        "bootstrap_interval_scope": "Descriptive player-bootstrap quantiles; not guaranteed calibrated near zero latent variance or correlation boundaries.",
        "scope": "descriptive latent-normal correlation; independent known period SEs; weak heterogeneity may make rho unidentified",
    }
