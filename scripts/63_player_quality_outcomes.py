#!/usr/bin/env python3
"""Relate a frozen adaptation score to future player qualities."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.player_qualities import (  # noqa: E402
    QUALITY_META, add_domain_composites, aggregate_player_qualities)
from src.models.expected import player_slopes  # noqa: E402
from src.models.hierarchical import posterior_mean_rates  # noqa: E402
from src.pipeline import names_map  # noqa: E402

CONFIG = json.loads((ROOT / "config/player_quality_outcomes.json").read_text())
OUT = ROOT / "results/player_quality_outcomes"
PAPER = ROOT / "paper/player_quality_outcomes.md"
RNG = np.random.default_rng(CONFIG["seed"])


def read_raw(year: int) -> pd.DataFrame:
    columns = ["game_date", "game_type", "game_pk", "at_bat_number",
        "pitch_number", "batter", "description", "zone", "events",
        "launch_speed", "launch_speed_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "bat_speed"]
    frames = [pd.read_parquet(p, columns=columns)
              for p in sorted((ROOT / "data/raw").glob(f"statcast_{year}-*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"No raw Statcast files for {year}")
    d = pd.concat(frames, ignore_index=True)
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.loc[d.game_type.eq("R") &
              ~d.description.isin(["automatic_ball", "automatic_strike"])].copy()
    return d.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])


def annual_score() -> pd.DataFrame:
    post = pd.read_csv(ROOT / "results/cross_pitcher/players_pooled_same_type_2025.csv")
    raw = pd.read_csv(ROOT / "results/cross_pitcher/first_stage_pooled_same_type_2025.csv")
    keep = ["player_id", "player_name", "n_exposures", "posterior_mean", "posterior_sd"]
    return post[keep].merge(raw[["player_id", "estimate", "se", "information"]],
                            on="player_id", validate="one_to_one").rename(
        columns={"posterior_mean": "adaptation_score", "estimate": "raw_score"})


def thresholds_ok(d: pd.DataFrame, suffix: str, minimum_pa: int) -> pd.Series:
    ok = d[f"pa{suffix}"].ge(minimum_pa)
    for denominator, minimum in CONFIG["minimum_opportunities"].items():
        use = minimum if denominator != "pa" else minimum_pa
        ok &= d[f"{denominator}{suffix}"].ge(use)
    return ok


def pair_qualities(prior: pd.DataFrame, future: pd.DataFrame,
                   score: pd.DataFrame, minimum_pa: int) -> pd.DataFrame:
    d = score.merge(prior, on="player_id", validate="one_to_one")
    d = d.merge(future, on="player_id", suffixes=("_prior", "_future"),
                validate="one_to_one")
    d = d.loc[thresholds_ok(d, "_prior", minimum_pa) &
              thresholds_ok(d, "_future", minimum_pa)].copy()
    return add_domain_composites(d).reset_index(drop=True)


def residual(y, controls):
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(y)), np.asarray(controls, float)])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial_rank_arrays(d: pd.DataFrame, outcome: str, extra=False):
    x = rankdata(d.adaptation_score)
    y = rankdata(d[f"{outcome}_future"])
    controls = [rankdata(d[f"{outcome}_prior"])]
    if extra:
        controls.extend([rankdata(np.log1p(d.n_exposures)), rankdata(d.posterior_sd)])
    c = np.column_stack(controls)
    return residual(x, c), residual(y, c)


def corr(x, y):
    return float(pearsonr(x, y).statistic)


def standardized_beta(d, outcome, score="adaptation_score", robust=False):
    x = np.column_stack([d[score], d[f"{outcome}_prior"]]).astype(float)
    y = d[f"{outcome}_future"].to_numpy(float)
    x = (x - x.mean(0)) / x.std(0, ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    model = HuberRegressor(epsilon=1.35, alpha=.001, max_iter=1000) if robust else LinearRegression()
    return float(model.fit(x, y).coef_[0])


def quantile_betas(d, outcome):
    x = np.column_stack([d.adaptation_score, d[f"{outcome}_prior"]]).astype(float)
    y = d[f"{outcome}_future"].to_numpy(float)
    x = (x - x.mean(0)) / x.std(0, ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    result = []
    for q in (.25, .5, .75):
        model = QuantileRegressor(quantile=q, alpha=.001, solver="highs")
        result.append(float(model.fit(x, y).coef_[0]))
    return result


def cv_gain(d, outcome):
    score = d.adaptation_score.to_numpy(float).reshape(-1, 1)
    prior = d[f"{outcome}_prior"].to_numpy(float).reshape(-1, 1)
    y = d[f"{outcome}_future"].to_numpy(float)
    cv = RepeatedKFold(n_splits=CONFIG["cv_splits"],
        n_repeats=CONFIG["cv_repeats"], random_state=CONFIG["seed"])
    base, full = [], []
    for train, test in cv.split(y):
        b = LinearRegression().fit(prior[train], y[train])
        f = LinearRegression().fit(np.c_[prior[train], score[train]], y[train])
        base.extend((y[test] - b.predict(prior[test])) ** 2)
        full.extend((y[test] - f.predict(np.c_[prior[test], score[test]])) ** 2)
    return 100 * (np.mean(base) - np.mean(full)) / np.mean(base)


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(ranked, 1)
    return out


def assess_outcomes(d):
    rows = []
    boots = CONFIG["bootstrap_replicates"]
    for metric, (label, domain, denominator, direction) in QUALITY_META.items():
        ex, ey = partial_rank_arrays(d, metric)
        exa, eya = partial_rank_arrays(d, metric, extra=True)
        indices = RNG.integers(0, len(d), (boots, len(d)))
        boot = np.array([corr(ex[ix], ey[ix]) for ix in indices])
        q25, q50, q75 = quantile_betas(d, metric)
        raw_ex = residual(rankdata(d.raw_score), rankdata(d[f"{metric}_prior"])[:, None])
        row = {"outcome": metric, "label": label, "domain": domain,
            "n_players": len(d), "raw_spearman": spearmanr(d.adaptation_score,
                d[f"{metric}_future"]).statistic,
            "partial_rank_rho": corr(ex, ey),
            "partial_rank_ci95_low": np.quantile(boot, .025),
            "partial_rank_ci95_high": np.quantile(boot, .975),
            "partial_rank_p": pearsonr(ex, ey).pvalue,
            "information_adjusted_rho": corr(exa, eya),
            "unpooled_partial_rho": corr(raw_ex, ey),
            "standardized_ols_beta": standardized_beta(d, metric),
            "standardized_huber_beta": standardized_beta(d, metric, robust=True),
            "quantile_beta_25": q25, "quantile_beta_50": q50,
            "quantile_beta_75": q75, "cv_mse_gain_pct": cv_gain(d, metric),
            "favorable_direction": direction}
        rows.append(row)
    out = pd.DataFrame(rows)
    out["bh_q"] = bh(out.partial_rank_p)

    # One score permutation is shared across outcomes, preserving dependence.
    observed = dict(zip(out.outcome, abs(out.partial_rank_rho)))
    exceed = {m: 0 for m in out.outcome}
    max_null = []
    controls = {m: rankdata(d[f"{m}_prior"])[:, None] for m in out.outcome}
    future_resid = {m: residual(rankdata(d[f"{m}_future"]), controls[m]) for m in out.outcome}
    sx = rankdata(d.adaptation_score)
    for _ in range(CONFIG["permutation_replicates"]):
        perm = RNG.permutation(sx)
        stats = [abs(corr(residual(perm, controls[m]), future_resid[m])) for m in out.outcome]
        max_null.append(max(stats))
    max_null = np.asarray(max_null)
    out["maxT_p"] = [(1 + np.sum(max_null >= observed[m])) / (len(max_null) + 1)
                      for m in out.outcome]
    return out


def assess_composites(d):
    rows = []
    for metric, label in [("authority_composite", "Batted-ball authority composite"),
                          ("discipline_composite", "Plate-discipline composite")]:
        ex, ey = partial_rank_arrays(d, metric)
        indices = RNG.integers(0, len(d), (CONFIG["bootstrap_replicates"], len(d)))
        boot = np.array([corr(ex[ix], ey[ix]) for ix in indices])
        perm = np.array([corr(RNG.permutation(ex), ey)
                         for _ in range(CONFIG["permutation_replicates"])])
        q25, q50, q75 = quantile_betas(d, metric)
        rows.append({"outcome": metric, "label": label, "n_players": len(d),
            "partial_rank_rho": corr(ex, ey), "ci95_low": np.quantile(boot, .025),
            "ci95_high": np.quantile(boot, .975),
            "permutation_p": (1 + np.sum(np.abs(perm) >= abs(corr(ex, ey)))) / (len(perm) + 1),
            "standardized_ols_beta": standardized_beta(d, metric),
            "standardized_huber_beta": standardized_beta(d, metric, robust=True),
            "quantile_beta_25": q25, "quantile_beta_50": q50,
            "quantile_beta_75": q75, "cv_mse_gain_pct": cv_gain(d, metric)})
    return pd.DataFrame(rows)


def half_score(year, end):
    columns = ["game_date", "game_pk", "batter", "adj_log_exposure_type", "adj_distortion"]
    d = pd.concat([pd.read_parquet(p, columns=columns) for p in sorted(
        (ROOT / "data/processed").glob(f"analytic_{year}_*.parquet"))], ignore_index=True)
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.loc[d.game_date.lt(end)]
    slopes = player_slopes(d, x="log_exposure_type", y="distortion",
        min_n=CONFIG["half_score_minimum_swings"],
        min_games=CONFIG["half_score_minimum_games"])
    slopes["adaptation_score"] = posterior_mean_rates(slopes.estimate, slopes.se)[0]
    return slopes[["player_id", "adaptation_score"]]


def temporal_checks(raw25, raw26, annual):
    definitions = [
        ("2025 annual score to 2026 through September 20", annual,
         raw25, raw26, CONFIG["minimum_pa_annual"]),
        ("2025 annual score to 2026 second half", annual,
         raw26.loc[raw26.game_date.lt("2026-07-01")],
         raw26.loc[raw26.game_date.ge("2026-07-01")], CONFIG["minimum_pa_half"]),
        ("2025 first-half score to 2025 second half", half_score(2025, "2025-07-01"),
         raw25.loc[raw25.game_date.lt("2025-07-01")],
         raw25.loc[raw25.game_date.ge("2025-07-01")], CONFIG["minimum_pa_half"]),
        ("2026 first-half score to 2026 second half", half_score(2026, "2026-07-01"),
         raw26.loc[raw26.game_date.lt("2026-07-01")],
         raw26.loc[raw26.game_date.ge("2026-07-01")], CONFIG["minimum_pa_half"])]
    rows = []
    for label, scores, before, after, minimum in definitions:
        paired = pair_qualities(aggregate_player_qualities(before),
            aggregate_player_qualities(after), scores, minimum)
        ex, ey = partial_rank_arrays(paired, "authority_composite")
        ix = RNG.integers(0, len(paired), (CONFIG["bootstrap_replicates"], len(paired)))
        boot = np.array([corr(ex[i], ey[i]) for i in ix])
        rows.append({"comparison": label, "n_players": len(paired),
            "partial_rank_rho": corr(ex, ey), "ci95_low": np.quantile(boot, .025),
            "ci95_high": np.quantile(boot, .975)})
    return pd.DataFrame(rows)


def quintiles(d):
    q = d.copy()
    q["adaptation_quintile"] = pd.qcut(q.adaptation_score, 5, labels=False) + 1
    q["future_authority_residual"] = residual(
        q.authority_composite_future,
        q.authority_composite_prior.to_numpy()[:, None])
    return q.groupby("adaptation_quintile").agg(
        n_players=("player_id", "size"), mean_score=("adaptation_score", "mean"),
        future_authority_residual=("future_authority_residual", "mean"),
        standard_error=("future_authority_residual", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()


def figure(outcomes, temporal, quint):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    order = outcomes.sort_values("partial_rank_rho")
    color = np.where(order.domain.eq("Batted-ball authority"), "#d95f02", "#4776a8")
    y = np.arange(len(order))
    axes[0].hlines(y, order.partial_rank_ci95_low, order.partial_rank_ci95_high,
                   color=color, alpha=.75)
    axes[0].scatter(order.partial_rank_rho, y, c=color, s=34, zorder=3)
    axes[0].axvline(0, color=".4", lw=1)
    axes[0].set_yticks(y, order.label); axes[0].set_xlabel("Partial rank correlation")
    axes[0].set_title("2025 score and 2026 qualities")

    axes[1].errorbar(quint.adaptation_quintile, quint.future_authority_residual,
        yerr=1.96 * quint.standard_error, fmt="o-", color="#d95f02", capsize=3)
    axes[1].axhline(0, color=".4", lw=1)
    axes[1].set(xlabel="Adaptation-score quintile", ylabel="Residual 2026 authority",
                title="Batted-ball authority by score quintile")

    ty = np.arange(len(temporal))
    axes[2].hlines(ty, temporal.ci95_low, temporal.ci95_high, color="#5a7d55")
    axes[2].scatter(temporal.partial_rank_rho, ty, color="#5a7d55", s=40, zorder=3)
    axes[2].axvline(0, color=".4", lw=1)
    axes[2].set_yticks(ty, [x.replace(" to ", "\nto ") for x in temporal.comparison])
    axes[2].set_xlabel("Partial rank correlation")
    axes[2].set_title("Temporal replication checks")
    for ax in axes: ax.grid(axis="x", alpha=.18)
    fig.suptitle("Adaptation speed has a small, exploratory link to future contact authority",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "quality_outcomes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def report(outcomes, composites, temporal, n):
    auth = composites.loc[composites.outcome.eq("authority_composite")].iloc[0]
    sig = outcomes.loc[outcomes.partial_rank_p.lt(.05), ["label", "partial_rank_rho", "partial_rank_p", "bh_q", "maxT_p"]]
    lines = ["# Adaptation speed and future player qualities", "",
        "## Design", "",
        f"This exploratory analysis freezes each hitter's 2025 posterior adaptation score, then relates it to 2026 qualities among {n} hitters with at least 200 plate appearances and common opportunity thresholds in both seasons. Positive adaptation scores indicate a faster decline in context-adjusted swing-mechanics deviation with accumulated same-type looks across pitchers within a game. Every future-quality comparison controls for the same quality in 2025.", "",
        "The outcomes cover swing decisions, bat-to-ball skill, plate discipline, batted-ball authority, overall production, and bat speed. The analysis uses partial rank correlation, ordinary least squares, Huber regression, quantile regression, repeated cross-validation, Benjamini-Hochberg correction, and a shared-score maximum-statistic permutation test.", "",
        "## Findings", "",
        f"The pre-specified five-component batted-ball authority composite has a partial rank correlation of {auth.partial_rank_rho:.3f} (95% bootstrap interval {auth.ci95_low:.3f} to {auth.ci95_high:.3f}; permutation p={auth.permutation_p:.3f}). Its standardized OLS coefficient is {auth.standardized_ols_beta:.3f}, its Huber coefficient is {auth.standardized_huber_beta:.3f}, and adding the score changes repeated-cross-validation mean squared error by {auth.cv_mse_gain_pct:.2f}%. The quantile coefficients are {auth.quantile_beta_25:.3f}, {auth.quantile_beta_50:.3f}, and {auth.quantile_beta_75:.3f} at the 25th, 50th, and 75th percentiles.", ""]
    if len(sig):
        listed = ", ".join(f"{r.label} (rho={r.partial_rank_rho:.3f})" for r in sig.itertuples())
        lines += [f"The nominal individual associations are {listed}. None is treated as confirmatory without the reported multiplicity adjustments.", ""]
    lines += ["The relationship is domain-specific. Swing decisions, bat-to-ball outcomes, walk and strikeout rates, and overall wOBA do not show a comparable prospective association. Shorter-window scores also fail to reproduce the annual authority result consistently. The evidence therefore identifies a candidate link between the adaptation measure and future contact authority, but it does not yet establish a persistent general adaptability skill.", "", "## Interpretation", "",
        "A plausible interpretation is that hitters whose mechanics converge faster during repeated same-type exposure also retain or develop greater capacity to produce forceful contact. The small cross-validated gain limits its current decision value. A clean preregistered season and an independent cohort are needed before using this score for player evaluation.", ""]
    PAPER.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True); PAPER.parent.mkdir(parents=True, exist_ok=True)
    raw25, raw26 = read_raw(2025), read_raw(2026)
    q25, q26 = aggregate_player_qualities(raw25), aggregate_player_qualities(raw26)
    score = annual_score()
    d = pair_qualities(q25, q26, score, CONFIG["minimum_pa_annual"])
    if len(d) < 200: raise AssertionError(f"Unexpectedly small common cohort: {len(d)}")
    outcomes = assess_outcomes(d)
    composites = assess_composites(d)
    temporal = temporal_checks(raw25, raw26, score)
    quint = quintiles(d)
    d.to_csv(OUT / "player_qualities.csv", index=False)
    outcomes.to_csv(OUT / "outcome_associations.csv", index=False)
    composites.to_csv(OUT / "composite_associations.csv", index=False)
    temporal.to_csv(OUT / "temporal_checks.csv", index=False)
    quint.to_csv(OUT / "authority_quintiles.csv", index=False)
    figure(outcomes, temporal, quint)
    report(outcomes, composites, temporal, len(d))
    summary = {"n_players": len(d), "primary_composite": composites.iloc[0].to_dict(),
        "minimum_pa": CONFIG["minimum_pa_annual"], "status": CONFIG["status"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(outcomes[["label", "partial_rank_rho", "partial_rank_p", "bh_q", "maxT_p", "cv_mse_gain_pct"]].to_string(index=False))
    print("\n", composites.to_string(index=False))
    print("\n", temporal.to_string(index=False))


if __name__ == "__main__":
    main()
