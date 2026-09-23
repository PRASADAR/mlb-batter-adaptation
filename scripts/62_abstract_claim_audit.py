#!/usr/bin/env python3
"""Audit same-type swing adjustment across pitchers and later contact."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deja-swing-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.lagged_mechanics import add_lagged_mechanics
from src.features.prospective import KEY, CONTACT

OUT = ROOT / "results/abstract_claim_audit"
CONFIG = ROOT / "config/abstract_claim_audit.json"
OUTCOMES = ["adj_distortion", "adj_absz_attack_angle",
    "adj_absz_attack_direction", "adj_absz_swing_path_tilt",
    "adj_absz_bat_speed", "adj_absz_swing_length"]
TYPE_GROUP = ["game_pk", "batter", "pitch_type"]
PITCHER_TYPE_GROUP = ["game_pk", "batter", "pitcher", "pitch_type"]
PITCHER_HISTORY = ["context_logit", "log_pair_pitches",
    "prior_pair_contact_rate", "prior_pair_swing_rate"]
COMMON_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_swings() -> tuple[pd.DataFrame, list[Path]]:
    paths = [p for year in [2025, 2026] for p in sorted(
        (ROOT / "data/processed").glob(f"analytic_{year}_*.parquet"))]
    if len(paths) != 14:
        raise FileNotFoundError("Expected 14 frozen monthly analytic swing files")
    columns = list(dict.fromkeys(KEY + PITCHER_TYPE_GROUP + ["game_date", "season",
        "exposure_type", "exposure_pitcher", "irrelevant_exposure",
        "pa_prior_pitches"] + OUTCOMES))
    d = pd.concat([pd.read_parquet(p, columns=columns) for p in paths],
                  ignore_index=True)
    if d.duplicated(KEY).any():
        raise AssertionError("Analytic swing keys must be unique")
    d["game_date"] = pd.to_datetime(d.game_date)
    return d, paths


def within_type_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    d = frame.copy()
    sizes = d.groupby(TYPE_GROUP, sort=False, dropna=False).game_pk.transform("size")
    d = d.loc[sizes.ge(2)].copy()
    d["log_same_type"] = np.log1p(d.exposure_type)
    d["log_same_pitcher_type"] = np.log1p(d.exposure_pitcher)
    d["log_other_type"] = np.log1p(d.irrelevant_exposure)
    d["within_pa"] = d.pa_prior_pitches / 5
    fields = ["log_same_type", "log_same_pitcher_type", "log_other_type",
              "within_pa"] + OUTCOMES
    means = d.groupby(TYPE_GROUP, sort=False, dropna=False)[fields].transform("mean")
    for field in fields:
        d[f"center_{field}"] = d[field] - means[field]
    designs = {
        "type_fixed_effect": ["log_same_type"],
        "type_fe_plus_general_exposure": ["log_same_type", "log_other_type",
                                           "within_pa"],
        "type_fe_plus_same_pitcher_control": ["log_same_type",
            "log_same_pitcher_type", "log_other_type", "within_pa"],
    }
    return d, {name: np.column_stack([
        d[f"center_{field}"].to_numpy(float) for field in design])
        for name, design in designs.items()}


def player_bootstrap(x: np.ndarray, y: np.ndarray, player: np.ndarray,
                     reps: int, seed: int) -> tuple[float, float]:
    ids, code = np.unique(player, return_inverse=True)
    k = x.shape[1]
    xx = np.zeros((len(ids), k, k), dtype=float)
    xy = np.zeros((len(ids), k), dtype=float)
    np.add.at(xx, code, x[:, :, None] * x[:, None, :])
    np.add.at(xy, code, x * y[:, None])
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(ids), size=(reps, len(ids)))
    a = xx[picks].sum(axis=1)
    b = xy[picks].sum(axis=1)
    draws = np.linalg.solve(a, b[..., None])[:, 0, 0]
    return tuple(float(v) for v in np.quantile(draws, [.025, .975]))


def population_changes(swings: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for year in [2025, 2026]:
        d, designs = within_type_design(swings.loc[swings.season.eq(year)])
        print(f"Within-type mechanics {year}: {len(d):,} repeated tracked swings",
              flush=True)
        for j, outcome in enumerate(OUTCOMES):
            y = d[f"center_{outcome}"].to_numpy(float)
            for k, (variant, x) in enumerate(designs.items()):
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                low, high = player_bootstrap(x, y, d.batter.to_numpy(),
                    config["population_bootstrap_replicates"],
                    config["seed"] + year * 100 + j * 2 + k)
                rows.append({"season": year, "outcome": outcome,
                    "model": variant, "tracked_swings": len(d),
                    "type_groups": int(d.groupby(TYPE_GROUP, dropna=False).ngroups),
                    "games": int(d.game_pk.nunique()),
                    "hitters": int(d.batter.nunique()),
                    "beta_deviation_per_log_dose": float(beta[0]),
                    "ci95_low": low, "ci95_high": high,
                    "x_information": float(np.sum(x[:, 0] ** 2))})
    return pd.DataFrame(rows)


def read_2026_raw() -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((ROOT / "data/raw").glob("statcast_2026-*.parquet"))
    columns = KEY + ["game_date", "game_type", "batter", "pitcher",
                     "pitch_type", "description"]
    d = pd.concat([pd.read_parquet(p, columns=columns) for p in paths],
                  ignore_index=True)
    d = d.loc[d.game_type.eq("R") & ~d.description.isin([
        "automatic_ball", "automatic_strike"])].copy()
    if d.duplicated(KEY).any():
        raise AssertionError("Raw 2026 pitch keys must be unique")
    return d, paths


def read_forecasts() -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((ROOT / "results/selected_shape_forecast").glob(
        "predictions_2026_*.parquet"))
    if len(paths) != 7:
        raise FileNotFoundError("Frozen 2026 pre-delivery context forecasts are required")
    d = pd.concat([pd.read_parquet(p, columns=KEY +
        ["game_date", "batter", "pitcher", "endpoint", "contact", "p_context"])
        for p in paths], ignore_index=True)
    d = d.loc[d.endpoint.isin(["contact_per_pitch", "contact_given_swing"])].copy()
    if d.duplicated(KEY + ["endpoint"]).any():
        raise AssertionError("Context pitch-endpoint keys must be unique")
    d["game_date"] = pd.to_datetime(d.game_date)
    return d, paths


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def improvement(frame: pd.DataFrame, base: str, candidate: str,
                cohort: str, endpoint: str, config: dict, seed: int) -> dict:
    y = frame.contact.to_numpy(float)
    p = frame[base].to_numpy(float)
    q = frame[candidate].to_numpy(float)
    diff = logloss(y, p) - logloss(y, q)
    out = {"endpoint": endpoint, "cohort": cohort, "base": base,
        "candidate": candidate, "pitches": len(frame),
        "hitters": int(frame.batter.nunique()),
        "games": int(frame.game_pk.nunique()),
        "log_loss_gain": float(diff.mean()),
        "brier_gain": float(np.mean((y-p)**2 - (y-q)**2))}
    for unit in ["game_pk", "batter"]:
        agg = pd.DataFrame({"unit": frame[unit].to_numpy(),
            "gain": diff}).groupby("unit").gain.agg(["sum", "size"])
        rng = np.random.default_rng(seed + (0 if unit == "game_pk" else 10_000))
        picks = rng.integers(len(agg), size=(
            config["forecast_bootstrap_replicates"], len(agg)))
        draws = (agg["sum"].to_numpy()[picks].sum(axis=1) /
            agg["size"].to_numpy()[picks].sum(axis=1))
        low, high = np.quantile(draws, [.025, .975])
        out[f"{unit}_ci95_low"] = float(low)
        out[f"{unit}_ci95_high"] = float(high)
    return out


def fit_chronological_model(train: pd.DataFrame, test: pd.DataFrame,
                            terms: list[str], config: dict
                            ) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=config["forecast_logistic_c"], max_iter=600, solver="lbfgs"))
    model.fit(train[terms], train.contact.astype(int))
    p = model.predict_proba(test[terms])[:, 1]
    coefficients = model.named_steps["logisticregression"].coef_[0]
    return p, coefficients


def lagged_forecasts(swings: pd.DataFrame, config: dict
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame, list[Path]]:
    raw, raw_paths = read_2026_raw()
    tracked = swings.loc[swings.season.eq(2026), KEY + ["adj_distortion"]]
    history = add_lagged_mechanics(raw, tracked,
                                   config["lagged_mechanics_clip_sd"])
    frozen, forecast_paths = read_forecasts()
    d = frozen.merge(history, on=KEY, validate="many_to_one",
        suffixes=("_forecast", ""))
    if (d.game_date_forecast.ne(d.game_date).any() or
        d.batter_forecast.ne(d.batter).any() or
        d.pitcher_forecast.ne(d.pitcher).any() or
        d.contact.ne(d.description.isin(CONTACT).astype(int)).any()):
        raise AssertionError("Frozen forecasts and delivered-pitch history must align")
    d["context_logit"] = logit(d.p_context.clip(1e-5, 1-1e-5))
    for name in ["pair", "type", "pair_type", "other_type"]:
        d[f"log_{name}_pitches"] = np.log1p(d[f"prior_{name}_pitch_n"])
        d[f"log_{name}_tracked"] = np.log1p(d[f"prior_{name}_tracked_n"])
    category = d.pitch_type.where(d.pitch_type.isin(COMMON_TYPES), "OTHER")
    type_dummies = []
    for code in COMMON_TYPES[1:] + ["OTHER"]:
        name = f"current_type_{code}"
        d[name] = category.eq(code).astype("int8")
        type_dummies.append(name)
    cutoff = pd.Timestamp(config["forecast_train_end_exclusive"])
    rows, fits, prediction_rows = [], [], []
    transfer_rows, transfer_fits = [], []
    context_terms = PITCHER_HISTORY + type_dummies
    type_terms = context_terms + ["log_type_pitches",
        "prior_type_contact_rate", "prior_type_swing_rate",
        "log_type_tracked"]
    other_terms = context_terms + ["log_pair_type_pitches",
        "log_other_type_pitches", "prior_other_type_contact_rate",
        "prior_other_type_swing_rate", "log_other_type_tracked"]
    models = {
        "pitch_context": context_terms,
        "type_outcome_history": type_terms,
        "type_prior_mechanics": type_terms + ["prior_type_dev_mean"],
        "type_mechanics_plus_last": type_terms + ["prior_type_dev_mean",
                                                   "prior_type_dev_last"],
        "other_pitcher_outcomes": other_terms,
        "other_pitcher_mechanics": other_terms + ["prior_other_type_dev_mean"],
    }
    if not set(models).isdisjoint(d.columns):
        raise AssertionError("Prediction output columns must not replace inputs")
    for i, endpoint in enumerate([
        config["forecast_primary_outcome"], config["forecast_secondary_outcome"]]):
        local = d.loc[d.endpoint.eq(endpoint)].copy()
        train = local.loc[local.game_date.lt(cutoff)]
        test = local.loc[local.game_date.ge(cutoff)].copy()
        if (train.empty or test.empty or
            train.game_date.max() >= test.game_date.min()):
            raise AssertionError("The training period must precede the test period")
        for model_name, terms in models.items():
            test[model_name], beta = fit_chronological_model(
                train, test, terms, config)
            fits.append({"endpoint": endpoint, "model": model_name,
                "training_pitches": len(train),
                "training_hitters": int(train.batter.nunique()),
                **{f"scaled_beta_{term}": float(beta[j])
                    for j, term in enumerate(terms)}})
        cohorts = [("all_later", test),
            ("prior_same_type_tracked", test.loc[test.prior_type_tracked_n.ge(1)]),
            ("prior_two_same_type_tracked", test.loc[
                test.prior_type_tracked_n.ge(2)])]
        pairs = [("pitch_context", "type_outcome_history"),
            ("type_outcome_history", "type_prior_mechanics"),
            ("type_prior_mechanics", "type_mechanics_plus_last")]
        for j, (cohort_name, subset) in enumerate(cohorts):
            if len(subset) < 1000:
                raise AssertionError("Too few later pitches with same-type history")
            for k, (base, candidate) in enumerate(pairs):
                rows.append(improvement(subset, base, candidate,
                    cohort_name, endpoint, config,
                    config["seed"] + i * 100 + j * 10 + k))
        for j, (cohort_name, subset) in enumerate([
            ("other_pitcher_prior_tracked", test.loc[
                test.prior_other_type_tracked_n.ge(1)]),
            ("first_type_from_pitcher_with_other_history", test.loc[
                test.prior_pair_type_pitch_n.eq(0) &
                test.prior_other_type_tracked_n.ge(1)])]):
            if len(subset) < 1000:
                raise AssertionError("Too few later other-pitcher histories")
            transfer_rows.append(improvement(subset,
                "other_pitcher_outcomes", "other_pitcher_mechanics",
                cohort_name, endpoint, config,
                config["seed"] + 500 + i * 100 + j))
        transfer_fits.extend(row for row in fits[-len(models):]
            if row["model"].startswith("other_pitcher"))
        prediction_rows.append(test[KEY + ["game_date", "batter", "pitcher",
            "endpoint", "contact", "p_context", "pitch_type",
            "prior_type_pitch_n", "prior_type_tracked_n",
            "prior_type_dev_mean", "prior_other_type_pitch_n",
            "prior_other_type_tracked_n", "pitch_context",
            "type_outcome_history", "type_prior_mechanics",
            "other_pitcher_outcomes", "other_pitcher_mechanics"]])
        print(f"Later {endpoint}: {len(test):,} pitches", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(fits),
            pd.concat(prediction_rows, ignore_index=True),
            pd.DataFrame(transfer_rows), pd.DataFrame(transfer_fits),
            raw_paths + forecast_paths)


def make_figure(population: pd.DataFrame, forecast: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    names = {"adj_distortion": "Multivariate deviation",
        "adj_absz_attack_angle": "Attack angle",
        "adj_absz_attack_direction": "Attack direction",
        "adj_absz_swing_path_tilt": "Swing-path tilt",
        "adj_absz_bat_speed": "Bat speed",
        "adj_absz_swing_length": "Swing length"}
    ordered = list(names)
    for i, outcome in enumerate(ordered):
        for year, color, offset in [(2025, "#ad612a", -.12),
                                     (2026, "#187c87", .12)]:
            row = population.loc[population.season.eq(year) &
                population.outcome.eq(outcome) &
                population.model.eq("type_fe_plus_general_exposure")].iloc[0]
            y = len(ordered) - i - 1 + offset
            axes[0].hlines(y, row.ci95_low, row.ci95_high, color=color, lw=2)
            axes[0].scatter(row.beta_deviation_per_log_dose, y, color=color,
                            s=35, label=str(year) if i == 0 else None)
    axes[0].axvline(0, color="#52636b", lw=1)
    axes[0].set(yticks=range(len(ordered)),
        yticklabels=[names[x] for x in ordered][::-1],
        xlabel="Change in adjusted deviation per log dose",
        title="Same pitch type within hitter and game")
    axes[0].legend(frameon=False)
    endpoints = [("contact_per_pitch", "Contact per pitch"),
                 ("contact_given_swing", "Contact after swing")]
    for i, (endpoint, label) in enumerate(endpoints):
        row = forecast.loc[forecast.endpoint.eq(endpoint) &
            forecast.cohort.eq("prior_same_type_tracked") &
            forecast.base.eq("type_outcome_history") &
            forecast.candidate.eq("type_prior_mechanics")].iloc[0]
        y = len(endpoints) - i - 1
        axes[1].hlines(y, row.batter_ci95_low, row.batter_ci95_high,
                       color="#187c87", lw=2)
        axes[1].scatter(row.log_loss_gain, y, color="#187c87", s=55)
    axes[1].axvline(0, color="#52636b", lw=1)
    axes[1].set(yticks=range(len(endpoints)),
        yticklabels=[x[1] for x in endpoints][::-1],
        xlabel="Later log-loss gain from same-type mechanics",
        title="Chronological pitch-type-conditioned test")
    for ax in axes:
        ax.grid(axis="x", color="#e6ecef")
    fig.tight_layout()
    fig.savefig(OUT / "claim_audit.png", dpi=190)
    plt.close(fig)


def make_report(population: pd.DataFrame, gains: pd.DataFrame,
                transfer_gains: pd.DataFrame, metadata: dict) -> None:
    def p(year, outcome="adj_distortion", model="type_fe_plus_general_exposure"):
        return population.loc[population.season.eq(year) &
            population.outcome.eq(outcome) & population.model.eq(model)].iloc[0]
    def f(endpoint, cohort="prior_same_type_tracked"):
        return gains.loc[gains.endpoint.eq(endpoint) & gains.cohort.eq(cohort) &
            gains.base.eq("type_outcome_history") &
            gains.candidate.eq("type_prior_mechanics")].iloc[0]
    def t(endpoint, cohort="other_pitcher_prior_tracked"):
        return transfer_gains.loc[transfer_gains.endpoint.eq(endpoint) &
            transfer_gains.cohort.eq(cohort)].iloc[0]
    def fmt(row, field, lo, hi):
        return f"{row[field]:+.6f} [{row[lo]:+.6f}, {row[hi]:+.6f}]"

    cross_summary = json.loads((ROOT / "results/cross_pitcher/summary.json").read_text())
    cross_models = pd.read_csv(ROOT / "results/cross_pitcher/model_comparison.csv")
    rank = cross_models.loc[cross_models.scope.eq("all_pitchers") &
        cross_models.outcome.eq("distortion")].iloc[0]
    persistence = cross_summary["persistence"]["pooled_same_type"]
    transport = cross_summary["holdout_prediction"]["pooled_same_type"]
    pop_rows, sensitivity_rows, lag_rows, transfer_rows = [], [], [], []
    for year in [2025, 2026]:
        row, sensitivity = p(year), p(year,
            model="type_fe_plus_same_pitcher_control")
        pop_rows.append(f"| {year} | {int(row.tracked_swings):,} | "
            f"{int(row.type_groups):,} | {fmt(row, 'beta_deviation_per_log_dose', 'ci95_low', 'ci95_high')} |")
        sensitivity_rows.append(f"| {year} | {fmt(sensitivity, 'beta_deviation_per_log_dose', 'ci95_low', 'ci95_high')} |")
    for endpoint, label in [("contact_per_pitch", "Contact per pitch"),
                            ("contact_given_swing", "Contact after swing")]:
        for cohort, cohort_label in [("all_later", "All later pitches"),
                ("prior_same_type_tracked", "Earlier same-type tracked swing"),
                ("prior_two_same_type_tracked", "Two earlier same-type tracked swings")]:
            row = f(endpoint, cohort)
            lag_rows.append(f"| {label} | {cohort_label} | {int(row.pitches):,} | "
                f"{fmt(row, 'log_loss_gain', 'batter_ci95_low', 'batter_ci95_high')} | "
                f"{fmt(row, 'log_loss_gain', 'game_pk_ci95_low', 'game_pk_ci95_high')} |")
        row = t(endpoint)
        transfer_rows.append(f"| {label} | {int(row.pitches):,} | "
            f"{fmt(row, 'log_loss_gain', 'batter_ci95_low', 'batter_ci95_high')} |")
    text = "\n".join([
        "# Abstract claim audit: Same-Type Mechanical Adjustment Across Pitchers", "",
        "## Data and design", "",
        "The project uses frozen public MLB Statcast pitch-level and bat-tracking records. Bat tracking began in the second half of 2023, according to the [MLB Statcast glossary](https://www.mlb.com/glossary/statcast). The expected-swing reference was fitted before 2025 and produces a context-adjusted multivariate swing deviation for tracked swings. This audit uses the existing 2025 and 2026 analytic swings, all delivered 2026 pitches, and frozen pitch-context forecasts. No new data were necessary because another partial day would not provide an independent replication season.", "",
        "The corrected exposure counts every earlier delivered pitch of the same Statcast type to the hitter in the game, regardless of pitcher. Takes and untracked swings contribute to the count, and the current pitch does not. The population test compares tracked swings within a hitter-game-pitch-type group. The frozen swing reference already adjusts for pitcher identity, pitch location, movement, velocity, handedness, count, and other observed context. The controlled model also includes exposure to other pitch types and progress within the plate appearance. Hitter resampling supplies intervals. A negative coefficient means that realized mechanics move closer to the context-specific expected swing as same-type exposure accumulates.", "",
        "## Systematic change in swing geometry", "",
        "| Season | Tracked swings in repeated type groups | Hitter-game-type groups | Controlled slope [95% hitter interval] |",
        "|---|---:|---:|---:|", *pop_rows, "",
        "The [full table](../results/abstract_claim_audit/population_changes.csv) reports the multivariate outcome and all five component outcomes in both seasons. It also reports a stricter sensitivity that adds same-pitcher, same-type exposure as a separate control:", "",
        "| Season | Same-type coefficient after same-pitcher exposure control [95% hitter interval] |",
        "|---|---:|", *sensitivity_rows, "",
        "The pooled same-type association replicates across seasons. The stricter coefficient isolates variation beyond repetition against the current pitcher and should be used to judge a strong cross-pitcher transfer claim. Because the two histories are highly related within games, that sensitivity is less precise and estimates a different contrast.", "",
        "## Does earlier same-type geometry predict later contact?", "",
        "For every 2026 outcome pitch, a chronological feature builder summarizes only earlier swings of the realized pitch type by that hitter in the game, pooling across pitchers. Penalized logistic models fit before July 1 and are evaluated on later pitches without refitting. Both models receive the frozen context probability, realized pitch type, current-pitcher history, and earlier same-type contact and swing rates. The candidate adds the mean earlier same-type mechanical deviation. Contact per delivered pitch is primary and contact after a swing is secondary. Positive log-loss gain favors the mechanics model. This test is prospectively ordered, but it is pitch-type-conditioned because the realized type selects the relevant history.", "",
        "| Outcome | Later subset | Pitches | Added mechanics gain [95% hitter interval] | [95% game interval] |",
        "|---|---|---:|---:|---:|", *lag_rows, "",
        "The [complete forecast comparisons](../results/abstract_claim_audit/lagged_forecast_gains.csv) include the gain from adding prior outcomes and a last-observed-mechanics sensitivity. The [pitch-level predictions](../results/abstract_claim_audit/predictions_later.parquet) preserve matched rows.", "",
        "A strict transfer diagnostic removes same-pitcher swings from the mechanical history. It evaluates later pitches only when the hitter has an earlier tracked swing of that type from another pitcher:", "",
        "| Outcome | Later pitches | Added other-pitcher mechanics gain [95% hitter interval] |",
        "|---|---:|---:|", *transfer_rows, "",
        "The [strict transfer table](../results/abstract_claim_audit/conditional_type_gains.csv) also reports the first pitch of that type from the current pitcher when another-pitcher tracked history exists.", "",
        "## Is the response a persistent individual trait?", "",
        f"The hierarchical curves already use same-type exposure pooled across pitchers. Their raw 2025-to-2026 Spearman correlation is {rank.rho_cross_season:+.3f} across {int(rank.shared_cross_season)} hitters. The measurement-error model estimates latent correlation {persistence['latent_correlation']:+.3f} with a 90% player-bootstrap range [{persistence['bootstrap_ci90'][0]:+.3f}, {persistence['bootstrap_ci90'][1]:+.3f}], and the model flags the correlation as weakly identified. The 2025 and 2026 split-half rank correlations are {rank.split_2025_rho:+.3f} and {rank.split_2026_rho:+.3f}. Applying 2025 player curves to 2026 changes centered mechanical prediction error by {transport['gain_pct']:+.4f}% relative to a population curve, with a 90% game-bootstrap interval [{transport['game_bootstrap_ci90'][0]:+.4f}%, {transport['game_bootstrap_ci90'][1]:+.4f}%]. These results do not support stable hitter rankings.", "",
        "The [interactive player curves](../notebooks/player_curves.html) and [curve table](../results/cross_pitcher/player_curves.csv) use the corrected exposure definition for every eligible hitter. A hierarchy can estimate a curve and its uncertainty even when temporal validation does not support treating the ordering as established talent.", "",
        "![Population mechanical slopes and lagged contact prediction](../results/abstract_claim_audit/claim_audit.png)", "",
        "## Verdict for the supplied abstract", "",
        "The corrected analysis supports a population statement: swing geometry moves systematically toward its context-specific expectation as hitters accumulate same-type looks within a game, pooling those looks across pitchers. It does not establish that the movement is a causal learning response, that the isolated other-pitcher component replicates, that hitter rankings persist, or that earlier mechanics improve later contact prediction. The [frozen protocol](../config/abstract_claim_audit.json), [source hashes and counts](../results/abstract_claim_audit/summary.json), and [lagged-history unit test](../tests/test_lagged_mechanics.py) make the audit reproducible.", "",
    ])
    if re.search(r"—|\bnot only\b|\bbut also\b", text, re.I):
        raise AssertionError("Report includes disallowed phrasing")
    (ROOT / "paper/abstract_claim_audit.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    swings, analytic_paths = read_swings()
    print(f"Read {len(swings):,} tracked 2025-2026 swings", flush=True)
    population = population_changes(swings, config)
    gains, fits, predictions, transfer_gains, transfer_fits, other_paths = (
        lagged_forecasts(swings, config))
    population.to_csv(OUT / "population_changes.csv", index=False)
    gains.to_csv(OUT / "lagged_forecast_gains.csv", index=False)
    fits.to_csv(OUT / "lagged_forecast_coefficients.csv", index=False)
    transfer_gains.to_csv(OUT / "conditional_type_gains.csv", index=False)
    transfer_fits.to_csv(OUT / "conditional_type_coefficients.csv", index=False)
    predictions.to_parquet(OUT / "predictions_later.parquet", index=False)
    make_figure(population, gains)
    metadata = {"protocol": config, "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(CONFIG),
        "feature_sha256": digest(ROOT / "src/features/lagged_mechanics.py"),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in
                    analytic_paths + other_paths},
        "analytic_swings": len(swings),
        "later_prediction_rows": len(predictions),
        "claim_boundary": "Pooled same-type mechanics association across pitchers; no causal or persistent individual-skill claim"}
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_report(population, gains, transfer_gains, metadata)
    print(population.loc[population.outcome.eq("adj_distortion")].to_string(index=False),
          flush=True)
    print(gains.loc[gains.base.eq("type_outcome_history") &
        gains.candidate.eq("type_prior_mechanics")].to_string(index=False),
        flush=True)
    print(transfer_gains.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
