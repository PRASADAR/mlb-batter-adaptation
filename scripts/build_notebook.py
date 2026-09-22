#!/usr/bin/env python3
"""Create the executable, offline research walkthrough from reproducible code."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(root=ROOT):
    root = Path(root).resolve()
    config = json.loads((root / 'config/analysis.json').read_text())
    title = config.get('title', 'Deja Swing: The Search for Batter Adaptation')
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

    md(f'''# {title}

### Research question and scope

This notebook examines whether repeated exposure to a pitch is associated with changes in swing geometry. Public Statcast records are used to reconstruct pitch sequences and estimate distributions over hitter-specific correction associations. Negative controls and later-season tests assess the interpretation and persistence of these associations.

Player estimates are reported as posterior distributions, with uncertainty in comparisons and ranks. The original primary model is evaluated on a temporal holdout. Any additional analyses specified after inspecting that holdout are labeled exploratory.

**Run:** select **Kernel → Restart Kernel and Run All Cells**. The default walkthrough runs offline from the frozen data and generated model outputs included in this project. It regenerates the scientific figures, runs a small synthetic example, and demonstrates real-player comparisons. The optional full pipeline refits models using frozen data and takes longer.

**Reading time:** approximately 15 minutes. **Units:** positive $\\lambda$ means lower unexpected swing deviation per one unit of $\\log(1 + \\text{{prior exposure}})$. It does not mean a percentage of error corrected, and it is not an identified causal learning rate.''')
    code('''from pathlib import Path
import sys, json, subprocess, io, os, tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deja-swing-matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

# Works from either the repository root or the notebooks/ directory.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "config/analysis.json").exists() and (p / "src/pipeline.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook inside the Deja Swing repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = json.loads((ROOT / "config/analysis.json").read_text())
SEED = CONFIG["seed"]
pd.set_option("display.max_columns", 12)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

def table(name):
    return pd.read_csv(ROOT / "results/tables" / name)

def document(name):
    return json.loads((ROOT / name).read_text())

def posterior(season=2025):
    with np.load(ROOT / f"results/posterior/adaptation_{season}.npz") as archive:
        return dict(archive)

print(f"Project: {ROOT.name} | Seed: {SEED} | Default mode: frozen offline replay")''')
    md('''## 1. Reproducing the analysis

The repository separates source retrieval, current-pitch baseline fitting, adaptation estimation, and validation. The frozen snapshot and source manifests record the observed data coverage.

`RUN_FULL_PIPELINE = True` below recomputes features, models, validation, and checksums from frozen raw partitions. It does not fetch new data. Default execution uses the saved outputs.''')
    code('''RUN_FULL_PIPELINE = False
if RUN_FULL_PIPELINE:
    subprocess.run([sys.executable, "-m", "src.pipeline", "all"], cwd=ROOT, check=True)
    for entry in ["17_supplementary_validation.py", "18_matched_exposure.py", "19_percentile_stability.py",
                  "11_pairwise_comparisons.py", "build_report.py"]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / entry)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "src.pipeline", "freeze"], cwd=ROOT, check=True)

required = ["results/tables/batter_adaptation_posteriors.csv",
            "results/tables/robustness.csv", "results/tables/persistence.json",
            "results/posterior/adaptation_2025.npz", "results/tables/percentile_stability.csv"]
missing = [path for path in required if not (ROOT / path).exists()]
if missing:
    raise FileNotFoundError("Missing generated outputs; run make reproduce first: " + ", ".join(missing))

sample = document("results/logs/sample.json")
display(Markdown(f"**Frozen sample:** {sample['raw_pitches']:,} pitch-level records across "
                 f"{sample['raw_games']:,} games, {sample['start']} through {sample['end']}."))
display(pd.DataFrame({"Season": list(sample["seasons"]),
                      "Pitch-level records": list(sample["seasons"].values())}).set_index("Season"))''')
    code('''primary = table("robustness.csv").query("specification == 'exposure_pitcher'")
transport = document("results/tables/holdout_prediction.json")
loss_verb = "reduce" if transport["mse_improvement_pct"] >= 0 else "increase"
loss_ci = transport["ci90"] if loss_verb == "reduce" else [-transport["ci90"][1], -transport["ci90"][0]]
result_lines = ["## Main results", "",
    "Positive correction associations mean less unexpected deviation with repeated exposure."]
for row in primary.itertuples():
    result_lines.append(f"- **{int(row.season)}:** {row.mu_median:.3f} deviation SD per log-exposure unit "
                       f"(90% credible interval {row.mu_low:.3f} to {row.mu_high:.3f}; {int(row.players)} hitters).")
component_rank = table("percentile_stability.csv").query(
    "specification == 'absz_swing_length__log_exposure_pitcher__n500' and method == 'pooled' and endpoint == 'spearman'").iloc[0]
result_lines += ["", f"Individual 2025 slopes **{loss_verb}** 2026 conditional prediction MSE by "
    f"**{abs(transport['mse_improvement_pct']):.3f}%** (90% interval "
    f"{loss_ci[0]:.3f}% to {loss_ci[1]:.3f}%).",
    "", "These estimates describe exposure associations. The controls, measurement limitations, "
    "and persistence checks below assess their interpretation as adaptation.",
    "", f"**Exploratory component result:** Swing-length score ranks correlate "
    f"**{component_rank.estimate:.3f}** across seasons among {int(component_rank.shared_players)} hitters "
    f"with at least 500 swings in both periods (90% interval {component_rank.ci90_low:.3f} to "
    f"{component_rank.ci90_high:.3f}). The complete percentile search, failed internal replication, "
    "and similarly stable future-exposure control are reported in Section 11."]
display(Markdown("\\n".join(result_lines)))''')
    code('''from src.visualization.figures import make_figures
FIGURES = make_figures(ROOT)

def show_figure(name):
    display(Image(filename=str(FIGURES[name]), width=1150))

print(f"Regenerated {len(FIGURES)} figures as PNG and vector PDF in results/figures/.")
show_figure("01_research_timeline")''')
    md('''The current-pitch baseline trains before July 2024. July–December 2024 calibrates the residual covariance. The 2025 nuisance regressions hold out whole games; other games later in 2025 may train a given fold. For the original primary model, **2026 is a temporal holdout**, predicted using 2025 nuisance models with fixed choices. This separation does not extend to additional exploratory specifications chosen after inspecting 2026.

Powers and Yurko (2026), *Swinging, Fast and Slow*, establish the importance of context when interpreting swing tracking. This project studies conditional exposure slopes and their uncertainty. Steinhardt and Borowiak (2025), *Hitter and Catcher Adaptation in Major League Baseball*, already study current-pitch-adjusted performance, repeated pitches, and heterogeneous adjustment. The extension considered here combines mechanics-based deviations, uncertainty in player comparisons, and negative controls. See [the literature review](../literature/related_work.md) for overlaps and differences.''')
    md('''## 2. Tracking coverage

A hitter can miss repeatedly while changing swing geometry. The coverage audit counts every recorded pitch, swing, whiff, and contact swing before model filtering. Empty measurements remain missing. Availability can vary by season and contact status.''')
    code('''show_figure("02_tracking_coverage")
coverage = table("data_coverage_season.csv")
display(coverage[["group", "field", "n_whiffs", "populated_whiffs", "pct_whiffs"]]
        .rename(columns={"group": "season"}).head(8))''')
    md('''## 3. Pitch histories and exposure

This runnable example starts from a real raw-data partition, takes one complete game, and rebuilds pitch histories before filtering to swings. Previous-pitch features reset at the plate-appearance boundary. Repeated exposure to the same pitcher and pitch type can continue across plate appearances within the game. All counts exclude the current pitch. Automatic balls and strikes are administrative events and are excluded from the physical exposure history.

The continuous shape kernel also uses only earlier pitches. For the original analysis, its physical scales and candidate memory half-lives were fixed before examining holdout outcomes.''')
    code('''from src.features.sequences import prepare_pitches
raw_file = sorted((ROOT / "data/raw").glob("statcast_*.parquet"))[0]
raw_window = pd.read_parquet(raw_file)
game_id = raw_window.game_pk.iloc[0]
game = prepare_pitches(raw_window.loc[raw_window.game_pk.eq(game_id)].copy())
pa_sizes = game.groupby("at_bat_number").size()
pa_id = pa_sizes[pa_sizes.ge(5)].index[0] if (pa_sizes >= 5).any() else pa_sizes.idxmax()
sequence = game.loc[game.at_bat_number.eq(pa_id),
    ["game_pk", "at_bat_number", "pitch_number", "pitch_type", "description",
     "lag1_pitch_type", "exposure_pa", "exposure_pitcher", "kernel_exposure"]]
assert game.groupby(["game_pk", "at_bat_number"]).head(1).lag1_pitch_type.isna().all()
assert game.groupby(["game_pk", "batter", "pitcher", "pitch_type"]).head(1).exposure_pitcher.eq(0).all()
display(sequence)
print("Verified: PA boundaries reset history; first comparable pitch has zero prior exposures.")
del raw_window''')
    md('''## 4. Expected swings and deviation

Each tracked swing is compared with a baseline prediction using the current pitch, count, game context, hitter, and pitcher. The baseline does not receive previous-pitch information. Physical component residuals remain available in degrees, mph, and feet.

A separate 2024 calibration sample determines residual scales and a regularized covariance matrix. Their Mahalanobis distance becomes a multivariate deviation score, expressed in calibration standard deviations. The covariance accounts for correlations and differences in component scale. The score does not establish an optimal swing or directly measure motor error.''')
    code('''components = CONFIG["swing_components"]
cols = ["game_date", "batter", "pitch_type"]
for component in components:
    cols += [component, f"expected_{component}", f"resid_{component}"]
baseline_examples = pd.read_parquet(sorted((ROOT / "data/processed").glob("residuals_2025*.parquet"))[0], columns=cols).head(6)
component = components[0]
display(baseline_examples[["game_date", "batter", "pitch_type", component,
                           f"expected_{component}", f"resid_{component}"]])
baseline_info = document("results/models/baseline.json")
display(Markdown(f"**Baseline training swings:** {baseline_info['current_model']['training_n']:,}. "
                 f"**Covariance calibration swings:** {baseline_info['calibration_n']:,}. "
                 f"**Components:** {', '.join(components)}."))
display(table("baseline_drift.csv")[["season", "n", "mean_deviation", "sd_deviation"]])''')
    md('''The frozen baseline can drift across seasons. The table above audits the average and spread of its deviation score; seasonal shifts can reflect context, players, or measurement changes as well as mechanics. The relationship between deviation and swing quality requires separate validation.''')
    md('''## 5. Sequence prediction

Two models share the same baseline training period. The second additionally receives previous-pitch features. The chart reports paired prediction-loss changes on later seasons, with game-bootstrap intervals. Improved prediction supports an association with pitch history. A causal effect on swing mechanics is not identified.''')
    code('''show_figure("03_sequence_prediction_gain")
display(table("sequence_distortion.csv").query("season == 2026")
        [["component", "n", "mse_gain_pct", "gain_ci90_low", "gain_ci90_high"]])''')
    md(r'''## 6. Hierarchical exposure model

The primary exposure is the number of earlier pitches of the same type from the same pitcher in the same game. Outcome and log exposure are separately residualized using current-pitch and sequence context. Each hitter receives a conditional slope and a game-cluster robust standard error. The reported correction association is the negative slope.

The second stage treats these estimates as noisy measurements:

$$
\widehat\lambda_i\mid\lambda_i\sim N(\lambda_i,s_i^2),\qquad
\lambda_i\mid\mu,\tau\sim N(\mu,\tau^2),
$$
$$
\mu\sim N(0,0.2^2),\qquad \tau\sim\operatorname{HalfNormal}(0.15).
$$

The population mean is integrated analytically and the population spread by checked one-dimensional numerical quadrature. Independent conditional posterior draws propagate both hyperparameters. The method uses independent posterior draws; MCMC chain diagnostics such as R-hat and divergences do not apply. This remains a two-stage approximation conditional on estimated first-stage SEs; uncertainty from fitting the baseline and nuisance models is not fully propagated.''')
    code('''from src.models.hierarchical import fit_hierarchy, summarize_posterior, compare_hitters
POSTERIOR = posterior(2025)
leaderboard = table("batter_adaptation_posteriors.csv")
players_2025 = leaderboard.loc[leaderboard.season.eq(2025)].copy()
diagnostics = document("results/posterior/adaptation_2025_diagnostics.json")
display(pd.Series({key: diagnostics[key] for key in [
    "grid_size", "tau_upper", "tail_mass_last_1pct",
    "max_mean_change_at_double_resolution", "independent_draws", "mu_mean_mcse"]}, name="Numerical diagnostic").to_frame())
show_figure("04_exposure_response")''')
    md(r'''The left panel is an observed association in the analytic sample. The right panel is the model-implied conditional change, $-\mu\log(1+E)$, anchored at zero exposure. The curve describes a conditional difference from zero exposure. It does not recover an absolute mechanical trajectory. Pitcher responses, unobserved intent, and selection into later exposures can contribute to the association.''')
    code('''show_figure("05_player_and_rank_uncertainty")
show_figure("13_player_posterior_distributions")''')
    md('''Players shown here are sampled across posterior means to illustrate uncertainty. Their inclusion is descriptive. A posterior rank interval can span much of the eligible player pool despite differences in point estimates. “League median” in the tables means the draw-specific median of the included model population.''')
    code('''show_figure("07_shrinkage_and_information")''')
    md('''## 7. Pairwise player comparisons

Change `PLAYER_A` and `PLAYER_B` to any displayed player names. The comparison uses paired draws from the joint posterior, preserving shared population uncertainty. It returns a difference distribution, credible intervals, and the probability that A's conditional association exceeds B's.

The default names come from the eligible-player table. Optional dropdowns are provided when `ipywidgets` is installed. The comparison also runs without interactive controls.''')
    code('''available_names = sorted(players_2025.player_name.astype(str).unique())
preferred_names = ["Aaron Judge", "Shohei Ohtani", "Juan Soto", "Mookie Betts",
                   "Bobby Witt Jr.", "Vladimir Guerrero Jr.", "José Ramírez", "Kyle Tucker"]
defaults = [name for name in preferred_names if name in available_names]
defaults += [name for name in available_names if name not in defaults]
PLAYER_A, PLAYER_B = defaults[:2]
name_to_id = dict(zip(players_2025.player_name, players_2025.player_id))

def compare_named_hitters(name_a, name_b, show=True):
    if name_a == name_b:
        raise ValueError("Choose two different hitters for a meaningful comparison.")
    comparison = compare_hitters(POSTERIOR, name_to_id[name_a], name_to_id[name_b])
    if show:
        probability = comparison["probability_a_greater"]
        probability_text = "<0.1%" if probability < .001 else ">99.9%" if probability > .999 else f"{probability:.1%}"
        display(Markdown(f"### {name_a} vs. {name_b} (2025)\\n"
            f"P(λA > λB) = **{probability_text}**. "
            f"Median difference **{comparison['median_difference']:.4f}**; "
            f"90% credible interval **[{comparison['cri90'][0]:.4f}, {comparison['cri90'][1]:.4f}]**. "
            "These probabilities compare conditional exposure associations. They do not establish differences in learning skill."))
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.hist(comparison["difference_draws"], bins=65, density=True, color="#008E8A", alpha=.75)
        ax.axvline(0, color="#D85C45", linewidth=2)
        ax.set(xlabel=f"λ({name_a}) − λ({name_b})", ylabel="Posterior density",
               title="Uncertainty about a conditional difference")
        ax.spines[["top", "right"]].set_visible(False)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        display(Image(data=buffer.getvalue()))
        plt.close(fig)
    return comparison

comparison = compare_named_hitters(PLAYER_A, PLAYER_B)''')
    code('''# Optional UI; the preceding cell remains the fully executable static fallback.
try:
    import ipywidgets as widgets
except ImportError:
    print("Optional dropdowns unavailable. Edit PLAYER_A / PLAYER_B and rerun the preceding cell.")
else:
    choice_a = widgets.Dropdown(options=available_names, value=PLAYER_A, description="Hitter A:")
    choice_b = widgets.Dropdown(options=available_names, value=PLAYER_B, description="Hitter B:")
    button = widgets.Button(description="Compare hitters", button_style="info")
    widget_output = widgets.Output()
    def update_comparison(_):
        with widget_output:
            widget_output.clear_output(wait=True)
            if choice_a.value == choice_b.value:
                print("Choose two different hitters for a meaningful comparison.")
            else:
                compare_named_hitters(choice_a.value, choice_b.value)
    button.on_click(update_comparison)
    display(widgets.HBox([choice_a, choice_b, button]), widget_output)''')
    code('''show_figure("06_pairwise_probability")''')
    md('''## 8. Parameter recovery simulations

The validation generator creates raw swing outcomes with heterogeneous player rates, unequal counts, changing pitch difficulty, within-group noise, heteroscedasticity, and missing observations. Missingness is ignorable given included covariates. The simulations do not test nonignorable tracking failures.

Groups and exposure opportunities are generated before outcome noise. First swings are sampled without selection on extreme errors, avoiding apparent improvement caused by that form of regression to the mean. Each replicate fits linear first-stage regressions with cluster-robust standard errors and the same measurement-error hierarchy. These simulations do not reproduce or validate the full flexible expected-swing and nuisance-model pipeline.

The next cell reruns a small example from scratch. Its four experiments per scenario demonstrate the code but are insufficient to establish calibration. The archived validation below uses the configured full number of independent experiments.''')
    code('''from src.evaluation.recovery import run_simulations
example_run = run_simulations(replicates=4, players=12, draws=800, seed=SEED + 99)
assert example_run["summary"]["validation_run"] is False
display(example_run["replicates"].groupby("scenario")
        [["coverage90", "rmse", "unpooled_rmse", "mean_bias"]].mean())
print("Runnable demonstration only: 4 independent experiments per scenario.")''')
    code('''simulation_summary = document("results/tables/simulation_summary.json")
display(pd.DataFrame({scenario: simulation_summary[scenario]
                     for scenario in ["heterogeneous", "zero_learning"]}).T)
show_figure("08_simulation_recovery_calibration")
show_figure("09_zero_learning_guardrail")''')
    md('''Nominal 90% intervals need not achieve exactly 90% coverage. The archived results report observed coverage, its Monte Carlo error across independent experiments, and any overconfidence. Pairwise reliability is measured against **known synthetic truth**, with uncertainty obtained by resampling independent experiments. This evaluates synthetic comparisons. It does not establish calibration of MLB comparisons against later-season truth.''')
    md('''## 9. Sensitivity analyses and negative controls

Sensitivity analyses use within-PA, within-game, similar-shape, and transition exposures; component-specific deviations; sample restrictions; and alternative priors.

Shuffled residual exposure, future exposure, and exposure to other pitch types serve as negative controls. Future pitch counts can correlate with game progression and selection even though they cannot cause earlier adjustments. A similar result for these controls would weaken the adaptation interpretation.''')
    code('''show_figure("10_robustness_negative_controls")
robustness = table("robustness.csv")
display(robustness.loc[robustness.season.eq(2026) & robustness.specification.isin([
    "exposure_pitcher", "shuffled_within_batter_game", "future_exposure", "irrelevant_exposure"]),
    ["specification", "players", "mu_median", "mu_low", "mu_high", "p_mu_positive"]])''')
    code('''show_figure("12_outcome_link_and_predictive_check")''')
    md('''The outcome plot checks whether the deviation proxy is related to whiffs. An association with whiffs does not establish that all deviations are mechanical errors. The posterior predictive check targets the **first-stage slope estimates**, because that is the outcome of the Bayesian measurement model. Raw-swing, whiff, and miss-distance generative checks would require additional generative models.''')
    md('''### Outcome validation and sample selection

A classifier trained on 2025 predicts 2026 whiffs from current-pitch context, then adds observed swing-component deviations. This measures held-out discrimination using contemporaneous mechanics. The required measurements are observed during the swing, so it cannot serve as a pre-pitch whiff forecast.

Miss distance provides a bat-ball proximity endpoint observed only on a selected subset of tracked whiffs. **Distances for contacts remain missing.** Its log transform is standardized using 2025 whiffs; the resulting conditional exposure slope cannot establish unconditional error learning. The inclusion audit separately shows how tracking, sanity bounds, and shape availability vary across exposure opportunities.''')
    code('''show_figure("15_outcome_sensitivity_and_selection")
display(table("whiff_validation.csv")[["model", "test_season", "test_n", "log_loss", "auc"]])''')
    md('''### Matched-pitch comparisons

This supplementary restriction matches within hitter, game, pitcher, pitch type, and exact count. Speed, movement, front-plane location and release point must satisfy fixed physical tolerances. Each swing is used at most once; outcomes and their missingness never choose a match. The full pairing audit is included in the frozen data.

Negative later-minus-earlier changes mean declining deviation. These pair-weighted changes have a different estimand from the player hierarchy, and the selected matched opportunities do not represent all swings. Game-bootstrap intervals condition on the fitted models. See [matching rules and limitations](../paper/matched_comparisons.md).''')
    code('''matched = table("matched_exposure_summary.csv")
display(matched[["season", "outcome", "n_pairs", "games", "players",
    "mean_change_later_minus_earlier", "ci90_low", "ci90_high"]])
display(table("matched_exposure_balance.csv")[["season", "covariate",
    "mean_absolute_difference", "max_absolute_difference", "unit"]])''')
    md('''## 10. Temporal persistence

Persistence is tested in a joint latent-normal model that treats estimated first-stage SEs as known. Its player bootstrap can reveal wide uncertainty or boundary behavior when true between-player variation is weak. An ordinary correlation of noisy point estimates does not account for estimation error. The reported bootstrap percentiles are descriptive and need not have nominal coverage near variance or correlation boundaries. A false boundary or weak-identification flag does not establish precise estimation.

A separate holdout test asks whether 2025 player-specific slopes improve 2026 centered residual loss over the population slope. This tests conditional slope transport. It does not evaluate prospective forecasts of uncentered raw swings or future OPS or wOBA.''')
    code('''show_figure("11_temporal_persistence")
persistence = document("results/tables/persistence.json")
holdout = document("results/tables/holdout_prediction.json")
display(pd.Series({
    "Shared hitters": persistence["n_players"],
    "Latent correlation": persistence["latent_correlation"],
    "90% correlation interval": persistence["bootstrap_ci90"],
    "Boundary fit": persistence["boundary_fit"],
    "Bootstrap fits with |correlation| > 0.98": persistence["bootstrap_diagnostics"]["abs_rho_above_0_98_count"],
    "2026 loss improvement (%)": holdout["mse_improvement_pct"],
    "90% loss-improvement interval": holdout["ci90"],
}, name="Temporal validation").to_frame())''')
    md('''### Pairwise agreement across seasons

This diagnostic draws reproducible random pairs from the shared hitter population, computes each pair's 2025 and 2026 posterior comparison probabilities, and averages the later probabilities within bins of earlier probability. Pair orientation is fixed by identifier, never chosen from the apparent winner.

**The 2026 posterior is an uncertain estimate.** Selection into both seasons, estimation noise, and genuine change prevent this plot from establishing empirical calibration. The curve describes agreement. Pairs share players, so independent-pair error bars would be inappropriate. The table's same-order probability combines annual marginal probabilities under an independent-fits approximation. A joint model of persistent skill is not fitted.''')
    code('''show_figure("14_temporal_pairwise_agreement")
display(table("temporal_pairwise_agreement.csv")[["n_pairs", "mean_probability_2025",
    "mean_probability_2026", "mean_probability_same_order"]])''')
    md('''### Within-year reliability

The supplementary 2025 check splits complete games at July 1 and applies the same 40-swing, eight-game eligibility rule to each half. The measurement-error likelihood adjusts for approximate slope SEs. Shared nuisance functions mean the halves are not fully independent. This evaluates conditional split-half reliability and does not provide a prospective forecast. A boundary correlation must be read as weakly identified. The [supplementary report](../paper/supplementary_validation.md) also includes descriptive pitcher sequence diagnostics with game-bootstrap intervals.''')
    code('''split_half = document("results/tables/split_half_2025_persistence.json")
display(pd.Series({key: split_half[key] for key in ["n_players", "split_date",
    "latent_correlation", "bootstrap_ci90", "boundary_fit", "weak_identification", "bootstrap_successes"]},
    name="Within-2025 reliability").to_frame())
if split_half.get("weak_identification"):
    display(Markdown("**The split-half correlation is weakly identified.** The likelihood profile "
        "and bootstrap support a wide range of correlations. The point estimate "
        "does not establish reliable individual differences."))''')
    md('''### Exploratory later-season performance

The next table compares regressions using prior performance and swing summaries with otherwise identical regressions that add the 2025 posterior mean correction association. Each player's outcome is held out during fitting, but the regressions **use other players' 2026 outcomes** as training targets. This is a **player-held-out association test**. It is not a forecast available before 2026.

The analysis uses point estimates without propagating adaptation measurement error. Its xwOBA endpoint is restricted to recorded batted balls. These exploratory results have a different validation scope from the independently specified temporal transport test above.''')
    code('''future_performance = table("future_performance.csv")
display(future_performance[["target", "model", "n_players", "mse"]])''')
    md('''## 11. Exploratory percentile stability

This section was specified after the original 2026 results had been inspected. It examines the proposal that relative player positions or broad percentile groups may be more stable than slope magnitudes. It is a post-holdout exploratory analysis.

Full-sample comparisons use all available observations in each season; the 2026 snapshot ends on September 20 and does not cover the complete season.

Converting a score to its percentile preserves its order and therefore its Spearman correlation. Partial pooling can change the order because players have different estimation precision. Different outcomes and exposure definitions can also measure different associations. The search compares 84 specifications, two score implementations, and five endpoints, retaining every result. The endpoints are Spearman correlation, Kendall correlation, quartile trend, decile trend, and top-quartile retention. Multiplicity adjustments cover the complete full-sample endpoint family.

The next cell checks percentile invariance using the actual primary player estimates. The saved search also includes six sensitivities requiring at least 500 contributing swings in each season.''')
    code('''from scipy.stats import spearmanr
percentile_results = table("percentile_stability.csv")
percentile_summary = document("results/tables/percentile_stability_summary.json")
rank_pairs = pd.read_csv(ROOT / "results/tables/percentile_full_year_pairs.csv.gz")
PRIMARY_SPEC = "distortion__log_exposure_pitcher"
example_pairs = rank_pairs.loc[rank_pairs.specification.eq(PRIMARY_SPEC)]
raw_rho = spearmanr(example_pairs.estimate_a, example_pairs.estimate_b).statistic
percentile_rho = spearmanr(example_pairs.estimate_a.rank(pct=True),
                           example_pairs.estimate_b.rank(pct=True)).statistic
assert np.isclose(raw_rho, percentile_rho)
display(pd.Series({"Correlation of slope ranks": raw_rho,
                   "Correlation after percentile conversion": percentile_rho,
                   "Specifications": percentile_summary["specifications"],
                   "Full-sample endpoint tests": percentile_summary["full_year_tests"],
                   "Positive non-control Spearman results with BH q ≤ 0.05":
                       percentile_summary["positive_spearman_bh_significant"]},
                  name="Exploratory rank search").to_frame())
show_figure("percentile_stability")''')
    md('''### Internal replication and the complete search

Candidate selection used March through June in both years. The selection rule chose one positive non-control Spearman result using adjusted significance, then the permutation p value and correlation. That candidate was then evaluated on July through September in both years, alongside the primary score. These are disjoint games, but their observations had contributed to previous full-sample analyses and share fitted nuisance functions. This is an internal replication check, not an untouched test sample.

The first table reports this selected candidate and its internal replication. The second shows the strongest full-sample rank results with their multiplicity corrections. Full-sample results were computed after the candidate was fixed and cannot replace it retrospectively. Bootstrap intervals are pointwise and condition on estimated scores. Stable sample sizes and standard errors can contribute to pooled-rank associations.''')
    code('''selection = document("results/tables/percentile_selection.json")["selected"]
discovery = table("percentile_discovery.csv")
confirmation = table("percentile_confirmation.csv")
if selection is not None:
    selected_rows = pd.concat([
        discovery.loc[discovery.specification.eq(selection["specification"]) &
                      discovery.method.eq(selection["method"])],
        confirmation.loc[confirmation.specification.eq(selection["specification"]) &
                         confirmation.method.eq(selection["method"])]])
    display(selected_rows[["stage", "specification", "method", "shared_players",
                           "estimate", "ci90_low", "ci90_high", "bh_q_positive"]])
rank_results = percentile_results.loc[percentile_results.endpoint.eq("spearman") &
                                      ~percentile_results.negative_control]
display(rank_results.sort_values(["bh_q_positive", "estimate"], ascending=[True, False])
        [["specification", "method", "shared_players", "estimate", "ci90_low",
          "ci90_high", "bh_q_positive", "by_q_positive"]].head(12))
display(Markdown("The [full percentile report](../paper/percentile_stability.md) documents "
                 "all specifications, null results, inference limits, and reproduction commands."))''')
    md('''## 12. Interpretation and limitations

The research assessment generated from the saved results follows below. It distinguishes aggregate exposure associations from evidence of persistent individual skill. The project cannot claim an identified motor-learning rate from observational pitch histories alone.

Limitations include tracking selection, endogenous pitcher strategy, residual confounding, the interpretation of a deviation from an expected swing, finite-cluster slope approximations, and incomplete propagation of baseline-model uncertainty. Memory-kernel comparisons do not by themselves estimate a personal forgetting rate. Synthetic recovery cannot establish causal identification or calibration in MLB data.''')
    code('''notes = ROOT / "paper/results_notes.md"
if notes.exists():
    results_text = notes.read_text()
    for filename in ["percentile_stability.md", "supplementary_validation.md", "matched_comparisons.md"]:
        results_text = results_text.replace(f"]({filename})", f"](../paper/{filename})")
    display(Markdown(results_text))
else:
    display(Markdown("Read the generated tables above. No research conclusion is substituted for missing results notes."))

print("Reproduction artifacts:")
for folder in ["data/checksums.json", "results/tables", "results/posterior", "results/figures", "paper/abstract.md"]:
    print(" •", folder)''')
    md('''## Supporting documents

- [Methods and identification notes](../paper/methods_notes.md)
- [Related work and primary sources](../literature/related_work.md)
- [Exploratory percentile analysis](../paper/percentile_stability.md)
- [Data provenance and reconstruction](../data/README.md)
- [Generated SSAC abstract](../paper/abstract.md)
- [Project README and full reproduction commands](../README.md)

Player evaluation or development applications require evidence from negative controls, temporal persistence, and measurement validation. The estimated distributions should be interpreted within those limits.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python (Deja Swing)', 'language':'python', 'name':'deja-swing'},
        'language_info': {'name':'python', 'version':'3.12', 'mimetype':'text/x-python', 'file_extension':'.py'},
        'title': title,
    })
    nbf.validate(notebook)
    target = root / 'notebooks/deja_swing.ipynb'; target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook,target)
    return target


if __name__ == '__main__':
    print(build_notebook())
