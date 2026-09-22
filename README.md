# Deja Swing
### The search for batter adaptation

This repository examines whether repeated exposure to a pitch is associated with changes in swing geometry. It uses public MLB Statcast data, temporal validation, and probability distributions over hitter-specific adjustment slopes.

Start with **[the executed Jupyter notebook](notebooks/deja_swing.ipynb)** or its **[HTML reading edition](notebooks/deja_swing.html)**. The notebook presents the analysis and runnable examples. The production analysis is implemented in `src/`. The [abstract PDF](paper/abstract.pdf) is generated from [the exact Markdown source](paper/abstract.md).

## Research question and estimand

The model estimates whether a hitter's **unexpected swing deviation declines with prior exposure to the same pitch type from the same pitcher within a game**, after adjusting for observed current-pitch and sequence context. It produces partially pooled distributions, pairwise comparisons, and uncertain ranks. The conditional exposure slope is a candidate measure of adjustment. Its causal interpretation as a motor-learning rate is not identified.

A positive rate denotes a decline in calibration-standardized deviation per unit of `log(1 + prior exposures)`. It does not measure the percentage of remaining error corrected. Aggregate association and persistent individual skill require separate tests. Swing atypicality also requires validation before it can be interpreted as mechanical error.

## Main empirical results

<!-- RESULTS_START -->
Analyzed **2,433,934 pitch-level records across 8,271 games**, 2023-07-14 through 2026-09-20. The primary parameter is a conditional exposure association, not a causal learning rate.

| Period | Eligible hitters | Population correction association | 90% credible interval |
|---|---:|---:|---:|
| 2025 | 587 | 0.0351 | [0.0276, 0.0427] |
| 2026 | 591 | 0.0298 | [0.0224, 0.0374] |

Measurement-error-adjusted cross-season correlation: **-0.282**, 90% player-bootstrap interval **[-0.999, 0.355]**, across 465 shared hitters. The interval crosses zero; 14/100 bootstrap fits have |correlation| > 0.98. These are descriptive bootstrap percentiles and may not have nominal coverage near boundaries.

Knowing individual 2025 slopes **increases** 2026 conditional prediction MSE by **0.0047%**, 90% interval **[0.0011%, 0.0081%]**. This centers within heldout hitters and tests slope transport, not prospective raw-swing forecasting.

Negative controls must be read alongside the primary result:

| 2026 exposure specification | Mean correction association | 90% interval |
|---|---:|---:|
| future_exposure | -0.0559 | [-0.0620, -0.0494] |
| irrelevant_exposure | -0.0040 | [-0.0108, 0.0032] |
| shuffled_within_batter_game | 0.0070 | [0.0004, 0.0141] |

In 60 heterogeneous-rate experiments, nominal 90% intervals covered **87.89%** of known rates. Partial pooling lowered RMSE from **0.0691** to **0.0548**. In zero-learning experiments, the population 90% interval excluded zero **4/60** times. This validates only the simulated model; coverage is imperfect.

**Primary-model conclusion:** Repeated exposure is associated with modestly reduced swing deviation, but the primary individual slopes do not support a persistent talent metric in this snapshot. Their held-out prediction is slightly worse than the population-slope benchmark. The improvement in whiff classification provides evidence that contemporaneous mechanics contain predictive information. It does not establish individual learning speed.

**Exploratory percentile analysis:** The search covered 84 specifications and 840 endpoint tests. There were **11 positive non-control Spearman results with BH q ≤ 0.05**. Sorting by BH q and then decreasing correlation gives `absz_swing_length__log_kernel_h2_m5` (pooled): correlation **0.153**, pointwise 90% interval **[0.080, 0.229]**, BH q=0.0339, BY q=0.2476. This is a result from the searched family.

The component findings concentrate in swing length. Among 203 hitters with at least 500 contributing swings in each season, the partially pooled swing-length score had correlation **0.195**, 90% interval **[0.086, 0.307]**, BH q=0.0437. However, **0 non-control rank results pass BY q ≤ 0.05**, and the structurally related future-exposure control is similarly stable: correlation **0.156**, interval **[0.076, 0.225]**, BH q=0.0339. These patterns do not isolate adaptation from stable selection or model structure.

The candidate selected from early-season partitions was `absz_swing_length__log_kernel_h1_m5` (unpooled): Spearman **0.181**, BH q=0.0351. In the later-season internal replication, its correlation was **-0.012**, 90% interval **[-0.093, 0.076]**, BH q=0.7736. The selected relationship did not replicate. Percentile conversion alone leaves Spearman correlation unchanged. The search was specified after inspecting the original 2026 analysis; its findings are exploratory. Full-sample associations cannot establish persistent latent skill without replication. See the [complete percentile report](paper/percentile_stability.md).
<!-- RESULTS_END -->

Quantitative statements in the abstract use the saved outputs. The report includes negative controls and null results. The analysis does not establish which player learns fastest.

## Contribution and prior work

[Powers and Yurko's *Swinging, Fast and Slow*](https://github.com/saberpowers/swinging-fast-and-slow) establishes why bat metrics depend on pitch context and measurement position. [Steinhardt and Borowiak's *Hitter and Catcher Adaptation in Major League Baseball*](https://sabr.org/journal/article/hitter-and-catcher-adaptation-in-major-league-baseball/) already studies repeated-pitch adaptation. [Prasad's SSAC sequencing paper](https://www.sloansportsconference.com/research-papers/decoding-mlb-pitch-sequencing-strategies-via-directed-graph-embeddings) analyzes sequential setup structure.

The contribution is the combination of mechanical deviation, measurement-error pooling, probabilistic comparisons, negative controls, and a later-season persistence test. Earlier research has already studied hitter adaptation. The [related-work review](literature/related_work.md) details overlaps and source-specific limitations. [Official SSAC requirements and verification notes](literature/ssac_requirements.md) are documented separately.

## Data and measurement

- Public Baseball Savant regular-season pitch exports covering the requested July 2023–September 2026 window; complete-day analysis ends September 20, 2026. Individual source requests and independent MLB schedule checks establish actual coverage.
- Frozen **raw partitions and final analytic Parquet files are included**, with [SHA256 checksums](data/checksums.json). The downloader supplements the included analyzed dataset.
- Tracking availability is measured among all pitches, swings, whiffs, and contact, by year/type/hitter in [coverage tables](results/tables/data_coverage.csv). The audit includes bunts; the primary swing model excludes them and applies documented broad instrument bounds.
- Five components: attack angle, circular attack direction, swing-path tilt, bat speed and swing length. Intercept variables and miss distance are audited. Miss distance remains conditional on whiffs and is never assumed zero on contact.
- Current-pitch expectations use earlier data; calibration-only residual covariance defines a standardized Mahalanobis deviation. Physical component residuals remain available. Outcome association is tested separately.
- Histories count **all delivered pitches, including takes**, before restricting to tracked swings. Automatic-ball/strike administrative events remain in the source audit but are excluded from physical exposure histories. No PA-level lag crosses a PA boundary. Raw future-schedule columns are never model inputs.

See [data provenance](data/README.md) and [detailed methods](paper/methods_notes.md). Publicly accessible MLB data retains MLB's rights; the MIT code license does not relicense source data.

## Exposure, inference and comparisons

The primary exposure counts previous same-type pitches from the same pitcher in the current game. Alternatives include within-PA, within-game type, repeated sequence, continuous shape similarity, and fixed within-game memory kernels. Raw and saturating exposure transforms provide functional-form sensitivity. No across-game forgetting or causal state-space parameter is claimed.

Per-player conditional slopes and game-cluster standard errors enter a normal measurement-error hierarchy. Population mean and heterogeneity are integrated by deterministic quadrature; independent draws propagate hyperparameter uncertainty. This avoids chain-convergence uncertainty, but still approximates first-stage uncertainty and dependence. Grid diagnostics, posterior predictive checks, priors and simulation coverage are saved.

`results/posterior/comparisons_YEAR.npz` additionally stores every strict pairwise probability and the complete discrete rank probability distribution for each included hitter. The legacy `n_exposures` column counts tracked analytic swings contributing to a player slope. Prior-pitch counts are recorded separately. The results table reports credible intervals, probabilities above the population mean and selected-player median, quartile/decile probabilities, and 50%/90% rank intervals. Smaller samples generally provide less information and receive more shrinkage. Eligible players can still differ substantially in estimation precision.

```python
from src.pipeline import load_posterior
from src.models.hierarchical import compare_hitters
posterior = load_posterior('adaptation_2025')
# Substitute any two IDs present in posterior['player_ids'].
a, b = posterior['player_ids'][:2]
comparison = compare_hitters(posterior, int(a), int(b))
print(comparison['probability_a_greater'], comparison['cri90'])
```

## Validation and limits

| Question | Implemented check |
|---|---|
| Does history explain more mechanics? | Paired current/history model losses on later seasons, game-bootstrap intervals |
| Is deviation relevant to contact? | Later-season whiff prediction with and without residual components; conditional miss-distance summaries |
| Can the statistical model recover known rates? | Independent raw-data simulations with unequal counts, missingness, clustered noise and zero learning |
| Are probabilistic comparisons calibrated? | Known-truth simulation reliability; empirical temporal agreement labeled separately from calibration |
| Does adjustment persist? | Measurement-error-aware cross-season and split-half correlation; later-season conditional slope prediction |
| Are percentile relationships more stable? | Post-holdout search of 84 specifications, all rank/group endpoints, multiplicity adjustments and internal replication |
| Could exposure reflect game selection? | Future/irrelevant/shuffled controls, selection tables, strict current-pitch matching, shape and inning restrictions |
| Does the definition drive results? | Component, kernel, functional-form, prior and swing-speed sensitivity |
| Is there incremental outcome utility? | Exploratory player-held-out next-season whiff/xwOBA regressions |

The 2025 nuisance models hold out complete games but are not forward-only. The original 2026 primary prediction functions use only prior years. Additional analyses specified after inspecting 2026 are exploratory. Separately labeled exploratory player-level outcome regressions use other players' 2026 outcomes for their training folds. Retrospective backfill and the 2026 ABS zone definition change limit transport interpretation. The simulations validate a simpler correctly specified slope model. They do not validate the entire bat-measurement pipeline. Finite-cluster undercoverage, unknown optimal swing geometry, missingness, shared-game dependence and endogenous pitcher response preclude a causal talent claim.

[Matched-pitch comparisons](paper/matched_comparisons.md) fix tolerances before reading paired effects and never use outcomes to select matches. [Supplementary validation](paper/supplementary_validation.md) reports within-2025 reliability and descriptive pitcher sequence-prediction diagnostics. These checks preserve the primary model and do not identify a causal hitter or pitcher skill.

The [exploratory percentile search](paper/percentile_stability.md) compares unpooled and partially pooled ranks, broad percentile groups, component outcomes, exposure definitions, and a 500-swing sensitivity. It reports the complete search and a candidate selected from early-season data before a late-season internal check. It was specified after the original 2026 results were inspected.

## Reproduce

Python 3.12 is the tested runtime (install it before creating a fresh environment). A fresh clone includes the frozen data, executed notebook, figures, tables and posterior draws; the default notebook requires no network access. Allow disk space for the scientific environment and approximately 1.3 GB of frozen data and research artifacts.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m ipykernel install --prefix .venv --name deja-swing --display-name "Python (Deja Swing)"
.venv/bin/jupyter lab notebooks/deja_swing.ipynb
```

Choose **Python (Deja Swing)** and **Restart Kernel → Run All**. The first runnable cell locates the repository from either the project root or the notebook directory. The notebook shows static results by default; dropdown comparison controls work in a live Jupyter session. `RUN_FULL_PIPELINE` is off by default because model refitting is substantially slower than replaying the delivered results.

```bash
make test        # mathematical, sequence, leakage and notebook contracts
make audit       # inspect the existing frozen raw sample
make features    # reconstruct histories from all pitches
make models      # baseline, deviation, nuisance and player posteriors
make validation  # controls, recovery, persistence and outcome checks
make exploratory # bounded post-holdout percentile search and internal replication
make results     # publication plots and dataset checksums
make abstract    # numerical abstract from computed tables
make pdf         # exact-text PDF and programmatic validation
make notebook    # regenerate, execute every cell, export HTML
```

`make reproduce` runs the complete offline research sequence. `make data` explicitly contacts MLB to retrieve/check the fixed target snapshot; existing validated partitions are reused. To refresh sources use the downloader's explicit `--force` flag, preserve old hashes, and treat the result as a new research version. Seeds and hyperparameters are in `config/analysis.json`; some feature constants are deliberately fixed and documented in the source. Cached models are hash-checked against training inputs and settings before reuse. The full CPU refit can take several hours on a laptop; the shipped notebook replay avoids that fit and is much faster. A full environment version record is in `results/logs/environment.json`.

## Repository map

`src/data` → download/audit; `src/features` → histories and trajectories; `src/models` → expected mechanics and hierarchy; `src/evaluation` → recovery and utility; `src/visualization` → figures; `src/pipeline.py` → production stages. `scripts/` contains entry points and document/notebook builders. `notebooks/` presents tested runnable examples and executed research. `paper/` contains the brief, methods, results, abstract and full-paper outline.

## Baseball use and publication status

The framework can identify adjustment hypotheses for further player-development measurement and quantify uncertainty in player comparisons. It should not be used to rank talent or prescribe pitching strategy without the falsification and persistence evidence required for that application.

This is the existing `mlb-batter-adaptation` GitHub Desktop repository. After reviewing the finished commit, use **Publish repository** in GitHub Desktop and choose public visibility. The [publication checklist](paper/publication_checklist.md) explains how to verify the public link and data before using it in the SSAC submission. The abstract's scientific assessment and numerical word count are saved in `paper/abstract_metadata.json`. Validated PDF layout does not establish that the research satisfies every scientific criterion for the competition.

Please cite `CITATION.cff`, MLB Baseball Savant, and the methodological sources. Original software is MIT licensed.
