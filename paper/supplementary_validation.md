# Supplementary validation

This supplement uses frozen main-pipeline residuals and does not refit or select the main models. Choices were fixed before inspecting these diagnostics: a July 1, 2025 split; 40 eligible swings and eight games per hitter in each half; 100 player-bootstrap replicates; and 100 swings/eight games with 500 game-bootstrap replicates for descriptive pitcher summaries. The 2026 holdout was not used to choose these settings.

## Within-season reliability

All swings from a game remain in one half. Early dates are 2025-03-18–2025-06-30; late dates are 2025-07-01–2025-09-28. The primary first-stage estimator is reused unchanged: negative slope of adjusted swing distortion on adjusted log(1 + prior pitches of the same type seen from the same pitcher in the game), with a hitter intercept and game-cluster sandwich standard errors. Positive slopes in the saved tables therefore mean decreasing distortion with exposure, not a percentage learning rate. Residual exposure information must exceed the estimator's fixed threshold of one.

There are 497 eligible early-half hitters, 491 eligible late-half hitters, and **406 matched hitters**. Raw Pearson correlation is **-0.095** and Spearman correlation is **-0.082**. A bivariate latent-normal measurement-error likelihood, treating the period slope SEs as known and independent, estimates correlation **0.012**, with a 90% player-bootstrap interval **[-0.999, 0.999]** (100/100 successful fits). The fitted correlation is weakly identified; its point estimate must not be interpreted as evidence of reliable individual differences.

Fitted latent SDs are 0.048068 early and 0.041058 late, versus median first-stage SEs 0.123308 and 0.135166. Their signal-to-noise SD ratios are 0.3898 and 0.3038. Model-implied signal fractions at the median SE are 13.1915% early and 8.4475% late. The likelihood-profile, bootstrap, or optimizer diagnostic flags weak identification despite non-negligible point-estimate variances. The original strict numerical boundary rule is retained in the JSON; it is not a sufficient identification check. The reporting-only weak-identification flag combines a period latent SD below 10% of its median SE, broad fixed-rho likelihood support, bootstrap near-zero variance frequency, and optimizer diagnostics. These are heuristic warnings, not calibrated hypothesis tests. Numerical optimization was corrected by analytically profiling means, scaling by median SE, and using multiple nonzero variance starts with tighter tolerances; the likelihood, data, eligibility, split, bootstrap count, and seed are unchanged. Bootstrap quantiles near a variance or correlation boundary are descriptive and are not guaranteed calibrated confidence limits.

These are conditional reliability diagnostics. The 2025 nuisance predictions hold out whole games but share fitted nuisance functions across halves, so estimation-error independence is approximate and this is not a prospective early-to-late forecast. Nuisance-fit uncertainty and cross-hitter dependence are omitted; eligibility and survival into both halves remain selection mechanisms. A measurement-error correction cannot establish persistent skill when heterogeneity or interval estimates are weak. No causal learning interpretation follows from this split.

Outputs: `results/tables/split_half_2025_pairs.csv`, `split_half_2025_bootstrap.csv`, and `split_half_2025_persistence.json`.

## Exploratory pitcher diagnostics

`results/tables/pitcher_sequence_diagnostics.csv` reports unpooled, descriptive results for 580 pitchers in 2025 and 560 in 2026. The predictive metric compares current-pitch-only and current-plus-history expected-swing models, both fitted before July 2024. Each component's squared residual is divided by its pre-2025 calibration variance; their equal-weight mean forms a standardized loss. Reported sequence gain is 100 × (current loss − history loss) / current loss. This loss comparison is distinct from the Mahalanobis distortion outcome used in the adaptation model.

Paired game-cluster bootstrap resampling within each pitcher gives 90% pointwise intervals for predictive gain. The table also reports mean adjusted distortion with an intercept-only game-cluster standard error and a t(G−1) interval, solely as a residual-calibration diagnostic. Minimum eligibility is 100 analytic swings across eight games per pitcher-season, fixed without looking at gains.

Pitcher identity, opponent mix, selection into tracked swings, counts, changing strategy, and model misspecification can affect these quantities. The intervals do not include shared model-fit uncertainty, partial pooling, or multiplicity adjustment. These tables are **not a causal pitcher-deception estimate, a pitcher skill leaderboard, or evidence of persistent pitcher setup ability**. No pitcher is selected for an interpretive case study based on a favorable result. Full settings and limitations are in `results/tables/pitcher_sequence_diagnostics.json`.

## Reproduction

Run `python scripts/17_supplementary_validation.py` after the main model stage finishes. Input Parquet hashes are saved in `results/logs/supplementary_validation_inputs.json`. All computations are deterministic under seed 20260921. The supplementary script does not write any analytic data or main-model artifacts.
