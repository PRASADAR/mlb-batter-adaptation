# Percentile stability: a post-holdout exploratory search

## Status and question

This analysis was requested after the original 2026 validation had been inspected. It is an explicitly exploratory search across a fixed set of 84 specifications, not a replacement for the frozen primary analysis or a new untouched holdout. It asks whether relative positions and broad groups are more reproducible than individual slope magnitudes. Every specification and null result is retained.

The file label `full_year` denotes all available records within each season: the complete 2025 regular season versus the 2026 regular season through September 20. It does not denote a completed 2026 season.

A monotone percentile transformation preserves ordering. Consequently, converting an unpooled slope into its empirical percentile leaves Spearman and Kendall correlation unchanged. The distinct comparisons here are partial-pooling changes to the order, coarse percentile groups, different fixed exposure/outcome definitions, and a higher-information population.

## Fixed search space and estimation

The search crosses 12 previously specified exposure transformations with six saved outcomes (Mahalanobis distortion and five component absolute standardized residuals), giving 72 combinations. Four existing restrictions use the primary exposure and distortion outcome: bat speed at least 50 mph, innings at most six, innings at most three, and a similar pitch within three preceding pitches. Two negative controls use future and physically irrelevant exposure. Six full-sample sensitivities require at least 500 contributing swings in each season for the primary exposure and each outcome. Total: 84 specifications. No cutoff is selected from 2026 results.

The ordinary eligibility rule is 40 swings and eight games per period, with residual exposure information at least one, matching the first-stage estimator. At least 20 shared eligible hitters are needed for a rank comparison. Positive scores mean a decrease in the stated residual outcome with increasing stated exposure. Same-pitcher exposure is specifically the count of prior pitches of the same pitch type from that pitcher to the hitter in the game. It is not a causal learning fraction.

Game-cluster slope SEs and frozen nuisance-adjusted quantities are reused. Expected-swing and nuisance models are not refitted. Unpooled estimates are compared with deterministic posterior means under the same Normal population mean prior (SD 0.2) and HalfNormal heterogeneity prior (scale 0.15) as the primary model. Quadrature is refined and checked, avoiding Monte Carlo noise in tiny shrunken rank differences. Rankings of posterior means still do not imply well-separated latent player distributions.

Within each period, percentile = (average rank − 0.5) / number of eligible hitters. Quartile and decile cutpoints are fixed at the usual fractions using the earlier period alone; later eligibility never determines the earlier ranks or cutpoints. The 500-swing sensitivity retains the same earlier-period ranking definition and hierarchy, but restricts the matched comparison population in both years.

Conditional miss distance is omitted from this frozen-model search. There is no saved nuisance-adjusted miss-distance outcome, and regressions on raw miss distance would not provide the same current-pitch-conditioned estimand. Obtaining it requires a separately specified nuisance fit, outside these 84 cases.

## Inference and multiplicity

Each full-sample specification has two score implementations and five endpoints: Spearman correlation, Kendall correlation, an ordered quartile trend, an ordered decile trend, and top-quartile retention. Ordered group trends correlate the earlier group number with the later percentile. This creates **840 estimable endpoint tests** in one full-sample multiplicity family. The group trends use their ordinal spacing and are supplementary to rank correlations.

All endpoints have 1000 paired-player bootstrap resamples and 90% intervals, conditional on estimated scores and group definitions. They do not propagate nuisance-model or hierarchy-refit uncertainty. Permutations (19999 per full-sample comparison) shuffle later-period player labels within the matched population. Positive-direction and two-sided Monte Carlo p values include the standard plus-one correction. For retention, the null is the later top-quartile fraction in that matched population, which need not equal 25% after roster selection. BH q values cover all searched endpoints; a BY adjustment is also saved as a conservative dependence sensitivity. Intervals are pointwise, not simultaneous. Player resampling does not account for shared-game or fitted-model dependence across hitters.

## Internal partition check

Before computing these new rank results, discovery was restricted to March–June 2025 versus March–June 2026. A single candidate and score method were selected by the smallest positive-direction discovery BH q, then p, then largest positive Spearman, with deterministic ties. Negative controls and the full-sample-only high-information sensitivity were excluded from candidate selection. The discovery multiplicity family includes both score implementations for all 78 applicable specifications.

Only after that selection was saved were July–September 2025 versus July–September 2026 comparisons computed for the candidate specification and the primary specification, with both score methods reported and a separate confirmation BH family. These partition-specific rank comparisons had not previously been computed. The underlying observations and full-sample 2026 results had already contributed to earlier analyses, and 2025 nuisance fits share information across halves. This is an internal replication diagnostic on disjoint game records, not independent external confirmation or an untouched holdout. Full-sample search results were computed after this internal selection and check.

The fixed discovery rule selected `absz_swing_length__log_kernel_h1_m5` using unpooled scores: discovery Spearman 0.181, BH q=0.0351.

- `distortion__log_exposure_pitcher`, unpooled: rho=0.033, 90% player-bootstrap interval [-0.054, 0.120], permutation p=0.25900, BH q=0.5180, n=367.
- `distortion__log_exposure_pitcher`, pooled: rho=0.041, 90% player-bootstrap interval [-0.042, 0.122], permutation p=0.22130, BH q=0.5180, n=367.
- `absz_swing_length__log_kernel_h1_m5`, unpooled: rho=-0.012, 90% player-bootstrap interval [-0.093, 0.076], permutation p=0.58020, BH q=0.7736, n=367.
- `absz_swing_length__log_kernel_h1_m5`, pooled: rho=-0.052, 90% player-bootstrap interval [-0.141, 0.027], permutation p=0.84240, BH q=0.8424, n=367.

## Full-sample results

There are **11 positive non-control Spearman results with BH q at most 0.05**. The leading full-sample non-control Spearman entry under the report sorting rule (smallest BH q, then largest rho) is `absz_swing_length__log_kernel_h2_m5` (pooled): rho=0.153, 90% interval [0.080, 0.229], p=0.00040, BH q=0.0339, n=465. This entry is descriptive of the searched family and is not retrospectively substituted for the candidate selected in discovery.

The primary outcome and a useful component comparison are:

- `distortion__log_exposure_pitcher`, unpooled: rho=-0.051, 90% interval [-0.133, 0.027], BH q=0.9031, n=465.
- `distortion__log_exposure_pitcher`, pooled: rho=-0.070, 90% interval [-0.149, 0.004], BH q=0.9448, n=465.
- `absz_swing_length__log_exposure_pitcher`, unpooled: rho=0.138, 90% interval [0.063, 0.215], BH q=0.0352, n=465.
- `absz_swing_length__log_exposure_pitcher`, pooled: rho=0.145, 90% interval [0.067, 0.221], BH q=0.0339, n=465.
- `absz_swing_length__log_exposure_pitcher__n500`, unpooled: rho=0.190, 90% interval [0.080, 0.299], BH q=0.0437, n=203.
- `absz_swing_length__log_exposure_pitcher__n500`, pooled: rho=0.195, 90% interval [0.086, 0.307], BH q=0.0437, n=203.

The positive component associations are small and appear in unpooled scores as well as pooled scores. Swing length also has a positive association in the fixed population with at least 500 contributing swings in both seasons. The corresponding primary distortion comparison remains negative. **0 of the 840 full-sample endpoints survive positive-direction BY adjustment at 0.05.** Ordinary BH results are therefore sensitive to the chosen treatment of dependence in this large, overlapping search family.

Negative-control rank results: log_future_exposure unpooled: rho=0.124, q=0.0507; log_future_exposure pooled: rho=0.156, q=0.0339; log_irrelevant_exposure unpooled: rho=-0.020, q=0.8415; log_irrelevant_exposure pooled: rho=-0.023, q=0.8415.

The future-exposure control has a positive pooled association at least as large as the strongest non-control association selected by the report sorting rule. The early-season discovery candidate fails the late-season internal replication check under both score implementations. Together these results limit the interpretation: the search finds modest descriptive component associations, but does not validate a stable latent adaptability ranking. Percentile conversion does not repair the primary persistence result, and the selected component findings warrant a new external test rather than a positive causal conclusion.

The full tables include eligible and shared player counts, median contributing swings/games, residual exposure information, SEs, shrinkage, score–SE correlations, and SE persistence. Stable precision can alter pooled rankings; a pooled rank association requires scrutiny alongside unpooled ranks, the 500-swing sensitivity, negative controls, and the internal partition check. A positive full-sample correlation alone does not establish a persistent causal adaptation skill.

## Secondary precision diagnostic

Only the primary specification, the discovery-selected candidate, and the full-sample pooled specification with the smallest BH q, then p, then largest rho are examined here. The additional p-value tie-breaker selects a different case from the report sorting rule above. This is a descriptive post-selection check, not another search or confirmatory test. Scores and both seasons' log contributing-swing counts and log first-stage SEs are rank-transformed. Earlier and later score ranks are separately residualized against an intercept and these ranked controls; their residual correlation is reported. Paired-player bootstrap resampling refits these small regressions but does not refit the frozen nuisance models or hierarchy. Counts and SEs can themselves relate to player skill, so this adjustment neither proves confounding nor identifies a causal learning rate. Intervals do not correct for selecting these cases.

- `distortion__log_exposure_pitcher`, unpooled: unadjusted rho=-0.051; precision-adjusted rank correlation=-0.049, conditional 90% interval [-0.127, 0.030].
- `distortion__log_exposure_pitcher`, pooled: unadjusted rho=-0.070; precision-adjusted rank correlation=-0.066, conditional 90% interval [-0.139, 0.010].
- `absz_swing_length__log_kernel_h1_m5`, unpooled: unadjusted rho=0.100; precision-adjusted rank correlation=0.099, conditional 90% interval [0.021, 0.179].
- `absz_swing_length__log_kernel_h1_m5`, pooled: unadjusted rho=0.096; precision-adjusted rank correlation=0.095, conditional 90% interval [0.022, 0.176].
- `absz_swing_length__log_exposure_pitcher`, unpooled: unadjusted rho=0.138; precision-adjusted rank correlation=0.143, conditional 90% interval [0.059, 0.217].
- `absz_swing_length__log_exposure_pitcher`, pooled: unadjusted rho=0.145; precision-adjusted rank correlation=0.152, conditional 90% interval [0.073, 0.224].

## Files and reproduction

Run `python scripts/19_percentile_stability.py` to reproduce the bounded search. Ordinary notebook replay should read the saved tables and call `src.visualization.percentiles.render_percentile_stability`, which performs no model fitting or search.

- `results/tables/percentile_stability.csv`: all full-sample endpoint tests, estimates, intervals, and multiplicity corrections.
- `percentile_discovery.csv`, `percentile_confirmation.csv`, and `percentile_selection.json`: ordered internal search and replication results.
- `percentile_groups.csv`: all fixed quartile and decile groups with later percentile summaries.
- `percentile_player_scores.csv.gz` and `percentile_full_year_pairs.csv.gz`: player-level scores, uncertainty inputs, eligibility, and matched records.
- `percentile_hierarchy_diagnostics.csv`: exact pooling and information diagnostics for every fitted period/specification.
- `percentile_precision_diagnostics.csv`: selected-case partial rank associations after count/SE adjustment.
- `results/logs/percentile_stability_inputs.json`: frozen input hashes and fixed settings.
- `results/figures/percentile_stability.png` and `.pdf`: full-family rank results and the primary fixed-group comparison.

No frozen primary model, analytic dataset, or primary conclusion is overwritten by this script. This search can support a revision only if the observed evidence, uncertainty, multiplicity, and replication warrant it; null and contradictory results remain visible.
