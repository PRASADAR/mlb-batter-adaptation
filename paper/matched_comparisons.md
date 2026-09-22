# Matched current-pitch comparisons

This supplementary restriction was specified before inspecting its paired outcome changes. It does not change the primary models. The specification is frozen in `results/logs/matched_exposure_specification.json`.

Match exactly within season, hitter, game, pitcher, pitch type, balls, and strikes. A later tracked swing must have greater prior same-pitcher-type exposure. Eligible earlier swings must differ by at most 2 mph in velocity, 0.15 ft in each movement component, 0.35 ft in Euclidean front-of-plate location, and 0.25 ft in Euclidean x/z release location. Front-plane location has the pipeline's common measurement-plane adjustment.

Process swings chronologically. Match each later swing to the available earlier swing with the smallest Euclidean covariate distance after dividing each coordinate by its fixed tolerance. Exact distance ties favor the most recent earlier swing. Neither swing can be used again. Outcomes and outcome missingness never affect matching. Missing matching covariates do exclude a swing.

The contrast is distortion on the later swing minus distortion on the earlier swing, reported for the original and nuisance-adjusted deviation score. Negative changes mean less deviation; they are not a percentage correction, not divided by exposure increments, and not estimates of a causal learning rate. Each matched pair receives equal weight, so hitters with more eligible pairs contribute more; this is not the same estimand as the hierarchical population mean. The uncertainty intervals resample complete games 2,000 times separately by season, conditional on the fitted nuisance models, observed hitter pool, and selected matches; players are not independently resampled.

The paired audit preserves pitch identifiers, covariates, exposure increments, outcomes, distances, and changes. The balance table reports both signed and absolute differences, because signed balance alone can conceal mismatch. Exact count matching and no replacement limit reuse but do not remove unmeasured intent, changing pitcher strategy, regression to the mean, survivorship, or tracking selection. Two-strike foul sequences and repeated same-count opportunities can be overrepresented. These strict pairs do not represent all swings or all hitters; agreement with a primary result would be a descriptive robustness check, not causal identification.

Specification frozen: 2026-09-22T00:02:26.868890+00:00. SHA256: `4f3994788ad088169bae1f11190be88f5a9301b172da3a7e1f9abb77a9f9ea1b`.

## Generated paired changes

| Season | Outcome | Pairs | Games | Hitters | Mean later − earlier | 90% game-bootstrap interval |
|---|---|---:|---:|---:|---:|---|
| 2025 | distortion | 752 | 639 | 366 | -0.0470 | [-0.1058, 0.0123] |
| 2025 | adj_distortion | 752 | 639 | 366 | -0.0394 | [-0.0981, 0.0200] |
| 2026 | distortion | 668 | 565 | 353 | -0.0779 | [-0.1429, -0.0109] |
| 2026 | adj_distortion | 668 | 565 | 353 | -0.0734 | [-0.1387, -0.0065] |

Selection fractions and exposure increments are reported in `results/tables/matched_exposure_summary.csv`; covariate balance is in `results/tables/matched_exposure_balance.csv`.
