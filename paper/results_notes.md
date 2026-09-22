# Computed results and scientific status

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

The candidate selected from early-season partitions was `absz_swing_length__log_kernel_h1_m5` (unpooled): Spearman **0.181**, BH q=0.0351. In the later-season internal replication, its correlation was **-0.012**, 90% interval **[-0.093, 0.076]**, BH q=0.7736. The selected relationship did not replicate. Percentile conversion alone leaves Spearman correlation unchanged. The search was specified after inspecting the original 2026 analysis; its findings are exploratory. Full-sample associations cannot establish persistent latent skill without replication. See the [complete percentile report](percentile_stability.md).

## Sequence information

| season | component | n | rmse_current | rmse_history | mse_gain_pct | gain_ci90_low | gain_ci90_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | attack_angle | 159801 | 8.7474 | 8.7422 | 0.11742 | 0.062765 | 0.18326 |
| 2024 | attack_direction | 159801 | 13.87 | 13.875 | -0.069787 | -0.13468 | 0.0056848 |
| 2024 | swing_path_tilt | 159801 | 3.639 | 3.6415 | -0.13937 | -0.25621 | -0.0092801 |
| 2024 | bat_speed | 159801 | 6.5471 | 6.5487 | -0.051115 | -0.11013 | 0.0083361 |
| 2024 | swing_length | 159801 | 0.54873 | 0.54854 | 0.070909 | -0.0043911 | 0.15189 |
| 2025 | attack_angle | 328151 | 8.9513 | 8.9512 | 0.0013967 | -0.039677 | 0.042751 |
| 2025 | attack_direction | 328151 | 14.05 | 14.053 | -0.041514 | -0.086806 | 0.0029969 |
| 2025 | swing_path_tilt | 328151 | 4.0372 | 4.0412 | -0.19631 | -0.28816 | -0.10472 |
| 2025 | bat_speed | 328151 | 6.7873 | 6.7899 | -0.078832 | -0.11637 | -0.034437 |
| 2025 | swing_length | 328151 | 0.58124 | 0.58059 | 0.2253 | 0.17048 | 0.28234 |
| 2026 | attack_angle | 316797 | 9.0212 | 9.0098 | 0.25252 | 0.20912 | 0.2978 |
| 2026 | attack_direction | 316797 | 14.188 | 14.194 | -0.086053 | -0.1362 | -0.03015 |
| 2026 | swing_path_tilt | 316797 | 4.3115 | 4.3048 | 0.31101 | 0.21153 | 0.42391 |
| 2026 | bat_speed | 316797 | 6.917 | 6.9137 | 0.094067 | 0.048012 | 0.14242 |
| 2026 | swing_length | 316797 | 0.6069 | 0.60649 | 0.13556 | 0.077413 | 0.19348 |

## Whiff classification

| model | train_season | test_season | test_n | log_loss | auc |
| --- | --- | --- | --- | --- | --- |
| current_pitch | 2025 | 2026 | 316713 | 0.43728 | 0.78132 |
| plus_swing_deviations | 2025 | 2026 | 316713 | 0.4095 | 0.81706 |

## Exploratory later-season outcomes

| target | model | n_players | mse | scope |
| --- | --- | --- | --- | --- |
| whiff | prior_performance | 404 | 0.0012008 | Exploratory player-held-out regression predicting 2026 from 2025; no measurement-error propagation; xwOBA among batted balls only. |
| whiff | plus_adaptation | 404 | 0.0012031 | Exploratory player-held-out regression predicting 2026 from 2025; no measurement-error propagation; xwOBA among batted balls only. |
| xwoba | prior_performance | 404 | 0.0016874 | Exploratory player-held-out regression predicting 2026 from 2025; no measurement-error propagation; xwOBA among batted balls only. |
| xwoba | plus_adaptation | 404 | 0.0016917 | Exploratory player-held-out regression predicting 2026 from 2025; no measurement-error propagation; xwOBA among batted balls only. |

## Complete sensitivity results

| specification | season | players | mu_median | mu_low | mu_high | p_mu_positive | tau_median | tau_low | tau_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exposure_pitcher | 2025 | 587 | 0.035075 | 0.02764 | 0.042735 | 1 | 0.036065 | 0.019251 | 0.049237 |
| exposure_pa | 2025 | 587 | 0.03067 | 0.017962 | 0.044257 | 0.99967 | 0.073883 | 0.051653 | 0.094827 |
| exposure_type | 2025 | 587 | 0.025308 | 0.019526 | 0.031478 | 1 | 0.032092 | 0.021137 | 0.041962 |
| kernel_exposure | 2025 | 587 | 0.055319 | 0.045419 | 0.065822 | 1 | 0.058908 | 0.042244 | 0.074789 |
| exposure_sequence | 2025 | 587 | 0.025282 | 0.017614 | 0.033436 | 1 | 0.044631 | 0.030875 | 0.057505 |
| future_exposure | 2025 | 587 | -0.045926 | -0.052285 | -0.039267 | 0 | 0.048737 | 0.040836 | 0.057218 |
| irrelevant_exposure | 2025 | 587 | -0.0051979 | -0.011328 | 0.0013527 | 0.096667 | 0.031192 | 0.017153 | 0.042439 |
| kernel_h0.5_m5 | 2025 | 586 | 0.076592 | 0.061794 | 0.092232 | 1 | 0.091069 | 0.06755 | 0.11406 |
| kernel_h0.5_m20 | 2025 | 587 | 0.05681 | 0.045586 | 0.068616 | 1 | 0.068984 | 0.051418 | 0.08624 |
| kernel_h1_m5 | 2025 | 586 | 0.076482 | 0.063279 | 0.090502 | 1 | 0.080421 | 0.058707 | 0.10138 |
| kernel_h2_m5 | 2025 | 587 | 0.077869 | 0.065989 | 0.090577 | 1 | 0.067656 | 0.046026 | 0.087483 |
| kernel_h2_m20 | 2025 | 587 | 0.055538 | 0.046559 | 0.065113 | 1 | 0.051622 | 0.035806 | 0.066338 |
| linear_exposure_pitcher | 2025 | 587 | 0.011058 | 0.0085869 | 0.013687 | 1 | 0.01492 | 0.010941 | 0.018823 |
| saturating_exposure_pitcher | 2025 | 582 | 0.076399 | 0.06072 | 0.093054 | 1 | 0.077909 | 0.042108 | 0.10658 |
| component:attack_angle | 2025 | 587 | 0.023212 | 0.01835 | 0.028397 | 1 | 0.02192 | 0.0081698 | 0.031913 |
| component:attack_direction | 2025 | 587 | 0.023243 | 0.018338 | 0.028483 | 1 | 0.023389 | 0.0094506 | 0.033578 |
| component:swing_path_tilt | 2025 | 587 | 0.013171 | 0.0076481 | 0.018959 | 0.99967 | 0.03771 | 0.029948 | 0.045768 |
| component:bat_speed | 2025 | 587 | 0.011579 | 0.0063438 | 0.017193 | 0.99933 | 0.017063 | 0.0029252 | 0.029343 |
| component:swing_length | 2025 | 587 | 0.022413 | 0.01728 | 0.027889 | 1 | 0.026652 | 0.014947 | 0.03624 |
| shuffled_within_batter_game | 2025 | 587 | 0.0074526 | 0.00054694 | 0.014849 | 0.96567 | 0.018432 | 0.0024541 | 0.036051 |
| similar_shape_recent | 2025 | 551 | 0.039189 | 0.029016 | 0.049797 | 1 | 0.053965 | 0.032991 | 0.071944 |
| competitive_swing_speed_50 | 2025 | 587 | 0.020399 | 0.014052 | 0.02712 | 1 | 0.036376 | 0.025693 | 0.046506 |
| early_in_game | 2025 | 574 | 0.035037 | 0.026639 | 0.043986 | 1 | 0.049611 | 0.035323 | 0.063224 |
| first_three_innings | 2025 | 490 | 0.029335 | 0.015631 | 0.043889 | 0.999 | 0.076722 | 0.052721 | 0.099505 |
| prior_tau_scale:0.075 | 2025 | 587 | 0.035057 | 0.027822 | 0.042746 | 1 | 0.035808 | 0.019278 | 0.04902 |
| prior_tau_scale:0.3 | 2025 | 587 | 0.035056 | 0.027807 | 0.04276 | 1 | 0.036247 | 0.019932 | 0.049433 |
| exposure_pitcher | 2026 | 591 | 0.029821 | 0.022368 | 0.037352 | 1 | 0.045755 | 0.033375 | 0.057545 |
| exposure_pa | 2026 | 590 | 0.028935 | 0.016137 | 0.042543 | 0.99967 | 0.083771 | 0.063821 | 0.10378 |
| exposure_type | 2026 | 591 | 0.018821 | 0.012842 | 0.025175 | 1 | 0.037156 | 0.027809 | 0.04639 |
| kernel_exposure | 2026 | 591 | 0.048266 | 0.038718 | 0.058382 | 1 | 0.057115 | 0.039789 | 0.073414 |
| exposure_sequence | 2026 | 591 | 0.023166 | 0.015674 | 0.03108 | 1 | 0.033825 | 0.012849 | 0.049132 |
| future_exposure | 2026 | 591 | -0.055875 | -0.062032 | -0.049406 | 0 | 0.043979 | 0.035462 | 0.052865 |
| irrelevant_exposure | 2026 | 591 | -0.0040486 | -0.010845 | 0.0031941 | 0.17533 | 0.052275 | 0.043342 | 0.061685 |
| kernel_h0.5_m5 | 2026 | 587 | 0.064891 | 0.050764 | 0.079789 | 1 | 0.076956 | 0.046839 | 0.10287 |
| kernel_h0.5_m20 | 2026 | 591 | 0.051209 | 0.040472 | 0.062468 | 1 | 0.06045 | 0.039082 | 0.079567 |
| kernel_h1_m5 | 2026 | 590 | 0.062959 | 0.050407 | 0.076334 | 1 | 0.071635 | 0.047225 | 0.09375 |
| kernel_h2_m5 | 2026 | 591 | 0.063873 | 0.052279 | 0.076146 | 1 | 0.068945 | 0.048713 | 0.088251 |
| kernel_h2_m20 | 2026 | 591 | 0.047976 | 0.039139 | 0.057321 | 1 | 0.055437 | 0.041038 | 0.069608 |
| linear_exposure_pitcher | 2026 | 591 | 0.0098064 | 0.0072508 | 0.012479 | 1 | 0.017218 | 0.013585 | 0.020964 |
| saturating_exposure_pitcher | 2026 | 584 | 0.065423 | 0.04979 | 0.081812 | 1 | 0.0959 | 0.068973 | 0.12168 |
| component:attack_angle | 2026 | 591 | 0.026132 | 0.021689 | 0.030903 | 1 | 0.0095288 | 0.0010995 | 0.02175 |
| component:attack_direction | 2026 | 591 | 0.025831 | 0.020731 | 0.031232 | 1 | 0.034151 | 0.026848 | 0.041659 |
| component:swing_path_tilt | 2026 | 591 | 0.017974 | 0.012834 | 0.023338 | 1 | 0.030413 | 0.021927 | 0.038671 |
| component:bat_speed | 2026 | 591 | 0.010407 | 0.0053494 | 0.015801 | 0.999 | 0.023663 | 0.0092681 | 0.034221 |
| component:swing_length | 2026 | 591 | 0.024973 | 0.019507 | 0.030785 | 1 | 0.038458 | 0.030265 | 0.046861 |
| shuffled_within_batter_game | 2026 | 591 | 0.0070114 | 0.00036052 | 0.01407 | 0.96033 | 0.024787 | 0.0049207 | 0.040719 |
| similar_shape_recent | 2026 | 538 | 0.036175 | 0.02631 | 0.046481 | 1 | 0.051258 | 0.031428 | 0.068243 |
| competitive_swing_speed_50 | 2026 | 589 | 0.017829 | 0.011747 | 0.024188 | 1 | 0.034042 | 0.023264 | 0.044078 |
| early_in_game | 2026 | 557 | 0.032885 | 0.024647 | 0.041607 | 1 | 0.052596 | 0.039852 | 0.065278 |
| first_three_innings | 2026 | 496 | 0.036861 | 0.023803 | 0.050678 | 1 | 0.067649 | 0.041257 | 0.090755 |
| prior_tau_scale:0.075 | 2026 | 591 | 0.029803 | 0.022529 | 0.037414 | 1 | 0.045515 | 0.033361 | 0.057318 |
| prior_tau_scale:0.3 | 2026 | 591 | 0.029794 | 0.022504 | 0.037422 | 1 | 0.045916 | 0.033794 | 0.057734 |

## Scope

Consult methods_notes.md and the executed notebook. These results do not identify causal motor learning. Statistical heterogeneity and narrow player intervals do not by themselves establish persistent adaptation or coaching utility. Negative-control failure weakens the interpretation even if a population association is precisely estimated.

Future exposure is structurally related to past exposure through the total number of same-type pitches in a game. Its opposite-sign association highlights timing and selection sensitivity; it is not an independent randomized placebo and does not prove that hitters never learn. Shuffling exposure residuals within hitter-game retains between-game structure, so its remaining association warns about that component of the primary estimate.

## Matched current-pitch restriction

| season | outcome | n_input_swings | n_covariate_eligible_swings | n_pairs | n_pairs_with_outcome | games | players | fraction_eligible_swings_matched | mean_exposure_increment | mean_change_later_minus_earlier | ci90_low | ci90_high | bootstrap_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | distortion | 328082 | 328082 | 752 | 752 | 639 | 366 | 0.0045842 | 1.7713 | -0.046974 | -0.10581 | 0.012304 | 2000 |
| 2025 | adj_distortion | 328082 | 328082 | 752 | 752 | 639 | 366 | 0.0045842 | 1.7713 | -0.039378 | -0.098103 | 0.019975 | 2000 |
| 2026 | distortion | 316713 | 316713 | 668 | 668 | 565 | 353 | 0.0042183 | 1.7425 | -0.077934 | -0.14286 | -0.010881 | 2000 |
| 2026 | adj_distortion | 316713 | 316713 | 668 | 668 | 565 | 353 | 0.0042183 | 1.7425 | -0.073442 | -0.13869 | -0.0064724 | 2000 |

## Whiff-conditional miss distance

| season | n_whiffs | n_players | mu_median | mu_low | mu_high | scope |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 77033 | 479 | 0.063367 | 0.05166 | 0.075747 | Conditional on tracked whiffs; contacts are missing, not zero. Log-distance normalized to 2025 SD; not unconditional mechanical learning. |
| 2026 | 73608 | 477 | 0.051698 | 0.040915 | 0.063242 | Conditional on tracked whiffs; contacts are missing, not zero. Log-distance normalized to 2025 SD; not unconditional mechanical learning. |

See [supplementary validation](supplementary_validation.md) for the weakly identified split-half fit and descriptive pitcher diagnostics, and [matched comparisons](matched_comparisons.md) for pairing rules, balance and selection.
