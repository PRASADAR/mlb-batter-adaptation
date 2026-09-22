# Deja Swing: Can Bat Tracking Measure Hitter Adaptation?

## Introduction

A hitter can change his swing while the box score records another whiff. We test whether public bat tracking detects changes associated with repeated pitch exposure and whether those changes form a persistent player skill. A reliable measure could inform player development and advance scouting.

## Methods

We analyzed 2,433,934 regular-season MLB pitch records from 2023-07-14 through 2026-09-20. We predicted five swing dimensions from the current pitch and player context, then measured each swing's unexpected deviation. We estimated how that deviation varied with prior pitches of the same type from the same pitcher within a game. A hierarchical model partially pooled noisy hitter slopes and produced posterior comparison and rank distributions. Earlier seasons trained the swing model; 2025 supplied game-group cross-fitting, and 2026 tested the original specification. Negative controls and known-truth simulations examined validity. An exploratory search specified after inspecting 2026 compared 84 exposure, outcome, and sample definitions.

## Results

The conditional population exposure estimate was 0.035 (90% credible interval 0.028 to 0.043) in 2025 and 0.030 (0.022 to 0.037) in 2026, in calibration-standardized deviation units per log-exposure unit. Pitch history reduced 2026 attack-angle prediction error by 0.25% (90% game-bootstrap interval 0.21% to 0.30%). Observed mechanics improved contemporaneous whiff classification log loss by 6.35%. Evidence for a stable individual rate was weak: the measurement-error-adjusted cross-season correlation was -0.28 (90% player-bootstrap range -1.00 to 0.35), and individual 2025 slopes increased 2026 conditional prediction error by 0.0047% (90% interval 0.0011% to 0.0081%). In the exploratory search, swing-length ranks correlated 0.195 among 203 hitters with at least 500 swings in each season (90% interval 0.086 to 0.307). The future-exposure control correlated 0.156; the early-season-selected candidate did not replicate later (-0.012). Of 11 positive rank correlations passing Benjamini-Hochberg adjustment, 0 passed the more conservative Benjamini-Yekutieli adjustment.

## Conclusion

Bat tracking reveals small, exposure-associated changes in swing geometry and permits uncertainty-aware player comparisons. Current evidence does not establish a persistent hitter adaptation skill. Teams can use the component findings to select hypotheses for independent validation before using them in player decisions.
