# Adaptation speed and future player qualities

## Design

This exploratory analysis freezes each hitter's 2025 posterior adaptation score, then relates it to 2026 qualities among 261 hitters with at least 200 plate appearances and common opportunity thresholds in both seasons. Positive adaptation scores indicate a faster decline in context-adjusted swing-mechanics deviation with accumulated same-type looks across pitchers within a game. Every future-quality comparison controls for the same quality in 2025.

The outcomes cover swing decisions, bat-to-ball skill, plate discipline, batted-ball authority, overall production, and bat speed. The analysis uses partial rank correlation, ordinary least squares, Huber regression, quantile regression, repeated cross-validation, Benjamini-Hochberg correction, and a shared-score maximum-statistic permutation test.

## Findings

The pre-specified five-component batted-ball authority composite has a partial rank correlation of 0.131 (95% bootstrap interval 0.009 to 0.249; permutation p=0.035). Its standardized OLS coefficient is 0.069, its Huber coefficient is 0.040, and adding the score changes repeated-cross-validation mean squared error by 0.53%. The quantile coefficients are 0.057, 0.009, and 0.052 at the 25th, 50th, and 75th percentiles.

The nominal individual associations are Hard-hit rate (rho=0.130), Mean exit velocity (rho=0.139), Mean bat speed (rho=0.149). None is treated as confirmatory without the reported multiplicity adjustments.

The relationship is domain-specific. Swing decisions, bat-to-ball outcomes, walk and strikeout rates, and overall wOBA do not show a comparable prospective association. Shorter-window scores also fail to reproduce the annual authority result consistently. The evidence therefore identifies a candidate link between the adaptation measure and future contact authority, but it does not yet establish a persistent general adaptability skill.

## Interpretation

A plausible interpretation is that hitters whose mechanics converge faster during repeated same-type exposure also retain or develop greater capacity to produce forceful contact. The small cross-validated gain limits its current decision value. A clean preregistered season and an independent cohort are needed before using this score for player evaluation.
