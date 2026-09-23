# Deja Swing

### Estimating batter adaptation speed from swing mechanics

This repository measures how a hitter's swing changes as he accumulates same-type looks across pitchers within a game. A context model first estimates the swing mechanics expected on each pitch. The adaptation score measures how quickly the hitter's realized multivariate swing deviation declines with repeated exposure.

Start with the [focused executable notebook](notebooks/adaptability_submission.ipynb), its [HTML reading view](notebooks/adaptability_submission.html), or the [interactive player-ranking explorer](notebooks/adaptability_rankings.html).

## Main result

Faster estimated adaptation in 2025 has a small positive association with 2026 batted-ball authority after controlling for each hitter's 2025 authority. In the common 261-hitter cohort:

- Partial rank correlation: **0.131**
- 95% bootstrap interval: **0.009 to 0.249**
- Permutation p-value: **0.035**
- Repeated cross-validation MSE improvement: **0.53%**

The authority composite averages five standardized measures: hard-hit rate, barrel rate, mean exit velocity, 90th-percentile exit velocity, and expected wOBA on contact. The components were defined as a contact-authority domain rather than selected for their observed correlations.

The relationship is specific to contact authority. Chase rate, zone-swing rate, whiff rate, strikeout rate, walk rate, and overall wOBA do not show comparable prospective relationships. No individual outcome survives correction across all 12 comparisons, and half-season score replications remain inconclusive. The result is therefore an exploratory external correlate, not proof of a persistent general talent.

## Player rankings

The ranking explorer covers 587 eligible hitters in 2025 and 591 in 2026. It reports:

- posterior adaptation rate and 90% interval;
- point rank and posterior rank interval;
- probability that the adaptation rate is positive;
- contributing tracked-swing exposure count;
- fitted annual adaptation curves; and
- future-authority residual for hitters in the external-validation cohort.

Point ranks order posterior means. They are not exact talent ranks. The median 2025 posterior rank interval spans 508 places, which demonstrates the uncertainty in ordering individual hitters.

The complete ranking table is [available as CSV](results/adaptability_rankings/player_rankings.csv).

## Methods

The expected-swing model adjusts for measured pitch context. Exposure counts earlier delivered pitches of the same Statcast pitch type seen by the hitter in the game, including pitches from other pitchers. Hitter-specific conditional slopes use game-clustered standard errors and enter a normal measurement-error hierarchy for partial pooling.

The player-quality analysis freezes the 2025 score and evaluates later qualities after controlling for the same 2025 quality. It reports partial rank correlation, ordinary least squares, Huber regression, quantile regression, repeated cross-validation, bootstrap intervals, Benjamini-Hochberg correction, and a shared-score maximum-statistic permutation test.

The full outcome definitions and findings are in the [player-quality report](paper/player_quality_outcomes.md). The [same-type mechanics audit](paper/abstract_claim_audit.md) documents the population exposure result and the lagged contact checks.

## Reproduction

Python 3.12 is the tested runtime.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m ipykernel install --prefix .venv --name deja-swing --display-name "Python (Deja Swing)"
make submission
```

Useful targets:

```bash
make cross-pitcher       # same-type scores and player curves
make claim-audit         # repeated-exposure mechanics and lagged-contact audit
make player-qualities    # future player-quality associations
make rankings            # figures, ranking table, and standalone explorer
make submission-notebook # focused executed notebook and HTML
make test                # unit and artifact checks
```

The frozen public Statcast partitions support offline replay. `make data` is the only target that contacts MLB.

## Repository map

- `notebooks/adaptability_submission.ipynb`: focused runnable research notebook
- `notebooks/adaptability_rankings.html`: interactive player explorer
- `scripts/20_cross_pitcher_adaptation.py`: same-type cross-pitcher scores
- `scripts/62_abstract_claim_audit.py`: mechanics and lagged-outcome audit
- `scripts/63_player_quality_outcomes.py`: external player-quality analysis
- `scripts/64_adaptability_rankings.py`: rankings and explorer
- `results/player_quality_outcomes/`: complete outcome tables
- `results/adaptability_rankings/`: rankings and figures
- `paper/`: methods and evidence-aligned writing

The code is MIT licensed. Public MLB data retains MLB's rights. Please cite `CITATION.cff` and Baseball Savant.
