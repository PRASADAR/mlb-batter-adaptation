#!/usr/bin/env python3
"""Build the submission abstract from frozen empirical results."""
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads((ROOT / path).read_text())

def row(table, specification, season):
    return table.loc[table.specification.eq(specification) & table.season.eq(season)].iloc[0]

sample = read_json('results/logs/sample.json')
robustness = pd.read_csv(ROOT / 'results/tables/robustness.csv')
sequence = pd.read_csv(ROOT / 'results/tables/sequence_distortion.csv')
whiff = pd.read_csv(ROOT / 'results/tables/whiff_validation.csv').set_index('model')
persistence = read_json('results/tables/persistence.json')
transport = read_json('results/tables/holdout_prediction.json')
rank = pd.read_csv(ROOT / 'results/tables/percentile_stability.csv')
selection = read_json('results/tables/percentile_selection.json')['selected']
confirmation = pd.read_csv(ROOT / 'results/tables/percentile_confirmation.csv')

primary_2025 = row(robustness, 'exposure_pitcher', 2025)
primary_2026 = row(robustness, 'exposure_pitcher', 2026)
attack = sequence.loc[sequence.season.eq(2026) & sequence.component.eq('attack_angle')].iloc[0]
whiff_gain = 100 * (1 - whiff.loc['plus_swing_deviations', 'log_loss'] / whiff.loc['current_pitch', 'log_loss'])
high_information = rank.loc[
    rank.specification.eq('absz_swing_length__log_exposure_pitcher__n500') &
    rank.method.eq('pooled') & rank.endpoint.eq('spearman')].iloc[0]
future_control = rank.loc[
    rank.specification.eq('distortion__log_future_exposure') &
    rank.method.eq('pooled') & rank.endpoint.eq('spearman')].iloc[0]
replication = confirmation.loc[
    confirmation.specification.eq(selection['specification']) &
    confirmation.method.eq(selection['method']) &
    confirmation.endpoint.eq('spearman')].iloc[0]
rank_count = int((rank.endpoint.eq('spearman') & ~rank.negative_control &
                  rank.estimate.gt(0) & rank.bh_q_positive.le(.05)).sum())
by_count = int((rank.endpoint.eq('spearman') & ~rank.negative_control &
                rank.estimate.gt(0) & rank.by_q_positive.le(.05)).sum())
loss = -transport['mse_improvement_pct']
loss_ci = [-transport['ci90'][1], -transport['ci90'][0]]
rho_ci = persistence['bootstrap_ci90']

text = f'''# Deja Swing: Can Bat Tracking Measure Hitter Adaptation?

## Introduction

A hitter can change his swing while the box score records another whiff. We test whether public bat tracking detects changes associated with repeated pitch exposure and whether those changes form a persistent player skill. A reliable measure could inform player development and advance scouting.

## Methods

We analyzed {sample['raw_pitches']:,} regular-season MLB pitch records from {sample['start']} through {sample['end']}. We predicted five swing dimensions from the current pitch and player context, then measured each swing's unexpected deviation. We estimated how that deviation varied with prior pitches of the same type from the same pitcher within a game. A hierarchical model partially pooled noisy hitter slopes and produced posterior comparison and rank distributions. Earlier seasons trained the swing model; 2025 supplied game-group cross-fitting, and 2026 tested the original specification. Negative controls and known-truth simulations examined validity. An exploratory search specified after inspecting 2026 compared 84 exposure, outcome, and sample definitions.

## Results

The conditional population exposure estimate was {primary_2025.mu_median:.3f} (90% credible interval {primary_2025.mu_low:.3f} to {primary_2025.mu_high:.3f}) in 2025 and {primary_2026.mu_median:.3f} ({primary_2026.mu_low:.3f} to {primary_2026.mu_high:.3f}) in 2026, in calibration-standardized deviation units per log-exposure unit. Pitch history reduced 2026 attack-angle prediction error by {attack.mse_gain_pct:.2f}% (90% game-bootstrap interval {attack.gain_ci90_low:.2f}% to {attack.gain_ci90_high:.2f}%). Observed mechanics improved contemporaneous whiff classification log loss by {whiff_gain:.2f}%. Evidence for a stable individual rate was weak: the measurement-error-adjusted cross-season correlation was {persistence['latent_correlation']:.2f} (90% player-bootstrap range {rho_ci[0]:.2f} to {rho_ci[1]:.2f}), and individual 2025 slopes increased 2026 conditional prediction error by {loss:.4f}% (90% interval {loss_ci[0]:.4f}% to {loss_ci[1]:.4f}%). In the exploratory search, swing-length ranks correlated {high_information.estimate:.3f} among {int(high_information.shared_players)} hitters with at least 500 swings in each season (90% interval {high_information.ci90_low:.3f} to {high_information.ci90_high:.3f}). The future-exposure control correlated {future_control.estimate:.3f}; the early-season-selected candidate did not replicate later ({replication.estimate:.3f}). Of {rank_count} positive rank correlations passing Benjamini-Hochberg adjustment, {by_count} passed the more conservative Benjamini-Yekutieli adjustment.

## Conclusion

Bat tracking reveals small, exposure-associated changes in swing geometry and permits uncertainty-aware player comparisons. Current evidence does not establish a persistent hitter adaptation skill. Teams can use the component findings to select hypotheses for independent validation before using them in player decisions.
'''

(ROOT / 'paper/abstract.md').write_text(text)
plain = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
words = len(plain.split())
assert words < 500, words
assert not re.search(r'—|\bnot only\b|\bbut also\b', text, flags=re.I)
info = {
    'word_count_including_title_headings_author': words,
    'title_and_body_word_count': words,
    'aggressive_alternate_title': 'Deja Swing: Why Apparent Hitter Adjustment May Not Be a Skill',
    'academic_alternate_title': 'Temporal Reliability of Exposure-Associated Swing Geometry in Major League Baseball',
    'persistent_skill_claim_supported': False,
    'negative_control_flag': True,
    'source_tables': ['sample.json', 'robustness.csv', 'sequence_distortion.csv',
                      'whiff_validation.csv', 'persistence.json', 'holdout_prediction.json',
                      'percentile_stability.csv', 'percentile_selection.json',
                      'percentile_confirmation.csv'],
}
(ROOT / 'paper/abstract_metadata.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info, indent=2))
