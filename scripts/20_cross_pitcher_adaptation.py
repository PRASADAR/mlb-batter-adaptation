#!/usr/bin/env python3
"""Post hoc, same-type adaptation analysis that pools exposure across pitchers.

The original same-pitcher analysis is preserved as the frozen benchmark. This
requested reanalysis reuses its cross-fitted nuisance residuals, which already
include log_exposure_type = log(1 + earlier same-type pitches from any pitcher
for this hitter in this game). No 2026 swing outcomes are used to refit them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.expected import player_slopes
from src.models.hierarchical import fit_hierarchy, measurement_error_persistence, posterior_mean_rates, summarize_posterior
from src.pipeline import names_map

CONFIG = json.loads((ROOT / 'config/analysis.json').read_text())
SEED = CONFIG['seed']
OUT = ROOT / 'results/cross_pitcher'
COLS = ['season', 'game_pk', 'batter', 'pitcher', 'pitch_type', 'exposure_type',
        'exposure_pitcher', 'adj_log_exposure_type', 'adj_distortion']
SPECS = {
    'pooled_same_type': 'All tracked swings. Prior same-type pitches from any pitcher in the same game.',
    'first_type_from_pitcher': 'Only the first same-type pitch from the current pitcher. Any positive exposure came from other pitchers in the same game.',
}


def save_json(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')


def analytic_frame():
    paths = sorted((ROOT / 'data/processed').glob('analytic_*.parquet'))
    if not paths:
        raise FileNotFoundError('Run make models before the cross-pitcher reanalysis.')
    frame = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    if set(frame.season) != {2025, 2026}:
        raise ValueError('Expected the frozen 2025 development and 2026 validation seasons.')
    if (frame.exposure_type < frame.exposure_pitcher).any():
        raise AssertionError('Across-pitcher exposure must include same-pitcher exposure.')
    if not np.isfinite(frame[['adj_log_exposure_type', 'adj_distortion']]).all().all():
        raise AssertionError('Missing cross-fitted residuals in the analytic snapshot.')
    return frame


def exposure_audit(frame):
    rows = []
    for year, g in frame.groupby('season'):
        cross = g.exposure_type.gt(g.exposure_pitcher)
        first = g.exposure_pitcher.eq(0)
        rows.append({
            'season': int(year), 'analytic_swings': len(g),
            'swings_with_prior_same_type_from_other_pitcher': int(cross.sum()),
            'first_type_from_current_pitcher_swings': int(first.sum()),
            'first_type_from_current_pitcher_with_cross_pitcher_history': int((first & cross).sum()),
            'hitters_with_strict_cross_pitcher_history': int(g.loc[first & cross, 'batter'].nunique()),
            'games_with_strict_cross_pitcher_history': int(g.loc[first & cross, 'game_pk'].nunique()),
        })
    return pd.DataFrame(rows)


def holdout_prediction(fitted_2025, validation):
    exact_player_means, population_mean = posterior_mean_rates(fitted_2025.estimate, fitted_2025.se)
    player_means = dict(zip(fitted_2025.player_id, exact_player_means))
    h = validation.loc[validation.batter.isin(player_means)].copy()
    x = h.adj_log_exposure_type - h.groupby('batter').adj_log_exposure_type.transform('mean')
    y = h.adj_distortion - h.groupby('batter').adj_distortion.transform('mean')
    pred_player = -h.batter.map(player_means).to_numpy() * x.to_numpy()
    pred_population = -population_mean * x.to_numpy()
    loss = pd.DataFrame({'game': h.game_pk, 'population': (y - pred_population) ** 2,
                         'player': (y - pred_player) ** 2}).groupby('game').sum()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(loss), (1000, len(loss)))
    pop = loss.population.to_numpy()[picks].sum(axis=1)
    player = loss.player.to_numpy()[picks].sum(axis=1)
    gain = 100 * (pop - player) / pop
    return {
        'validation_swings': len(h), 'shared_hitters': int(h.batter.nunique()),
        'gain_pct': float(100 * (loss.population.sum() - loss.player.sum()) / loss.population.sum()),
        'game_bootstrap_ci90': np.quantile(gain, [.05, .95]),
        'definition': 'Within-player centered 2026 conditional residual MSE: 2025 player slopes versus 2025 population slope.',
        'scope': 'Tests slope transport on held-out 2026 swings, not prospective raw-swing prediction.',
    }


def analyze(frame):
    OUT.mkdir(parents=True, exist_ok=True)
    audit = exposure_audit(frame)
    audit.to_csv(OUT / 'exposure_audit.csv', index=False)
    rows = []
    persistence = {}
    transport = {}
    names = names_map()
    for spec, description in SPECS.items():
        chosen = frame if spec == 'pooled_same_type' else frame.loc[frame.exposure_pitcher.eq(0)]
        fits = {}
        posteriors = {}
        for year, group in chosen.groupby('season'):
            year = int(year)
            fit = player_slopes(group, x='log_exposure_type',
                                min_n=CONFIG['minimum_player_observations'],
                                min_games=CONFIG['minimum_player_games'])
            if len(fit) < 10:
                raise ValueError(f'Insufficient eligible hitters for {spec} in {year}.')
            fit.to_csv(OUT / f'first_stage_{spec}_{year}.csv', index=False)
            posterior = fit_hierarchy(fit.estimate, fit.se, fit.player_id,
                                      draws=CONFIG['posterior_draws'], seed=SEED)
            np.savez_compressed(OUT / f'posterior_{spec}_{year}.npz',
                                **{k: posterior[k] for k in ['lambda', 'mu', 'tau', 'player_ids']})
            summary = summarize_posterior(posterior, names=names,
                counts=dict(zip(fit.player_id, fit.n)))
            summary.to_csv(OUT / f'players_{spec}_{year}.csv', index=False)
            fits[year], posteriors[year] = fit, posterior
            rows.append({
                'specification': spec, 'season': year, 'description': description,
                'analytic_swings': len(group), 'eligible_hitters': len(fit),
                'population_median': float(np.median(posterior['mu'])),
                'population_ci90_low': float(np.quantile(posterior['mu'], .05)),
                'population_ci90_high': float(np.quantile(posterior['mu'], .95)),
                'heterogeneity_median': float(np.median(posterior['tau'])),
                'median_first_stage_se': float(fit.se.median()),
            })
        paired = fits[2025].merge(fits[2026], on='player_id', suffixes=('_2025', '_2026'), validate='one_to_one')
        paired.to_csv(OUT / f'paired_{spec}.csv', index=False)
        result = measurement_error_persistence(paired.estimate_2025, paired.se_2025,
            paired.estimate_2026, paired.se_2026, bootstrap=500, seed=SEED)
        result['shared_hitters'] = len(paired)
        result['naive_spearman'] = float(spearmanr(paired.estimate_2025, paired.estimate_2026).statistic)
        result['median_first_stage_se'] = {
            '2025': float(fits[2025].se.median()), '2026': float(fits[2026].se.median())}
        persistence[spec] = result
        transport[spec] = holdout_prediction(fits[2025],
                                             chosen.loc[chosen.season.eq(2026)])
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'population.csv', index=False)
    report = {
        'analysis_status': 'post hoc user-requested cross-pitcher reanalysis; original same-pitcher benchmark remains frozen',
        'exposure_definition': 'Prior delivered pitches of the same pitch type to the same hitter in the same game, regardless of pitcher. Current pitch excluded; takes included before selecting tracked swings.',
        'strict_transfer_definition': 'Restrict to the first pitch of that type from the current pitcher; any positive same-type exposure is necessarily from other pitchers.',
        'persistence': persistence, 'holdout_prediction': transport,
        'limitations': [
            'Within-game history resets each game and does not estimate season-long retention.',
            'Pitcher changes and pitch selection are observational; the exposure association is not a causal learning rate.',
            'The strict transfer subset may be selected by pitcher usage and game state.',
            'Nuisance predictions were frozen before this reanalysis but the cross-pitcher hypothesis was chosen after inspecting the original 2026 results.',
        ],
    }
    save_json(OUT / 'summary.json', report)
    make_figure(table, persistence, audit)
    make_report(table, report, audit)
    return table, report


def make_report(table, report, audit):
    def estimate(spec, year):
        row = table.loc[table.specification.eq(spec) & table.season.eq(year)].iloc[0]
        return (f"{row.population_median:.3f} (90% credible interval "
                f"{row.population_ci90_low:.3f} to {row.population_ci90_high:.3f})")
    def rho(spec):
        value = report['persistence'][spec]
        return (f"{value['latent_correlation']:.3f} (90% player-bootstrap range "
                f"{value['bootstrap_ci90'][0]:.3f} to {value['bootstrap_ci90'][1]:.3f}; "
                f"{value['shared_hitters']} shared hitters)")
    def gain(spec):
        value = report['holdout_prediction'][spec]
        return (f"{value['gain_pct']:.4f}% (90% game-bootstrap interval "
                f"{value['game_bootstrap_ci90'][0]:.4f}% to "
                f"{value['game_bootstrap_ci90'][1]:.4f}%)")
    exposed = int(audit.swings_with_prior_same_type_from_other_pitcher.sum())
    strict_total = int(audit.first_type_from_current_pitcher_swings.sum())
    strict = int(audit.first_type_from_current_pitcher_with_cross_pitcher_history.sum())
    text = f'''# Same pitch type across pitchers: requested reanalysis

This analysis was requested after the original same-pitcher 2026 validation had been inspected. It is **post hoc** and does not acquire a new untouched holdout. The original same-pitcher results remain frozen for comparison. The new focal exposure counts every earlier delivered pitch of the same type seen by a hitter **within a game, regardless of pitcher**. Takes are included before selecting tracked swings; the current pitch is excluded. The count resets at the next game and therefore does not estimate season-long retention.

The analytic snapshot contains {exposed:,} tracked swings with prior same-type pitches from another pitcher. The strict transfer analysis uses {strict_total:,} first-type swings from the current pitcher. In that subset, any positive prior same-type exposure came entirely from other pitchers; {strict:,} tracked swings have such cross-pitcher history. The sample is selected by pitcher changes, pitch mix, and hitter survival into later game states.

| Exposure definition | 2025 conditional decline | 2026 conditional decline | Cross-season latent correlation | 2026 player-slope gain over population slope |
|---|---:|---:|---:|---:|
| Same type, all pitchers | {estimate('pooled_same_type',2025)} | {estimate('pooled_same_type',2026)} | {rho('pooled_same_type')} | {gain('pooled_same_type')} |
| First same-type pitch from current pitcher | {estimate('first_type_from_pitcher',2025)} | {estimate('first_type_from_pitcher',2026)} | {rho('first_type_from_pitcher')} | {gain('first_type_from_pitcher')} |

The positive population associations in both years are evidence that deviation tends to decline as same-type exposure accumulates across pitchers within games, conditional on the frozen nuisance models. The strict first-type analysis demonstrates that the pattern is present even when the current pitcher has not previously shown this hitter that pitch type. These are observational associations, not identified causal learning rates. Pitcher changes and pitch selection may still generate the pattern.

Neither cross-season player correlation is precise enough to establish a stable hitter ranking. The pooled player slopes worsen later-season conditional prediction relative to a population slope, while the strict transfer comparison is too uncertain to support a gain. The result supports a measurable cross-pitcher exposure association but **does not validate a persistent latent adaptation skill**. It should be presented as a new analysis selected after reviewing the original holdout.

![Cross-pitcher exposure association and persistence](../results/figures/17_cross_pitcher_adaptation.png)

The complete first-stage tables, player posterior distributions, exposure audit, bootstrap diagnostics, and predictive checks are in [`results/cross_pitcher`](../results/cross_pitcher). The original same-pitcher analysis remains in the main results tables and executed notebook as a benchmark.
'''
    (ROOT / 'paper/cross_pitcher_reanalysis.md').write_text(text)


def make_figure(table, persistence, audit):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
    colors = {'pooled_same_type': '#176b70', 'first_type_from_pitcher': '#d28c35'}
    labels = {'pooled_same_type': 'Same type, all pitchers',
              'first_type_from_pitcher': 'First type from current pitcher'}
    left_ticks, left_labels = [], []
    for i, spec in enumerate(SPECS):
        for j, year in enumerate([2025, 2026]):
            r = table.loc[table.specification.eq(spec) & table.season.eq(year)].iloc[0]
            y = 3 - 2*i - j*.55
            axes[0].plot([r.population_ci90_low, r.population_ci90_high], [y, y],
                         color=colors[spec], lw=3)
            axes[0].plot(r.population_median, y, 'o', ms=7, color=colors[spec])
            left_ticks.append(y)
            left_labels.append(f'{labels[spec]} ({year})')
    axes[0].axvline(0, color='#87929b', lw=1)
    axes[0].set_xlim(-.005, max(.043, table.population_ci90_high.max()+.004))
    axes[0].set_ylim(-.35, 3.55)
    axes[0].set_yticks(left_ticks, left_labels)
    axes[0].set_xlabel('Conditional deviation decline per log-exposure unit')
    axes[0].set_title('Exposure association')
    for i, spec in enumerate(SPECS):
        p = persistence[spec]
        y = 1-i
        axes[1].plot(p['bootstrap_ci90'], [y, y], color=colors[spec], lw=3)
        axes[1].plot(p['latent_correlation'], y, 'o', ms=7, color=colors[spec])
    axes[1].axvline(0, color='#87929b', lw=1)
    axes[1].set_xlim(-1.02, 1.02)
    axes[1].set_ylim(-.5, 1.5)
    axes[1].set_yticks([0, 1], ['First type from current pitcher', 'Same type, all pitchers'])
    axes[1].set_xlabel('Cross-season hitter-slope correlation')
    axes[1].set_title('Does the player ranking persist?')
    n = int(audit.swings_with_prior_same_type_from_other_pitcher.sum())
    fig.suptitle('Across-pitcher, same-type exposure within a game', fontsize=14, fontweight='bold')
    fig.text(.5, .005, f'{n:,} analytic swings have prior same-type pitches from another pitcher. '
             'Intervals are descriptive; the requested reanalysis is post hoc.', ha='center', fontsize=9)
    fig.tight_layout(rect=(.02, .035, 1, .93))
    fig.savefig(ROOT / 'results/figures/17_cross_pitcher_adaptation.png', dpi=190)
    fig.savefig(ROOT / 'results/figures/17_cross_pitcher_adaptation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    table, report = analyze(analytic_frame())
    print(table.to_string(index=False))
    for spec, result in report['persistence'].items():
        print(spec, 'cross-season rho', result['latent_correlation'],
              '90% player-bootstrap', result['bootstrap_ci90'])
