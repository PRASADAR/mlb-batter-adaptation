#!/usr/bin/env python3
"""Transparent post hoc model comparison for cross-pitcher adaptation.

No 2026 outcome is used to choose the development-selected candidate. The
largest 2026 prediction gain is reported only as a descriptive search winner.
All tested models and null results are retained.
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
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.expected import player_slopes
from src.models.hierarchical import posterior_mean_rates

CONFIG = json.loads((ROOT / 'config/analysis.json').read_text())
SEED = CONFIG['seed']
OUT = ROOT / 'results/cross_pitcher'
OUTCOMES = ['distortion'] + [f'absz_{x}' for x in CONFIG['swing_components']]
SCOPES = ['all_pitchers', 'first_type_from_pitcher']
COLS = ['season', 'game_date', 'game_pk', 'batter', 'exposure_pitcher',
        'adj_log_exposure_type'] + [f'adj_{x}' for x in OUTCOMES]


def fdr(p, dependence=False):
    p = np.asarray(p, float)
    m = len(p)
    factor = sum(1 / np.arange(1, m + 1)) if dependence else 1.0
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * m * factor / np.arange(1, m + 1))[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(q, 1)
    return result


def read_data():
    paths = sorted((ROOT / 'data/processed').glob('analytic_*.parquet'))
    if not paths: raise FileNotFoundError('Run make models first.')
    data = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    data['late'] = pd.to_datetime(data.game_date).dt.month.ge(7)
    return data


def fit_slopes(frame, outcome):
    return player_slopes(frame, x='log_exposure_type', y=outcome,
        min_n=CONFIG['minimum_player_observations'],
        min_games=CONFIG['minimum_player_games'])


def rank_correlation(a, b):
    paired = a.merge(b, on='player_id', suffixes=('_a', '_b'), validate='one_to_one')
    if len(paired) < 10: return len(paired), np.nan, paired
    return len(paired), float(spearmanr(paired.estimate_a, paired.estimate_b).statistic), paired


def positive_permutation_p(paired, observed, rng, replicates=2000):
    x = rankdata(paired.estimate_a.to_numpy())
    y = rankdata(paired.estimate_b.to_numpy())
    x = (x - x.mean()) / np.linalg.norm(x - x.mean())
    y = (y - y.mean()) / np.linalg.norm(y - y.mean())
    hits = 0
    for _ in range(replicates):
        hits += (x @ y[rng.permutation(len(y))]) >= observed
    return (hits + 1) / (replicates + 1)


def gain_and_interval(development, validation, outcome):
    exact_player_means, population_mean = posterior_mean_rates(development.estimate, development.se)
    player_mean = dict(zip(development.player_id, exact_player_means))
    h = validation.loc[validation.batter.isin(player_mean)].copy()
    x = h.adj_log_exposure_type - h.groupby('batter').adj_log_exposure_type.transform('mean')
    y = h[f'adj_{outcome}'] - h.groupby('batter')[f'adj_{outcome}'].transform('mean')
    pop = -population_mean * x.to_numpy()
    individual = -h.batter.map(player_mean).to_numpy() * x.to_numpy()
    loss = pd.DataFrame({'game': h.game_pk, 'population': (y - pop) ** 2,
                         'individual': (y - individual) ** 2}).groupby('game').sum()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(loss), (1000, len(loss)))
    baseline = loss.population.to_numpy()[picks].sum(axis=1)
    candidate = loss.individual.to_numpy()[picks].sum(axis=1)
    bootstrap = 100 * (baseline - candidate) / baseline
    gain = 100 * (loss.population.sum() - loss.individual.sum()) / loss.population.sum()
    return float(gain), np.quantile(bootstrap, [.05, .95]), len(h)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_data()
    rng = np.random.default_rng(SEED)
    rows = []
    for scope in SCOPES:
        frame = data if scope == 'all_pitchers' else data.loc[data.exposure_pitcher.eq(0)]
        for outcome in OUTCOMES:
            full = {int(year): fit_slopes(g, outcome) for year, g in frame.groupby('season')}
            halves = {(int(year), bool(late)): fit_slopes(g, outcome)
                      for (year, late), g in frame.groupby(['season', 'late'])}
            n25, rho25, _ = rank_correlation(halves[2025, False], halves[2025, True])
            n26, rho26, _ = rank_correlation(halves[2026, False], halves[2026, True])
            shared, rho, paired = rank_correlation(full[2025], full[2026])
            p = positive_permutation_p(paired, rho, rng)
            gain, ci, n_validation = gain_and_interval(full[2025],
                frame.loc[frame.season.eq(2026)], outcome)
            rows.append({
                'scope': scope, 'outcome': outcome,
                'eligible_2025': len(full[2025]), 'eligible_2026': len(full[2026]),
                'shared_cross_season': shared, 'rho_cross_season': rho,
                'one_sided_permutation_p': p,
                'split_2025_shared': n25, 'split_2025_rho': rho25,
                'split_2026_shared': n26, 'split_2026_rho': rho26,
                'validation_swings': n_validation, 'gain_2026_pct': gain,
                'gain_ci90_low': ci[0], 'gain_ci90_high': ci[1],
            })
            print(scope, outcome, f'cross-season rho={rho:.3f}',
                  f'2026 prediction gain={gain:.5f}%', flush=True)
    table = pd.DataFrame(rows)
    table['bh_q_positive'] = fdr(table.one_sided_permutation_p)
    table['by_q_positive'] = fdr(table.one_sided_permutation_p, dependence=True)
    table.to_csv(OUT / 'model_comparison.csv', index=False)
    discovery = table.sort_values(['split_2025_rho', 'shared_cross_season'],
                                  ascending=False).iloc[0]
    descriptive = table.sort_values(['gain_2026_pct', 'rho_cross_season'],
                                    ascending=False).iloc[0]
    result = {
        'status': 'post hoc model search; no new untouched future-season test',
        'candidate_family': 'Two scopes by six prespecified available deviation outcomes; same-type exposure across pitchers in one game for every candidate.',
        'development_selection_rule': 'Maximum March-June versus July-September 2025 unpooled Spearman correlation, then largest cross-season shared sample.',
        'development_selected': {'scope': discovery.scope, 'outcome': discovery.outcome},
        'descriptive_2026_best': {'scope': descriptive.scope, 'outcome': descriptive.outcome},
        'descriptive_selection_warning': 'The largest 2026 gain was selected on 2026 data; its pointwise interval is not a post-selection confirmation interval.',
        'multiplicity': 'One-sided cross-season rank permutation tests, 2000 permutations per candidate; BH and BY across all 12 candidates.',
        'validated_persistent_skill': False,
    }
    (OUT / 'model_search_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    make_figure(table)
    make_report(table, result)
    return table


def make_figure(table):
    view = table.sort_values('gain_2026_pct')
    names = [f"{r.scope.replace('first_type_from_pitcher','strict').replace('all_pitchers','pooled')}: "
             f"{r.outcome.replace('absz_','').replace('_',' ')}" for r in view.itertuples()]
    y = np.arange(len(view))
    colors = ['#d28c35' if scope == 'first_type_from_pitcher' else '#176b70'
              for scope in view.scope]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    axes[0].axvline(0, color='#9ba7ae', lw=1)
    axes[0].barh(y, view.rho_cross_season, color=colors, alpha=.85)
    axes[0].set_xlabel('Cross-season slope rank correlation')
    axes[0].set_yticks(y, names)
    axes[0].set_title('Hitter ordering')
    axes[1].axvline(0, color='#9ba7ae', lw=1)
    for i, row in enumerate(view.itertuples()):
        axes[1].plot([row.gain_ci90_low, row.gain_ci90_high], [i, i], color=colors[i], lw=2)
        axes[1].plot(row.gain_2026_pct, i, 'o', color=colors[i])
    axes[1].set_xlabel('2026 conditional MSE gain over population rate (%)')
    axes[1].set_title('Later-season transport')
    fig.suptitle('Cross-pitcher adaptation model search', fontsize=14, fontweight='bold')
    fig.text(.5, .015, 'All candidate models and negative results shown. Intervals are pointwise; search is post hoc.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(.03, .035, 1, .94))
    fig.savefig(ROOT / 'results/figures/18_cross_pitcher_model_search.png', dpi=180)
    fig.savefig(ROOT / 'results/figures/18_cross_pitcher_model_search.pdf')
    plt.close(fig)


def make_report(table, summary):
    selected = table.loc[table.scope.eq(summary['development_selected']['scope']) &
                         table.outcome.eq(summary['development_selected']['outcome'])].iloc[0]
    best = table.loc[table.scope.eq(summary['descriptive_2026_best']['scope']) &
                     table.outcome.eq(summary['descriptive_2026_best']['outcome'])].iloc[0]
    top = table.sort_values('gain_2026_pct', ascending=False)
    lines = ['# Across-pitcher model comparison', '',
        'This search compares the same within-game, across-pitcher pitch-type exposure using multivariate swing deviation and each available standardized swing component. It crosses two sample scopes with six outcomes. The first scope uses all tracked swings. The strict scope uses only the first pitch of that type from the current pitcher, where a positive exposure count came from other pitchers.', '',
        'The search was conducted after the original 2026 results were inspected. The model selected by 2025 split-half correlation was chosen without using the 2026 search scores, but the candidate family and criterion were still set post hoc. The largest 2026 predictive gain is a descriptive winner selected on the same data used to score it. Neither is an untouched independent validation.', '',
        '| Scope | Swing outcome | 2025 split-half rank rho | 2026 split-half rank rho | 2025-2026 rank rho | Positive-rank BY q | 2026 MSE gain, % | 90% game-bootstrap interval, % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in top.itertuples():
        lines.append(f'| {r.scope} | {r.outcome} | {r.split_2025_rho:.3f} | '
                     f'{r.split_2026_rho:.3f} | {r.rho_cross_season:.3f} | '
                     f'{r.by_q_positive:.3f} | {r.gain_2026_pct:.5f} | '
                     f'{r.gain_ci90_low:.5f} to {r.gain_ci90_high:.5f} |')
    lines += ['', f"The development-selected model is **{selected.scope} / {selected.outcome}**. "
        f"Its full-year cross-season rank correlation is {selected.rho_cross_season:.3f}; "
        f"its 2026 prediction gain is {selected.gain_2026_pct:.5f}%.", '',
        f"The largest descriptive 2026 gain is **{best.scope} / {best.outcome}** at "
        f"{best.gain_2026_pct:.5f}%, with a pointwise game-bootstrap interval "
        f"{best.gain_ci90_low:.5f}% to {best.gain_ci90_high:.5f}%. Its 2026 "
        f"split-half rank correlation is {best.split_2026_rho:.3f}. This selected "
        'gain cannot demonstrate persistence by itself.', '',
        f"Across the search, {(table.by_q_positive <= .05).sum()} positive cross-season "
        'rank correlations pass the dependence-robust BY adjustment. Candidate curves '
        'are model-implied conditional responses, not verified individual learning '
        'trajectories. The present data do not establish a persistent player-level '
        'adaptation skill.', '',
        '![All model comparisons](../results/figures/18_cross_pitcher_model_search.png)', '']
    (ROOT / 'paper/cross_pitcher_model_search.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    run()
