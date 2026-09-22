"""Publication figures computed only from frozen, generated research artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deja-swing-matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

NAVY = '#17324D'
TEAL = '#008E8A'
CORAL = '#D85C45'
GOLD = '#C28F2C'
MUTED = '#607387'
PALE = '#E8F2F0'
BG = '#FAFCFD'
LABELS = {
    'attack_angle': 'Attack angle', 'attack_direction': 'Attack direction',
    'swing_path_tilt': 'Swing-path tilt', 'bat_speed': 'Bat speed',
    'swing_length': 'Swing length', 'miss_distance': 'Miss distance',
    'intercept_ball_minus_batter_pos_x_inches': 'Horizontal intercept',
    'intercept_ball_minus_batter_pos_y_inches': 'Depth intercept',
}


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.labelcolor': NAVY, 'text.color': NAVY,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#B8C6D0',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': BG, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'grid.alpha': .22, 'grid.color': '#A2B3C0', 'axes.axisbelow': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _table(root, name):
    return pd.read_csv(root / 'results/tables' / name)


def _json(root, name):
    return json.loads((root / name).read_text())


def _number(value, specification='.2f'):
    """Make weakly identified/null JSON diagnostic values displayable."""
    try:
        return format(float(value), specification) if np.isfinite(float(value)) else 'unavailable'
    except (TypeError, ValueError):
        return 'unavailable'


def _posterior(root, season=2025):
    with np.load(root / f'results/posterior/adaptation_{season}.npz') as p:
        return dict(p)


def _save(fig, out, name, note=None):
    if note:
        fig.text(.025, .012, note, fontsize=8, color=MUTED, va='bottom')
    fig.savefig(out / f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    return out / f'{name}.png'


def _selected(root, season=2025, n=8):
    t = _table(root, 'batter_adaptation_posteriors.csv')
    t = t[t.season.eq(season)].sort_values('posterior_mean').reset_index(drop=True)
    positions = np.unique(np.linspace(0, len(t) - 1, min(n, len(t))).round().astype(int))
    return t.iloc[positions].copy()


def plot_timeline(root, out):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(-.1, 4.1); ax.set_ylim(-.2, 1.7); ax.axis('off')
    steps = [
        ('2023 H2–2024 H1', 'FIT EXPECTED SWINGS', 'Current pitch + hitter context\nCompare model with pitch history', NAVY),
        ('2024 H2', 'CALIBRATE DISTORTION', 'Freeze residual covariance\nand outcome standardization', TEAL),
        ('2025', 'ESTIMATE ASSOCIATIONS', 'Game-group cross-fitting\nFit player slope distributions', CORAL),
        ('2026', 'TEMPORAL HOLDOUT', 'Frozen nuisance models\nTest signal transport and persistence', GOLD),
    ]
    for i, (date, heading, body, color) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + .01, .12), .91, .94, boxstyle='round,pad=.025,rounding_size=.045', facecolor=color, edgecolor='none'))
        ax.text(i + .46, .84, date, ha='center', color='white', fontsize=12, fontweight='bold')
        ax.text(i + .46, .60, heading, ha='center', color='white', fontsize=8.5, fontweight='bold')
        ax.text(i + .46, .35, body, ha='center', color='white', fontsize=8.1, linespacing=1.8)
        if i < 3:
            ax.annotate('', xy=(i + 1.01, 1.25), xytext=(i + .40, 1.25), arrowprops={'arrowstyle': '->', 'color': MUTED, 'lw': 1.8})
    ax.set_title('Training, calibration and temporal validation', loc='left', pad=8, fontsize=17)
    return _save(fig, out, '01_research_timeline', '2025 nuisance folds hold out games. The original primary model uses 2026 for temporal validation. Subsequent specifications are exploratory.')


def plot_coverage(root, out):
    t = _table(root, 'data_coverage_season.csv')
    fields = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), sharey=True)
    cmap = colors.LinearSegmentedColormap.from_list('coverage', ['#F9EEEE', '#D5EAE5', TEAL])
    for ax, metric, title in zip(axes, ['pct_whiffs', 'pct_contact'], ['Whiffs', 'Contact swings']):
        pivot = t.pivot(index='field', columns='group', values=metric).reindex(fields)
        a = pivot.to_numpy()
        im = ax.imshow(a, vmin=0, vmax=100, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)), pivot.columns.astype(str)); ax.set_yticks(range(len(fields)), [LABELS[x] for x in fields])
        ax.set_title(title, loc='left', pad=12)
        for (i, j), value in np.ndenumerate(a):
            ax.text(j, i, 'NA' if not np.isfinite(value) else f'{value:.1f}%', ha='center', va='center', color='white' if value > 78 else NAVY, fontsize=9)
        ax.tick_params(length=0)
    fig.suptitle('Coverage of swing-tracking measurements', x=.02, ha='left', fontsize=17, fontweight='bold')
    fig.subplots_adjust(top=.86, bottom=.13, left=.20, right=.85, wspace=.15)
    cax = fig.add_axes([.885, .245, .014, .49])
    fig.colorbar(im, cax=cax, label='Populated records (%)')
    return _save(fig, out, '02_tracking_coverage', 'Denominators include recorded swings without tracking. Tracking availability alone does not establish that a metric measures error.')


def plot_sequence_gain(root, out):
    t = _table(root, 'sequence_distortion.csv')
    t = t[t.season.ge(2025)].copy()
    components = list(t.component.drop_duplicates())
    fig, ax = plt.subplots(figsize=(10, 4.9))
    for season, offset, color in [(2025, -.12, TEAL), (2026, .12, CORAL)]:
        g = t[t.season.eq(season)].set_index('component').reindex(components)
        mean = g.mse_gain_pct.to_numpy(); lo = g.gain_ci90_low.to_numpy(); hi = g.gain_ci90_high.to_numpy()
        ax.errorbar(mean, np.arange(len(components)) + offset, xerr=[np.maximum(0, mean-lo), np.maximum(0, hi-mean)], fmt='o', color=color, capsize=3, lw=2, label=f'{season}' + (' holdout' if season == 2026 else ' development'))
    ax.axvline(0, color=MUTED, lw=1); ax.grid(axis='x')
    ax.set_yticks(range(len(components)), [LABELS.get(x, x) for x in components])
    ax.set_xlabel('Reduction in squared prediction error when pitch history is added (%)')
    ax.set_title('Incremental prediction from pitch history', loc='left', pad=15, fontsize=16)
    ax.legend(frameon=False, loc='best'); fig.tight_layout(rect=(0, .075, 1, 1))
    return _save(fig, out, '03_sequence_prediction_gain', 'Paired game-bootstrap 90% intervals. Negative values mean the history model predicts worse; predictive gain is not a causal sequence effect.')


def plot_exposure(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    selection = _table(root, 'selection.csv')
    for year, color in [(2025, TEAL), (2026, CORAL)]:
        g = selection[selection.season.eq(year) & selection.exposure_pitcher.le(8) & selection.n.ge(50)]
        axes[0].plot(g.exposure_pitcher, g.distortion, 'o-', color=color, label=str(year), lw=2)
        p = _posterior(root, year); exposures = np.linspace(0, 8, 100)
        curve = -p['mu'][:, None] * np.log1p(exposures)[None, :]
        lo, mid, hi = np.quantile(curve, [.05, .5, .95], axis=0)
        axes[1].plot(exposures, mid, color=color, lw=2, label=str(year))
        axes[1].fill_between(exposures, lo, hi, color=color, alpha=.15)
    axes[0].set_title('Observed association', loc='left'); axes[1].set_title('Conditional population slope', loc='left')
    axes[0].set_ylabel('Mean unexpected swing deviation\n(calibration SD)')
    axes[1].set_ylabel('Model-implied change in deviation\n(relative to zero prior exposures)')
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=.8); ax.grid(axis='y'); ax.set_xlabel('Prior pitches of this type from this pitcher, in game'); ax.legend(frameon=False)
    fig.suptitle('Exposure curves: the raw association and the adjusted model', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, .11, 1, .95))
    return _save(fig, out, '04_exposure_response', 'Right: −μ log(1 + exposure), anchored at zero, with 90% credible bands. Observational associations; this is not a percentage of error corrected.')


def plot_player_uncertainty(root, out):
    t = _selected(root, 2025, 9).sort_values('posterior_mean', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={'width_ratios': [1.25, 1]})
    y = np.arange(len(t))
    for ax in axes:
        ax.set_yticks(y); ax.invert_yaxis(); ax.grid(axis='x')
    axes[0].set_yticklabels(t.player_name)
    axes[1].set_yticklabels([])
    for _, row in t.reset_index(drop=True).iterrows():
        i = list(t.player_id).index(row.player_id)
        axes[0].plot([row.cri90_low, row.cri90_high], [i, i], color=TEAL, lw=2)
        axes[0].plot([row.cri50_low, row.cri50_high], [i, i], color=TEAL, lw=6, solid_capstyle='round')
        axes[0].plot(row.posterior_median, i, 'o', color=NAVY, ms=4)
        axes[1].plot([row.rank_cri90_low, row.rank_cri90_high], [i, i], color=CORAL, lw=2)
        axes[1].plot([row.rank_cri50_low, row.rank_cri50_high], [i, i], color=CORAL, lw=6, solid_capstyle='round')
        axes[1].plot(row.rank_median, i, 'o', color=NAVY, ms=4)
    axes[0].axvline(0, color=MUTED, lw=1)
    axes[0].set_title('Conditional correction association', loc='left'); axes[1].set_title('Rank among included hitters', loc='left')
    axes[0].set_xlabel('λ · deviation SD per log(1 + exposure)'); axes[1].set_xlabel('Posterior rank (1 = largest λ)')
    axes[1].set_xlim(0, int(t.rank_population_size.iloc[0])+1)
    fig.suptitle('Player estimates and rank uncertainty', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, .08, 1, .95))
    return _save(fig, out, '05_player_and_rank_uncertainty', 'Illustrative 2025 conditional associations, not established persistent learning skill. Players span posterior means; thick: 50% interval, thin: 90%.')


def plot_pairwise(root, out):
    t = _selected(root, 2025, 7).sort_values('posterior_mean', ascending=False)
    p = _posterior(root, 2025); indices = [int(np.flatnonzero(p['player_ids'] == i)[0]) for i in t.player_id]
    lam = p['lambda'][:, indices]
    matrix = (lam[:, :, None] > lam[:, None, :]).mean(axis=0)
    np.fill_diagonal(matrix, .5)
    fig, ax = plt.subplots(figsize=(9.1, 7.1))
    cmap = colors.LinearSegmentedColormap.from_list('comparison', [CORAL, '#F6F9FA', TEAL])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(t)), t.player_name, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(len(t)), t.player_name, fontsize=9)
    for (i, j), value in np.ndenumerate(matrix):
        label = 'self' if i == j else '<1%' if value < .005 else '>99%' if value >= .995 else f'{value:.0%}'
        ax.text(j, i, label, ha='center', va='center', color='white' if value < .12 or value > .88 else NAVY)
    ax.set_title('Pairwise posterior probabilities: P(row λ > column λ)', loc='left', pad=18, fontsize=15)
    fig.colorbar(im, ax=ax, shrink=.75, label='Posterior probability')
    fig.tight_layout(rect=(0, .065, 1, 1))
    return _save(fig, out, '06_pairwise_probability', 'Conditional 2025 comparisons among illustrative hitters. Probabilities depend on the model and estimated first-stage uncertainty.')


def plot_information(root, out):
    t = _table(root, 'batter_adaptation_posteriors.csv'); t = t[t.season.eq(2025)]
    first = _table(root, 'first_stage_2025.csv')
    t = t.merge(first[['player_id', 'estimate', 'se']], on='player_id')
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    c = np.log10(t.n_exposures.to_numpy())
    points = axes[0].scatter(t.estimate, t.posterior_mean, c=c, cmap='viridis', s=22, alpha=.75, edgecolor='none')
    limit = max(abs(t.estimate).max(), abs(t.posterior_mean).max()) * 1.05
    axes[0].plot([-limit, limit], [-limit, limit], '--', color=MUTED, lw=1)
    axes[0].axhline(_posterior(root)['mu'].mean(), color=CORAL, lw=1, label='Population posterior mean')
    axes[0].set_xscale('symlog',linthresh=.1); axes[0].set_yscale('symlog',linthresh=.1)
    axes[0].set_xlabel('Unpooled conditional slope estimate'); axes[0].set_ylabel('Partially pooled posterior mean')
    axes[0].set_title('Noisy estimates move toward the population', loc='left', fontsize=11)
    axes[1].scatter(t.information, t.cri90_high - t.cri90_low, c=c, cmap='viridis', s=22, alpha=.75, edgecolor='none')
    axes[1].set_xscale('log'); axes[1].set_xlabel('Residual exposure information (Σ centered x²)')
    axes[1].set_ylabel('Width of the 90% credible interval'); axes[1].set_title('Unequal information yields unequal uncertainty', loc='left', fontsize=11)
    for ax in axes: ax.grid()
    fig.subplots_adjust(left=.08, bottom=.20, right=.83, top=.85, wspace=.35)
    cax = fig.add_axes([.88, .29, .014, .47])
    fig.colorbar(points, cax=cax, label='log₁₀ tracked analytic swings')
    fig.suptitle('Partial pooling and unequal information', x=.02, ha='left', fontsize=16, fontweight='bold')
    return _save(fig, out, '07_shrinkage_and_information', 'Left axes: symmetric-log scale, linear within ±0.1; all hitters retained. Information depends on exposure variation and sample size.')


def plot_recovery(root, out):
    with np.load(root / 'results/posterior/simulation_example.npz') as f: ex = dict(f)
    cal = _table(root, 'pairwise_calibration.csv')
    summary = _json(root, 'results/tables/simulation_summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    mean = ex['posterior'].mean(axis=0); low, high = np.quantile(ex['posterior'], [.05, .95], axis=0)
    axes[0].errorbar(ex['truth'], mean, yerr=[np.maximum(0, mean-low), np.maximum(0, high-mean)], fmt='o', color=TEAL, ecolor='#ACD8D3', ms=4, lw=1)
    limits = [min(ex['truth'].min(), low.min())-.02, max(ex['truth'].max(), high.max())+.02]
    axes[0].plot(limits, limits, '--', color=MUTED)
    axes[0].set_xlabel('Known synthetic correction rate'); axes[0].set_ylabel('Posterior mean and 90% interval')
    axes[0].set_title('Parameter recovery in one replicate', loc='left')
    axes[1].plot([0, 1], [0, 1], '--', color=MUTED)
    axes[1].errorbar(cal.mean_probability, cal.empirical_frequency, yerr=[np.maximum(0, cal.empirical_frequency-cal.frequency_ci90_low), np.maximum(0, cal.frequency_ci90_high-cal.empirical_frequency)], fmt='o-', color=CORAL, capsize=3)
    axes[1].set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xlabel='Posterior P(player A > player B)', ylabel='Frequency A truly exceeds B')
    axes[1].set_title(f'{summary["replicates_per_scenario"]} experiments: comparison calibration', loc='left')
    for ax in axes: ax.grid()
    fig.suptitle('Parameter recovery and calibration in simulation', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, .13, 1, .94))
    return _save(fig, out, '08_simulation_recovery_calibration', 'Simulation only. Calibration bars bootstrap independent experiments; pairs within an experiment are dependent. This is not empirical MLB calibration.')


def plot_null(root, out):
    t = _table(root, 'simulation_recovery.csv')
    z = t[t.scenario.eq('zero_learning')]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    order = np.argsort(z.mu_mean.to_numpy()); q = z.iloc[order].reset_index(drop=True)
    for i, r in q.iterrows():
        color = CORAL if r.mu_interval_excludes_zero else TEAL
        axes[0].plot([r.mu90_low, r.mu90_high], [i, i], color=color, alpha=.8, lw=1.5)
    axes[0].scatter(q.mu_mean, np.arange(len(q)), color=NAVY, s=6)
    axes[0].axvline(0, color=NAVY, lw=1); axes[0].set_xlabel('Population μ posterior and 90% interval'); axes[0].set_ylabel('Independent simulation (sorted for display)')
    axes[0].set_title('True correction rate = 0', loc='left')
    categories = ['Heterogeneous', 'Zero learning']; x = np.arange(2)
    for offset, col, color, name in [(-.16, 'rmse', TEAL, 'Partially pooled'), (.16, 'unpooled_rmse', CORAL, 'Unpooled')]:
        values = [t.loc[t.scenario.eq(s), col].mean() for s in ['heterogeneous', 'zero_learning']]
        axes[1].bar(x + offset, values, .30, color=color, label=name)
    axes[1].set_xticks(x, categories); axes[1].set_ylabel('Mean root mean squared parameter error')
    axes[1].set_title('Pooling improves parameter estimation', loc='left'); axes[1].legend(frameon=False); axes[1].grid(axis='y')
    fig.suptitle('Parameter recovery under zero learning', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, .10, 1, .94))
    return _save(fig, out, '09_zero_learning_guardrail', 'Simulated trajectories are sampled before response noise; no selection on extreme first-swing errors. Coral intervals exclude the true zero.')


def plot_robustness(root, out):
    t = _table(root, 'robustness.csv')
    spec_labels = {
        'exposure_pitcher': 'Same pitcher + pitch type (primary)', 'exposure_pa': 'Same pitch type within PA',
        'exposure_type': 'Same pitch type within game', 'kernel_exposure': 'Similar pitch shape + memory',
        'exposure_sequence': 'Same pitch-type transition', 'shuffled_within_batter_game': 'Shuffled history within hitter–game',
        'future_exposure': 'Future pitches (negative control)', 'irrelevant_exposure': 'Other pitch types (negative control)',
    }
    fig, ax = plt.subplots(figsize=(11.8, 6.1))
    for year, offset, color in [(2025, -.12, TEAL), (2026, .12, CORAL)]:
        g = t[t.season.eq(year) & t.specification.isin(spec_labels)].set_index('specification').reindex(spec_labels)
        ax.errorbar(g.mu_median, np.arange(len(spec_labels)) + offset, xerr=[np.maximum(0,g.mu_median-g.mu_low),np.maximum(0,g.mu_high-g.mu_median)], fmt='o', color=color, capsize=3, label=str(year))
    ax.axvline(0, color=MUTED, lw=1); ax.axhspan(4.5, 7.6, color='#F8EEE9', zorder=0)
    ax.set_yticks(range(len(spec_labels)), list(spec_labels.values())); ax.invert_yaxis(); ax.grid(axis='x')
    ax.set_xlabel('Population conditional correction association μ (90% credible interval)')
    ax.set_title('Exposure definitions and control analyses', loc='left', pad=16, fontsize=16)
    ax.legend(frameon=False); fig.tight_layout(rect=(0,.10,1,1))
    return _save(fig, out, '10_robustness_negative_controls', 'Future and unrelated exposure share game structure and selection. Similar effects in negative controls weaken any adaptation interpretation.')


def plot_persistence(root, out):
    t = _table(root, 'persistence_pairs.csv'); p = _json(root, 'results/tables/persistence.json')
    h = _json(root, 'results/tables/holdout_prediction.json')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), gridspec_kw={'width_ratios':[1.45,1]})
    axes[0].errorbar(t.estimate_a, t.estimate_b, xerr=1.645*t.se_a, yerr=1.645*t.se_b, fmt='none', ecolor='#DCE5EB', alpha=.55, lw=.5, zorder=1)
    axes[0].scatter(t.estimate_a,t.estimate_b,color=TEAL,s=15,alpha=.65,zorder=2)
    axes[0].set_xscale('symlog',linthresh=.1); axes[0].set_yscale('symlog',linthresh=.1)
    axes[0].axvline(0,color=MUTED,lw=.8);axes[0].axhline(0,color=MUTED,lw=.8);axes[0].grid()
    axes[0].set(xlabel='2025 unpooled conditional λ',ylabel='2026 unpooled conditional λ',title='Player slopes in successive seasons')
    ci = p.get('bootstrap_ci90') or [None, None]; rho = p.get('latent_correlation')
    holdout_ci = h.get('ci90') or [None, None]
    axes[1].axis('off')
    content = [
        ('MATCHED HITTERS', f'{p["n_players"]:,}'),
        ('LATENT CORRELATION', f'{_number(rho, ".3f")}  ·  90% bootstrap range\n[{_number(ci[0], ".3f")}, {_number(ci[1], ".3f")}]'),
        ('RAW PEARSON CORRELATION', _number(p.get('raw_correlation'), '.3f')),
        ('2026 MSE CHANGE FROM INDIVIDUAL SLOPES', f'{_number(-h["mse_improvement_pct"], "+.3f")}%\n90% CI [{_number(-holdout_ci[1], "+.3f")}, {_number(-holdout_ci[0], "+.3f")}]%'),
    ]
    for i,(label,value) in enumerate(content):
        yy=[.95,.74,.43,.24][i]
        axes[1].text(.03,yy,label,color=MUTED,fontsize=9,fontweight='bold',transform=axes[1].transAxes)
        axes[1].text(.03,yy-.06,value,color=NAVY,fontsize=12,transform=axes[1].transAxes,va='top',linespacing=1.4)
    if p.get('boundary_fit') or p.get('weak_identification'):
        axes[1].text(.03,-.035,'Correlation is weakly identified.',color=CORAL,fontsize=9,transform=axes[1].transAxes,wrap=True)
    fig.suptitle('Cross-season stability of conditional exposure slopes',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,.13,1,.94))
    return _save(fig,out,'11_temporal_persistence','Symmetric-log axes (linear within ±0.1). Bootstrap quantiles are descriptive near correlation boundaries.\nPositive MSE change means worse prediction. The test measures centered slope transport.')


def plot_outcomes_ppc(root,out):
    outcome=_table(root,'outcome_link.csv'); ppc=_table(root,'posterior_predictive.csv')
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    for year,color in [(2025,TEAL),(2026,CORAL)]:
        g=outcome[outcome.season.eq(year)].sort_values('deviation')
        axes[0].plot(g.deviation,100*g.whiff_rate,'o-',color=color,lw=2,label=str(year))
    axes[0].set(xlabel='Unexpected swing deviation (decile mean)',ylabel='Whiff rate (%)',title='Swing deviation and observed whiff rates')
    axes[0].legend(frameon=False);axes[0].grid()
    g=ppc[ppc.season.eq(2025)].reset_index(drop=True)
    for i,r in g.iterrows():
        axes[1].plot([r.rep_low,r.rep_high],[i,i],color=TEAL,lw=7,solid_capstyle='round',alpha=.3)
        axes[1].plot(r.rep_median,i,'o',color=TEAL)
        axes[1].plot(r.observed,i,'D',color=CORAL,ms=6)
    axes[1].set_yticks(range(len(g)),[x.replace('_',' ') for x in g.statistic]);axes[1].set_xlabel('Distributional statistic of first-stage λ estimates')
    axes[1].set_title('Second-stage posterior predictive check');axes[1].grid(axis='x')
    axes[1].plot([],[],'o',color=TEAL,label='Replicated median + 90% band');axes[1].plot([],[],'D',color=CORAL,label='Observed');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Outcome associations and posterior predictive checks',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,.12,1,.94))
    return _save(fig,out,'12_outcome_link_and_predictive_check','Deviation is not proven mechanical error. PPC checks the slope measurement model, not a generative model of raw swings, whiffs, or miss distance.')


def plot_posteriors(root,out):
    t=_selected(root,2025,7).sort_values('posterior_mean')
    p=_posterior(root);indices=[int(np.flatnonzero(p['player_ids']==i)[0]) for i in t.player_id]
    values=p['lambda'][:,indices]
    lo,hi=np.quantile(values,[.001,.999]);xx=np.linspace(lo,hi,500)
    fig,ax=plt.subplots(figsize=(11,6))
    for i,(name,idx) in enumerate(zip(t.player_name,indices)):
        yy=gaussian_kde(p['lambda'][:,idx])(xx);yy=yy/yy.max()*.8
        color=plt.get_cmap('viridis')(.15+.65*i/max(1,len(t)-1))
        ax.fill_between(xx,i,i+yy,color=color,alpha=.6,lw=0)
        ax.plot(xx,i+yy,color=color,lw=1.4)
    ax.axvline(0,color=MUTED,lw=1,linestyle='--');ax.set_yticks(np.arange(len(t))+.12,t.player_name)
    ax.set_xlabel('λ · deviation SD per log(1 + exposure)');ax.set_title('Posterior distributions across selected hitters',loc='left',fontsize=17,pad=16)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x');fig.tight_layout(rect=(0,.09,1,1))
    return _save(fig,out,'13_player_posterior_distributions','Conditional 2025 associations; hitters span posterior means for illustration. Each density is scaled to its own peak; displayed areas are not comparable.')


def plot_temporal_pairwise(root, out):
    """Temporal agreement of two uncertain posteriors, never a truth calibration.

    A reproducible sample of at most 5,000 unordered pairs is oriented by player
    identifier, without selecting on either season's performance. Probability
    calculations use up to 1,000 independent marginal posterior draws per year.
    The display has no sampling-uncertainty bars because sampled pairs share
    hitters and the later-season posterior is not observed ground truth.
    """
    p25, p26 = _posterior(root, 2025), _posterior(root, 2026)
    ids = np.intersect1d(p25['player_ids'], p26['player_ids'])
    if len(ids) < 2:
        raise ValueError('Temporal pairwise comparison requires at least two shared hitters')
    seed = _json(root, 'config/analysis.json')['seed']
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(len(ids), 1)
    keep = rng.choice(len(a), min(5000, len(a)), replace=False)
    a, b = a[keep], b[keep]
    probabilities = []
    for posterior in (p25, p26):
        indices = [int(np.flatnonzero(posterior['player_ids'] == player)[0]) for player in ids]
        draw_ids = rng.choice(len(posterior['lambda']), min(1000, len(posterior['lambda'])), replace=False)
        samples = posterior['lambda'][draw_ids][:, indices]
        probabilities.append((samples[:, a] > samples[:, b]).mean(axis=0))
    pair = pd.DataFrame({'player_a': ids[a], 'player_b': ids[b], 'p2025': probabilities[0], 'p2026': probabilities[1]})
    pair['probability_same_order'] = pair.p2025 * pair.p2026 + (1-pair.p2025) * (1-pair.p2026)
    pair['bin'] = np.minimum((10 * pair.p2025).astype(int), 9)
    summary = pair.groupby('bin').agg(n_pairs=('p2025','size'), mean_probability_2025=('p2025','mean'),
        mean_probability_2026=('p2026','mean'), mean_probability_same_order=('probability_same_order','mean')).reset_index()
    summary['bin_low'] = summary['bin']/10
    summary['bin_high'] = (summary['bin']+1)/10
    summary['shared_players'] = len(ids)
    summary['total_pairs_sampled'] = len(pair)
    summary['interpretation'] = 'Conditional annual associations; same-order probability assumes independent annual fits; later probability is not observed truth or calibration'
    summary.to_csv(root/'results/tables/temporal_pairwise_agreement.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
    axes[0].plot([0,1],[0,1],'--',color='#ADBBC5',lw=1,label='Identical posterior separation')
    axes[0].axhline(.5,color=MUTED,lw=1,label='No later-season separation')
    axes[0].plot(summary.mean_probability_2025,summary.mean_probability_2026,'o-',color=TEAL,lw=2,ms=7)
    axes[0].set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean 2025 posterior P(λA > λB), binned',
                ylabel='Mean 2026 posterior P(λA > λB)',title='Cross-season comparison probabilities')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left');axes[0].grid()
    axes[1].bar(summary.mean_probability_2025,summary.n_pairs,width=.065,color=CORAL,alpha=.8)
    axes[1].set(xlim=(-.02,1.02),xlabel='Mean 2025 posterior P(λA > λB), binned',ylabel='Number of sampled player pairs',title='Number of comparisons in each probability bin')
    axes[1].grid(axis='y')
    fig.suptitle('Temporal agreement of uncertain comparisons',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,.13,1,.94))
    return _save(fig,out,'14_temporal_pairwise_agreement',
        f'Conditional λ comparisons: {len(pair):,} pairs among {len(ids):,} shared hitters. Later posterior probability is not truth; this is not empirical calibration.')


def plot_outcome_sensitivity(root, out):
    """Outcome validity, a selected miss-distance endpoint, and selection audit."""
    whiff = _table(root, 'whiff_validation.csv')
    miss = _table(root, 'miss_distance_sensitivity.csv').sort_values('season')
    selection = _table(root, 'tracking_selection.csv')
    inclusion_column = 'included_pct' if 'included_pct' in selection else 'inclusion_pct'
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.1), gridspec_kw={'width_ratios': [.90, 1.05, 1.15]})
    ordered = whiff.set_index('model').reindex(['current_pitch', 'plus_swing_deviations'])
    bars = axes[0].bar([0, 1], ordered.log_loss, width=.58, color=[NAVY, TEAL])
    axes[0].bar_label(bars, labels=[f'{value:.4f}' for value in ordered.log_loss], padding=5, fontsize=10)
    axes[0].set_xticks([0, 1], ['Current\npitch', '+ swing\ndeviations'])
    axes[0].set_ylim(0, float(ordered.log_loss.max()) * 1.17)
    axes[0].set_ylabel('2026 log loss (lower is better)')
    axes[0].set_title('Held-out whiff discrimination', loc='left', fontsize=11)
    axes[0].grid(axis='y')
    y = np.arange(len(miss))
    axes[1].errorbar(miss.mu_median, y,
        xerr=[np.maximum(0,miss.mu_median-miss.mu_low), np.maximum(0,miss.mu_high-miss.mu_median)],
        fmt='o', color=CORAL, capsize=4, lw=2)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_yticks(y, [str(int(value)) for value in miss.season])
    axes[1].set_ylim(-.65, max(.65, len(miss)-.35))
    axes[1].set_xlabel('Conditional miss-distance μ\n(90% credible interval)')
    axes[1].set_title('Only tracked whiffs', loc='left', fontsize=11)
    axes[1].grid(axis='x')
    palette = {2023:'#8299AA', 2024:'#AC8D6B', 2025:TEAL, 2026:CORAL}
    for season, group in selection.groupby('season', sort=True):
        group = group.sort_values('exposure_bucket')
        axes[2].plot(group.exposure_bucket, group[inclusion_column], 'o-', ms=3,
                     lw=1.7, color=palette.get(int(season), NAVY), label=str(int(season)))
    axes[2].set_xticks([0,2,4,6,8,10], ['0','2','4','6','8','10+'])
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel('Prior comparable pitches (exposure bucket)')
    axes[2].set_ylabel('Swings meeting analytic inclusion criteria (%)')
    axes[2].set_title('Tracking inclusion by exposure', loc='left', fontsize=11)
    axes[2].legend(frameon=False, ncol=2, fontsize=8, loc='best'); axes[2].grid(axis='y')
    fig.suptitle('Outcome validity and tracking selection', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0,.19,1,.94))
    return _save(fig,out,'15_outcome_sensitivity_and_selection',
        'Whiff classification uses contemporaneous mechanics, not a pre-pitch forecast. Miss distance is conditional on tracked whiffs; contacts are missing, not zero.\n'
        'Miss-distance units: 2025 SD of log(1 + distance) per log(1 + exposure). No unconditional error-learning claim follows from this selected endpoint.')


def make_figures(root):
    """Regenerate all PNG/PDF figures from actual frozen pipeline outputs.

    Required artifacts are intentionally not faked or silently skipped. Run
    `python -m src.pipeline all` first if a file is missing.
    """
    from src.visualization.percentiles import plot_percentile_stability
    root=Path(root).resolve();out=root/'results/figures';out.mkdir(parents=True,exist_ok=True)
    set_style();result={}
    for fn in [plot_timeline,plot_coverage,plot_sequence_gain,plot_exposure,plot_player_uncertainty,plot_pairwise,plot_information,plot_recovery,plot_null,plot_robustness,plot_persistence,plot_outcomes_ppc,plot_posteriors,plot_temporal_pairwise,plot_outcome_sensitivity]:
        path=fn(root,out);result[path.stem]=path
    path=plot_percentile_stability(root,out);result[path.stem]=path
    return result


if __name__=='__main__':
    make_figures(Path(__file__).resolve().parents[2])
