#!/usr/bin/env python3
"""Bounded supplementary diagnostics on frozen main-pipeline outputs.

Run only after `python -m src.pipeline models` completes. This script does not
refit expected-swing or nuisance models, and never tunes against 2026.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.expected import player_slopes
from src.models.hierarchical import measurement_error_persistence

SEED = 20260921
SPLIT_DATE = '2025-07-01'
MIN_SWINGS = 40
MIN_GAMES = 8
PERSISTENCE_BOOTSTRAPS = 100
PITCHER_MIN_SWINGS = 100
PITCHER_MIN_GAMES = 8
PITCHER_BOOTSTRAPS = 500


def write_json(path: Path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + '\n')


def require_complete_main_outputs():
    marker = ROOT/'results/tables/batter_adaptation_posteriors.csv'
    required = [marker, ROOT/'results/tables/first_stage_2025.csv',
                ROOT/'results/tables/first_stage_2026.csv', ROOT/'results/models/nuisance.json',
                ROOT/'results/models/baseline.json', ROOT/'results/posterior/adaptation_2026.npz']
    files = []
    for year in (2025, 2026):
        for month in range(3, 10):
            files.append(ROOT/f'data/processed/analytic_{year}_{month:02d}.parquet')
    missing = [str(p.relative_to(ROOT)) for p in required + files if not p.exists()]
    if missing:
        raise FileNotFoundError('Wait for the main model run to finish. Missing: ' + ', '.join(missing))
    if marker.stat().st_mtime < max(p.stat().st_mtime for p in files):
        raise RuntimeError('Analytic files are newer than the completed-model marker; wait for the main run.')
    return files


def add_identification_diagnostics(result: dict, pairs: pd.DataFrame):
    """Reporting-only signal/noise diagnostics; never alter fitted estimates.

    A ratio below .1 means fitted latent variance is below 1% of the median
    first-stage measurement variance. This is an explicitly heuristic warning,
    not a calibrated test or a criterion for selecting/refitting a model.
    """
    ratios={}
    reliability={}
    median_ses={}
    for period,suffix in [('early','a'),('late','b')]:
        se=float(pairs[f'se_{period}'].median())
        tau=float(result[f'latent_sd_{suffix}'])
        median_ses[period]=se
        ratios[period]=tau/se
        reliability[period]=tau*tau/(tau*tau+se*se)
    result['median_first_stage_se']=median_ses
    result['latent_sd_to_median_se_ratio']=ratios
    result['implied_signal_fraction_at_median_se']=reliability
    result['weak_identification_ratio_threshold']=.1
    result['supplementary_low_signal_warning']=bool(min(ratios.values())<.1)
    result['shared_function_weak_identification']=bool(result.get('weak_identification',False))
    result['weak_identification']=result['supplementary_low_signal_warning'] or result['shared_function_weak_identification']
    result['weak_identification_diagnostic_scope']='Reporting-only union of shared-function numerical/profile/bootstrap diagnostics and any period latent SD <10% of its median first-stage SE, equivalent to latent variance <1% of median measurement variance. These heuristic warnings do not alter estimates, intervals, eligibility or model choices.'
    return result


def split_half(analytic: pd.DataFrame):
    """Date-based 2025 split with entire games assigned to one side."""
    data = analytic.loc[analytic.season.eq(2025)].copy()
    dates = pd.to_datetime(data.game_date)
    game_dates = pd.DataFrame({'game_pk': data.game_pk, 'date': dates}).groupby('game_pk').date
    if (game_dates.nunique() != 1).any():
        raise ValueError('A game spans different recorded dates; specify its full-game split assignment before proceeding.')
    early = dates.lt(SPLIT_DATE)
    if set(data.loc[early,'game_pk']) & set(data.loc[~early,'game_pk']):
        raise AssertionError('A game appears in both halves.')
    first = player_slopes(data.loc[early], min_n=MIN_SWINGS, min_games=MIN_GAMES)
    second = player_slopes(data.loc[~early], min_n=MIN_SWINGS, min_games=MIN_GAMES)
    pairs = first.merge(second, on='player_id', suffixes=('_early','_late'), validate='one_to_one')
    if len(pairs) < 10:
        raise ValueError('Fewer than ten matched eligible hitters; split-half persistence is not estimable under the fixed rule.')
    names = pd.read_csv(ROOT/'data/raw/players.csv').rename(columns={'batter':'player_id'})
    pairs = pairs.merge(names[['player_id','player_name']],on='player_id',how='left',validate='one_to_one')
    result = measurement_error_persistence(pairs.estimate_early, pairs.se_early,
        pairs.estimate_late, pairs.se_late, bootstrap=PERSISTENCE_BOOTSTRAPS, seed=SEED)
    draws = pd.DataFrame(result.pop('bootstrap_draws'),columns=['latent_correlation','latent_sd_early','latent_sd_late'])
    result['naive_spearman'] = float(spearmanr(pairs.estimate_early,pairs.estimate_late).statistic)
    result.update(year=2025, split_date=SPLIT_DATE, minimum_swings_per_half=MIN_SWINGS,
                  minimum_games_per_half=MIN_GAMES, minimum_residual_exposure_information=1.0,
                  n_eligible_early=len(first), n_eligible_late=len(second),
                  n_analytic_swings_early=int(early.sum()), n_analytic_swings_late=int((~early).sum()),
                  n_games_early=int(data.loc[early,'game_pk'].nunique()),
                  n_games_late=int(data.loc[~early,'game_pk'].nunique()),
                  first_date_early=str(dates[early].min().date()),last_date_early=str(dates[early].max().date()),
                  first_date_late=str(dates[~early].min().date()),last_date_late=str(dates[~early].max().date()),
                  holdout_2026_used_for_split_half=False, seed=SEED,
                  interpretation='Descriptive split-half reliability of conditional negative exposure slopes; not causal adaptation.',
                  limitations=['2025 game-group cross-fitting shares estimated nuisance functions across halves; period errors are only approximately independent.',
                      'First-stage cluster SEs are treated as known; nuisance-model uncertainty and cross-hitter dependence are not propagated.',
                      'Player bootstrap resamples matched eligible hitters; survival and playing-time selection are not removed.',
                      'The latent-normal correlation can approach its numerical boundary when between-hitter heterogeneity is weak.',
                      'This conditional residual diagnostic is not a strict prospective early-to-late forecast.'])
    add_identification_diagnostics(result,pairs)
    return pairs, draws, result


def pitcher_names():
    records=[]
    for path in sorted((ROOT/'data/raw').glob('statcast_*.parquet')):
        records.append(pd.read_parquet(path,columns=['pitcher','player_name']).drop_duplicates('pitcher'))
    return pd.concat(records,ignore_index=True).drop_duplicates('pitcher').set_index('pitcher').player_name.to_dict()


def pitcher_diagnostics(analytic: pd.DataFrame, baseline: dict):
    """Descriptive sequence predictability and residual calibration by pitcher.

    Each component's squared residual is standardized by the fixed pre-2025
    covariance-calibration SD. The combined metric weights components equally;
    it is not the Mahalanobis adaptation outcome or a causal pitcher effect.
    """
    components=baseline['components']
    scales=np.asarray(baseline['residual_scale'],dtype=float)
    frame=analytic[['season','pitcher','game_pk','adj_distortion']].copy()
    a=analytic[[f'resid_{component}' for component in components]].to_numpy(dtype=float)
    b=analytic[[f'sequence_resid_{component}' for component in components]].to_numpy(dtype=float)
    frame['current_loss']=np.mean((a/scales)**2,axis=1)
    frame['history_loss']=np.mean((b/scales)**2,axis=1)
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna()
    names=pitcher_names()
    rows=[]
    for (season,pitcher),g in frame.groupby(['season','pitcher'],sort=True):
        n,ng=len(g),g.game_pk.nunique()
        if n<PITCHER_MIN_SWINGS or ng<PITCHER_MIN_GAMES:
            continue
        game=g.groupby('game_pk').agg(current_loss=('current_loss','sum'),history_loss=('history_loss','sum'),n=('current_loss','size'))
        la,lb=game.current_loss.to_numpy(),game.history_loss.to_numpy()
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(season),int(pitcher)]))
        indices=rng.integers(0,ng,size=(PITCHER_BOOTSTRAPS,ng))
        bootstrap_gain=100*(la[indices].sum(axis=1)-lb[indices].sum(axis=1))/la[indices].sum(axis=1)
        mean=float(g.adj_distortion.mean())
        score=(g.adj_distortion-mean).groupby(g.game_pk).sum().to_numpy()
        se=float(np.sqrt(ng/(ng-1)*(score@score)/(n*n)))
        critical=float(t.ppf(.95,ng-1))
        rows.append(dict(season=int(season),pitcher=int(pitcher),pitcher_name=names.get(int(pitcher),str(pitcher)),
            n_swings=n,n_games=int(ng),current_standardized_mse=float(la.sum()/n),history_standardized_mse=float(lb.sum()/n),
            sequence_mse_gain_pct=float(100*(la.sum()-lb.sum())/la.sum()),
            sequence_gain_ci90_low=float(np.quantile(bootstrap_gain,.05)),sequence_gain_ci90_high=float(np.quantile(bootstrap_gain,.95)),
            mean_adjusted_distortion=mean,adjusted_distortion_cluster_se=se,
            adjusted_distortion_ci90_low=mean-critical*se,adjusted_distortion_ci90_high=mean+critical*se,
            interpretation='observational predictive/calibration diagnostic; not causal pitcher setup skill'))
    table=pd.DataFrame(rows)
    metadata=dict(components=components,calibration_residual_scales=scales.tolist(),minimum_swings=PITCHER_MIN_SWINGS,
        minimum_games=PITCHER_MIN_GAMES,game_bootstrap_replicates=PITCHER_BOOTSTRAPS,seed=SEED,
        loss='Equal-weight mean squared component residual divided by pre-2025 calibration SD squared.',
        adjusted_distortion_interval='Game-cluster intercept-only sandwich standard error with t(G-1) 90% interval.',
        qualifying_pitchers_by_season={str(int(year)):len(g) for year,g in table.groupby('season')},
        limitations=['Current-only and history models were fitted before July 2024; differences reflect predictive associations and model fit, not causal pitch-sequence effects.',
                     'Conditional nuisance residual means are calibration diagnostics, not pitcher deception parameters.',
                     'Pitcher-specific batter mix, count mix, tracking selection, workload, and strategy can influence these descriptive summaries.',
                     'Intervals are pointwise, unpooled, and do not adjust for comparisons across pitchers.',
                     'Shared fitted-model uncertainty is not propagated; no pitcher ranking or persistence claim is made.'])
    return table,metadata


def methods_note(split: dict, pitcher_meta: dict):
    ci=split['bootstrap_ci90']
    if split['weak_identification']:
        boundary='The fitted correlation is weakly identified; its point estimate must not be interpreted as evidence of reliable individual differences.'
        weak_periods=' and '.join(period for period,ratio in split['latent_sd_to_median_se_ratio'].items() if ratio<split['weak_identification_ratio_threshold'])
        identification=(f'Negligible fitted {weak_periods} variance makes the correlation uninformative.' if weak_periods else
                        'The likelihood-profile, bootstrap, or optimizer diagnostic flags weak identification despite non-negligible point-estimate variances.')
    else:
        boundary='The latent-correlation fit is at or near a numerical/weak-heterogeneity boundary.' if split['boundary_fit'] else 'The fit did not trigger the preset boundary diagnostic.'
        identification='The configured identification diagnostics do not flag weak identification; the bootstrap interval and shared-nuisance limitations still govern interpretation.'
    return f'''# Supplementary validation

This supplement uses frozen main-pipeline residuals and does not refit or select the main models. Choices were fixed before inspecting these diagnostics: a July 1, 2025 split; 40 eligible swings and eight games per hitter in each half; 100 player-bootstrap replicates; and 100 swings/eight games with 500 game-bootstrap replicates for descriptive pitcher summaries. The 2026 holdout was not used to choose these settings.

## Within-season reliability

All swings from a game remain in one half. Early dates are {split['first_date_early']}–{split['last_date_early']}; late dates are {split['first_date_late']}–{split['last_date_late']}. The primary first-stage estimator is reused unchanged: negative slope of adjusted swing distortion on adjusted log(1 + prior pitches of the same type seen from the same pitcher in the game), with a hitter intercept and game-cluster sandwich standard errors. Positive slopes in the saved tables therefore mean decreasing distortion with exposure, not a percentage learning rate. Residual exposure information must exceed the estimator's fixed threshold of one.

There are {split['n_eligible_early']} eligible early-half hitters, {split['n_eligible_late']} eligible late-half hitters, and **{split['n_players']} matched hitters**. Raw Pearson correlation is **{split['raw_correlation']:.3f}** and Spearman correlation is **{split['naive_spearman']:.3f}**. A bivariate latent-normal measurement-error likelihood, treating the period slope SEs as known and independent, estimates correlation **{split['latent_correlation']:.3f}**, with a 90% player-bootstrap interval **[{ci[0]:.3f}, {ci[1]:.3f}]** ({split['bootstrap_successes']}/{split['bootstrap_requested']} successful fits). {boundary}

Fitted latent SDs are {split['latent_sd_a']:.6f} early and {split['latent_sd_b']:.6f} late, versus median first-stage SEs {split['median_first_stage_se']['early']:.6f} and {split['median_first_stage_se']['late']:.6f}. Their signal-to-noise SD ratios are {split['latent_sd_to_median_se_ratio']['early']:.4f} and {split['latent_sd_to_median_se_ratio']['late']:.4f}. Model-implied signal fractions at the median SE are {100*split['implied_signal_fraction_at_median_se']['early']:.4f}% early and {100*split['implied_signal_fraction_at_median_se']['late']:.4f}% late. {identification} The original strict numerical boundary rule is retained in the JSON; it is not a sufficient identification check. The reporting-only weak-identification flag combines a period latent SD below 10% of its median SE, broad fixed-rho likelihood support, bootstrap near-zero variance frequency, and optimizer diagnostics. These are heuristic warnings, not calibrated hypothesis tests. Numerical optimization was corrected by analytically profiling means, scaling by median SE, and using multiple nonzero variance starts with tighter tolerances; the likelihood, data, eligibility, split, bootstrap count, and seed are unchanged. Bootstrap quantiles near a variance or correlation boundary are descriptive and are not guaranteed calibrated confidence limits.

These are conditional reliability diagnostics. The 2025 nuisance predictions hold out whole games but share fitted nuisance functions across halves, so estimation-error independence is approximate and this is not a prospective early-to-late forecast. Nuisance-fit uncertainty and cross-hitter dependence are omitted; eligibility and survival into both halves remain selection mechanisms. A measurement-error correction cannot establish persistent skill when heterogeneity or interval estimates are weak. No causal learning interpretation follows from this split.

Outputs: `results/tables/split_half_2025_pairs.csv`, `split_half_2025_bootstrap.csv`, and `split_half_2025_persistence.json`.

## Exploratory pitcher diagnostics

`results/tables/pitcher_sequence_diagnostics.csv` reports unpooled, descriptive results for {pitcher_meta['qualifying_pitchers_by_season'].get('2025',0)} pitchers in 2025 and {pitcher_meta['qualifying_pitchers_by_season'].get('2026',0)} in 2026. The predictive metric compares current-pitch-only and current-plus-history expected-swing models, both fitted before July 2024. Each component's squared residual is divided by its pre-2025 calibration variance; their equal-weight mean forms a standardized loss. Reported sequence gain is 100 × (current loss − history loss) / current loss. This loss comparison is distinct from the Mahalanobis distortion outcome used in the adaptation model.

Paired game-cluster bootstrap resampling within each pitcher gives 90% pointwise intervals for predictive gain. The table also reports mean adjusted distortion with an intercept-only game-cluster standard error and a t(G−1) interval, solely as a residual-calibration diagnostic. Minimum eligibility is 100 analytic swings across eight games per pitcher-season, fixed without looking at gains.

Pitcher identity, opponent mix, selection into tracked swings, counts, changing strategy, and model misspecification can affect these quantities. The intervals do not include shared model-fit uncertainty, partial pooling, or multiplicity adjustment. These tables are **not a causal pitcher-deception estimate, a pitcher skill leaderboard, or evidence of persistent pitcher setup ability**. No pitcher is selected for an interpretive case study based on a favorable result. Full settings and limitations are in `results/tables/pitcher_sequence_diagnostics.json`.

## Reproduction

Run `python scripts/17_supplementary_validation.py` after the main model stage finishes. Input Parquet hashes are saved in `results/logs/supplementary_validation_inputs.json`. All computations are deterministic under seed {SEED}. The supplementary script does not write any analytic data or main-model artifacts.
'''


def main():
    files=require_complete_main_outputs()
    baseline=json.loads((ROOT/'results/models/baseline.json').read_text())
    components=baseline['components']
    columns=['game_date','game_pk','batter','pitcher','season','adj_log_exposure_pitcher','adj_distortion']
    columns += [f'{prefix}_{c}' for prefix in ['resid','sequence_resid'] for c in components]
    analytic=pd.concat([pd.read_parquet(p,columns=columns) for p in files],ignore_index=True)
    out=ROOT/'results/tables'
    out.mkdir(parents=True,exist_ok=True)
    print('Estimating fixed 2025 split-half reliability.',flush=True)
    pairs,draws,split=split_half(analytic)
    pairs.to_csv(out/'split_half_2025_pairs.csv',index=False)
    draws.to_csv(out/'split_half_2025_bootstrap.csv',index=False)
    write_json(out/'split_half_2025_persistence.json',split)
    print('Summarizing descriptive pitcher sequence predictability.',flush=True)
    pitcher_table,pitcher_meta=pitcher_diagnostics(analytic,baseline)
    pitcher_table.to_csv(out/'pitcher_sequence_diagnostics.csv',index=False)
    write_json(out/'pitcher_sequence_diagnostics.json',pitcher_meta)
    (ROOT/'paper').mkdir(exist_ok=True)
    (ROOT/'paper/supplementary_validation.md').write_text(methods_note(split,pitcher_meta))
    provenance=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    provenance.append(dict(path='results/models/baseline.json',sha256=hashlib.sha256((ROOT/'results/models/baseline.json').read_bytes()).hexdigest()))
    (ROOT/'results/logs').mkdir(exist_ok=True)
    write_json(ROOT/'results/logs/supplementary_validation_inputs.json',provenance)
    print(json.dumps({key:split[key] for key in ['n_players','raw_correlation','naive_spearman','latent_correlation','bootstrap_ci90','boundary_fit']}),flush=True)
    print(f'Saved {len(pitcher_table)} pitcher-season diagnostics and supplementary methods note.',flush=True)


if __name__=='__main__':
    main()
