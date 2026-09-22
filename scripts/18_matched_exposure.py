#!/usr/bin/env python3
"""Prespecified, descriptive matching restriction; never a causal estimator.

Match only physical current-pitch covariates and prior exposure/order. Outcome
values never enter eligibility, distance, tie breaking, or pairing decisions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['season', 'batter', 'game_pk', 'pitcher', 'pitch_type', 'balls', 'strikes']
PITCH_KEY = ['game_pk', 'at_bat_number', 'pitch_number']
COVARIATES = ['release_speed', 'pfx_x', 'pfx_z', 'plate_x_front', 'plate_z_front', 'release_pos_x', 'release_pos_z']
SCALES = np.array([2.0, .15, .15, .35, .35, .25, .25])
OUTCOMES = ['distortion', 'adj_distortion']
SPECIFICATION = {
    'version': 1,
    'seasons': [2025, 2026],
    'exact_strata': KEYS,
    'chronological_order': PITCH_KEY,
    'speed_max_difference_mph': 2.0,
    'pfx_x_max_difference_ft': .15,
    'pfx_z_max_difference_ft': .15,
    'front_plane_location_max_distance_ft': .35,
    'release_xz_max_distance_ft': .25,
    'exposure_requirement': 'later exposure_pitcher strictly exceeds earlier exposure_pitcher',
    'distance': 'Euclidean covariate distance divided by fixed coordinate tolerances; no data-estimated scales',
    'covariates': COVARIATES,
    'coordinate_scales': SCALES.tolist(),
    'matching': 'Greedy chronological; closest eligible unmatched earlier swing; ties choose most recent; no replacement',
    'outcome_eligibility': 'None; outcome values and missingness never enter matching',
    'endpoints': OUTCOMES,
    'contrast': 'later minus earlier; negative means a decline; not divided by exposure increment',
    'uncertainty': '2000 game-cluster bootstrap replicates within each season; paired swings kept together',
    'scope': 'Descriptive matched restriction among tracked swings; not causal identification',
}


def specification_record():
    canonical = json.dumps(SPECIFICATION, sort_keys=True, separators=(',', ':'))
    return {'specified_utc': datetime.now(timezone.utc).isoformat(),
            'specification_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
            'specification': SPECIFICATION}


def freeze_specification(root):
    path = root / 'results/logs/matched_exposure_specification.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = specification_record()
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['specification_sha256'] != record['specification_sha256']:
            raise ValueError('Existing matched-analysis specification differs; do not silently change it after seeing outcomes')
        return previous
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def match_swings(frame):
    """Return an auditable set of nonoverlapping pairs chosen without outcomes."""
    required = list(dict.fromkeys(KEYS + PITCH_KEY + COVARIATES + ['exposure_pitcher']))
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing matching columns: {sorted(missing)}')
    if frame.duplicated(PITCH_KEY).any():
        raise ValueError('Duplicate physical pitch identifiers are not allowed')
    finite = np.isfinite(frame[COVARIATES + ['exposure_pitcher']].to_numpy(float)).all(axis=1)
    eligible = finite & frame[KEYS].notna().all(axis=1).to_numpy()
    d = frame.loc[eligible].sort_values(PITCH_KEY, kind='stable').reset_index(drop=True)
    rows = []
    for _, group in d.groupby(KEYS, sort=True, dropna=False):
        if len(group) < 2:
            continue
        group = group.reset_index(drop=True)
        x = group[COVARIATES].to_numpy(float)
        exposure = group.exposure_pitcher.to_numpy(float)
        unmatched = []
        for later in range(len(group)):
            candidates = np.asarray(unmatched, dtype=int)
            if len(candidates):
                delta = x[later] - x[candidates]
                allowed = ((np.abs(delta[:, 0]) <= 2.0)
                           & (np.abs(delta[:, 1]) <= .15)
                           & (np.abs(delta[:, 2]) <= .15)
                           & (np.hypot(delta[:, 3], delta[:, 4]) <= .35)
                           & (np.hypot(delta[:, 5], delta[:, 6]) <= .25)
                           & (exposure[later] > exposure[candidates]))
                candidates = candidates[allowed]
            if not len(candidates):
                unmatched.append(later)
                continue
            distances = np.sqrt(np.sum(((x[later]-x[candidates])/SCALES)**2, axis=1))
            # lexsort's last key is primary; equal distance favors recent order.
            chosen = int(np.lexsort((-candidates, distances))[0])
            earlier = int(candidates[chosen])
            unmatched.remove(earlier)
            early_row, late_row = group.iloc[earlier], group.iloc[later]
            pair = {key: late_row[key] for key in KEYS}
            pair.update({
                'earlier_at_bat_number': early_row.at_bat_number,
                'earlier_pitch_number': early_row.pitch_number,
                'later_at_bat_number': late_row.at_bat_number,
                'later_pitch_number': late_row.pitch_number,
                'earlier_exposure': exposure[earlier], 'later_exposure': exposure[later],
                'exposure_increment': exposure[later]-exposure[earlier],
                'covariate_distance': float(distances[chosen]),
                'plate_location_distance_ft': float(np.hypot(*(x[later,3:5]-x[earlier,3:5]))),
                'release_location_distance_ft': float(np.hypot(*(x[later,5:7]-x[earlier,5:7]))),
            })
            for cov in COVARIATES:
                pair[f'earlier_{cov}'] = float(early_row[cov])
                pair[f'later_{cov}'] = float(late_row[cov])
                pair[f'delta_{cov}'] = float(late_row[cov])-float(early_row[cov])
            # Read outcomes only AFTER the pair has been fixed.
            for outcome in OUTCOMES:
                before = float(early_row.get(outcome, np.nan))
                after = float(late_row.get(outcome, np.nan))
                pair[f'earlier_{outcome}'] = before
                pair[f'later_{outcome}'] = after
                pair[f'delta_{outcome}'] = after-before
            rows.append(pair)
    result = pd.DataFrame(rows)
    result.attrs['n_input_swings'] = len(frame)
    result.attrs['n_covariate_eligible_swings'] = len(d)
    return result


def bootstrap_means(pairs, columns, seed, replicates=2000):
    """Percentile intervals from resampling games; repeated pairs stay together."""
    grouped = pairs.groupby('game_pk')[columns]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.count().to_numpy(float)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for start in range(0, replicates, 200):
        index = rng.integers(0, len(sums), size=(min(200,replicates-start), len(sums)))
        denominator = counts[index].sum(axis=1)
        numerator = sums[index].sum(axis=1)
        bootstrap.append(np.divide(numerator, denominator, out=np.full_like(numerator,np.nan), where=denominator>0))
    values = np.concatenate(bootstrap)
    return dict(zip(columns, np.nanquantile(values, [.05,.95], axis=0).T))


def read_analytic_partition(path, columns):
    """The compact pipeline stores corrected front-plane coordinates as plate_x/z.

    prepare_pitches explicitly overwrites plate_x and plate_z with the common
    front-plane values before analytic partitioning. This is an alias of the
    prespecified measurement, not a substitution of a different pitch location.
    """
    schema = set(pq.read_schema(path).names)
    aliases = {name: name.removesuffix('_front') for name in ['plate_x_front','plate_z_front'] if name not in schema}
    loaded = pd.read_parquet(path, columns=[aliases.get(name,name) for name in columns])
    return loaded.rename(columns={source:target for target,source in aliases.items()})


def run(root=ROOT, seed=20260921):
    root = Path(root).resolve()
    record = freeze_specification(root)  # Freeze BEFORE loading analytic rows.
    markers = [root/'results/posterior/adaptation_2025.npz', root/'results/posterior/adaptation_2026.npz',
               root/'results/tables/batter_adaptation_posteriors.csv']
    if not all(path.exists() for path in markers):
        raise RuntimeError('Primary annual posterior markers are not complete; finish the primary models before this supplementary analysis')
    selected_columns = list(dict.fromkeys(KEYS + PITCH_KEY + COVARIATES + ['exposure_pitcher'] + OUTCOMES))
    pairs_by_season, summary_rows, balance_rows = [], [], []
    for season in SPECIFICATION['seasons']:
        paths = sorted((root/'data/processed').glob(f'analytic_{season}*.parquet'))
        if not paths:
            raise FileNotFoundError(f'No analytic monthly partitions for {season}')
        swings = pd.concat([read_analytic_partition(path, selected_columns) for path in paths], ignore_index=True)
        pairs = match_swings(swings)
        counts = dict(pairs.attrs)
        del swings
        if pairs.empty:
            raise ValueError(f'Prespecified matching found no pairs for {season}; do not relax tolerances after seeing this')
        pairs['pair_id'] = [f'{season}-{i:06d}' for i in range(len(pairs))]
        pairs_by_season.append(pairs)
        delta_columns = [f'delta_{name}' for name in OUTCOMES+COVARIATES]
        intervals = bootstrap_means(pairs, delta_columns, seed+season)
        for outcome in OUTCOMES:
            valid = pairs.loc[pairs[f'delta_{outcome}'].notna()]
            lo, hi = intervals[f'delta_{outcome}']
            summary_rows.append({
                'season':season, 'outcome':outcome, **counts, 'n_pairs':len(pairs),
                'n_pairs_with_outcome':len(valid), 'games':int(valid.game_pk.nunique()),
                'players':int(valid.batter.nunique()),
                'fraction_eligible_swings_matched':2*len(pairs)/counts['n_covariate_eligible_swings'],
                'mean_exposure_increment':float(pairs.exposure_increment.mean()),
                'mean_change_later_minus_earlier':float(valid[f'delta_{outcome}'].mean()),
                'ci90_low':float(lo), 'ci90_high':float(hi), 'bootstrap_replicates':2000,
            })
        for cov in COVARIATES:
            before, after, delta = pairs[f'earlier_{cov}'], pairs[f'later_{cov}'], pairs[f'delta_{cov}']
            pooled_sd = float(np.sqrt((before.var(ddof=1)+after.var(ddof=1))/2))
            lo, hi = intervals[f'delta_{cov}']
            balance_rows.append({
                'season':season,'covariate':cov,'unit':'mph' if cov=='release_speed' else 'ft',
                'earlier_mean':float(before.mean()),'later_mean':float(after.mean()),
                'mean_difference':float(delta.mean()),'mean_absolute_difference':float(delta.abs().mean()),
                'max_absolute_difference':float(delta.abs().max()),'pooled_sd':pooled_sd,
                'standardized_mean_difference':float(delta.mean()/pooled_sd) if pooled_sd>0 else None,
                'ci90_low':float(lo),'ci90_high':float(hi),
            })
    audit = pd.concat(pairs_by_season, ignore_index=True)
    summary, balance = pd.DataFrame(summary_rows), pd.DataFrame(balance_rows)
    audit.to_parquet(root/'data/processed/matched_exposure_pairs.parquet', index=False, compression='zstd')
    summary.to_csv(root/'results/tables/matched_exposure_summary.csv', index=False)
    balance.to_csv(root/'results/tables/matched_exposure_balance.csv', index=False)
    result = {'specification':record,'seed':seed,'summary':summary_rows,'balance':balance_rows,
              'audit_file':'data/processed/matched_exposure_pairs.parquet',
              'interpretation':'Selected matched swings; later-minus-earlier change, not a causal or per-exposure learning rate'}
    (root/'results/tables/matched_exposure.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_notes(root, record, summary)
    print(summary.to_string(index=False))
    return result


def write_notes(root, record, summary=None):
    text = '''# Matched current-pitch comparisons

This supplementary restriction was specified before inspecting its paired outcome changes. It does not change the primary models. The specification is frozen in `results/logs/matched_exposure_specification.json`.

Match exactly within season, hitter, game, pitcher, pitch type, balls, and strikes. A later tracked swing must have greater prior same-pitcher-type exposure. Eligible earlier swings must differ by at most 2 mph in velocity, 0.15 ft in each movement component, 0.35 ft in Euclidean front-of-plate location, and 0.25 ft in Euclidean x/z release location. Front-plane location has the pipeline's common measurement-plane adjustment.

Process swings chronologically. Match each later swing to the available earlier swing with the smallest Euclidean covariate distance after dividing each coordinate by its fixed tolerance. Exact distance ties favor the most recent earlier swing. Neither swing can be used again. Outcomes and outcome missingness never affect matching. Missing matching covariates do exclude a swing.

The contrast is distortion on the later swing minus distortion on the earlier swing, reported for the original and nuisance-adjusted deviation score. Negative changes mean less deviation; they are not a percentage correction, not divided by exposure increments, and not estimates of a causal learning rate. Each matched pair receives equal weight, so hitters with more eligible pairs contribute more; this is not the same estimand as the hierarchical population mean. The uncertainty intervals resample complete games 2,000 times separately by season, conditional on the fitted nuisance models, observed hitter pool, and selected matches; players are not independently resampled.

The paired audit preserves pitch identifiers, covariates, exposure increments, outcomes, distances, and changes. The balance table reports both signed and absolute differences, because signed balance alone can conceal mismatch. Exact count matching and no replacement limit reuse but do not remove unmeasured intent, changing pitcher strategy, regression to the mean, survivorship, or tracking selection. Two-strike foul sequences and repeated same-count opportunities can be overrepresented. These strict pairs do not represent all swings or all hitters; agreement with a primary result would be a descriptive robustness check, not causal identification.
'''
    text += f"\nSpecification frozen: {record['specified_utc']}. SHA256: `{record['specification_sha256']}`.\n"
    if summary is not None:
        text += '\n## Generated paired changes\n\n| Season | Outcome | Pairs | Games | Hitters | Mean later − earlier | 90% game-bootstrap interval |\n|---|---|---:|---:|---:|---:|---|\n'
        for _, row in summary.iterrows():
            text += f"| {int(row.season)} | {row.outcome} | {int(row.n_pairs_with_outcome):,} | {int(row.games):,} | {int(row.players):,} | {row.mean_change_later_minus_earlier:.4f} | [{row.ci90_low:.4f}, {row.ci90_high:.4f}] |\n"
        text += '\nSelection fractions and exposure increments are reported in `results/tables/matched_exposure_summary.csv`; covariate balance is in `results/tables/matched_exposure_balance.csv`.\n'
    (root/'paper/matched_comparisons.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-only',action='store_true',help='Freeze matching rules and methods notes without loading any outcomes')
    arguments=parser.parse_args()
    if arguments.spec_only:
        frozen=freeze_specification(ROOT);write_notes(ROOT,frozen)
        print('Matching specification frozen without loading outcomes:',frozen['specification_sha256'])
    else:
        run()
