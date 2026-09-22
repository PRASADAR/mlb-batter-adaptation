#!/usr/bin/env python3
"""Audit empirical coverage for every requested bat variable and denominator."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.statcast import load_pitches, coverage_table, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, BAT_FIELDS, verify_partition_integrity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(ROOT/'data/raw/statcast.parquet'))
    a=p.parse_args()
    data=load_pitches(a.input)
    out=ROOT/'results/tables'
    out.mkdir(exist_ok=True,parents=True)
    input_path=Path(a.input)
    integrity=verify_partition_integrity(input_path if input_path.is_dir() else input_path.parent)
    if len(integrity):
        integrity.to_csv(out/'data_coverage_integrity.csv',index=False)
        if not integrity.sha256_verified.all():
            raise ValueError('A raw partition fails its recorded SHA256.')
    tables=[]
    for group in [None,'season','pitch_type','batter']:
        coverage=coverage_table(data,group)
        coverage.to_csv(out/f"data_coverage_{group or 'overall'}.csv",index=False)
        tables.append(coverage)
    pd.concat(tables,ignore_index=True).to_csv(out/'data_coverage.csv',index=False)
    ranges=[]
    for field in BAT_FIELDS:
        series=pd.to_numeric(data[field],errors='coerce') if field in data else pd.Series(dtype=float)
        observed=series.dropna()
        finite=observed[np.isfinite(observed)]
        quant=finite.quantile([0,.01,.5,.99,1])
        ranges.append(dict(field=field,n_populated=len(observed),n_nonfinite=int((~np.isfinite(observed)).sum()),
                           minimum=quant.get(0),p01=quant.get(.01),median=quant.get(.5),p99=quant.get(.99),maximum=quant.get(1)))
    pd.DataFrame(ranges).to_csv(out/'data_coverage_ranges.csv',index=False)
    summary=data.assign(season=data.game_date.astype(str).str[:4], swing=data.description.isin(SWING_DESCRIPTIONS), whiff=data.description.isin(WHIFF_DESCRIPTIONS)).groupby('season').agg(pitches=('game_pk','size'),games=('game_pk','nunique'),batters=('batter','nunique'),swings=('swing','sum'),whiffs=('whiff','sum'),first_date=('game_date','min'),last_date=('game_date','max'))
    summary.to_csv(out/'data_sample.csv')
    data.assign(season=data.game_date.astype(str).str[:4]).groupby(['season','description'],dropna=False).size().rename('records').reset_index().to_csv(out/'data_coverage_events.csv',index=False)
    print(summary.to_string())
    print(tables[0][['field','pct_all_pitches','pct_swings','pct_whiffs','pct_contact']].to_string(index=False))
if __name__=='__main__':
    main()
