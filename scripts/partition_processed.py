"""Convert older combined derived files to identical monthly partitions."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline import write_processed
for kind in ['residuals','analytic']:
    for yr in [2024,2025,2026]:
        p=ROOT/f'data/processed/{kind}_{yr}.parquet'
        if p.exists():
            d=pd.read_parquet(p)
            write_processed(d,kind,yr)
            print(f'{kind} {yr}: {len(d):,} rows partitioned losslessly')
