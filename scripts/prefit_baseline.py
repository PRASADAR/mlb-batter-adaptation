"""Optional warm-start of identical cached baselines while later data download."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.expected import fit_predict
cfg=json.loads((ROOT/'config/analysis.json').read_text())
s=pd.concat([pd.read_parquet(ROOT/f'data/processed/swings_{yr}.parquet') for yr in [2023,2024]],ignore_index=True)
components=cfg['swing_components']
s=s.loc[s.game_date.lt('2024-07-01')&s[components].notna().all(axis=1)&s.bat_speed.between(20,110)&s.swing_length.between(1,12)].copy()
s['direction_sin']=np.sin(np.radians(s.attack_direction));s['direction_cos']=np.cos(np.radians(s.attack_direction))
targets=[x for x in components if x!='attack_direction']+['direction_sin','direction_cos']
for history,label in [(False,'current'),(True,'history')]:
    print(f'Fitting cached {label} baseline',flush=True)
    _,_,meta=fit_predict(s,s.iloc[:100],targets,history=history,seed=cfg['seed'],iterations=cfg['catboost_iterations'],max_train=cfg['max_training_swings'],cache_path=ROOT/f'results/models/{label}_fit.cbm')
    print(meta,flush=True)
