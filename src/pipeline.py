"""Reproducible research pipeline. Run: python -m src.pipeline [features|models|validation|all]."""
from __future__ import annotations
import argparse, json, hashlib, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.features.sequences import prepare_pitches, BAT_FIELDS
from src.models.expected import swing_baseline, nuisance_residuals, player_slopes
from src.models.hierarchical import fit_hierarchy, summarize_posterior, measurement_error_persistence

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config/analysis.json').read_text())
SEED=CONFIG['seed']
EXPOSURES=['exposure_pitcher','exposure_pa','exposure_type','kernel_exposure',
           'exposure_sequence','future_exposure','irrelevant_exposure',
           'kernel_h0.5_m5','kernel_h0.5_m20','kernel_h1_m5','kernel_h2_m5','kernel_h2_m20']

def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def write_json(path,obj):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj,indent=2,default=default,allow_nan=False)+'\n')

def save_posterior(p,name):
    np.savez_compressed(ROOT/f'results/posterior/{name}.npz',**{k:p[k] for k in ['lambda','mu','tau','player_ids','observed_predictive']})
    write_json(ROOT/f'results/posterior/{name}_diagnostics.json',p['diagnostics'])

def load_posterior(name='adaptation_2025'):
    with np.load(ROOT/f'results/posterior/{name}.npz') as p: return dict(p)

def names_map():
    f=ROOT/'data/raw/players.csv'
    if not f.exists(): return {}
    a=pd.read_csv(f)
    idcol=next(x for x in ['player_id','batter','id'] if x in a)
    nmcol=next(x for x in ['player_name','fullName','name','full_name'] if x in a)
    return dict(zip(a[idcol].astype(int),a[nmcol]))

def feature_season(yr):
    from src.features.sequences import BASE_NUM,BASE_CAT,HISTORY_NUM,HISTORY_CAT
    paths=sorted((ROOT/'data/raw').glob(f'statcast_{yr}-*.parquet'))
    raw=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).drop_duplicates(['game_pk','at_bat_number','pitch_number'])
    raw=raw.loc[pd.to_datetime(raw.game_date).lt('2026-09-21')].copy()
    info={'season':yr,'pitches':len(raw),'games':int(raw.game_pk.nunique()),'administrative_events':int(raw.description.isin(['automatic_ball','automatic_strike']).sum()),'start':str(pd.to_datetime(raw.game_date).min().date()),'end':str(pd.to_datetime(raw.game_date).max().date())}
    log(f'All-pitch exposure and trajectory features: {yr}, {len(raw):,} pitches')
    d=prepare_pitches(raw);del raw;gc.collect()
    sel=d.groupby(['season','is_swing','is_whiff']).agg(n=('game_pk','size'),tracked=('bat_speed','count')).reset_index()
    sel.to_csv(ROOT/f'results/tables/filter_counts_{yr}.csv',index=False)
    keep=set(BASE_NUM+BASE_CAT+HISTORY_NUM+HISTORY_CAT+BAT_FIELDS+EXPOSURES+['game_pk','game_date','at_bat_number','pitch_number','season','description','is_swing','is_whiff','is_contact','shape_valid','similar_pitch_gap','estimated_woba_using_speedangle','launch_speed','launch_angle','events'])
    d=d.loc[d.is_swing,sorted(keep&set(d.columns))].copy()
    for c in d.select_dtypes('float64'):d[c]=d[c].astype('float32')
    d.to_parquet(ROOT/f'data/processed/swings_{yr}.parquet',index=False,compression='zstd')
    return info


def features():
    infos=[feature_season(yr) for yr in [2023,2024,2025,2026]]
    summary={'raw_pitches':sum(x['pitches'] for x in infos),'raw_games':sum(x['games'] for x in infos),'administrative_events':sum(x['administrative_events'] for x in infos),'delivered_pitch_records':sum(x['pitches']-x['administrative_events'] for x in infos),'start':min(x['start'] for x in infos),'end':max(x['end'] for x in infos),'seasons':{str(x['season']):x['pitches'] for x in infos}}
    write_json(ROOT/'results/logs/sample.json',summary)
    log(f'Frozen {summary["raw_pitches"]:,} pitch-level records')


def read_swings():
    return pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'data/processed').glob('swings_*.parquet'))],ignore_index=True)

def write_processed(frame,kind,year):
    """Monthly lossless partitions stay below GitHub's per-file limit."""
    directory=ROOT/'data/processed';old=set(directory.glob(f'{kind}_{year}*.parquet'));written=set()
    for month,g in frame.groupby(pd.to_datetime(frame.game_date).dt.month,sort=True):
        p=directory/f'{kind}_{year}_{int(month):02d}.parquet'
        g.to_parquet(p,index=False,compression='zstd');written.add(p)
    for p in old-written:p.unlink()


def models():
    s=read_swings();log(f'Baseline from {len(s):,} observed swings')
    e,m,meta,models_=swing_baseline(s,CONFIG['swing_components'],CONFIG)
    del s;gc.collect()
    m.to_csv(ROOT/'results/tables/sequence_distortion.csv',index=False)
    e.groupby('season').agg(n=('distortion','size'),mean_deviation=('distortion','mean'),sd_deviation=('distortion','std')).reset_index().to_csv(ROOT/'results/tables/baseline_drift.csv',index=False)
    write_json(ROOT/'results/models/baseline.json',meta)
    for label,model in zip(['current','history'],models_): model.save_model(str(ROOT/f'results/models/{label}.cbm'))
    # Retain all covariance-calibration rows separately for transparent whitening.
    for yr,g in e.groupby('season'): write_processed(g,'residuals',yr)
    a=e.loc[e.season.ge(2025)&e.shape_valid].copy().reset_index(drop=True)
    del e;gc.collect()
    for f in EXPOSURES: a[f'log_{f}']=np.log1p(a[f])
    a['linear_exposure_pitcher']=a.exposure_pitcher
    a['saturating_exposure_pitcher']=1-np.exp(-.35*a.exposure_pitcher)
    # Alternatives remain component-specific; no hidden rotation of the error state.
    targets=['distortion']+[f'absz_{x}' for x in CONFIG['swing_components']]+[f'log_{x}' for x in EXPOSURES]+['linear_exposure_pitcher','saturating_exposure_pitcher']
    log('Game-group cross-fitting nuisance models; 2026 predictions trained on 2025 only')
    a,meta_n=nuisance_residuals(a,targets,CONFIG)
    write_json(ROOT/'results/models/nuisance.json',meta_n)
    for yr,g in a.groupby('season'): write_processed(g,'analytic',yr)
    log('Partial pooling of hitter slopes')
    names=names_map();summ=[]
    for yr,g in a.groupby('season'):
        slopes=player_slopes(g,min_n=CONFIG['minimum_player_observations'],min_games=CONFIG['minimum_player_games'])
        slopes.to_csv(ROOT/f'results/tables/first_stage_{yr}.csv',index=False)
        p=fit_hierarchy(slopes.estimate,slopes.se,slopes.player_id,draws=CONFIG['posterior_draws'],seed=SEED)
        save_posterior(p,f'adaptation_{yr}')
        t=summarize_posterior(p,names=names,counts=dict(zip(slopes.player_id,slopes.n)))
        t=t.merge(slopes[['player_id','games','information']],on='player_id')
        t['season']=int(yr);summ.append(t)
    pd.concat(summ).to_csv(ROOT/'results/tables/batter_adaptation_posteriors.csv',index=False)
    log('Model fitting complete')


def posterior_row(p,label,year,n):
    mu=np.asarray(p['mu']);tau=np.asarray(p['tau'])
    return {'specification':label,'season':int(year),'players':n,'mu_median':np.median(mu),'mu_low':np.quantile(mu,.05),'mu_high':np.quantile(mu,.95),'p_mu_positive':np.mean(mu>0),'tau_median':np.median(tau),'tau_low':np.quantile(tau,.05),'tau_high':np.quantile(tau,.95)}


def robustness(a):
    rows=[]
    def slopes(frame,**kwargs):
        return player_slopes(frame,min_n=CONFIG['minimum_player_observations'],min_games=CONFIG['minimum_player_games'],**kwargs)
    for yr,g in a.groupby('season'):
        for x in EXPOSURES:
            b=slopes(g,x=f'log_{x}')
            p=load_posterior(f'adaptation_{int(yr)}') if x=='exposure_pitcher' else fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED)
            rows.append(posterior_row(p,x,yr,len(b)))
        for form in ['linear_exposure_pitcher','saturating_exposure_pitcher']:
            b=slopes(g,x=form);p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED)
            rows.append(posterior_row(p,form,yr,len(b)))
        for comp in CONFIG['swing_components']:
            b=slopes(g,y=f'absz_{comp}')
            p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED)
            rows.append(posterior_row(p,f'component:{comp}',yr,len(b)))
        # Permute residual exposure within hitter-game, preserving game's selection.
        rng=np.random.default_rng(SEED); sh=g.copy()
        sh['adj_log_exposure_pitcher']=sh.groupby(['batter','game_pk'])['adj_log_exposure_pitcher'].transform(lambda x:rng.permutation(x.to_numpy()))
        b=slopes(sh); p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED)
        rows.append(posterior_row(p,'shuffled_within_batter_game',yr,len(b)))
        for label,mask in [('similar_shape_recent',g.similar_pitch_gap.le(3)),('competitive_swing_speed_50',g.bat_speed.ge(50)),('early_in_game',g.inning.le(6)),('first_three_innings',g.inning.le(3))]:
            b=slopes(g.loc[mask])
            if len(b)>10:
                p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED)
                rows.append(posterior_row(p,label,yr,len(b)))
        b=slopes(g)
        for scale in [.075,.3]:
            p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=SEED,tau_scale=scale)
            rows.append(posterior_row(p,f'prior_tau_scale:{scale}',yr,len(b)))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/robustness.csv',index=False)


def validation():
    from src.evaluation.recovery import run_simulations
    a=pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'data/processed').glob('analytic_*.parquet'))],ignore_index=True)
    log('Alternative exposures, component definitions, negative controls, restrictions and priors')
    robustness(a)
    t1=pd.read_csv(ROOT/'results/tables/first_stage_2025.csv');t2=pd.read_csv(ROOT/'results/tables/first_stage_2026.csv')
    j=t1.merge(t2,on='player_id',suffixes=('_a','_b'))
    log(f'Measurement-error-aware temporal persistence for {len(j)} shared hitters')
    per=measurement_error_persistence(j.estimate_a,j.se_a,j.estimate_b,j.se_b,bootstrap=100,seed=SEED)
    per['shared_players']=len(j);per['naive_spearman']=float(spearmanr(j.estimate_a,j.estimate_b).statistic)
    write_json(ROOT/'results/tables/persistence.json',per)
    j.to_csv(ROOT/'results/tables/persistence_pairs.csv',index=False)
    # Does knowing individual slopes improve later-season within-player predictions?
    p=load_posterior('adaptation_2025'); pm=dict(zip(p['player_ids'],p['lambda'].mean(axis=0)))
    h=a.loc[a.season.eq(2026)&a.batter.isin(pm)].copy()
    xx=h.adj_log_exposure_pitcher-h.groupby('batter').adj_log_exposure_pitcher.transform('mean')
    yy=h.adj_distortion-h.groupby('batter').adj_distortion.transform('mean')
    pred_ind=-h.batter.map(pm).to_numpy()*xx;pred_pop=-p['mu'].mean()*xx
    loss=pd.DataFrame({'game':h.game_pk,'pop':(yy-pred_pop)**2,'individual':(yy-pred_ind)**2}).groupby('game').agg(pop=('pop','sum'),individual=('individual','sum'),n=('pop','size'))
    rng=np.random.default_rng(SEED); ix=rng.integers(0,len(loss),(1000,len(loss)))
    gain=100*(loss['pop'].to_numpy()[ix].sum(axis=1)-loss['individual'].to_numpy()[ix].sum(axis=1))/loss['pop'].to_numpy()[ix].sum(axis=1)
    pred={'n':len(h),'mse_improvement_pct':float(100*(loss['pop'].sum()-loss['individual'].sum())/loss['pop'].sum()),'ci90':np.quantile(gain,[.05,.95]).tolist(),'note':'Within-player centered later-season residual loss; conditional slope transport, not prospective raw-swing prediction.'}
    write_json(ROOT/'results/tables/holdout_prediction.json',pred)
    # Outcome validity: relation to whiff, and miss distance conditional on whiff.
    a['miss_distance_on_whiff']=a.miss_distance.where(a.is_whiff)
    a['distortion_bin']=pd.qcut(a.distortion,10,duplicates='drop')
    outcome=a.groupby(['season','distortion_bin'],observed=True).agg(n=('game_pk','size'),deviation=('distortion','mean'),whiff_rate=('is_whiff','mean'),miss_distance_whiffs=('miss_distance_on_whiff','mean')).reset_index()
    outcome['distortion_bin']=outcome.distortion_bin.astype(str)
    outcome.to_csv(ROOT/'results/tables/outcome_link.csv',index=False)
    selection=a.groupby(['season','exposure_pitcher']).agg(n=('batter','size'),hitters=('batter','nunique'),speed=('release_speed','mean'),zone_z=('zone_z','mean'),strikes=('strikes','mean'),inning=('inning','mean'),distortion=('distortion','mean'),whiff_rate=('is_whiff','mean')).reset_index()
    selection.to_csv(ROOT/'results/tables/selection.csv',index=False)
    # Posterior predictive checking at the level the second-stage model describes.
    ppc=[]
    for yr in [2025,2026]:
        po=load_posterior(f'adaptation_{yr}');b=pd.read_csv(ROOT/f'results/tables/first_stage_{yr}.csv')
        obs=b.estimate.to_numpy();rep=po['observed_predictive']
        for nm,fn in [('mean',lambda z:np.mean(z,axis=-1)),('sd',lambda z:np.std(z,axis=-1)),('max_abs',lambda z:np.max(np.abs(z),axis=-1))]:
            rr=fn(rep);oo=float(fn(obs));ppc.append({'season':yr,'statistic':nm,'observed':oo,'rep_low':np.quantile(rr,.05),'rep_median':np.median(rr),'rep_high':np.quantile(rr,.95),'p_rep_greater':np.mean(rr>oo)})
    pd.DataFrame(ppc).to_csv(ROOT/'results/tables/posterior_predictive.csv',index=False)
    log('Parameter recovery and no-learning experiments (separate synthetic worlds)')
    sim=run_simulations(replicates=CONFIG['simulation_replicates'],players=30,draws=1000,seed=SEED)
    sim['replicates'].to_csv(ROOT/'results/tables/simulation_recovery.csv',index=False)
    sim['calibration'].to_csv(ROOT/'results/tables/pairwise_calibration.csv',index=False)
    write_json(ROOT/'results/tables/simulation_summary.json',sim['summary'])
    ex=sim['example'];np.savez_compressed(ROOT/'results/posterior/simulation_example.npz',truth=ex['truth'],estimates=ex['estimates'],ses=ex['ses'],counts=ex['counts'],posterior=ex['posterior']['lambda'])
    from src.evaluation.outcomes import evaluate_outcomes
    log('Outcome validity and exploratory future-performance utility')
    evaluate_outcomes(a,ROOT,SEED)
    log('Validation complete')


def freeze():
    paths=sorted(p for p in (ROOT/'data/raw').iterdir() if p.suffix in {'.parquet','.csv','.json'} and p.name not in {'statcast.parquet','snapshot.json'})+sorted((ROOT/'data/processed').glob('*.parquet'))
    entries=[]
    for p in paths:
        if p.name=='statcast.parquet':continue
        entries.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    write_json(ROOT/'data/checksums.json',entries)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['features','models','validation','freeze','all'],default='all',nargs='?');args=ap.parse_args()
    for key,fn in [('features',features),('models',models),('validation',validation),('freeze',freeze)]:
        if args.stage in [key,'all']:fn()
