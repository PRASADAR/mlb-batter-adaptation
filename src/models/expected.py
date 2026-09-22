"""Chronologically held-out swing baseline and cross-fitted nuisance regressions."""
from __future__ import annotations
import numpy as np
import hashlib, json
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.covariance import LedoitWolf
from src.features.sequences import model_frame


def fit_predict(train, test, target, history=False, seed=20260921, iterations=240, max_train=150000, cache_path=None, max_ctr_complexity=None, thread_count=4):
    """Fit only on train; scale multivariate responses only on train."""
    cols = [target] if isinstance(target,str) else list(target)
    if len(train)>max_train: train = train.sample(max_train,random_state=seed)
    xt,cat = model_frame(train,history)
    xv,_ = model_frame(test,history)
    y = train[cols].to_numpy(float)
    center = y.mean(axis=0); scale = y.std(axis=0).clip(.00001)
    yz = (y-center)/scale
    model = CatBoostRegressor(iterations=iterations,depth=6,learning_rate=.07,
        loss_function='MultiRMSE' if len(cols)>1 else 'RMSE',
        l2_leaf_reg=5,random_seed=seed,thread_count=thread_count,max_ctr_complexity=max_ctr_complexity,verbose=False,
        allow_writing_files=False,cat_features=cat)
    cache_settings={'parameters':model.get_params(),'features':list(xt.columns),'targets':cols}
    signature = hashlib.sha256(pd.util.hash_pandas_object(xt,index=False).to_numpy().tobytes()+y.tobytes()+json.dumps(cache_settings,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_path) if cache_path else None
    hit=False
    if cache and cache.exists() and cache.with_suffix('.json').exists():
        hit=json.loads(cache.with_suffix('.json').read_text()).get('signature')==signature
        if hit:model.load_model(str(cache))
    if not hit:
        model.fit(xt,yz if len(cols)>1 else yz.ravel())
        if cache:
            cache.parent.mkdir(parents=True,exist_ok=True)
            model.save_model(str(cache))
            cache.with_suffix('.json').write_text(json.dumps({'signature':signature,'columns':cols,'iterations':iterations,'seed':seed}))
    pred = np.asarray(model.predict(xv)).reshape(len(test),len(cols))*scale+center
    return pred,model,{'training_n':len(train),'training_start':str(train.game_date.min().date()),
                     'training_end':str(train.game_date.max().date()),'target_center':center.tolist(),'target_scale':scale.tolist()}


def swing_baseline(d,components,config):
    """2024 H1 and earlier fit; 2024 H2 covariance calibration; 2025/26 inference.

    Predictions for every analysis row and covariance row are out of training.
    Holdout 2026 is never used to choose features, covariance, or hyperparameters.
    """
    valid = d.is_swing & d[components].notna().all(axis=1)
    # Broad instrument sanity bounds, fixed before outcomes; no outcome trimming.
    valid &= d.bat_speed.between(20,110) & d.swing_length.between(1,12)
    s = d.loc[valid].copy()
    fit = s.game_date.lt('2024-07-01')
    cal = s.game_date.ge('2024-07-01') & s.game_date.lt('2025-01-01')
    analysis = s.game_date.ge('2025-01-01')
    if min(fit.sum(),cal.sum(),analysis.sum())<1000:
        raise ValueError('Snapshot must include baseline before July 2024, calibration July-Dec 2024 and analysis 2025+.')
    targets = list(components)
    if 'attack_direction' in targets:
        s['direction_sin'] = np.sin(np.radians(s.attack_direction))
        s['direction_cos'] = np.cos(np.radians(s.attack_direction))
        targets = [x for x in targets if x!='attack_direction']+['direction_sin','direction_cos']
    evaluation = s.loc[cal|analysis].copy()
    pred_a,model_a,meta_a = fit_predict(s.loc[fit],evaluation,targets,seed=config['seed'],iterations=config['catboost_iterations'],max_train=config['max_training_swings'],cache_path=Path(__file__).resolve().parents[2]/'results/models/current_fit.cbm')
    pred_b,model_b,meta_b = fit_predict(s.loc[fit],evaluation,targets,history=True,seed=config['seed'],iterations=config['catboost_iterations'],max_train=config['max_training_swings'],cache_path=Path(__file__).resolve().parents[2]/'results/models/history_fit.cbm')
    if 'attack_direction' in components:
        def restore(pred):
            return np.column_stack([np.degrees(np.arctan2(pred[:,-2],pred[:,-1])) if c=='attack_direction' else pred[:,targets.index(c)] for c in components])
        pred_a,pred_b=restore(pred_a),restore(pred_b)
    ra = evaluation[components].to_numpy()-pred_a
    rb = evaluation[components].to_numpy()-pred_b
    if 'attack_direction' in components:
        j=components.index('attack_direction')
        ra[:,j]=(ra[:,j]+180)%360-180
        rb[:,j]=(rb[:,j]+180)%360-180
    c = evaluation.game_date.lt('2025-01-01').to_numpy()
    # Whiten standardized residuals to avoid shrinking unlike physical units together.
    scale = ra[c].std(axis=0).clip(.0001)
    cov = LedoitWolf().fit(ra[c]/scale)
    z = (ra/scale-cov.location_)
    dist = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',z,cov.precision_,z),0))
    # Keep physical residuals separately; headline metric is in calibration SD.
    evaluation['distortion_raw'] = dist
    evaluation['distortion'] = (dist-dist[c].mean())/dist[c].std()
    for j,k in enumerate(components):
        evaluation[f'expected_{k}'] = pred_a[:,j]
        evaluation[f'resid_{k}'] = ra[:,j]
        evaluation[f'absz_{k}'] = np.abs(ra[:,j]/scale[j])
        evaluation[f'sequence_resid_{k}'] = rb[:,j]
    metrics=[]
    for yr,g in evaluation.groupby('season'):
        for k in components:
            a=g[f'resid_{k}'].to_numpy(); b=g[f'sequence_resid_{k}'].to_numpy()
            # Bootstrap game-level squared loss differences (paired).
            losses=pd.DataFrame({'game':g.game_pk,'a':a*a,'b':b*b}).groupby('game').agg(a=('a','sum'),b=('b','sum'),n=('a','size'))
            rng=np.random.default_rng(config['seed']); ix=rng.integers(0,len(losses),(500,len(losses)))
            la=losses.a.to_numpy(); lb=losses.b.to_numpy()
            gain=100*(la[ix].sum(axis=1)-lb[ix].sum(axis=1))/la[ix].sum(axis=1)
            metrics.append({'season':int(yr),'component':k,'n':len(g),'rmse_current':np.sqrt(np.mean(a*a)),
                            'rmse_history':np.sqrt(np.mean(b*b)),'mse_gain_pct':100*(np.mean(a*a)-np.mean(b*b))/np.mean(a*a),
                            'gain_ci90_low':np.quantile(gain,.05),'gain_ci90_high':np.quantile(gain,.95)})
    metadata={'current_model':meta_a,'history_model':meta_b,'calibration_n':int(c.sum()),
              'components':components,'residual_scale':scale.tolist(),'covariance':cov.covariance_.tolist(),
              'calibration_distance_mean':float(dist[c].mean()),'calibration_distance_sd':float(dist[c].std()),
              'excluded_swings_missing_or_sanity':int((d.is_swing&~valid).sum())}
    return evaluation, pd.DataFrame(metrics),metadata,(model_a,model_b)


def nuisance_residuals(d,targets,config):
    """Development folds hold out whole games; final season uses development only.

    Outcomes and exposure are separately predicted by current+history features.
    Future exposure exists ONLY as a falsification target, never as a predictor.
    """
    out=d.copy()
    a=out.season.eq(2025); b=out.season.ge(2026)
    if a.sum()<1000 or b.sum()<1000: raise ValueError('Need both 2025 and 2026 for temporal validation.')
    p=np.full((len(out),len(targets)),np.nan)
    loc=np.flatnonzero(a.to_numpy()); dev=out.iloc[loc]
    for fold,(tr,te) in enumerate(GroupKFold(3).split(dev,groups=dev.game_pk)):
        print(f'Nuisance fold {fold+1}/3: {len(tr):,} train, {len(te):,} validation',flush=True)
        pp,_,_=fit_predict(dev.iloc[tr],dev.iloc[te],targets,history=True,
                          seed=config['seed']+fold,iterations=180,max_train=config['max_training_swings'],max_ctr_complexity=config.get('nuisance_max_ctr_complexity',2),thread_count=config.get('nuisance_threads',8))
        p[loc[te]]=pp
    print(f'Final nuisance fit: {len(dev):,} development, {int(b.sum()):,} temporal holdout',flush=True)
    pp,_,meta=fit_predict(dev,out.loc[b],targets,history=True,seed=config['seed'],iterations=180,max_train=config['max_training_swings'],max_ctr_complexity=config.get('nuisance_max_ctr_complexity',2),thread_count=config.get('nuisance_threads',8))
    p[np.flatnonzero(b.to_numpy())]=pp
    meta['nuisance_max_ctr_complexity']=config.get('nuisance_max_ctr_complexity',2)
    meta['development_prediction_diagnostics']={t:{'mse':float(np.mean((out.loc[a,t].to_numpy()-p[a.to_numpy(),j])**2)), 'r2':float(1-np.mean((out.loc[a,t].to_numpy()-p[a.to_numpy(),j])**2)/np.var(out.loc[a,t].to_numpy()))} for j,t in enumerate(targets)}
    for j,t in enumerate(targets): out[f'adj_{t}']=out[t].to_numpy()-p[:,j]
    return out,meta


def player_slopes(d,x='log_exposure_pitcher',y='distortion',min_n=40,min_games=8):
    """DML-style residual slope with player intercept and game-cluster sandwich.

    Positive lambda means less deviation as prior exposure increases. The
    likelihood in the next stage is approximate; nuisance-fit uncertainty is
    not fully propagated. Low-information players are excluded transparently.
    """
    rows=[]
    for player,g in d.groupby('batter',sort=True):
        z=g[[f'adj_{x}',f'adj_{y}','game_pk']].dropna()
        n=len(z); ng=z.game_pk.nunique()
        if n<min_n or ng<min_games: continue
        xx=z.iloc[:,0].to_numpy(dtype=float,copy=True); yy=z.iloc[:,1].to_numpy(dtype=float,copy=True)
        xx-=xx.mean(); yy-=yy.mean(); info=xx@xx
        if info<1.: continue
        slope=float(xx@yy/info); e=yy-slope*xx
        score=pd.Series(xx*e).groupby(z.game_pk.to_numpy()).sum().to_numpy()
        se=float(np.sqrt(ng/(ng-1)*(n-1)/(n-2)*(score@score)/(info**2)))
        if np.isfinite(se) and se>1e-6:
            rows.append({'player_id':int(player),'estimate':-slope,'se':se,'n':n,'games':ng,'information':info})
    return pd.DataFrame(rows)
