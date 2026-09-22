"""Exploratory outcome validity, outcome-independent selection and future utility."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from src.features.sequences import model_frame


def summarize_player_outcomes(a):
    """Separate contact-quality xwOBA from exported strikeout zeros."""
    data=a.copy()
    data['batted_ball_xwoba']=data.estimated_woba_using_speedangle.where(data.description.eq('hit_into_play'))
    return data.groupby(['season','batter']).agg(n=('is_whiff','size'),whiff=('is_whiff','mean'),speed=('bat_speed','mean'),deviation=('distortion','mean'),xwoba=('batted_ball_xwoba','mean')).reset_index()


def evaluate_outcomes(a,root,seed=20260921):
    rows=[];tr=a.loc[a.season.eq(2025)];te=a.loc[a.season.eq(2026)]
    if len(tr)>150000:tr=tr.sample(150000,random_state=seed)
    for label,extra in [('current_pitch',[]),('plus_swing_deviations',[f'absz_{f}' for f in ['attack_angle','attack_direction','swing_path_tilt','bat_speed','swing_length']])]:
        xt,cat=model_frame(tr,extra_numeric=extra);xv,_=model_frame(te,extra_numeric=extra)
        model=CatBoostClassifier(iterations=160,depth=5,learning_rate=.08,verbose=False,thread_count=4,random_seed=seed,cat_features=cat,allow_writing_files=False)
        model.fit(xt,tr.is_whiff.astype(int));p=model.predict_proba(xv)[:,1]
        rows.append({'model':label,'train_season':2025,'test_season':2026,'test_n':len(te),'log_loss':log_loss(te.is_whiff,p),'auc':roc_auc_score(te.is_whiff,p)})
    pd.DataFrame(rows).to_csv(root/'results/tables/whiff_validation.csv',index=False)
    # Only train-season quantities enter this exploratory future-season regression.
    metrics=summarize_player_outcomes(a)
    x=metrics.loc[metrics.season.eq(2025)].merge(metrics.loc[metrics.season.eq(2026)],on='batter',suffixes=('_train','_test'))
    post=pd.read_csv(root/'results/tables/batter_adaptation_posteriors.csv')
    x=x.merge(post.loc[post.season.eq(2025),['player_id','posterior_mean']],left_on='batter',right_on='player_id')
    x=x.loc[(x.n_train>=100)&(x.n_test>=100)].copy()
    rows=[]
    for target in ['whiff','xwoba']:
        complete=x.dropna(subset=[f'{target}_train',f'{target}_test','speed_train','deviation_train','posterior_mean'])
        for label,cols in [('prior_performance',[f'{target}_train','speed_train','deviation_train']),('plus_adaptation',[f'{target}_train','speed_train','deviation_train','posterior_mean'])]:
            pred=np.zeros(len(complete));y=complete[f'{target}_test'].to_numpy()
            for i,j in KFold(5,shuffle=True,random_state=seed).split(complete):
                model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(complete.iloc[i][cols],y[i]);pred[j]=model.predict(complete.iloc[j][cols])
            rows.append({'target':target,'model':label,'n_players':len(complete),'mse':float(np.mean((y-pred)**2)),'scope':'Exploratory player-held-out regression predicting 2026 from 2025; no measurement-error propagation; xwOBA among batted balls only.'})
    pd.DataFrame(rows).to_csv(root/'results/tables/future_performance.csv',index=False)
    config=json.loads((root/'config/analysis.json').read_text())
    print('Whiff-conditional miss-distance sensitivity (censored endpoint)',flush=True)
    miss_distance_sensitivity(a,root,config)
    audit_tracking_selection(root,config['swing_components'])


def miss_distance_sensitivity(a,root,config):
    """Whiff-conditional sensitivity; selection precludes unconditional error claims."""
    from src.models.expected import nuisance_residuals,player_slopes
    from src.models.hierarchical import fit_hierarchy
    w=a.loc[a.is_whiff&a.miss_distance.notna()&a.miss_distance.ge(0)].copy().reset_index(drop=True)
    w['miss_error']=np.log1p(w.miss_distance)
    train=w.season.eq(2025)
    center=w.loc[train,'miss_error'].mean();scale=w.loc[train,'miss_error'].std()
    w['miss_error']=(w.miss_error-center)/scale
    w,_=nuisance_residuals(w,['miss_error','log_exposure_pitcher'],config)
    rows=[]
    for yr,g in w.groupby('season'):
        b=player_slopes(g,y='miss_error',min_n=config['minimum_player_observations'],min_games=config['minimum_player_games'])
        p=fit_hierarchy(b.estimate,b.se,b.player_id,draws=3000,seed=config['seed'])
        rows.append({'season':int(yr),'n_whiffs':len(g),'n_players':len(b),'mu_median':np.median(p['mu']),'mu_low':np.quantile(p['mu'],.05),'mu_high':np.quantile(p['mu'],.95),'scope':'Conditional on tracked whiffs; contacts are missing, not zero. Log-distance normalized to 2025 SD; not unconditional mechanical learning.'})
    pd.DataFrame(rows).to_csv(root/'results/tables/miss_distance_sensitivity.csv',index=False)


def audit_tracking_selection(root,components):
    rows=[]
    for yr in [2023,2024,2025,2026]:
        d=pd.read_parquet(root/f'data/processed/swings_{yr}.parquet')
        d['tracking_complete']=d[components].notna().all(axis=1)
        d['included']=d.tracking_complete&d.bat_speed.between(20,110)&d.swing_length.between(1,12)&d.shape_valid
        d['exposure_bucket']=d.exposure_pitcher.clip(upper=10)
        q=d.groupby(['season','exposure_bucket']).agg(swings=('batter','size'),tracking_complete_pct=('tracking_complete',lambda x:100*x.mean()),included_pct=('included',lambda x:100*x.mean()),whiff_rate=('is_whiff','mean'),mean_inning=('inning','mean')).reset_index()
        rows.append(q)
    pd.concat(rows).to_csv(root/'results/tables/tracking_selection.csv',index=False)
