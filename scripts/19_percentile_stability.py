#!/usr/bin/env python3
"""Post-holdout exploratory search for reproducible percentile relationships.

Frozen expected-swing and nuisance predictions are reused, never refitted.
Specification choices and an internal partition check are fixed in this file.
Default notebook replay should read the saved results, not execute this search.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau, spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.expected import player_slopes
from src.models.hierarchical import _tau_grid

SEED=20260922
MIN_N=40
MIN_GAMES=8
HIGH_INFORMATION_N=500
BOOTSTRAPS=1000
PERMUTATIONS=19999
DISCOVERY_PERMUTATIONS=9999
METHODS=['unpooled','pooled']
OUTCOMES=['distortion','absz_attack_angle','absz_attack_direction','absz_swing_path_tilt','absz_bat_speed','absz_swing_length']
EXPOSURES=['log_exposure_pitcher','log_exposure_pa','log_exposure_type','log_kernel_exposure','log_exposure_sequence',
 'log_kernel_h0.5_m5','log_kernel_h0.5_m20','log_kernel_h1_m5','log_kernel_h2_m5','log_kernel_h2_m20',
 'linear_exposure_pitcher','saturating_exposure_pitcher']
ENDPOINTS=['spearman','kendall','quartile_trend','decile_trend','top_quartile_retention']


def specs():
    out=[]
    for y in OUTCOMES:
        for x in EXPOSURES:
            out.append(dict(specification=f'{y}__{x}',x=x,y=y,restriction='all',high_information=False,negative_control=False))
    for restriction in ['speed50','innings6','innings3','recent_similar']:
        out.append(dict(specification=f'distortion__log_exposure_pitcher__{restriction}',x=EXPOSURES[0],y='distortion',restriction=restriction,high_information=False,negative_control=False))
    for x in ['log_future_exposure','log_irrelevant_exposure']:
        out.append(dict(specification=f'distortion__{x}',x=x,y='distortion',restriction='all',high_information=False,negative_control=True))
    for y in OUTCOMES:
        out.append(dict(specification=f'{y}__log_exposure_pitcher__n500',x=EXPOSURES[0],y=y,restriction='all',high_information=True,negative_control=False))
    assert len(out)==84
    return out


def write_json(path,value):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
        if isinstance(x,np.generic): x=x.item()
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    q=np.full(len(p),np.nan)
    valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid],kind='stable')]
    if len(order):
        adjusted=p[order]*len(order)/np.arange(1,len(order)+1)
        q[order]=np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def percentiles(values):
    a=np.asarray(values,dtype=float)
    return (rankdata(a,method='average')-.5)/len(a)


def correlation(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x.mean();y=y-y.mean()
    denominator=np.sqrt((x@x)*(y@y))
    return float(x@y/denominator) if denominator>0 else np.nan


def exact_pool(slopes):
    """Deterministic posterior means under the frozen normal hierarchy.

    Integrates the same priors as fit_hierarchy; avoids ranking Monte Carlo
    jitter when player distributions shrink almost to the common mean.
    """
    b=slopes.estimate.to_numpy(float);s=slopes.se.to_numpy(float)
    upper=max(.6,2*np.std(b),.05);n_grid=2401
    for _ in range(12):
        grid,density,weights,mu,_=_tau_grid(b,s,.2,.15,n_grid,upper)
        if density[-1]<1e-10 and weights[int(.99*n_grid):].sum()<1e-7: break
        upper*=2
    else: raise RuntimeError('Could not bound hierarchy quadrature tail.')
    for _ in range(5):
        if weights[:2].sum()<=.025: break
        n_grid=2*n_grid-1
        grid,density,weights,mu,_=_tau_grid(b,s,.2,.15,n_grid,upper)
    if weights[:2].sum()>.025: raise RuntimeError('Unresolved small-variance hierarchy posterior.')
    shrink=grid[:,None]**2/(grid[:,None]**2+s[None,:]**2)
    mean=np.einsum('i,ij->j',weights,shrink*b[None,:]+(1-shrink)*mu[:,None])
    fine,_,fw,fmu,_=_tau_grid(b,s,.2,.15,2*n_grid-1,upper)
    fs=fine[:,None]**2/(fine[:,None]**2+s[None,:]**2)
    fm=np.einsum('i,ij->j',fw,fs*b[None,:]+(1-fs)*fmu[:,None])
    difference=float(np.max(np.abs(mean-fm)))
    if difference>1e-6: raise RuntimeError('Posterior mean quadrature failed its refinement check.')
    out=slopes.copy()
    out['pooled_mean']=fm
    out['expected_shrinkage']=np.einsum('i,ij->j',fw,fs)
    out['unpooled_percentile']=percentiles(out.estimate)
    out['pooled_percentile']=percentiles(out.pooled_mean)
    meta=dict(players=len(out),mu_mean=float(fw@fmu),tau_mean=float(fw@fine),median_shrinkage=float(out.expected_shrinkage.median()),
              quadrature_max_mean_difference=difference,mu_prior_sd=.2,tau_halfnormal_scale=.15)
    return out,meta


def select_rows(frame,spec):
    mask=np.ones(len(frame),dtype=bool)
    if spec['restriction']=='speed50': mask=frame.bat_speed.ge(50)
    elif spec['restriction']=='innings6': mask=frame.inning.le(6)
    elif spec['restriction']=='innings3': mask=frame.inning.le(3)
    elif spec['restriction']=='recent_similar': mask=frame.similar_pitch_gap.le(3)
    return frame.loc[mask,['batter','game_pk',f"adj_{spec['x']}",f"adj_{spec['y']}"]]


def fit_scores(frame,period,all_specs):
    scores={};metas=[]
    for index,spec in enumerate(all_specs):
        if spec['high_information']: continue
        subset=select_rows(frame,spec)
        slopes=player_slopes(subset,x=spec['x'],y=spec['y'],min_n=MIN_N,min_games=MIN_GAMES)
        if len(slopes)<10:
            scores[spec['specification']]=slopes
            metas.append(dict(period=period,**spec,status='fewer_than_10_eligible_players',eligible_players=len(slopes),analytic_swings=len(subset)))
            continue
        table,meta=exact_pool(slopes)
        scores[spec['specification']]=table
        metas.append(dict(period=period,**spec,status='ok',eligible_players=len(table),analytic_swings=len(subset),**meta))
        if (index+1)%12==0: print(f'{period}: {index+1} specifications fitted.',flush=True)
    return scores,metas


def match_scores(scores_a,scores_b,spec):
    key=spec['specification'].removesuffix('__n500') if spec['high_information'] else spec['specification']
    a,b=scores_a[key],scores_b[key]
    if len(a)<10 or len(b)<10: return pd.DataFrame()
    matched=a.merge(b,on='player_id',suffixes=('_a','_b'),validate='one_to_one')
    if spec['high_information']:
        matched=matched.loc[matched.n_a.ge(HIGH_INFORMATION_N)&matched.n_b.ge(HIGH_INFORMATION_N)].copy()
    return matched


def observed_endpoints(x,y,pa,pb):
    quartile=np.minimum(4,np.floor(pa*4).astype(int)+1)
    decile=np.minimum(10,np.floor(pa*10).astype(int)+1)
    top=pa>=.75
    return np.array([correlation(rankdata(x),rankdata(y)),kendalltau(x,y).statistic,
        correlation(quartile,pb),correlation(decile,pb),np.mean(pb[top]>=.75) if top.any() else np.nan])


def rank_statistics(pairs,method,seed,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS,full=True):
    column='estimate' if method=='unpooled' else 'pooled_mean'
    x=pairs[f'{column}_a'].to_numpy(float);y=pairs[f'{column}_b'].to_numpy(float)
    pa=pairs[f'{method}_percentile_a'].to_numpy(float);pb=pairs[f'{method}_percentile_b'].to_numpy(float)
    n=len(x);rng=np.random.default_rng(seed)
    obs=observed_endpoints(x,y,pa,pb)
    if not full:
        obs=obs[:1]
    null=np.empty((permutations,len(obs)))
    xr=rankdata(x);yr=rankdata(y)
    xs=xr-xr.mean();ys=yr-yr.mean();scale=np.sqrt((xs@xs)*(ys@ys))
    q=np.minimum(4,np.floor(pa*4).astype(int)+1).astype(float);q-=q.mean()
    d=np.minimum(10,np.floor(pa*10).astype(int)+1).astype(float);d-=d.mean()
    pc=pb-pb.mean();qscale=np.sqrt((q@q)*(pc@pc));dscale=np.sqrt((d@d)*(pc@pc))
    top=pa>=.75
    for offset in range(0,permutations,256):
        size=min(256,permutations-offset)
        indices=rng.permuted(np.tile(np.arange(n),(size,1)),axis=1)
        null[offset:offset+size,0]=np.einsum('ij,j->i',ys[indices],xs)/scale if scale>0 else np.nan
        if full:
            null[offset:offset+size,1]=[kendalltau(xr,yr[idx]).statistic for idx in indices]
            null[offset:offset+size,2]=np.einsum('ij,j->i',pc[indices],q)/qscale if qscale>0 else np.nan
            null[offset:offset+size,3]=np.einsum('ij,j->i',pc[indices],d)/dscale if dscale>0 else np.nan
            null[offset:offset+size,4]=(pb[indices][:,top]>=.75).mean(axis=1) if top.any() else np.nan
    boot=np.empty((bootstraps,len(obs)))
    for j in range(bootstraps):
        idx=rng.integers(0,n,n)
        boot[j]=observed_endpoints(x[idx],y[idx],pa[idx],pb[idx])[:len(obs)]
    center=np.zeros(len(obs))
    if full: center[-1]=np.mean(pb>=.75)
    output=[]
    for j,endpoint in enumerate(ENDPOINTS[:len(obs)]):
        valid_boot=boot[np.isfinite(boot[:,j]),j]
        valid_null=null[np.isfinite(null[:,j]),j]
        estimable=np.isfinite(obs[j]) and len(valid_null)>0
        low,high=np.quantile(valid_boot,[.05,.95]) if len(valid_boot) and np.isfinite(obs[j]) else [np.nan,np.nan]
        output.append(dict(endpoint=endpoint,estimate=float(obs[j]),ci90_low=float(low),ci90_high=float(high),
            permutation_p_positive=float((1+(valid_null>=obs[j]-1e-14).sum())/(len(valid_null)+1)) if estimable else np.nan,
            permutation_p_two_sided=float((1+(np.abs(valid_null-center[j])>=abs(obs[j]-center[j])-1e-14).sum())/(len(valid_null)+1)) if estimable else np.nan,
            permutation_null_mean=float(center[j]),permutations=permutations,bootstrap_replicates=bootstraps))
    diagnostics=dict(shared_players=n,median_n_a=float(pairs.n_a.median()),median_n_b=float(pairs.n_b.median()),
        min_n_a=int(pairs.n_a.min()),min_n_b=int(pairs.n_b.min()),median_games_a=float(pairs.games_a.median()),median_games_b=float(pairs.games_b.median()),
        median_information_a=float(pairs.information_a.median()),median_information_b=float(pairs.information_b.median()),
        median_se_a=float(pairs.se_a.median()),median_se_b=float(pairs.se_b.median()),
        median_shrinkage_a=float(pairs.expected_shrinkage_a.median()),median_shrinkage_b=float(pairs.expected_shrinkage_b.median()),
        score_se_spearman_a=float(spearmanr(x,pairs.se_a).statistic),score_se_spearman_b=float(spearmanr(y,pairs.se_b).statistic),
        se_stability_spearman=float(spearmanr(pairs.se_a,pairs.se_b).statistic),top_quartile_baseline_n=int(top.sum()))
    return output,diagnostics


def adjust_family(frame):
    frame=frame.copy()
    frame['bh_q_positive']=bh_adjust(frame.permutation_p_positive)
    frame['bh_q_two_sided']=bh_adjust(frame.permutation_p_two_sided)
    m=int(frame.permutation_p_positive.notna().sum())
    frame['by_q_positive']=np.minimum(1,frame.bh_q_positive*np.sum(1/np.arange(1,m+1)))
    frame['family_tests']=m
    return frame


def group_summaries(pairs,method,seed):
    pa=pairs[f'{method}_percentile_a'].to_numpy();pb=pairs[f'{method}_percentile_b'].to_numpy()
    rng=np.random.default_rng(seed);rows=[]
    for bins in [4,10]:
        group=np.minimum(bins,np.floor(pa*bins).astype(int)+1)
        for level in range(1,bins+1):
            y=pb[group==level]
            if len(y):
                means=y[rng.integers(0,len(y),(BOOTSTRAPS,len(y)))].mean(axis=1)
                lo,hi=np.quantile(means,[.05,.95])
                mean=float(y.mean());top=float((y>=.75).mean())
            else: mean=lo=hi=top=np.nan
            rows.append(dict(bins=bins,baseline_group=level,n=len(y),mean_followup_percentile=mean,
                             mean_percentile_ci90_low=float(lo),mean_percentile_ci90_high=float(hi),top_quartile_followup_rate=top))
    return rows


def select_candidate(discovery):
    candidates=discovery.loc[~discovery.negative_control & discovery.estimate.gt(0)].copy()
    if not len(candidates): return None
    candidates=candidates.sort_values(['bh_q_positive','permutation_p_positive','estimate','specification','method'],ascending=[True,True,False,True,True],kind='stable')
    return candidates.iloc[0].to_dict()


def partial_rank_correlation(x,y,controls):
    """Partial Spearman association: rank scores and log-count/SE controls."""
    x=rankdata(x);y=rankdata(y);controls=np.asarray(controls,dtype=float)
    controls=np.column_stack([rankdata(column) for column in controls.T])
    center=controls.mean(axis=0);scale=controls.std(axis=0)
    design=np.column_stack([np.ones(len(x)),(controls-center)/np.where(scale>0,scale,1)])
    xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    yr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    return correlation(xr,yr)


def precision_diagnostics(pairs,summary,selection):
    """Selected cases only; no new searched hypothesis or p-value family."""
    roles={'distortion__log_exposure_pitcher':['primary']}
    if selection is not None:
        roles.setdefault(selection['specification'],[]).append('discovery_selected')
    pooled=summary.loc[summary.endpoint.eq('spearman')&summary.method.eq('pooled')&~summary.negative_control&summary.estimate.gt(0)]
    if len(pooled):
        best=pooled.sort_values(['bh_q_positive','permutation_p_positive','estimate','specification'],ascending=[True,True,False,True],kind='stable').iloc[0]
        roles.setdefault(best.specification,[]).append('best_full_sample_pooled_by_q_then_p_descriptive')
    rows=[]
    for spec,role in roles.items():
        match=pairs.loc[pairs.specification.eq(spec)]
        if len(match)<20: continue
        controls=np.log(match[['n_a','n_b','se_a','se_b']].to_numpy(float))
        for method in METHODS:
            column='estimate' if method=='unpooled' else 'pooled_mean'
            x=match[f'{column}_a'].to_numpy();y=match[f'{column}_b'].to_numpy()
            estimate=partial_rank_correlation(x,y,controls)
            rng=np.random.default_rng(SEED+int(hashlib.sha256(f'{spec}:{method}'.encode()).hexdigest()[:8],16))
            draws=[]
            for _ in range(BOOTSTRAPS):
                idx=rng.integers(0,len(match),len(match))
                draws.append(partial_rank_correlation(x[idx],y[idx],controls[idx]))
            draws=np.asarray(draws);draws=draws[np.isfinite(draws)]
            low,high=np.quantile(draws,[.05,.95]) if len(draws) else [np.nan,np.nan]
            rows.append(dict(specification=spec,role=';'.join(role),method=method,shared_players=len(match),
                unadjusted_spearman=correlation(rankdata(x),rankdata(y)),partial_rank_correlation=estimate,
                partial_ci90_low=float(low),partial_ci90_high=float(high),controls='log n_2025, log n_2026, log SE_2025, log SE_2026',
                bootstrap_replicates=BOOTSTRAPS,scope='Post-selection descriptive diagnostic; conditional score bootstrap, no selective inference or new validation claim.'))
    return pd.DataFrame(rows)


def run_comparisons(a,b,all_specs,stage,full=True):
    rows=[];groups=[];matches=[]
    for index,spec in enumerate(all_specs):
        pairs=match_scores(a,b,spec)
        if len(pairs)<20:
            rows.append(dict(stage=stage,**spec,method='not_estimable',endpoint='spearman',shared_players=len(pairs),estimate=np.nan,
                permutation_p_positive=np.nan,permutation_p_two_sided=np.nan,status='fewer_than_20_shared_eligible_players'))
            continue
        for j,method in enumerate(METHODS):
            seed=SEED+index*31+j+(100000 if stage=='full_year' else 200000 if stage=='confirmation' else 0)
            stats,diag=rank_statistics(pairs,method,seed,permutations=PERMUTATIONS if full else DISCOVERY_PERMUTATIONS,
                                       bootstraps=BOOTSTRAPS if full else 500,full=full)
            for value in stats: rows.append(dict(stage=stage,**spec,method=method,**value,**diag,status='ok'))
            if full:
                for value in group_summaries(pairs,method,seed+7): groups.append(dict(stage=stage,**spec,method=method,**value))
        if stage=='full_year':
            save=pairs.copy();save['specification']=spec['specification'];matches.append(save)
        if (index+1)%6==0: print(f'{stage}: {index+1}/{len(all_specs)} comparisons complete.',flush=True)
    return adjust_family(pd.DataFrame(rows)),pd.DataFrame(groups),pd.concat(matches,ignore_index=True) if matches else pd.DataFrame()


def save_scores(scores,period,accumulator):
    for spec,table in scores.items():
        if len(table):
            t=table.copy();t['specification']=spec;t['period']=period;accumulator.append(t)


def report(summary,discovery,confirmation,selection,metadata,precision=None):
    rank=summary.loc[summary.endpoint.eq('spearman')&~summary.negative_control].sort_values(['bh_q_positive','estimate'],ascending=[True,False])
    significant=rank.loc[rank.estimate.gt(0)&rank.bh_q_positive.le(.05)]
    top=rank.iloc[0]
    selected='No positive discovery candidate was available.' if selection is None else f"The fixed discovery rule selected `{selection['specification']}` using {selection['method']} scores: discovery Spearman {selection['estimate']:.3f}, BH q={selection['bh_q_positive']:.4f}."
    checks=[]
    for _,r in confirmation.loc[confirmation.endpoint.eq('spearman')].iterrows():
        checks.append(f"- `{r.specification}`, {r.method}: rho={r.estimate:.3f}, 90% player-bootstrap interval [{r.ci90_low:.3f}, {r.ci90_high:.3f}], permutation p={r.permutation_p_positive:.5f}, BH q={r.bh_q_positive:.4f}, n={int(r.shared_players)}.")
    negative=summary.loc[summary.negative_control&summary.endpoint.eq('spearman')]
    control='; '.join(f"{r.specification.replace('distortion__','')} {r.method}: rho={r.estimate:.3f}, q={r.bh_q_positive:.4f}" for _,r in negative.iterrows())
    focal=summary.loc[summary.specification.isin(['distortion__log_exposure_pitcher','absz_swing_length__log_exposure_pitcher','absz_swing_length__log_exposure_pitcher__n500'])&summary.endpoint.eq('spearman')]
    focal_text='\n'.join(f"- `{r.specification}`, {r.method}: rho={r.estimate:.3f}, 90% interval [{r.ci90_low:.3f}, {r.ci90_high:.3f}], BH q={r.bh_q_positive:.4f}, n={int(r.shared_players)}." for _,r in focal.iterrows())
    by_count=int((summary.estimate.gt(summary.permutation_null_mean)&summary.by_q_positive.le(.05)).sum())
    precision_text=''
    if precision is not None and len(precision):
        precision_text='\n'.join(f"- `{r.specification}`, {r.method}: unadjusted rho={r.unadjusted_spearman:.3f}; precision-adjusted rank correlation={r.partial_rank_correlation:.3f}, conditional 90% interval [{r.partial_ci90_low:.3f}, {r.partial_ci90_high:.3f}]." for _,r in precision.iterrows())
    return f'''# Percentile stability: a post-holdout exploratory search

## Status and question

This analysis was requested after the original 2026 validation had been inspected. It is an explicitly exploratory search across a fixed set of 84 specifications, not a replacement for the frozen primary analysis or a new untouched holdout. It asks whether relative positions and broad groups are more reproducible than individual slope magnitudes. Every specification and null result is retained.

The file label `full_year` denotes all available records within each season: the complete 2025 regular season versus the 2026 regular season through September 20. It does not denote a completed 2026 season.

A monotone percentile transformation preserves ordering. Consequently, converting an unpooled slope into its empirical percentile leaves Spearman and Kendall correlation unchanged. The distinct comparisons here are partial-pooling changes to the order, coarse percentile groups, different fixed exposure/outcome definitions, and a higher-information population.

## Fixed search space and estimation

The search crosses 12 previously specified exposure transformations with six saved outcomes (Mahalanobis distortion and five component absolute standardized residuals), giving 72 combinations. Four existing restrictions use the primary exposure and distortion outcome: bat speed at least 50 mph, innings at most six, innings at most three, and a similar pitch within three preceding pitches. Two negative controls use future and physically irrelevant exposure. Six full-sample sensitivities require at least 500 contributing swings in each season for the primary exposure and each outcome. Total: 84 specifications. No cutoff is selected from 2026 results.

The ordinary eligibility rule is 40 swings and eight games per period, with residual exposure information at least one, matching the first-stage estimator. At least 20 shared eligible hitters are needed for a rank comparison. Positive scores mean a decrease in the stated residual outcome with increasing stated exposure. Same-pitcher exposure is specifically the count of prior pitches of the same pitch type from that pitcher to the hitter in the game. It is not a causal learning fraction.

Game-cluster slope SEs and frozen nuisance-adjusted quantities are reused. Expected-swing and nuisance models are not refitted. Unpooled estimates are compared with deterministic posterior means under the same Normal population mean prior (SD 0.2) and HalfNormal heterogeneity prior (scale 0.15) as the primary model. Quadrature is refined and checked, avoiding Monte Carlo noise in tiny shrunken rank differences. Rankings of posterior means still do not imply well-separated latent player distributions.

Within each period, percentile = (average rank − 0.5) / number of eligible hitters. Quartile and decile cutpoints are fixed at the usual fractions using the earlier period alone; later eligibility never determines the earlier ranks or cutpoints. The 500-swing sensitivity retains the same earlier-period ranking definition and hierarchy, but restricts the matched comparison population in both years.

Conditional miss distance is omitted from this frozen-model search. There is no saved nuisance-adjusted miss-distance outcome, and regressions on raw miss distance would not provide the same current-pitch-conditioned estimand. Obtaining it requires a separately specified nuisance fit, outside these 84 cases.

## Inference and multiplicity

Each full-sample specification has two score implementations and five endpoints: Spearman correlation, Kendall correlation, an ordered quartile trend, an ordered decile trend, and top-quartile retention. Ordered group trends correlate the earlier group number with the later percentile. This creates **{int(summary.family_tests.max())} estimable endpoint tests** in one full-sample multiplicity family. The group trends use their ordinal spacing and are supplementary to rank correlations.

All endpoints have {BOOTSTRAPS} paired-player bootstrap resamples and 90% intervals, conditional on estimated scores and group definitions. They do not propagate nuisance-model or hierarchy-refit uncertainty. Permutations ({PERMUTATIONS} per full-sample comparison) shuffle later-period player labels within the matched population. Positive-direction and two-sided Monte Carlo p values include the standard plus-one correction. For retention, the null is the later top-quartile fraction in that matched population, which need not equal 25% after roster selection. BH q values cover all searched endpoints; a BY adjustment is also saved as a conservative dependence sensitivity. Intervals are pointwise, not simultaneous. Player resampling does not account for shared-game or fitted-model dependence across hitters.

## Internal partition check

Before computing these new rank results, discovery was restricted to March–June 2025 versus March–June 2026. A single candidate and score method were selected by the smallest positive-direction discovery BH q, then p, then largest positive Spearman, with deterministic ties. Negative controls and the full-sample-only high-information sensitivity were excluded from candidate selection. The discovery multiplicity family includes both score implementations for all 78 applicable specifications.

Only after that selection was saved were July–September 2025 versus July–September 2026 comparisons computed for the candidate specification and the primary specification, with both score methods reported and a separate confirmation BH family. These partition-specific rank comparisons had not previously been computed. The underlying observations and full-sample 2026 results had already contributed to earlier analyses, and 2025 nuisance fits share information across halves. This is an internal replication diagnostic on disjoint game records, not independent external confirmation or an untouched holdout. Full-sample search results were computed after this internal selection and check.

{selected}

{chr(10).join(checks)}

## Full-sample results

There are **{len(significant)} positive non-control Spearman results with BH q at most 0.05**. The leading full-sample non-control Spearman entry under the report sorting rule (smallest BH q, then largest rho) is `{top.specification}` ({top.method}): rho={top.estimate:.3f}, 90% interval [{top.ci90_low:.3f}, {top.ci90_high:.3f}], p={top.permutation_p_positive:.5f}, BH q={top.bh_q_positive:.4f}, n={int(top.shared_players)}. This entry is descriptive of the searched family and is not retrospectively substituted for the candidate selected in discovery.

The primary outcome and a useful component comparison are:

{focal_text}

The positive component associations are small and appear in unpooled scores as well as pooled scores. Swing length also has a positive association in the fixed population with at least 500 contributing swings in both seasons. The corresponding primary distortion comparison remains negative. **{by_count} of the 840 full-sample endpoints survive positive-direction BY adjustment at 0.05.** Ordinary BH results are therefore sensitive to the chosen treatment of dependence in this large, overlapping search family.

Negative-control rank results: {control}.

The future-exposure control has a positive pooled association at least as large as the strongest non-control association selected by the report sorting rule. The early-season discovery candidate fails the late-season internal replication check under both score implementations. Together these results limit the interpretation: the search finds modest descriptive component associations, but does not validate a stable latent adaptability ranking. Percentile conversion does not repair the primary persistence result, and the selected component findings warrant a new external test rather than a positive causal conclusion.

The full tables include eligible and shared player counts, median contributing swings/games, residual exposure information, SEs, shrinkage, score–SE correlations, and SE persistence. Stable precision can alter pooled rankings; a pooled rank association requires scrutiny alongside unpooled ranks, the 500-swing sensitivity, negative controls, and the internal partition check. A positive full-sample correlation alone does not establish a persistent causal adaptation skill.

## Secondary precision diagnostic

Only the primary specification, the discovery-selected candidate, and the full-sample pooled specification with the smallest BH q, then p, then largest rho are examined here. The additional p-value tie-breaker selects a different case from the report sorting rule above. This is a descriptive post-selection check, not another search or confirmatory test. Scores and both seasons' log contributing-swing counts and log first-stage SEs are rank-transformed. Earlier and later score ranks are separately residualized against an intercept and these ranked controls; their residual correlation is reported. Paired-player bootstrap resampling refits these small regressions but does not refit the frozen nuisance models or hierarchy. Counts and SEs can themselves relate to player skill, so this adjustment neither proves confounding nor identifies a causal learning rate. Intervals do not correct for selecting these cases.

{precision_text}

## Files and reproduction

Run `python scripts/19_percentile_stability.py` to reproduce the bounded search. Ordinary notebook replay should read the saved tables and call `src.visualization.percentiles.render_percentile_stability`, which performs no model fitting or search.

- `results/tables/percentile_stability.csv`: all full-sample endpoint tests, estimates, intervals, and multiplicity corrections.
- `percentile_discovery.csv`, `percentile_confirmation.csv`, and `percentile_selection.json`: ordered internal search and replication results.
- `percentile_groups.csv`: all fixed quartile and decile groups with later percentile summaries.
- `percentile_player_scores.csv.gz` and `percentile_full_year_pairs.csv.gz`: player-level scores, uncertainty inputs, eligibility, and matched records.
- `percentile_hierarchy_diagnostics.csv`: exact pooling and information diagnostics for every fitted period/specification.
- `percentile_precision_diagnostics.csv`: selected-case partial rank associations after count/SE adjustment.
- `results/logs/percentile_stability_inputs.json`: frozen input hashes and fixed settings.
- `results/figures/percentile_stability.png` and `.pdf`: full-family rank results and the primary fixed-group comparison.

No frozen primary model, analytic dataset, or primary conclusion is overwritten by this script. This search can support a revision only if the observed evidence, uncertainty, multiplicity, and replication warrant it; null and contradictory results remain visible.
'''


def main():
    started=time.time();all_specs=specs();base_specs=[s for s in all_specs if not s['high_information']]
    out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True)
    columns=['game_date','game_pk','batter','season','bat_speed','inning','similar_pitch_gap']
    columns += sorted({f"adj_{s[k]}" for s in all_specs for k in ['x','y']})
    files=sorted((ROOT/'data/processed').glob('analytic_202[56]_*.parquet'))
    if len(files)!=14: raise ValueError('Expected all fourteen frozen analytic monthly partitions.')
    provenance=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files]
    settings=dict(label='post-holdout exploratory',seed=SEED,specifications=all_specs,minimum_n=MIN_N,minimum_games=MIN_GAMES,high_information_n=HIGH_INFORMATION_N,
        bootstraps=BOOTSTRAPS,permutations=PERMUTATIONS,discovery_permutations=DISCOVERY_PERMUTATIONS,
        discovery='2025 March–June versus 2026 March–June',confirmation='2025 July–September versus 2026 July–September',
        selection_rule='Smallest discovery positive-direction BH q, then p, then largest positive Spearman; exclude negative controls and full-sample-only n500 variants.',
        disclosure='Earlier aggregate 2026 results are already inspected; new partition comparisons are internal diagnostics, not untouched data.')
    write_json(ROOT/'results/logs/percentile_stability_inputs.json',dict(inputs=provenance,settings=settings))
    data=pd.concat([pd.read_parquet(p,columns=columns) for p in files],ignore_index=True)
    dates=pd.to_datetime(data.game_date)
    if data.groupby('game_pk').game_date.nunique().gt(1).any(): raise ValueError('A game has multiple dates; resolve partition assignment before search.')
    frames={f'{year}_{part}':data.loc[data.season.eq(year)&(dates.dt.month.le(6) if part=='early' else dates.dt.month.ge(7))]
            for year in [2025,2026] for part in ['early','late']}
    if set(frames['2025_early'].game_pk)&set(frames['2025_late'].game_pk) or set(frames['2026_early'].game_pk)&set(frames['2026_late'].game_pk):
        raise AssertionError('Discovery and confirmation share a game.')
    saved_scores=[];all_meta=[]
    print('Post-holdout exploratory discovery on fixed early-season partitions.',flush=True)
    a,ma=fit_scores(frames['2025_early'],'2025_early',base_specs)
    b,mb=fit_scores(frames['2026_early'],'2026_early',base_specs)
    all_meta+=ma+mb;save_scores(a,'2025_early',saved_scores);save_scores(b,'2026_early',saved_scores)
    discovery,_,_=run_comparisons(a,b,base_specs,'discovery',full=False)
    discovery.to_csv(out/'percentile_discovery.csv',index=False)
    candidate=select_candidate(discovery)
    write_json(out/'percentile_selection.json',dict(selected=candidate,settings=settings,selection_precedes_confirmation=True))
    primary=base_specs[0]
    keys={primary['specification']}
    if candidate is not None: keys.add(candidate['specification'])
    confirmation_specs=[s for s in base_specs if s['specification'] in keys]
    print('Running the reserved internal partition checks after fixing the candidate.',flush=True)
    ca,ma=fit_scores(frames['2025_late'],'2025_late',confirmation_specs)
    cb,mb=fit_scores(frames['2026_late'],'2026_late',confirmation_specs)
    all_meta+=ma+mb;save_scores(ca,'2025_late',saved_scores);save_scores(cb,'2026_late',saved_scores)
    confirmation,_,_=run_comparisons(ca,cb,confirmation_specs,'confirmation',full=False)
    confirmation.to_csv(out/'percentile_confirmation.csv',index=False)
    del a,b,ca,cb
    print('Computing the complete 84-specification full-sample family.',flush=True)
    a,ma=fit_scores(data.loc[data.season.eq(2025)],'2025_full',base_specs)
    b,mb=fit_scores(data.loc[data.season.eq(2026)],'2026_full',base_specs)
    all_meta+=ma+mb;save_scores(a,'2025_full',saved_scores);save_scores(b,'2026_full',saved_scores)
    summary,groups,pairs=run_comparisons(a,b,all_specs,'full_year',full=True)
    summary.to_csv(out/'percentile_stability.csv',index=False)
    groups.to_csv(out/'percentile_groups.csv',index=False)
    pairs.to_csv(out/'percentile_full_year_pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.concat(saved_scores,ignore_index=True).to_csv(out/'percentile_player_scores.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(all_meta).to_csv(out/'percentile_hierarchy_diagnostics.csv',index=False)
    precision=precision_diagnostics(pairs,summary,candidate)
    precision.to_csv(out/'percentile_precision_diagnostics.csv',index=False)
    metadata=dict(runtime_seconds=time.time()-started,full_year_tests=len(summary),specifications=len(all_specs),
        positive_spearman_bh_significant=int((summary.endpoint.eq('spearman')&~summary.negative_control&summary.estimate.gt(0)&summary.bh_q_positive.le(.05)).sum()),
        unique_shared_counts=sorted(summary.shared_players.unique().tolist()))
    write_json(out/'percentile_stability_summary.json',metadata)
    (ROOT/'paper/percentile_stability.md').write_text(report(summary,discovery,confirmation,candidate,metadata,precision))
    from src.visualization.percentiles import render_percentile_stability
    render_percentile_stability(ROOT)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
