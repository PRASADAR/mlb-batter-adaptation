"""Generate results notes and replace README's numerical results block."""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def j(p):return json.loads((ROOT/p).read_text())
s=j('results/logs/sample.json');r=pd.read_csv(ROOT/'results/tables/robustness.csv');p=j('results/tables/persistence.json');h=j('results/tables/holdout_prediction.json');sim=j('results/tables/simulation_summary.json')
seq=pd.read_csv(ROOT/'results/tables/sequence_distortion.csv');whiff=pd.read_csv(ROOT/'results/tables/whiff_validation.csv');utility=pd.read_csv(ROOT/'results/tables/future_performance.csv')
main=r.loc[r.specification.eq('exposure_pitcher')]
gain=h['mse_improvement_pct'];loss_verb='reduces' if gain>=0 else 'increases'
loss_interval=h['ci90'] if gain>=0 else [-h['ci90'][1],-h['ci90'][0]]
lines=[f"Analyzed **{s['raw_pitches']:,} pitch-level records across {s['raw_games']:,} games**, {s['start']} through {s['end']}. The primary parameter is a conditional exposure association, not a causal learning rate.", '', '| Period | Eligible hitters | Population correction association | 90% credible interval |','|---|---:|---:|---:|']
for _,x in main.iterrows():lines.append(f"| {int(x.season)} | {int(x.players)} | {x.mu_median:.4f} | [{x.mu_low:.4f}, {x.mu_high:.4f}] |")
lines+=['',f"Measurement-error-adjusted cross-season correlation: **{p['latent_correlation']:.3f}**, 90% player-bootstrap interval **[{p['bootstrap_ci90'][0]:.3f}, {p['bootstrap_ci90'][1]:.3f}]**, across {p['shared_players']} shared hitters. The interval crosses zero; {p['bootstrap_diagnostics']['abs_rho_above_0_98_count']}/{p['bootstrap_successes']} bootstrap fits have |correlation| > 0.98. These are descriptive bootstrap percentiles and may not have nominal coverage near boundaries.", '',f"Knowing individual 2025 slopes **{loss_verb}** 2026 conditional prediction MSE by **{abs(gain):.4f}%**, 90% interval **[{loss_interval[0]:.4f}%, {loss_interval[1]:.4f}%]**. This centers within heldout hitters and tests slope transport, not prospective raw-swing forecasting.",'','Negative controls must be read alongside the primary result:','','| 2026 exposure specification | Mean correction association | 90% interval |','|---|---:|---:|']
for name in ['future_exposure','irrelevant_exposure','shuffled_within_batter_game']:
    x=r.loc[r.specification.eq(name)&r.season.eq(2026)].iloc[0];lines.append(f"| {name} | {x.mu_median:.4f} | [{x.mu_low:.4f}, {x.mu_high:.4f}] |")
lines+=['',f"In {sim['replicates_per_scenario']} heterogeneous-rate experiments, nominal 90% intervals covered **{100*sim['heterogeneous']['mean_coverage90']:.2f}%** of known rates. Partial pooling lowered RMSE from **{sim['heterogeneous']['mean_unpooled_rmse']:.4f}** to **{sim['heterogeneous']['mean_rmse']:.4f}**. In zero-learning experiments, the population 90% interval excluded zero **{sim['zero_learning']['mu_excludes_zero_count']}/{sim['replicates_per_scenario']}** times. This validates only the simulated model; coverage is imperfect."]
block='\n'.join(lines)
if h['ci90'][1]<0:
    block+='\n\n**Primary-model conclusion:** Repeated exposure is associated with modestly reduced swing deviation, but the primary individual slopes do not support a persistent talent metric in this snapshot. Their held-out prediction is slightly worse than the population-slope benchmark. The improvement in whiff classification provides evidence that contemporaneous mechanics contain predictive information. It does not establish individual learning speed.'
rank=pd.read_csv(ROOT/'results/tables/percentile_stability.csv')
rank_meta=j('results/tables/percentile_stability_summary.json')
selected=j('results/tables/percentile_selection.json')['selected']
replication=pd.read_csv(ROOT/'results/tables/percentile_confirmation.csv')
screen=rank.loc[rank.endpoint.eq('spearman')&~rank.negative_control].sort_values(['bh_q_positive','estimate'],ascending=[True,False])
best=screen.iloc[0]
high_info=screen.loc[screen.specification.eq('absz_swing_length__log_exposure_pitcher__n500')&screen.method.eq('pooled')].iloc[0]
control=rank.loc[rank.specification.eq('distortion__log_future_exposure')&rank.method.eq('pooled')&rank.endpoint.eq('spearman')].iloc[0]
by_hits=int((screen.estimate.gt(0)&screen.by_q_positive.le(.05)).sum())
block+=f"\n\n**Exploratory percentile analysis:** The search covered {rank_meta['specifications']} specifications and {rank_meta['full_year_tests']} endpoint tests. There were **{rank_meta['positive_spearman_bh_significant']} positive non-control Spearman results with BH q ≤ 0.05**. Sorting by BH q and then decreasing correlation gives `{best.specification}` ({best.method}): correlation **{best.estimate:.3f}**, pointwise 90% interval **[{best.ci90_low:.3f}, {best.ci90_high:.3f}]**, BH q={best.bh_q_positive:.4f}, BY q={best.by_q_positive:.4f}. This is a result from the searched family."
block+=f"\n\nThe component findings concentrate in swing length. Among {int(high_info.shared_players)} hitters with at least 500 contributing swings in each season, the partially pooled swing-length score had correlation **{high_info.estimate:.3f}**, 90% interval **[{high_info.ci90_low:.3f}, {high_info.ci90_high:.3f}]**, BH q={high_info.bh_q_positive:.4f}. However, **{by_hits} non-control rank results pass BY q ≤ 0.05**, and the structurally related future-exposure control is similarly stable: correlation **{control.estimate:.3f}**, interval **[{control.ci90_low:.3f}, {control.ci90_high:.3f}]**, BH q={control.bh_q_positive:.4f}. These patterns do not isolate adaptation from stable selection or model structure."
if selected is not None:
    check=replication.loc[replication.specification.eq(selected['specification'])&replication.method.eq(selected['method'])].iloc[0]
    block+=f"\n\nThe candidate selected from early-season partitions was `{selected['specification']}` ({selected['method']}): Spearman **{selected['estimate']:.3f}**, BH q={selected['bh_q_positive']:.4f}. In the later-season internal replication, its correlation was **{check.estimate:.3f}**, 90% interval **[{check.ci90_low:.3f}, {check.ci90_high:.3f}]**, BH q={check.bh_q_positive:.4f}. "
    block+=("This supplies an exploratory replication result requiring independent validation." if check.estimate>0 and check.bh_q_positive<=.05 else "The selected relationship did not replicate.")
block+=' Percentile conversion alone leaves Spearman correlation unchanged. The search was specified after inspecting the original 2026 analysis; its findings are exploratory. Full-sample associations cannot establish persistent latent skill without replication. See the [complete percentile report](paper/percentile_stability.md).'
readme=ROOT/'README.md';text=readme.read_text();text=re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->\n'+block+'\n<!-- RESULTS_END -->',text,flags=re.S);readme.write_text(text)
def markdown_table(df):
    fmt=lambda x: f'{x:.5g}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
notes='# Computed results and scientific status\n\n'+block.replace('(paper/percentile_stability.md)','(percentile_stability.md)')+'\n\n## Sequence information\n\n'+markdown_table(seq)+'\n\n## Whiff classification\n\n'+markdown_table(whiff)+'\n\n## Exploratory later-season outcomes\n\n'+markdown_table(utility)+'\n\n## Complete sensitivity results\n\n'+markdown_table(r)+'\n\n## Scope\n\nConsult methods_notes.md and the executed notebook. These results do not identify causal motor learning. Statistical heterogeneity and narrow player intervals do not by themselves establish persistent adaptation or coaching utility. Negative-control failure weakens the interpretation even if a population association is precisely estimated.\n'
notes+='\nFuture exposure is structurally related to past exposure through the total number of same-type pitches in a game. Its opposite-sign association highlights timing and selection sensitivity; it is not an independent randomized placebo and does not prove that hitters never learn. Shuffling exposure residuals within hitter-game retains between-game structure, so its remaining association warns about that component of the primary estimate.\n'
for filename,title in [('matched_exposure_summary.csv','Matched current-pitch restriction'),('miss_distance_sensitivity.csv','Whiff-conditional miss distance')]:
    source=ROOT/'results/tables'/filename
    if source.exists():notes+='\n## '+title+'\n\n'+markdown_table(pd.read_csv(source))+'\n'
notes+='\nSee [supplementary validation](supplementary_validation.md) for the weakly identified split-half fit and descriptive pitcher diagnostics, and [matched comparisons](matched_comparisons.md) for pairing rules, balance and selection.\n'
(ROOT/'paper/results_notes.md').write_text(notes)
print(block)
