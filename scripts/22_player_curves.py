#!/usr/bin/env python3
"""Generate every estimable hitter's across-pitcher adaptation curve.

Curves are posterior conditional associations anchored at zero prior pitches.
They are not observed causal learning trajectories or validated skill grades.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.expected import player_slopes
from src.models.hierarchical import fit_hierarchy
from src.pipeline import names_map

CONFIG = json.loads((ROOT / 'config/analysis.json').read_text())
SEED = CONFIG['seed']
OUT = ROOT / 'results/cross_pitcher'
GRID = np.arange(0, 9)


def load_analytic(outcome):
    columns = ['season', 'game_pk', 'batter', 'exposure_pitcher',
               'adj_log_exposure_type', f'adj_{outcome}']
    return pd.concat([pd.read_parquet(p, columns=columns)
        for p in sorted((ROOT / 'data/processed').glob('analytic_*.parquet'))], ignore_index=True)


def selected_posteriors(selection):
    scope, outcome = selection['scope'], selection['outcome']
    data = load_analytic(outcome)
    if scope == 'first_type_from_pitcher':
        data = data.loc[data.exposure_pitcher.eq(0)]
    posteriors = {}
    for year, group in data.groupby('season'):
        fit = player_slopes(group, x='log_exposure_type', y=outcome,
            min_n=CONFIG['minimum_player_observations'],
            min_games=CONFIG['minimum_player_games'])
        p = fit_hierarchy(fit.estimate, fit.se, fit.player_id,
                          draws=CONFIG['posterior_draws'], seed=SEED)
        year = int(year)
        np.savez_compressed(OUT / f'posterior_selected_{year}.npz',
            **{k:p[k] for k in ['lambda','mu','tau','player_ids']})
        posteriors[year] = p
    return posteriors, data


def pooled_posteriors():
    result = {}
    for year in [2025, 2026]:
        with np.load(OUT / f'posterior_pooled_same_type_{year}.npz') as archive:
            result[year] = dict(archive)
    return result


def curve_rows(posteriors, model, outcome, names):
    rows = []
    for year, p in posteriors.items():
        ids = p['player_ids'].astype(int)
        lam = p['lambda']
        for count in GRID:
            # Positive lambda means lower atypicality. The curve plots change
            # from no previous same-type pitch: -lambda*log(1+count).
            value = -lam * np.log1p(count)
            lower, median, upper = np.quantile(value, [.05,.5,.95], axis=0)
            rows.extend({'model': model, 'outcome': outcome, 'season': year,
                         'player_id': int(player), 'player_name': names.get(int(player), str(player)),
                         'prior_same_type_pitches': int(count),
                         'change_ci90_low': float(lower[i]),
                         'change_median': float(median[i]),
                         'change_ci90_high': float(upper[i])}
                        for i, player in enumerate(ids))
    return rows


def coverage(data, curves, names):
    all_rows = data.groupby(['season','batter']).size().rename('tracked_swings').reset_index()
    all_rows['player_id'] = all_rows.batter.astype(int)
    all_rows['player_name'] = all_rows.player_id.map(names).fillna(all_rows.player_id.astype(str))
    all_rows = all_rows.drop(columns='batter')
    for model in curves.model.unique():
        modeled = curves.loc[curves.model.eq(model), ['season','player_id']].drop_duplicates()
        hit = all_rows.merge(modeled.assign(eligible=True), on=['season','player_id'], how='left')
        hit['model'] = model
        hit['eligible'] = hit.eligible.eq(True)
        yield hit


def static_figure(curves):
    base = curves.loc[curves.model.eq('pooled_geometry')]
    eligible = set(base.loc[base.season.eq(2025), 'player_name']) & set(base.loc[base.season.eq(2026), 'player_name'])
    preferred = ['Aaron Judge','Shohei Ohtani','Juan Soto','Mookie Betts']
    names = [name for name in preferred if name in eligible]
    names += [name for name in sorted(eligible) if name not in names]
    names = names[:4]
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax, name in zip(axes.flat,names):
        for year,color in [(2025,'#176b70'),(2026,'#d28c35')]:
            d=base.loc[base.player_name.eq(name)&base.season.eq(year)].sort_values('prior_same_type_pitches')
            x=d.prior_same_type_pitches.to_numpy()
            ax.plot(x,d.change_median.to_numpy(),color=color,lw=2,label=str(year))
            ax.fill_between(x,d.change_ci90_low.to_numpy(),d.change_ci90_high.to_numpy(),
                            color=color,alpha=.14)
        ax.axhline(0,color='#9ba7ae',lw=.8)
        ax.set_title(name)
        ax.set_xlabel('Earlier same-type pitches in game')
        ax.set_ylabel('Change in swing deviation')
    axes.flat[0].legend(frameon=False)
    fig.suptitle('Illustrative cross-pitcher curves for preselected hitters',fontsize=14,fontweight='bold')
    fig.text(.5,.005,'Posterior conditional associations with 90% intervals; players selected by name, not result.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.93))
    fig.savefig(ROOT / 'results/figures/19_player_adaptation_curves.png',dpi=180)
    fig.savefig(ROOT / 'results/figures/19_player_adaptation_curves.pdf')
    plt.close(fig)
    return names


def interactive_html(curves, coverage_table):
    records = curves[['model','season','player_id','player_name','prior_same_type_pitches',
                      'change_ci90_low','change_median','change_ci90_high']].to_dict('records')
    player_list = coverage_table[['player_id','player_name']].drop_duplicates().sort_values('player_name')
    players = player_list.to_dict('records')
    payload = json.dumps({'curves':records,'players':players},ensure_ascii=False).replace('</','<\\/')
    best = json.loads((OUT / 'model_search_summary.json').read_text())['descriptive_2026_best']
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Deja Swing | Player curves</title><style>
body{font-family:system-ui,-apple-system,sans-serif;color:#203644;background:#f5f8f9;margin:0}main{max-width:1080px;margin:2.5rem auto;padding:2rem;background:white;border-radius:14px;box-shadow:0 8px 30px #16364715}h1{font-family:Georgia,serif;font-size:2.3rem;margin:.2rem 0}.sub{color:#526976;max-width:760px;line-height:1.5}.controls{display:flex;gap:1rem;flex-wrap:wrap;margin:1.6rem 0}label{font-weight:700;display:flex;flex-direction:column;gap:.4rem}select{min-width:240px;padding:.65rem;border:1px solid #aebdc4;border-radius:7px;background:white;font:inherit}#plot{height:560px}.note{font-size:.95rem;line-height:1.55;color:#455d69;border-top:1px solid #d8e2e6;padding-top:1rem}</style></head><body><main><h1>Deja Swing</h1><p class="sub">Explore posterior curves for every hitter with enough tracked swings. Exposure counts earlier pitches of the same type from any pitcher within the game. The selected component model is descriptive and was chosen after the later-season data were inspected.</p><div class="controls"><label>Hitter<select id="player"></select></label><label>Model<select id="model"><option value="pooled_geometry">Multivariate swing deviation</option><option value="selected_component">Selected component: __BEST__</option></select></label></div><div id="plot"></div><p id="status" class="note"></p><p class="note">Each curve is the model-implied change from no prior same-type pitch, with a 90% posterior interval. It is a conditional association, not a causal learning trajectory or a validated player grade. Histories reset each game. Unmodeled hitters appear in the selector with an insufficient-information message.</p></main><script>__PLOTLY__</script><script>const DATA=__DATA__;const colors={2025:'#176b70',2026:'#d28c35'};const player=document.getElementById('player'),model=document.getElementById('model'),status=document.getElementById('status');for(const p of DATA.players){const o=document.createElement('option');o.value=p.player_id;o.textContent=p.player_name;player.appendChild(o)}const preferred=DATA.players.find(p=>p.player_name==='Aaron Judge');if(preferred)player.value=preferred.player_id;function draw(){const id=Number(player.value),m=model.value;const rows=DATA.curves.filter(r=>r.player_id===id&&r.model===m);const traces=[];for(const year of [2025,2026]){const d=rows.filter(r=>r.season===year).sort((a,b)=>a.prior_same_type_pitches-b.prior_same_type_pitches);if(!d.length)continue;const x=d.map(r=>r.prior_same_type_pitches),c=colors[year];traces.push({x,y:d.map(r=>r.change_ci90_low),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});traces.push({x,y:d.map(r=>r.change_ci90_high),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:year===2025?'#176b7033':'#d28c3533',showlegend:false,hoverinfo:'skip'});traces.push({x,y:d.map(r=>r.change_median),mode:'lines+markers',name:String(year),line:{color:c,width:3},marker:{size:5}})}Plotly.react('plot',traces,{paper_bgcolor:'white',plot_bgcolor:'white',margin:{l:75,r:25,t:30,b:70},xaxis:{title:'Earlier same-type pitches in game',dtick:1,gridcolor:'#e6edef'},yaxis:{title:'Change in standardized swing deviation',zeroline:true,zerolinecolor:'#82949d',gridcolor:'#e6edef'},legend:{orientation:'h',y:1.12}},{responsive:true,displaylogo:false});status.textContent=rows.length?'The shaded bands show uncertainty. Compare years to inspect whether the fitted response persists; this plot alone cannot validate a latent skill.':'This hitter did not meet the minimum tracked-swing, game, and residual-information rules for this model.'}player.addEventListener('change',draw);model.addEventListener('change',draw);draw();</script></body></html>'''
    html = template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload)
    html = html.replace('__BEST__',f"{best['outcome'].replace('absz_','').replace('_',' ')} on first type from current pitcher")
    # The embedded Plotly bundle contains trailing spaces; keep the generated
    # artifact clean so its diff passes the repository whitespace check.
    html = '\n'.join(line.rstrip() for line in html.splitlines()) + '\n'
    (ROOT / 'notebooks/player_curves.html').write_text(html)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((OUT / 'model_search_summary.json').read_text())['descriptive_2026_best']
    names=names_map()
    pooled=pooled_posteriors()
    best,_=selected_posteriors(selected)
    rows=curve_rows(pooled,'pooled_geometry','distortion',names)
    rows+=curve_rows(best,'selected_component',selected['outcome'],names)
    curves=pd.DataFrame(rows).sort_values(['model','player_id','season','prior_same_type_pitches'])
    curves.to_csv(OUT / 'player_curves.csv',index=False)
    coverage_table=pd.concat(coverage(load_analytic('distortion'),curves,names),ignore_index=True)
    coverage_table.to_csv(OUT / 'curve_coverage.csv',index=False)
    illustrated=static_figure(curves)
    interactive_html(curves,coverage_table)
    summary={
        'focal_model':'pooled_geometry / distortion',
        'descriptive_selected_model':selected,
        'curve_grid_prior_pitches':GRID.tolist(),
        'modeled_hitter_seasons':curves.groupby(['model','season']).player_id.nunique().to_dict(),
        'coverage_file':'results/cross_pitcher/curve_coverage.csv',
        'curves_file':'results/cross_pitcher/player_curves.csv',
        'interactive_html':'notebooks/player_curves.html',
        'illustrative_players':illustrated,
        'interpretation':'Model-implied within-game conditional deviation curves. Not causal or verified persistent skill.',
    }
    # Tuple keys are formatted for portable JSON.
    summary['modeled_hitter_seasons']={f'{m}_{y}':int(n)
        for (m,y),n in summary['modeled_hitter_seasons'].items()}
    (OUT / 'player_curves_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
