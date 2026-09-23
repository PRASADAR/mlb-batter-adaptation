#!/usr/bin/env python3
"""Build uncertainty-aware adaptability rankings and an interactive explorer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/adaptability_rankings"
YEARS = (2025, 2026)


def residual(y, x):
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def annual_table(year: int) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"results/cross_pitcher/players_pooled_same_type_{year}.csv")
    d = d.sort_values(["posterior_mean", "player_id"], ascending=[False, True]).reset_index(drop=True)
    d["point_rank"] = np.arange(1, len(d) + 1)
    d["point_percentile"] = 100 * (len(d) - d.point_rank) / max(len(d) - 1, 1)
    d["rank_interval_width"] = d.rank_cri90_high - d.rank_cri90_low
    d["evidence"] = np.select(
        [d.cri90_low.gt(0), d.prob_positive.ge(.8)],
        ["90% interval above zero", "Positive tendency"], default="Unresolved")
    d["season"] = year
    return d


def build_rankings() -> pd.DataFrame:
    annual = pd.concat([annual_table(y) for y in YEARS], ignore_index=True)
    qualities = pd.read_csv(ROOT / "results/player_quality_outcomes/player_qualities.csv")
    qualities["future_authority_residual"] = residual(
        qualities.authority_composite_future,
        qualities.authority_composite_prior.to_numpy()[:, None])
    q = qualities[["player_id", "authority_composite_prior", "authority_composite_future",
                   "future_authority_residual"]]
    annual = annual.merge(q, on="player_id", how="left", validate="many_to_one")
    return annual


def make_figures(rankings: pd.DataFrame):
    d = rankings.loc[rankings.season.eq(2025)].copy()
    top = d.nsmallest(25, "point_rank").sort_values("posterior_mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    y = np.arange(len(top))
    axes[0].hlines(y, top.cri90_low, top.cri90_high, color="#8da0ae", lw=2)
    axes[0].scatter(top.posterior_mean, y, c=top.prob_positive,
                    cmap="YlOrRd", vmin=.5, vmax=1, s=48, zorder=3)
    axes[0].axvline(0, color="#455a64", lw=1)
    axes[0].set_yticks(y, top.player_name)
    axes[0].set_xlabel("Posterior adaptation rate")
    axes[0].set_title("Top 25 point estimates with 90% intervals")

    sc = axes[1].scatter(d.n_exposures, d.posterior_mean,
        c=d.prob_positive, cmap="viridis", vmin=.5, vmax=1,
        s=18 + 45 / (1 + 100 * d.posterior_sd), alpha=.78)
    axes[1].axhline(0, color="#455a64", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Tracked swing exposures, log scale")
    axes[1].set_ylabel("Posterior adaptation rate")
    axes[1].set_title("All 587 eligible hitters")
    bar = fig.colorbar(sc, ax=axes[1], shrink=.72)
    bar.set_label("Posterior probability rate is positive")
    fig.suptitle("Adaptability rankings require uncertainty, not a single ordering",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "adaptability_rankings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    q = d.dropna(subset=["future_authority_residual"]).copy()
    q["quintile"] = pd.qcut(q.posterior_mean, 5, labels=False) + 1
    means = q.groupby("quintile").future_authority_residual.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(q.posterior_mean, q.future_authority_residual,
               color="#7895a4", alpha=.35, s=25, label="Hitter")
    coef = np.polyfit(q.posterior_mean, q.future_authority_residual, 1)
    grid = np.linspace(q.posterior_mean.min(), q.posterior_mean.max(), 100)
    ax.plot(grid, np.polyval(coef, grid), color="#c7552d", lw=2.5,
            label="Linear summary")
    ax.errorbar(q.groupby("quintile").posterior_mean.mean(), means["mean"],
        yerr=1.96 * means["sem"], fmt="o", color="#172b3a", capsize=4,
        ms=7, label="Quintile mean and 95% interval")
    ax.axhline(0, color="#455a64", lw=1)
    ax.set(xlabel="2025 posterior adaptation rate",
           ylabel="2026 authority after controlling 2025 authority",
           title="Adaptation speed and subsequent batted-ball authority")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / "adaptability_authority.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def interactive_html(rankings: pd.DataFrame):
    curves = pd.read_csv(ROOT / "results/cross_pitcher/player_curves.csv")
    curves = curves.loc[curves.model.eq("pooled_geometry")]
    cols = ["season", "player_id", "player_name", "point_rank", "point_percentile",
        "n_exposures", "posterior_mean", "posterior_sd", "cri90_low", "cri90_high",
        "rank_median", "rank_cri90_low", "rank_cri90_high", "prob_positive",
        "evidence", "future_authority_residual"]
    payload = json.dumps({
        "rankings": rankings[cols].where(pd.notna(rankings[cols]), None).to_dict("records"),
        "curves": curves[["season", "player_id", "prior_same_type_pitches",
            "change_ci90_low", "change_median", "change_ci90_high"]].to_dict("records")
    }, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Deja Swing | Adaptability rankings</title><style>
:root{--ink:#172b3a;--muted:#5c707c;--paper:#fff;--bg:#eef3f4;--accent:#c7552d;--teal:#19757b}*{box-sizing:border-box}body{font-family:Inter,system-ui,-apple-system,sans-serif;color:var(--ink);background:var(--bg);margin:0}main{max-width:1280px;margin:24px auto;padding:28px;background:var(--paper);box-shadow:0 12px 36px #172b3a18;border-radius:16px}h1{font-family:Georgia,serif;font-size:2.35rem;margin:0 0 8px}.sub{color:var(--muted);line-height:1.55;max-width:920px}.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:22px 0}label{font-weight:700;font-size:.9rem;display:flex;flex-direction:column;gap:6px}select,input{font:inherit;padding:9px 11px;border:1px solid #adbbc1;border-radius:8px;background:#fff;min-width:220px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:12px 0}.card{background:#f5f8f8;border-radius:10px;padding:13px}.card b{display:block;font-size:1.25rem;margin-top:3px}.card span{color:var(--muted);font-size:.78rem}.plots{display:grid;grid-template-columns:1.35fr .65fr;gap:14px}.plot{height:480px;border:1px solid #e0e7e9;border-radius:10px}.tableWrap{max-height:520px;overflow:auto;margin-top:20px;border:1px solid #dce5e8;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:.86rem}th{position:sticky;top:0;background:#eaf0f1;cursor:pointer;text-align:left}th,td{padding:9px;border-bottom:1px solid #e4eaec}tr:hover{background:#f4f8f8}.note{color:var(--muted);line-height:1.5;font-size:.9rem}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.plots{grid-template-columns:1fr}.plot{height:420px}}</style></head><body><main><h1>Deja Swing adaptability rankings</h1><p class="sub">Search every eligible hitter, inspect annual posterior curves, and compare point rankings with their uncertainty. Positive adaptation means context-adjusted swing deviation declines faster as the hitter accumulates same-type looks across pitchers within a game.</p><div class="controls"><label>Hitter<select id="player"></select></label><label>Season<select id="season"><option>2025</option><option>2026</option></select></label><label>Filter table<input id="search" placeholder="Type a player name"></label></div><div class="cards"><div class="card"><span>Point rank</span><b id="rank"></b></div><div class="card"><span>Posterior rate</span><b id="rate"></b></div><div class="card"><span>90% rate interval</span><b id="rateci"></b></div><div class="card"><span>Median rank, 90% interval</span><b id="rankci"></b></div><div class="card"><span>Probability positive</span><b id="positive"></b></div></div><div class="plots"><div id="curve" class="plot"></div><div id="rankPlot" class="plot"></div></div><p id="authority" class="note"></p><div class="tableWrap"><table><thead><tr><th data-key="point_rank">Rank</th><th data-key="player_name">Player</th><th data-key="posterior_mean">Rate</th><th data-key="prob_positive">P(rate &gt; 0)</th><th data-key="rank_cri90_low">Best plausible rank</th><th data-key="rank_cri90_high">Worst plausible rank</th><th data-key="n_exposures">Exposures</th><th data-key="evidence">Evidence</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note">Point ranks order posterior means. They are descriptive, not exact talent ranks. Rank intervals propagate uncertainty in every hitter's rate. The future-authority value is available only for the common 261-hitter external-validation cohort and is residualized on 2025 authority.</p></main><script>__PLOTLY__</script><script>
const D=__DATA__,player=document.getElementById('player'),season=document.getElementById('season'),search=document.getElementById('search'),tbody=document.getElementById('rows'),rankEl=document.getElementById('rank'),rateEl=document.getElementById('rate'),rateciEl=document.getElementById('rateci'),rankciEl=document.getElementById('rankci'),positiveEl=document.getElementById('positive'),authorityEl=document.getElementById('authority');let sortKey='point_rank',ascending=true;const names=[...new Map(D.rankings.map(r=>[r.player_id,r.player_name])).entries()].sort((a,b)=>a[1].localeCompare(b[1]));for(const [id,name] of names){const o=document.createElement('option');o.value=id;o.textContent=name;player.appendChild(o)}const first=names.find(x=>x[1]==='Aaron Judge')||names[0];player.value=first[0];function fmt(x,d=3){return x==null?'NA':Number(x).toFixed(d)}function current(){return D.rankings.find(r=>r.player_id===Number(player.value)&&r.season===Number(season.value))}function draw(){const r=current();if(!r)return;rankEl.textContent=`${r.point_rank} of ${D.rankings.filter(x=>x.season===r.season).length}`;rateEl.textContent=fmt(r.posterior_mean);rateciEl.textContent=`${fmt(r.cri90_low)} to ${fmt(r.cri90_high)}`;rankciEl.textContent=`${Math.round(r.rank_median)} (${Math.round(r.rank_cri90_low)} to ${Math.round(r.rank_cri90_high)})`;positiveEl.textContent=`${(100*r.prob_positive).toFixed(1)}%`;authorityEl.textContent=r.future_authority_residual==null?'This hitter is outside the common player-quality validation cohort.':`Future authority residual: ${fmt(r.future_authority_residual)} standard-deviation units after controlling for prior authority.`;const c=D.curves.filter(x=>x.player_id===r.player_id&&x.season===r.season).sort((a,b)=>a.prior_same_type_pitches-b.prior_same_type_pitches);const x=c.map(x=>x.prior_same_type_pitches);Plotly.react('curve',[{x,y:c.map(x=>x.change_ci90_low),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},{x,y:c.map(x=>x.change_ci90_high),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'#19757b33',showlegend:false,hoverinfo:'skip'},{x,y:c.map(x=>x.change_median),mode:'lines+markers',line:{color:'#19757b',width:3},name:r.player_name}],{title:`${r.player_name}: fitted ${r.season} curve`,xaxis:{title:'Earlier same-type pitches in game',dtick:1},yaxis:{title:'Change in swing deviation',zeroline:true},margin:{l:68,r:20,t:55,b:65}},{responsive:true,displaylogo:false});Plotly.react('rankPlot',[{x:[r.posterior_mean],y:[r.rank_median],mode:'markers',marker:{size:12,color:'#c7552d'},error_y:{type:'data',symmetric:false,array:[r.rank_cri90_high-r.rank_median],arrayminus:[r.rank_median-r.rank_cri90_low],color:'#c7552d',thickness:2},name:'Rank uncertainty'}],{title:'Posterior rank uncertainty',xaxis:{title:'Posterior adaptation rate',range:[-.08,.13]},yaxis:{title:'Rank, 1 is highest',autorange:'reversed',range:[D.rankings.filter(x=>x.season===r.season).length,1]},margin:{l:65,r:20,t:55,b:65},showlegend:false},{responsive:true,displaylogo:false});renderTable()}function renderTable(){const s=search.value.toLowerCase(),yr=Number(season.value);let a=D.rankings.filter(r=>r.season===yr&&r.player_name.toLowerCase().includes(s));a.sort((x,y)=>{let u=x[sortKey],v=y[sortKey];if(typeof u==='string')return (ascending?1:-1)*u.localeCompare(v);return (ascending?1:-1)*((u??Infinity)-(v??Infinity))});tbody.innerHTML=a.map(r=>`<tr data-id="${r.player_id}"><td>${r.point_rank}</td><td>${r.player_name}</td><td>${fmt(r.posterior_mean)}</td><td>${(100*r.prob_positive).toFixed(1)}%</td><td>${Math.round(r.rank_cri90_low)}</td><td>${Math.round(r.rank_cri90_high)}</td><td>${r.n_exposures}</td><td>${r.evidence}</td></tr>`).join('');tbody.querySelectorAll('tr').forEach(tr=>tr.onclick=()=>{player.value=tr.dataset.id;draw();window.scrollTo({top:0,behavior:'smooth'})})}document.querySelectorAll('th').forEach(th=>th.onclick=()=>{if(sortKey===th.dataset.key)ascending=!ascending;else{sortKey=th.dataset.key;ascending=true}renderTable()});player.onchange=draw;season.onchange=draw;search.oninput=renderTable;draw();
</script></body></html>'''
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", payload)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (ROOT / "notebooks/adaptability_rankings.html").write_text(html)


def write_summary(rankings):
    d = rankings.loc[rankings.season.eq(2025)]
    summary = {"players_2025": int(len(d)),
        "players_2026": int(rankings.season.eq(2026).sum()),
        "rate_interval_above_zero_2025": int(d.cri90_low.gt(0).sum()),
        "median_rank_interval_width_2025": float(d.rank_interval_width.median()),
        "warning": "Point rankings are descriptive. Wide posterior rank intervals preclude exact talent ordering."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rankings = build_rankings()
    rankings.to_csv(OUT / "player_rankings.csv", index=False)
    make_figures(rankings)
    interactive_html(rankings)
    write_summary(rankings)
    print(rankings.loc[rankings.season.eq(2025), ["point_rank", "player_name",
        "posterior_mean", "cri90_low", "cri90_high", "rank_cri90_low",
        "rank_cri90_high", "prob_positive"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
