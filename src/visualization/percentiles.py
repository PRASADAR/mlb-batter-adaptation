"""Fast, offline rendering of the saved exploratory percentile search."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EXPOSURES=['log_exposure_pitcher','log_exposure_pa','log_exposure_type','log_kernel_exposure','log_exposure_sequence',
 'log_kernel_h0.5_m5','log_kernel_h0.5_m20','log_kernel_h1_m5','log_kernel_h2_m5','log_kernel_h2_m20',
 'linear_exposure_pitcher','saturating_exposure_pitcher']
EXPOSURE_LABELS=['Same pitcher and type','Same type within PA','Same type within game','Shape h=1, memory 20',
 'Repeated transition','Shape h=0.5, memory 5','Shape h=0.5, memory 20','Shape h=1, memory 5',
 'Shape h=2, memory 5','Shape h=2, memory 20','Linear pitcher exposure','Saturating pitcher exposure']
OUTCOMES=['distortion','absz_attack_angle','absz_attack_direction','absz_swing_path_tilt','absz_bat_speed','absz_swing_length']
OUTCOME_LABELS=['Overall\ndeviation','Attack\nangle','Attack\ndirection','Swing-path\ntilt','Bat\nspeed','Swing\nlength']


def plot_percentile_stability(root, out=None):
    """Read saved tables and return the PNG Path; no analysis is rerun.

    `out` may be a figure directory or a PNG/PDF path. Both PNG and PDF are
    exported alongside one another for notebook and publication use.
    """
    root=Path(root)
    output=Path(out) if out is not None else root/'results/figures'
    stem=output.with_suffix('') if output.suffix.lower() in {'.png','.pdf'} else output/'percentile_stability'
    stem.parent.mkdir(parents=True,exist_ok=True)
    results=pd.read_csv(root/'results/tables/percentile_stability.csv')
    groups=pd.read_csv(root/'results/tables/percentile_groups.csv')
    fig=plt.figure(figsize=(17,8.8),facecolor='#faf9f6')
    grid=fig.add_gridspec(1,3,width_ratios=[1.14,1.14,1.05],left=.145,right=.985,bottom=.18,top=.81,wspace=.12)
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    limit=.4
    for ax,method,title in zip(axes[:2],['unpooled','pooled'],['A. Unpooled slope ranks','B. Partially pooled slope ranks']):
        selected=results.loc[results.endpoint.eq('spearman')&results.method.eq(method)&results.restriction.eq('all')&~results.high_information&~results.negative_control]
        value=selected.pivot(index='x',columns='y',values='estimate').reindex(index=EXPOSURES,columns=OUTCOMES)
        q=selected.pivot(index='x',columns='y',values='bh_q_positive').reindex(index=EXPOSURES,columns=OUTCOMES)
        image=ax.imshow(value,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(np.arange(6),OUTCOME_LABELS,fontsize=8.4)
        ax.set_yticks(np.arange(12),EXPOSURE_LABELS if method=='unpooled' else ['']*12,fontsize=9)
        ax.tick_params(length=0,pad=8)
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=15,color='#1b2635')
        ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,12,1),minor=True)
        ax.grid(which='minor',color='#faf9f6',linewidth=2);ax.tick_params(which='minor',length=0)
        for i in range(12):
            for j in range(6):
                v=value.iloc[i,j]
                if pd.notna(v):
                    marker='*' if q.iloc[i,j]<=.05 and v>0 else ''
                    ax.text(j,i,f'{v:.2f}{marker}',ha='center',va='center',fontsize=8.2,
                            color='white' if abs(v)>.26 else '#233044')
        for spine in ax.spines.values(): spine.set_visible(False)
    bar=fig.colorbar(image,ax=axes[:2],orientation='horizontal',fraction=.042,pad=.105,aspect=40)
    bar.set_label('Spearman correlation from 2025 to 2026',fontsize=9)
    bar.ax.tick_params(labelsize=8)
    ax=axes[2]
    primary='distortion__log_exposure_pitcher'
    for method,color,offset,label in [('unpooled','#304e75',-.09,'Unpooled'),('pooled','#c05b32',.09,'Partially pooled')]:
        selected=groups.loc[groups.specification.eq(primary)&groups.method.eq(method)&groups.bins.eq(10)].sort_values('baseline_group')
        x=selected.baseline_group.to_numpy()+offset;y=selected.mean_followup_percentile.to_numpy()*100
        lo=selected.mean_percentile_ci90_low.to_numpy()*100;hi=selected.mean_percentile_ci90_high.to_numpy()*100
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt='o-',color=color,label=label,lw=1.6,markersize=4.5,capsize=2.5)
    ax.axhline(50,color='#89929c',lw=1,ls=':')
    ax.plot([1,10],[5,95],color='#b4bbc2',lw=1,ls='--',label='Identical percentile reference')
    ax.set_xlim(.45,10.55);ax.set_ylim(0,100)
    ax.set_xticks([1,3,5,7,10]);ax.set_xlabel('2025 conditional-slope decile',fontsize=10)
    ax.set_ylabel('Mean 2026 percentile',fontsize=10,labelpad=2)
    ax.set_title('C. Primary decile comparison',loc='left',fontsize=12,fontweight='bold',pad=15,color='#1b2635')
    ax.grid(axis='y',color='#e3e5e7',lw=.7);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#acb1b6')
    ax.legend(loc='upper left',fontsize=8,frameon=False)
    ax.set_facecolor('#faf9f6')
    fig.text(.145,.955,'Cross-season stability of player percentiles',fontsize=23,fontweight='bold',color='#182331')
    fig.text(.145,.908,'An exploratory search across exposure definitions and swing components',fontsize=12,color='#526170')
    fig.text(.145,.058,'* Positive-direction BH q ≤ 0.05 across the complete endpoint family. Cells show 72 combinations; all 84 specifications are saved.\nDecile intervals use within-group player bootstraps. The selected candidate failed internal replication; no endpoint survived BY adjustment.\n2026 data run through September 20. Aggregate 2026 results had already been inspected before this exploratory search.',fontsize=8.5,color='#526170',linespacing=1.4)
    fig.savefig(stem.with_suffix('.png'),dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix('.pdf'),facecolor=fig.get_facecolor())
    plt.close(fig)
    return stem.with_suffix('.png')


def render_percentile_stability(root):
    return plot_percentile_stability(root,Path(root)/'results/figures')
