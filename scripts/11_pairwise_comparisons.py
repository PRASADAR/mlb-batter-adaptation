#!/usr/bin/env python3
"""Export all strict pairwise probabilities and full discrete rank distributions."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import load_posterior


def full_comparison_tables(posterior):
    lam=posterior['lambda'];draws,n=lam.shape
    pair=np.empty((n,n))
    for start in range(0,n,16):
        pair[start:start+16]=(lam[:,start:start+16,None]>lam[:,None,:]).mean(axis=0)
    ranks=np.argsort(np.argsort(-lam,axis=1),axis=1)
    counts=np.zeros((n,n),dtype=np.uint32)
    np.add.at(counts,(np.tile(np.arange(n),draws),ranks.ravel()),1)
    rank_probability=counts/draws
    assert np.allclose(rank_probability.sum(axis=1),1)
    assert np.all(pair+pair.T<=1+1e-12)
    assert np.all(np.diag(pair)==0)
    return pair,rank_probability


def main():
    meta={}
    for year in [2025,2026]:
        p=load_posterior(f'adaptation_{year}')
        pair,ranks=full_comparison_tables(p)
        np.savez_compressed(ROOT/f'results/posterior/comparisons_{year}.npz',player_ids=p['player_ids'],pairwise_probability=pair,rank_probability=ranks,rank_values=np.arange(1,len(ranks)+1))
        meta[str(year)]={'players':len(ranks),'draws':len(p['lambda']),'pairwise_definition':'strict P(lambda_i > lambda_j); diagonal is zero','rank_definition':'columns correspond to rank_values;1 is largest conditional rate','scope':'Qualified hitters in this period, not a causal talent ranking'}
    (ROOT/'results/posterior/comparison_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
