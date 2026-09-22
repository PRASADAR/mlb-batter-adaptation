"""Meaningful safeguards for the exploratory rank search."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import importlib.util
from pathlib import Path

_path=Path(__file__).resolve().parents[1]/'scripts/19_percentile_stability.py'
_spec=importlib.util.spec_from_file_location('percentile_search',_path)
module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def test_percentiles_preserve_rank_correlation_even_with_ties():
    x=np.array([5,2,2,7,1,9,4],dtype=float)
    y=np.array([1,2,3,5,4,7,6],dtype=float)
    assert np.isclose(spearmanr(x,y).statistic,spearmanr(module.percentiles(x),module.percentiles(y)).statistic)


def test_bh_adjustment_handles_unsorted_values_and_missing_tests():
    q=module.bh_adjust([.04,.001,np.nan,.02,.8])
    np.testing.assert_allclose(q[[0,1,3,4]],[.0533333333333,.004,.04,.8])
    assert np.isnan(q[2])


def test_high_information_filter_requires_both_periods():
    spec=next(s for s in module.specs() if s['high_information'])
    key=spec['specification'].removesuffix('__n500')
    a=pd.DataFrame({'player_id':np.arange(12),'n':[600]*11+[40]})
    b=pd.DataFrame({'player_id':np.arange(12),'n':[40]+[600]*11})
    pairs=module.match_scores({key:a},{key:b},spec)
    assert set(pairs.player_id)==set(range(1,11))


def test_candidate_selection_excludes_negative_controls_and_nonpositive_results():
    d=pd.DataFrame([
        dict(specification='control',method='pooled',negative_control=True,estimate=.9,bh_q_positive=.001,permutation_p_positive=.0001),
        dict(specification='negative',method='pooled',negative_control=False,estimate=-.9,bh_q_positive=.002,permutation_p_positive=.0001),
        dict(specification='eligible',method='unpooled',negative_control=False,estimate=.2,bh_q_positive=.1,permutation_p_positive=.02)])
    assert module.select_candidate(d)['specification']=='eligible'


def test_specification_family_is_fixed_and_nonduplicated():
    s=module.specs()
    assert len(s)==84 and len({x['specification'] for x in s})==84
    assert sum(x['high_information'] for x in s)==6
    assert sum(x['negative_control'] for x in s)==2


def test_pooling_means_respect_symmetry_and_are_deterministic():
    slopes=pd.DataFrame({'player_id':[1,2,3,4],'estimate':[-.2,-.1,.1,.2],'se':[.2]*4,'n':[100]*4,'games':[20]*4,'information':[10]*4})
    first,_=module.exact_pool(slopes)
    second,_=module.exact_pool(slopes)
    np.testing.assert_allclose(first.pooled_mean,-first.pooled_mean.to_numpy()[::-1],atol=1e-12)
    np.testing.assert_array_equal(first.pooled_mean,second.pooled_mean)
    assert np.all(np.abs(first.pooled_mean)<np.abs(first.estimate))


def test_undefined_endpoints_do_not_become_small_permutation_pvalues():
    import warnings
    n=24
    p=pd.DataFrame({'estimate_a':np.ones(n),'estimate_b':np.arange(n,dtype=float),
        'unpooled_percentile_a':np.full(n,.5),'unpooled_percentile_b':module.percentiles(np.arange(n))})
    for suffix in ['a','b']:
        for name,value in [('n',100),('games',20),('information',10.),('se',.1),('expected_shrinkage',.5)]:
            p[f'{name}_{suffix}']=value
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows,_=module.rank_statistics(p,'unpooled',1,permutations=19,bootstraps=20,full=True)
    assert all(np.isnan(row['estimate']) for row in rows)
    assert all(np.isnan(row['permutation_p_positive']) and np.isnan(row['permutation_p_two_sided']) for row in rows)
    adjusted=module.adjust_family(pd.DataFrame(rows))
    assert adjusted.bh_q_positive.isna().all()
    assert adjusted.family_tests.eq(0).all()


def test_partial_rank_diagnostic_removes_known_precision_driven_order():
    rng=np.random.default_rng(902)
    precision=rng.normal(size=2000)
    x=5*precision+rng.normal(size=len(precision))
    y=5*precision+rng.normal(size=len(precision))
    controls=np.column_stack([precision,rng.normal(size=(len(precision),3))])
    assert spearmanr(x,y).statistic>.9
    assert abs(module.partial_rank_correlation(x,y,controls))<.1
