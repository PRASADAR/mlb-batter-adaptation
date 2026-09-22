"""Matching must respect design constraints and be invariant to outcomes."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_exposure',ROOT/'scripts/18_matched_exposure.py')
matched=importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)


def sample():
    rows=[]
    # First two rows cannot match (speed gap 3); third can match either, and
    # must choose the second, closer physical pitch regardless of outcomes.
    for i,speed in enumerate([90.,93.,92.,92.,92.,92.]):
        rows.append({'season':2025,'batter':1,'game_pk':1,'pitcher':2,'pitch_type':'FF',
            'balls':0,'strikes':0,'at_bat_number':1,'pitch_number':i+1,
            'release_speed':speed,'pfx_x':.5,'pfx_z':1.,'plate_x_front':0.,
            'plate_z_front':2.5,'release_pos_x':-2.,'release_pos_z':6.,
            'exposure_pitcher':i,'distortion':float(i),'adj_distortion':float(-i)})
    # Later rows fail respectively movement, plate-location, and release calipers.
    rows[3]['pfx_x']=.9
    rows[4]['plate_x_front']=.5
    rows[5]['release_pos_x']=-1.5
    return pd.DataFrame(rows)


def test_covariate_matching_is_outcome_invariant_and_respects_calipers():
    frame=sample()
    pairs=matched.match_swings(frame)
    assert len(pairs)==1
    assert pairs.iloc[0].earlier_pitch_number==2
    assert pairs.iloc[0].later_pitch_number==3
    changed=frame.copy()
    changed['distortion']=[900.,np.nan,-1000.,3.,50.,-9.]
    changed['adj_distortion']=np.arange(len(changed))[::-1]*1000
    again=matched.match_swings(changed)
    identity=['earlier_pitch_number','later_pitch_number','covariate_distance']
    pd.testing.assert_frame_equal(pairs[identity],again[identity])
    assert pairs.delta_release_speed.abs().le(2).all()
    assert pairs.delta_pfx_x.abs().le(.15).all()
    assert pairs.delta_pfx_z.abs().le(.15).all()
    assert pairs.plate_location_distance_ft.le(.35).all()
    assert pairs.release_location_distance_ft.le(.25).all()


def test_no_reuse_exact_count_and_increasing_exposure():
    frame=sample().iloc[:4].copy()
    for column in matched.COVARIATES:
        frame[column]=frame[column].iloc[0]
    # Counts differ for the last swing: it must not cross the exact-count stratum.
    frame.loc[3,'strikes']=1
    pairs=matched.match_swings(frame)
    assert len(pairs)==1
    ids=list(pairs.earlier_pitch_number)+list(pairs.later_pitch_number)
    assert len(ids)==len(set(ids))
    assert pairs.exposure_increment.gt(0).all()
    frame['exposure_pitcher']=0
    assert matched.match_swings(frame).empty


def test_compact_analytic_front_plane_alias_preserves_measurement(tmp_path):
    frame=sample()
    path=tmp_path/'analytic_2025_03.parquet'
    frame.rename(columns={'plate_x_front':'plate_x','plate_z_front':'plate_z'}).to_parquet(path,index=False)
    restored=matched.read_analytic_partition(path,list(frame.columns))
    pd.testing.assert_frame_equal(restored,frame)


def test_audit_deltas_use_same_precision_as_selection():
    frame=sample().iloc[:2].copy()
    frame['release_speed']=90.
    frame['pfx_x']=np.array([-.27,-.12000001],dtype='float32')
    pairs=matched.match_swings(frame)
    assert len(pairs)==1
    difference=float(frame.pfx_x.iloc[1])-float(frame.pfx_x.iloc[0])
    assert pairs.iloc[0].delta_pfx_x==difference
    assert pairs.iloc[0].delta_pfx_x<=.15
