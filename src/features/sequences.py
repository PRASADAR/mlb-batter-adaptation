"""Pitch histories built on ALL pitches, before restricting to tracked swings.

No fitted feature scale sees holdout data: the shape kernel uses fixed physical
scales (5 mph, 0.5 ft movement, 0.5 ft release) rather than full-sample z scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
WHIFF = {'swinging_strike', 'swinging_strike_blocked'}
CONTACT = {'foul', 'foul_tip', 'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
BAT_FIELDS = ['bat_speed', 'swing_length', 'attack_angle', 'attack_direction',
              'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches',
              'intercept_ball_minus_batter_pos_y_inches', 'miss_distance']
SHAPE = ['release_speed','pfx_x','pfx_z','release_pos_x','release_pos_z']
SHAPE_SCALE = np.array([5., .5, .5, .5, .5])
BASE_NUM = ['release_speed','plate_x','plate_z','pfx_x','pfx_z','release_pos_x',
            'release_pos_z','release_extension','release_spin_rate','spin_axis',
            'arm_angle','vaa','haa','zone_z','balls','strikes','outs_when_up',
            'inning','bat_score_diff','day_of_season']
BASE_CAT = ['batter','pitcher','pitch_type','stand','p_throws','home_team', 'count']
HISTORY_NUM = ['lag1_release_speed','lag2_release_speed','delta_release_speed',
               'delta_pfx_x','delta_pfx_z','delta_plate_x','delta_plate_z',
               'release_separation','tunnel_separation','late_separation',
               'lag1_whiff','pa_prior_pitches']
HISTORY_CAT = ['lag1_pitch_type','lag2_pitch_type','transition']


def prepare_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    if 'game_type' in d:
        d = d.loc[d.game_type.eq('R')].copy()
    # Automatic count penalties contain no delivered pitch or perceptual exposure.
    d=d.loc[~d.description.isin(['automatic_ball','automatic_strike'])].copy()
    d['game_date'] = pd.to_datetime(d.game_date).dt.normalize()
    d = d.dropna(subset=KEY+['batter','pitcher']).copy()
    if d.duplicated(KEY).any():
        raise ValueError('Duplicate pitch identifiers: resolve source overlap first.')
    d = d.sort_values(KEY, kind='stable').reset_index(drop=True)
    d['season'] = d.game_date.dt.year
    d['day_of_season'] = d.game_date.dt.dayofyear
    for f in set(BASE_NUM+SHAPE+BAT_FIELDS+['plate_x','plate_z','vx0','vy0','vz0','ax','ay','az','sz_top','sz_bot','launch_speed']):
        if f not in d: d[f] = np.nan
        d[f] = pd.to_numeric(d[f], errors='coerce')
    for f in BASE_CAT:
        if f not in d: d[f] = 'unknown'
    d['is_whiff'] = d.description.isin(WHIFF)
    d['is_contact'] = d.description.isin(CONTACT)
    d['is_swing'] = d.is_whiff | d.is_contact
    d['count'] = d.balls.fillna(-1).astype(int).astype(str) + '-' + d.strikes.fillna(-1).astype(int).astype(str)
    d['zone_z'] = (d.plate_z-d.sz_bot)/(d.sz_top-d.sz_bot).replace(0,np.nan)
    # Evaluate velocity at a common front-of-plate plane, using y0=50 ft.
    def time_at(y):
        discr = d.vy0**2 - 2*d.ay*(50-y)
        return (2*(50-y)/(-d.vy0+np.sqrt(discr.where(discr>=0)))).where(d.vy0<0)
    tf = time_at(17/12)
    d['vaa'] = np.degrees(np.arctan2(d.vz0+d.az*tf, -(d.vy0+d.ay*tf)))
    d['haa'] = np.degrees(np.arctan2(d.vx0+d.ax*tf, -(d.vy0+d.ay*tf)))
    # Absolute x/z trajectory anchored to published plate position. The 2026
    # midpoint plate-reference change is handled with 8.5 rather than 17 inches.
    tr = time_at(np.where(d.season.ge(2026),8.5/12,17/12))
    d['plate_x_front']=d.plate_x+d.vx0*(tf-tr)+.5*d.ax*(tf**2-tr**2)
    d['plate_z_front']=d.plate_z+d.vz0*(tf-tr)+.5*d.az*(tf**2-tr**2)
    for y in [23.8, 5.]:
        ty = time_at(y)
        d[f'traj_x_{y}'] = d.plate_x+d.vx0*(ty-tr)+.5*d.ax*(ty**2-tr**2)
        d[f'traj_z_{y}'] = d.plate_z+d.vz0*(ty-tr)+.5*d.az*(ty**2-tr**2)
    # All model location/delta features use the same front-of-plate plane.
    d['plate_x']=d.plate_x_front
    d['plate_z']=d.plate_z_front
    g = d.groupby(['game_pk','at_bat_number'], sort=False)
    d['pa_prior_pitches'] = g.cumcount()
    for lag in [1,2]:
        for f in ['pitch_type','release_speed','pfx_x','pfx_z','plate_x','plate_z','release_pos_x','release_pos_z','is_whiff','traj_x_23.8','traj_z_23.8','traj_x_5.0','traj_z_5.0']:
            d[f'lag{lag}_{f}'] = g[f].shift(lag)
    d['lag1_whiff'] = d.lag1_is_whiff.eq(True).astype(int)
    for f in ['release_speed','pfx_x','pfx_z','plate_x','plate_z']:
        d[f'delta_{f}'] = d[f]-d[f'lag1_{f}']
    d['release_separation'] = np.hypot(d.release_pos_x-d.lag1_release_pos_x,d.release_pos_z-d.lag1_release_pos_z)
    for name,y in [('tunnel_separation',23.8),('late_separation',5.0)]:
        d[name] = np.hypot(d[f'traj_x_{y}']-d[f'lag1_traj_x_{y}'],d[f'traj_z_{y}']-d[f'lag1_traj_z_{y}'])
    d['transition'] = d.lag1_pitch_type.fillna('START').astype(str)+'>'+d.pitch_type.fillna('UNK').astype(str)
    d = d.copy()
    # Every count excludes the current pitch and includes takes and untracked swings.
    specs = {
        'exposure_pa': ['game_pk','at_bat_number','batter','pitch_type'],
        'exposure_pitcher': ['game_pk','batter','pitcher','pitch_type'],
        'exposure_type': ['game_pk','batter','pitch_type'],
        'exposure_pitcher_all': ['game_pk','batter','pitcher'],
        'exposure_sequence': ['game_pk','batter','transition'],
    }
    for name,keys in specs.items(): d[name] = d.groupby(keys,dropna=False,sort=False).cumcount()
    # A future-exposure negative control is explicitly segregated from model inputs.
    keys = specs['exposure_pitcher']
    d['future_exposure'] = d.groupby(keys,dropna=False).pitch_number.transform('size')-1-d.exposure_pitcher
    d['irrelevant_exposure'] = d.groupby(['game_pk','batter']).cumcount()-d.exposure_type
    # Chronological similar-shape kernels within game. Group sizes are ~15 pitches.
    kernels = {(h,half): np.zeros(len(d)) for h in [.5,1.,2.] for half in [5.,20.]}
    since = np.full(len(d),np.nan)
    shape = d[SHAPE].to_numpy(float)/SHAPE_SCALE
    for idx in d.groupby(['game_pk','batter'],sort=False).indices.values():
        a = shape[idx]
        valid = np.isfinite(a).all(axis=1)
        dist = np.sum((a[:,None,:]-a[None,:,:])**2,axis=2)
        lag = np.arange(len(idx))[:,None]-np.arange(len(idx))[None,:]
        past = (lag>0)&valid[:,None]&valid[None,:]
        for (h,half),out in kernels.items():
            w = np.exp(-np.nan_to_num(dist,nan=1e6)/h)*np.exp(-np.maximum(lag,0)*np.log(2)/half)
            out[idx] = (w*past).sum(axis=1)
        close = past&(dist<1)
        since[idx] = np.where(close.any(axis=1),np.where(close,lag,1e9).min(axis=1),np.nan)
    for (h,half),out in kernels.items(): d[f'kernel_h{h:g}_m{half:g}'] = out
    d['similar_pitch_gap'] = since
    d['shape_valid'] = np.isfinite(shape).all(axis=1)
    for key in kernels:
        h,half=key
        d.loc[~d.shape_valid,f'kernel_h{h:g}_m{half:g}']=np.nan
    d['kernel_exposure'] = d['kernel_h1_m20']
    return d


def model_frame(d, history=False, extra_numeric=()):
    num = BASE_NUM+(HISTORY_NUM if history else [])+list(extra_numeric)
    cat = BASE_CAT+(HISTORY_CAT if history else [])
    x = pd.DataFrame(index=d.index)
    for f in num: x[f] = pd.to_numeric(d[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
    for f in cat: x[f] = d[f].fillna('missing').astype(str)
    return x, cat
