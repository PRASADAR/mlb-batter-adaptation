"""Source exports may include strikeout xwOBA values; contact quality must not."""
import pandas as pd
import pytest
from src.evaluation.outcomes import summarize_player_outcomes


def test_contact_quality_excludes_exported_strikeout_zeros():
    frame=pd.DataFrame({
        'season':[2025]*4,'batter':[1]*4,
        'description':['hit_into_play','hit_into_play','swinging_strike','foul_tip'],
        'estimated_woba_using_speedangle':[.4,.8,0.,0.],
        'is_whiff':[False,False,True,False],
        'bat_speed':[70.]*4,'distortion':[0.]*4,
    })
    summary=summarize_player_outcomes(frame).iloc[0]
    assert summary.xwoba==pytest.approx(.6)
    assert summary.whiff==pytest.approx(.25)
    assert summary.n==4
