"""Aggregate pitch-level records into interpretable batter qualities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prospective import CONTACT, WHIFF

STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk"}

QUALITY_META = {
    "chase_rate": ("Chase rate", "Swing decisions", "out_zone_pitches", -1),
    "zone_swing_rate": ("Zone swing rate", "Swing decisions", "in_zone_pitches", 1),
    "whiff_rate": ("Whiff rate", "Bat-to-ball", "swings", -1),
    "strikeout_rate": ("Strikeout rate", "Bat-to-ball", "pa", -1),
    "walk_rate": ("Walk rate", "Plate discipline", "pa", 1),
    "hard_hit_rate": ("Hard-hit rate", "Batted-ball authority", "batted_balls", 1),
    "barrel_rate": ("Barrel rate", "Batted-ball authority", "batted_balls", 1),
    "mean_exit_velocity": ("Mean exit velocity", "Batted-ball authority", "batted_balls", 1),
    "p90_exit_velocity": ("90th-percentile exit velocity", "Batted-ball authority", "batted_balls", 1),
    "xwoba_on_contact": ("Expected wOBA on contact", "Batted-ball authority", "batted_balls", 1),
    "woba": ("wOBA", "Overall production", "woba_denominator", 1),
    "mean_bat_speed": ("Mean bat speed", "Physical swing", "tracked_swings", 1),
}

AUTHORITY_COMPONENTS = ["hard_hit_rate", "barrel_rate",
                        "mean_exit_velocity", "p90_exit_velocity",
                        "xwoba_on_contact"]
DISCIPLINE_COMPONENTS = {
    "chase_rate": -1, "whiff_rate": -1,
    "strikeout_rate": -1, "walk_rate": 1,
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.gt(0)))


def aggregate_player_qualities(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hitter using transparent Statcast definitions."""
    required = {"batter", "description", "zone", "events", "launch_speed",
        "launch_speed_angle", "estimated_woba_using_speedangle", "woba_value",
        "woba_denom", "bat_speed"}
    missing = required.difference(frame)
    if missing:
        raise ValueError(f"Missing quality columns: {sorted(missing)}")
    d = frame.copy()
    d["swing"] = d.description.isin(CONTACT | WHIFF).astype("int8")
    d["whiff"] = d.description.isin(WHIFF).astype("int8")
    d["in_zone"] = d.zone.between(1, 9).astype("int8")
    d["out_zone"] = d.zone.between(11, 14).astype("int8")
    d["chase"] = (d.swing.eq(1) & d.out_zone.eq(1)).astype("int8")
    d["zone_swing"] = (d.swing.eq(1) & d.in_zone.eq(1)).astype("int8")
    d["batted_ball"] = d.launch_speed.notna().astype("int8")
    d["hard_hit"] = (d.launch_speed.ge(95) & d.batted_ball.eq(1)).astype("int8")
    d["barrel"] = (d.launch_speed_angle.eq(6) &
                    d.batted_ball.eq(1)).astype("int8")
    d["tracked_swing"] = d.bat_speed.notna().astype("int8")
    grouped = d.groupby("batter", observed=True)
    out = grouped.agg(pitches=("description", "size"),
        swings=("swing", "sum"), whiffs=("whiff", "sum"),
        in_zone_pitches=("in_zone", "sum"),
        out_zone_pitches=("out_zone", "sum"),
        chases=("chase", "sum"), zone_swings=("zone_swing", "sum"),
        batted_balls=("batted_ball", "sum"), hard_hits=("hard_hit", "sum"),
        barrels=("barrel", "sum"), tracked_swings=("tracked_swing", "sum"),
        mean_exit_velocity=("launch_speed", "mean"),
        xwoba_on_contact=("estimated_woba_using_speedangle", "mean"),
        mean_bat_speed=("bat_speed", "mean")).reset_index()
    p90 = grouped.launch_speed.quantile(.9).rename("p90_exit_velocity")
    out = out.merge(p90, on="batter", validate="one_to_one")

    pa = d.loc[d.events.notna()].copy()
    pa["strikeout"] = pa.events.isin(STRIKEOUT_EVENTS).astype("int8")
    pa["walk"] = pa.events.isin(WALK_EVENTS).astype("int8")
    pa["woba_num"] = pd.to_numeric(pa.woba_value, errors="coerce").fillna(0)
    pa["woba_den"] = pd.to_numeric(pa.woba_denom, errors="coerce").fillna(0)
    pa_summary = pa.groupby("batter", observed=True).agg(
        pa=("events", "size"), strikeouts=("strikeout", "sum"),
        walks=("walk", "sum"), woba_numerator=("woba_num", "sum"),
        woba_denominator=("woba_den", "sum")).reset_index()
    out = out.merge(pa_summary, on="batter", how="left", validate="one_to_one")
    out = out.rename(columns={"batter": "player_id"})
    for column in ["pa", "strikeouts", "walks", "woba_numerator",
                   "woba_denominator"]:
        out[column] = out[column].fillna(0)
    out["chase_rate"] = _safe_ratio(out.chases, out.out_zone_pitches)
    out["zone_swing_rate"] = _safe_ratio(out.zone_swings, out.in_zone_pitches)
    out["whiff_rate"] = _safe_ratio(out.whiffs, out.swings)
    out["strikeout_rate"] = _safe_ratio(out.strikeouts, out.pa)
    out["walk_rate"] = _safe_ratio(out.walks, out.pa)
    out["hard_hit_rate"] = _safe_ratio(out.hard_hits, out.batted_balls)
    out["barrel_rate"] = _safe_ratio(out.barrels, out.batted_balls)
    out["woba"] = _safe_ratio(out.woba_numerator, out.woba_denominator)
    return out


def add_domain_composites(paired: pd.DataFrame,
                          prior_suffix: str = "_prior",
                          future_suffix: str = "_future") -> pd.DataFrame:
    """Standardize both periods to prior-period scales and average domains."""
    d = paired.copy()
    for metric in set(AUTHORITY_COMPONENTS) | set(DISCIPLINE_COMPONENTS):
        prior = d[metric + prior_suffix].to_numpy(float)
        center, scale = float(np.nanmean(prior)), float(np.nanstd(prior, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise AssertionError(f"Cannot standardize {metric}")
        d[f"z_prior_{metric}"] = (prior - center) / scale
        d[f"z_future_{metric}"] = (
            d[metric + future_suffix].to_numpy(float) - center) / scale
    d["authority_composite_prior"] = d[
        [f"z_prior_{x}" for x in AUTHORITY_COMPONENTS]].mean(axis=1)
    d["authority_composite_future"] = d[
        [f"z_future_{x}" for x in AUTHORITY_COMPONENTS]].mean(axis=1)
    d["discipline_composite_prior"] = np.mean(np.column_stack([
        d[f"z_prior_{metric}"] * direction
        for metric, direction in DISCIPLINE_COMPONENTS.items()]), axis=1)
    d["discipline_composite_future"] = np.mean(np.column_stack([
        d[f"z_future_{metric}"] * direction
        for metric, direction in DISCIPLINE_COMPONENTS.items()]), axis=1)
    return d
