"""Features available before a pitch is delivered.

The current pitch type is unknown at prediction time. Exposure is consequently
averaged over the pitcher's repertoire, estimated strictly before the current
date. Current-day pitches only enter within-game exposure counts. Outcome and
pitch-profile histories update after the entire date, including doubleheaders.

Columns ending in ``_for_update`` use the realized type and are intended only
for fitting a curve after that pitch has been observed. They are deliberately
absent from every prediction feature allowlist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]
CONTACT = frozenset({"foul", "foul_tip", "hit_into_play", "hit_into_play_no_out",
                     "hit_into_play_score", "foul_bunt", "bunt_foul_tip", "foul_pitchout"})
WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt",
                   "swinging_pitchout"})
TAKE = frozenset({"ball", "blocked_ball", "called_strike", "hit_by_pitch", "pitchout",
                  "intent_ball"})
AUTOMATIC = frozenset({"automatic_ball", "automatic_strike"})
# A fixed vocabulary prevents future pitch types from changing earlier priors.
# Missing or unrecognized type codes enter UN, including in exposure histories.
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH",
               "FS", "FO", "SC", "KN", "EP", "FA", "FT", "PO", "IN", "UN")
PHYSICAL = ("release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z")
CONTEXT_NUM = ("balls", "strikes", "inning", "outs_when_up", "bat_score", "fld_score",
               "pitch_number", "at_bat_number")
PRE_PITCH_CAT = ("batter", "pitcher", "stand", "p_throws", "inning_topbot")
REPERTOIRE_FEATURES = tuple(f"repertoire_{code}" for code in PITCH_TYPES)
EXPOSURE_FEATURES = ("weighted_expected_log_other_h7", "weighted_expected_log_own_h7",
                     "expected_log_game_other", "expected_log_game_own",
                     "weighted_other_h7_second_moment")
HISTORY_FEATURES = (
    "prior_hitter_contact_rate", "prior_hitter_contact_given_swing",
    "prior_hitter_swing_rate", "log_prior_hitter_pitch_count",
    "log_prior_hitter_labeled_pitch_count", "log_prior_hitter_swing_count",
    "log_prior_all_stock_h7", "log_prior_pitcher_pitch_count",
) + tuple(f"prior_pitcher_{name}_{stat}" for name in PHYSICAL for stat in ("mean", "n"))
PRE_PITCH_NUM = CONTEXT_NUM + ("day_of_year", "base_1_occupied", "base_2_occupied",
                                "base_3_occupied") + HISTORY_FEATURES + REPERTOIRE_FEATURES
UPDATE_FEATURES = ("log_other_h7_for_update", "log_own_h7_for_update",
                   "log_game_other_for_update", "log_game_own_for_update",
                   "prior_all_stock_h7_for_update")
INPUT_COLUMNS = tuple(dict.fromkeys(KEY + ["game_date", "batter", "pitcher", "pitch_type",
    "description", "game_type"] + list(CONTEXT_NUM) + list(PRE_PITCH_CAT) +
    ["on_1b", "on_2b", "on_3b"] + list(PHYSICAL)))


def build_pre_pitch_features(raw: pd.DataFrame, half_life_days: float = 7,
                             repertoire_prior_strength: float = 20) -> pd.DataFrame:
    """Return delivered regular-season pitches with strictly prior features.

    All supplied raw columns are retained, so callers must select model inputs
    using PRE_PITCH_NUM, PRE_PITCH_CAT and, when appropriate, EXPOSURE_FEATURES.
    Missing optional context fields are explicit NaNs. Unrecognized outcome
    descriptions remain in exposure histories but have missing binary labels
    and ``label_valid=False``. They do not enter empirical contact/swing rates.

    Repertoire probabilities use cumulative prior-date pitcher type counts
    plus ``repertoire_prior_strength`` pseudopitches distributed according to
    cumulative prior-date league type frequencies. Before any history exists,
    the fixed PITCH_TYPES vocabulary has a uniform prior. Prior pitcher means
    use finite observations only and have a zero fallback accompanied by n=0.
    Batters' contact and swing rates use independent Beta(1, 1) smoothing.

    The historical column suffix h7 names the prespecified default. A different
    half_life_days changes the decay while preserving this stable API.
    """
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be positive and finite.")
    if not np.isfinite(repertoire_prior_strength) or repertoire_prior_strength <= 0:
        raise ValueError("repertoire_prior_strength must be positive and finite.")
    required = KEY + ["game_date", "batter", "pitcher", "pitch_type", "description"]
    missing = set(required).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing pre-pitch columns: {sorted(missing)}")
    keep = ~raw.description.isin(AUTOMATIC)
    if "game_type" in raw:
        keep &= raw.game_type.eq("R")
    frame = raw.loc[keep].copy()
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.normalize()
    if frame[KEY + ["game_date", "batter", "pitcher"]].isna().any().any():
        raise ValueError("Delivered pitches require nonmissing identifiers and dates.")
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate pitch identifiers must be resolved before feature construction.")
    frame = frame.sort_values(["game_date"] + KEY, kind="stable").reset_index(drop=True)
    for name in CONTEXT_NUM:
        if name not in frame:
            frame[name] = np.nan
    for name in PRE_PITCH_CAT:
        if name not in frame:
            frame[name] = pd.NA
    valid = frame.description.isin(CONTACT | WHIFF | TAKE).to_numpy()
    contact = frame.description.isin(CONTACT).to_numpy()
    swing = frame.description.isin(CONTACT | WHIFF).to_numpy()
    frame["label_valid"] = valid
    frame["is_contact"] = pd.array(np.where(valid, contact, None), dtype="boolean")
    frame["is_swing"] = pd.array(np.where(valid, swing, None), dtype="boolean")
    frame["is_whiff"] = pd.array(np.where(valid, swing & ~contact, None), dtype="boolean")
    frame["day_of_year"] = frame.game_date.dt.dayofyear.astype("int16")
    for base in (1, 2, 3):
        source = f"on_{base}b"
        frame[f"base_{base}_occupied"] = (frame[source].notna().astype("int8")
                                           if source in frame else np.nan)
    names = HISTORY_FEATURES + REPERTOIRE_FEATURES + EXPOSURE_FEATURES + UPDATE_FEATURES
    output = {name: np.zeros(len(frame), dtype=np.float32) for name in names}
    if frame.empty:
        return pd.concat([frame, pd.DataFrame(output)], axis=1)

    n_types = len(PITCH_TYPES)
    type_codes = pd.Categorical(frame.pitch_type, categories=PITCH_TYPES).codes.copy()
    type_codes[type_codes < 0] = PITCH_TYPES.index("UN")
    days = frame.game_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    batters = frame.batter.to_numpy()
    pitchers = frame.pitcher.to_numpy()
    games = frame.game_pk.to_numpy()
    physics = np.column_stack([pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if name in frame else np.full(len(frame), np.nan) for name in PHYSICAL])
    # Each stock state is (last observed date, vector); decaying is lazy.
    hitter_stocks: dict = {}
    pair_stocks: dict = {}
    # Hitter statistics: pitches, labeled pitches, contacts, swings.
    hitter_stats: dict = {}
    # Pitcher statistics: type counts, physical sums, physical counts.
    pitcher_stats: dict = {}
    league = np.zeros(n_types)
    day_starts = np.r_[0, np.flatnonzero(np.diff(days)) + 1, len(frame)]
    zero_types = np.zeros(n_types)
    zero_stats = np.zeros(4)

    def stock(states, key, date):
        previous = states.get(key)
        if previous is None:
            return np.zeros(n_types)
        return previous[1] * np.exp2(-(date - previous[0]) / half_life_days)

    for start, end in zip(day_starts[:-1], day_starts[1:]):
        day = int(days[start])
        day_b, b_idx = np.unique(batters[start:end], return_inverse=True)
        day_p, p_idx = np.unique(pitchers[start:end], return_inverse=True)
        pairs, pair_idx = np.unique(np.column_stack((b_idx, p_idx)), axis=0, return_inverse=True)
        b_stock = np.array([stock(hitter_stocks, b, day) for b in day_b])
        bp_stock = np.array([stock(pair_stocks, (day_b[b], day_p[p]), day) for b, p in pairs])
        bs = np.array([hitter_stats.get(b, zero_stats) for b in day_b])
        global_q = league / league.sum() if league.sum() else np.full(n_types, 1 / n_types)
        repertoires = np.empty((len(day_p), n_types))
        p_counts = np.empty(len(day_p))
        p_means = np.zeros((len(day_p), len(PHYSICAL)))
        p_ns = np.zeros_like(p_means)
        for i, p in enumerate(day_p):
            history = pitcher_stats.get(p)
            counts = zero_types if history is None else history[0]
            p_counts[i] = counts.sum()
            repertoires[i] = (counts + repertoire_prior_strength * global_q) / (
                p_counts[i] + repertoire_prior_strength)
            if history is not None:
                p_ns[i] = history[2]
                p_means[i] = np.divide(history[1], history[2], out=np.zeros(len(PHYSICAL)),
                                       where=history[2] > 0)
        pair_q = repertoires[pairs[:, 1]]
        own_log = np.log1p(bp_stock)
        other_log = np.log1p(np.maximum(b_stock[pairs[:, 0]] - bp_stock, 0))
        pair_values = {
            "weighted_expected_log_other_h7": np.sum(pair_q * other_log, axis=1),
            "weighted_expected_log_own_h7": np.sum(pair_q * own_log, axis=1),
            "weighted_other_h7_second_moment": np.sum(pair_q * other_log**2, axis=1),
        }
        for name, values in pair_values.items():
            output[name][start:end] = values[pair_idx]
        for j, name in enumerate(REPERTOIRE_FEATURES):
            output[name][start:end] = repertoires[p_idx, j]
        b_values = {
            "prior_hitter_contact_rate": (bs[:, 2] + 1) / (bs[:, 1] + 2),
            "prior_hitter_contact_given_swing": (bs[:, 2] + 1) / (bs[:, 3] + 2),
            "prior_hitter_swing_rate": (bs[:, 3] + 1) / (bs[:, 1] + 2),
            "log_prior_hitter_pitch_count": np.log1p(bs[:, 0]),
            "log_prior_hitter_labeled_pitch_count": np.log1p(bs[:, 1]),
            "log_prior_hitter_swing_count": np.log1p(bs[:, 3]),
            "log_prior_all_stock_h7": np.log1p(b_stock.sum(axis=1)),
            "prior_all_stock_h7_for_update": b_stock.sum(axis=1),
        }
        for name, values in b_values.items():
            output[name][start:end] = values[b_idx]
        output["log_prior_pitcher_pitch_count"][start:end] = np.log1p(p_counts[p_idx])
        for j, name in enumerate(PHYSICAL):
            output[f"prior_pitcher_{name}_mean"][start:end] = p_means[p_idx, j]
            output[f"prior_pitcher_{name}_n"][start:end] = p_ns[p_idx, j]
        day_types = type_codes[start:end]
        output["log_other_h7_for_update"][start:end] = other_log[pair_idx, day_types]
        output["log_own_h7_for_update"][start:end] = own_log[pair_idx, day_types]

        # Only these counts incorporate previously delivered current-game pitches.
        game_b, game_bp = {}, {}
        previous_game = None
        for local, absolute in enumerate(range(start, end)):
            if games[absolute] != previous_game:
                previous_game = games[absolute]
                game_b, game_bp = {}, {}
            b, p, pair = b_idx[local], p_idx[local], pair_idx[local]
            bg = game_b.setdefault(b, np.zeros(n_types))
            bpg = game_bp.setdefault(pair, np.zeros(n_types))
            gl_own = np.log1p(bpg)
            gl_other = np.log1p(np.maximum(bg - bpg, 0))
            output["expected_log_game_own"][absolute] = repertoires[p] @ gl_own
            output["expected_log_game_other"][absolute] = repertoires[p] @ gl_other
            t = day_types[local]
            output["log_game_other_for_update"][absolute] = gl_other[t]
            output["log_game_own_for_update"][absolute] = gl_own[t]
            bg[t] += 1
            bpg[t] += 1

        # All prior-date states update only after prediction for the whole date.
        increments_b = np.zeros_like(b_stock)
        increments_bp = np.zeros_like(bp_stock)
        increments_p = np.zeros_like(repertoires)
        np.add.at(increments_b, (b_idx, day_types), 1)
        np.add.at(increments_bp, (pair_idx, day_types), 1)
        np.add.at(increments_p, (p_idx, day_types), 1)
        stats_increment = np.zeros_like(bs)
        np.add.at(stats_increment[:, 0], b_idx, 1)
        np.add.at(stats_increment[:, 1], b_idx, valid[start:end])
        np.add.at(stats_increment[:, 2], b_idx, contact[start:end])
        np.add.at(stats_increment[:, 3], b_idx, swing[start:end])
        for i, b in enumerate(day_b):
            hitter_stocks[b] = (day, b_stock[i] + increments_b[i])
            hitter_stats[b] = bs[i] + stats_increment[i]
        for i, (b, p) in enumerate(pairs):
            pair_stocks[(day_b[b], day_p[p])] = (day, bp_stock[i] + increments_bp[i])
        phy = physics[start:end]
        observed = np.isfinite(phy)
        physical_sum = np.zeros((len(day_p), len(PHYSICAL)))
        physical_n = np.zeros_like(physical_sum)
        np.add.at(physical_sum, p_idx, np.where(observed, phy, 0))
        np.add.at(physical_n, p_idx, observed)
        for i, p in enumerate(day_p):
            old = pitcher_stats.get(p)
            current = (increments_p[i], physical_sum[i], physical_n[i])
            pitcher_stats[p] = current if old is None else tuple(a + b for a, b in zip(old, current))
        league += increments_p.sum(axis=0)
    return pd.concat([frame, pd.DataFrame(output)], axis=1)
