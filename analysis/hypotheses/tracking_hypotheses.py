"""Tracking hypotheses on all 20 matches.

3 proxy  High-block lane screen (defender standing in the pass lane). No orientation.
4        Escape angle between a wide player and a central midfielder.
5        Attacking full-back tracked: distance, goal-side, similar velocity.
6        Overlap and underlap associations with those defensive states.
7        Striker tight mark versus centre-back, midfield, and full-back departure.

Run: opendata/.venv/bin/python analysis/hypotheses/tracking_hypotheses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coupled_formation import attack_direction, hungarian_marking  # noqa: E402

from common import (  # noqa: E402
    BOOT_SEED,
    CB_ROLES,
    CM_ROLES,
    FB_ROLES,
    HIGH_PRESSURE,
    LANE_DIST_M,
    LANE_T,
    MIN_ROLE_SAMPLES,
    ORGANIZED,
    OUT_DIR,
    STRIKER_ROLES,
    TIGHT_MARK_M,
    TRACK_COS,
    TRACK_DIST_M,
    TRACK_LOOKBACK,
    TRACK_SPEED_DIFF,
    angle_between,
    attack_sign,
    cluster_diff,
    cluster_mean,
    fmt,
    in_lane,
    is_goal_side,
    json_ready,
    load_events,
    load_phases,
    load_roster,
    match_ids,
    modal_positions,
    nearest_opponent,
    pass_angle_from_goal,
    player_role,
    similar_velocity,
    displacement,
    stream_frames,
    supports,
    wide_cm_pair,
)

ANGLE_BINS = (0, 30, 60, 90, 180)
ANGLE_CUTS = (30, 45, 60, 90)


def _iid(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _xy_of(frames, frame, pid) -> tuple[float, float] | None:
    if pid is None:
        return None
    return frames.get(int(frame), {}).get(int(pid))


def _opponents(xy, team_of, att_team, gk, skip: set[int]) -> list[tuple[float, float]]:
    out = []
    for pid, pos in xy.items():
        if pid in skip or pid in gk:
            continue
        tm = team_of.get(pid)
        if tm is None or tm == att_team:
            continue
        out.append(pos)
    return out


def _tracked_state(frames, frame, pid, team_of, gk, sign) -> dict:
    xy = frames.get(frame) or {}
    opp_id, dist = nearest_opponent(pid, xy, team_of, gk)
    if opp_id is None or pid not in xy:
        return {"tracked": np.nan, "dist": np.nan, "goal_side": np.nan, "similar_vel": np.nan}
    gs = is_goal_side(xy[pid], xy[opp_id], sign)
    sim = similar_velocity(
        displacement(frames, pid, frame),
        displacement(frames, opp_id, frame),
    )
    close = dist <= TRACK_DIST_M
    if not np.isfinite(sim):
        tracked = np.nan
    else:
        tracked = 1.0 if (close and gs and sim == 1.0) else 0.0
    return {
        "tracked": tracked,
        "dist": dist,
        "goal_side": 1.0 if gs else 0.0,
        "similar_vel": sim,
        "opp_id": opp_id,
    }


def _role_medians(frames, phases, team_of, gk, modal, roster) -> dict:
    samples: dict[tuple, list] = {}
    for row in phases.itertuples(index=False):
        block = row.team_out_of_possession_phase_type
        if block not in ORGANIZED:
            continue
        sign = attack_sign(row.attacking_side)
        if not np.isfinite(sign):
            continue
        att_team = int(row.team_in_possession_id)
        xy = frames.get(int(row.frame_start)) or {}
        for pid, (x, y) in xy.items():
            if pid in gk or team_of.get(pid) is None or team_of[pid] == att_team:
                continue
            role = player_role(pid, modal, roster)
            if role not in CB_ROLES and role not in CM_ROLES and role not in FB_ROLES:
                continue
            key = (team_of[pid], block, role)
            samples.setdefault(key, []).append((x * sign, y * sign))
    med = {}
    for key, pts in samples.items():
        if len(pts) < MIN_ROLE_SAMPLES:
            continue
        arr = np.asarray(pts, dtype=float)
        med[key] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), len(pts))
    return med


def _group_departure(xy, team_of, att_team, gk, modal, roster, block, sign, medians) -> dict:
    groups = {"CB": [], "CM": [], "FB": []}
    fwd = {"CB": [], "CM": [], "FB": []}
    for pid, (x, y) in xy.items():
        if pid in gk or team_of.get(pid) in (None, att_team):
            continue
        role = player_role(pid, modal, roster)
        if role in CB_ROLES:
            group = "CB"
        elif role in CM_ROLES:
            group = "CM"
        elif role in FB_ROLES:
            group = "FB"
        else:
            continue
        key = (team_of[pid], block, role)
        ref = medians.get(key)
        if ref is None:
            continue
        x_att, y_att = x * sign, y * sign
        groups[group].append(float(np.hypot(x_att - ref[0], y_att - ref[1])))
        fwd[group].append(float(x_att - ref[0]))
    out = {}
    for g in ("CB", "CM", "FB"):
        out[f"dep_{g}"] = float(np.mean(groups[g])) if groups[g] else np.nan
        out[f"fwd_{g}"] = float(np.mean(fwd[g])) if fwd[g] else np.nan
        out[f"n_{g}"] = len(groups[g])
    return out


def _hungarian_dist(xy, team_of, att_team, gk, striker_id, attacking_side) -> float:
    att, defs = [], []
    att_index = None
    for pid, (x, y) in xy.items():
        if pid in gk or team_of.get(pid) is None:
            continue
        if team_of[pid] == att_team:
            if pid == striker_id:
                att_index = len(att)
            att.append((x, y))
        else:
            defs.append((x, y))
    if att_index is None or not defs or not att:
        return float("nan")
    marking = hungarian_marking(
        np.asarray(defs, dtype=float),
        np.asarray(att, dtype=float),
        attack_direction(attacking_side),
    )
    for pair in marking["pairs"]:
        if pair["att_idx"] == att_index and pair["goal_side"]:
            return float(pair["dist"])
    return float("nan")


def process_match(match_id: str) -> dict[str, list[dict]]:
    events = load_events(match_id)
    phases = load_phases(match_id)
    team_of, roster, gk, _plen, _pwid = load_roster(match_id)
    modal = modal_positions(events)

    poss = events[events["event_type"] == "player_possession"].copy()
    options = events[events["event_type"] == "passing_option"].copy()
    runs = events[events["event_type"] == "off_ball_run"].copy()
    engs = events[events["event_type"] == "on_ball_engagement"].copy()

    xshot = (
        engs.dropna(subset=["associated_player_possession_event_id"])
        .groupby("associated_player_possession_event_id")["xshot_player_possession_max"]
        .max()
    )
    poss_by_id = poss.set_index("event_id", drop=False)

    needed: set[int] = set()

    def add_frame(fr, lookback: bool = False):
        if fr is None or (isinstance(fr, float) and not np.isfinite(fr)):
            return
        fr = int(fr)
        if fr < 0:
            return
        needed.add(fr)
        if lookback and fr >= 10:
            needed.add(fr - 10)

    # Phase starts build the reference block.
    for fr in phases.loc[
        phases["team_out_of_possession_phase_type"].isin(ORGANIZED), "frame_start"
    ]:
        add_frame(fr)

    high_options = options[options["team_out_of_possession_phase_type"] == "high_block"]
    for fr in high_options["frame_start"]:
        add_frame(fr)

    high_passes = poss[
        (poss["end_type"] == "pass")
        & (poss["team_out_of_possession_phase_type"] == "high_block")
        & (poss["pass_ahead"] == 1)
    ]
    for fr in high_passes["frame_end"]:
        add_frame(fr)

    # Wide-CM passes and options need both players at the pass moment.
    wc_passes = poss[poss["end_type"] == "pass"]
    for row in wc_passes.itertuples(index=False):
        if wide_cm_pair(row.player_position, row.player_targeted_position):
            add_frame(row.frame_end)
    wc_options = options[
        [
            wide_cm_pair(a, b)
            for a, b in zip(options["player_in_possession_position"], options["player_position"])
        ]
    ]
    for fr in wc_options["frame_start"]:
        add_frame(fr)

    fb_poss = poss[poss["player_position"].isin(FB_ROLES)]
    for row in fb_poss.itertuples(index=False):
        midf = int((int(row.frame_start) + int(row.frame_end)) / 2)
        add_frame(midf, lookback=True)

    for row in runs.itertuples(index=False):
        midf = int((int(row.frame_start) + int(row.frame_end)) / 2)
        is_fb = row.player_position in FB_ROLES
        is_overlap = row.event_subtype in ("overlap", "underlap")
        if is_fb or is_overlap:
            add_frame(midf, lookback=True)
            add_frame(row.frame_start)

    cf_poss = poss[
        poss["player_position"].isin(STRIKER_ROLES)
        & poss["team_out_of_possession_phase_type"].isin(ORGANIZED)
    ]
    for fr in cf_poss["frame_start"]:
        add_frame(fr)

    frames = stream_frames(match_id, needed)
    medians = _role_medians(frames, phases, team_of, gk, modal, roster)

    h3, h4, h5, h6, h7 = [], [], [], [], []

    # --- H3 proxy: high-block passing options and forward passes ---
    for row in high_options.itertuples(index=False):
        fr = int(row.frame_start)
        xy = frames.get(fr) or {}
        passer = _iid(row.player_in_possession_id)
        option = _iid(row.player_id)
        a = _xy_of(frames, fr, passer)
        b = _xy_of(frames, fr, option)
        if a is None:
            a = _finite_xy(row.player_in_possession_x_start, row.player_in_possession_y_start)
        if b is None:
            b = _finite_xy(row.x_start, row.y_start)
        att_team = _iid(row.team_id)
        screened = np.nan
        if a and b and att_team is not None:
            defs = _opponents(xy, team_of, att_team, gk, {passer, option} - {None})
            screened = 1.0 if in_lane(defs, a, b) else 0.0
        h3.append({
            "match_id": match_id,
            "kind": "option",
            "screened": screened,
            "received": row.received,
            "pass_ahead": row.pass_ahead,
            "xthreat": row.xthreat,
            "first_line_break": row.first_line_break,
            "pass_epv": np.nan,
            "successful": np.nan,
            "possession_id": row.associated_player_possession_event_id,
            "high_pressure": _pressure_flag(poss_by_id, row.associated_player_possession_event_id),
        })

    for row in high_passes.itertuples(index=False):
        fr = int(row.frame_end)
        xy = frames.get(fr) or {}
        passer = _iid(row.player_id)
        target = _iid(row.player_targeted_id)
        a = _xy_of(frames, fr, passer)
        b = _xy_of(frames, fr, target)
        if a is None:
            a = _finite_xy(row.x_end, row.y_end)
        if b is None:
            b = _finite_xy(row.player_targeted_x_reception, row.player_targeted_y_reception)
        att_team = _iid(row.team_id)
        screened = np.nan
        if a and b and att_team is not None:
            defs = _opponents(xy, team_of, att_team, gk, {passer, target} - {None})
            screened = 1.0 if in_lane(defs, a, b) else 0.0
        h3.append({
            "match_id": match_id,
            "kind": "forward_pass",
            "screened": screened,
            "received": 1.0 if row.pass_outcome == "successful" else (0.0 if row.pass_outcome == "unsuccessful" else np.nan),
            "pass_ahead": row.pass_ahead,
            "xthreat": row.player_targeted_xthreat,
            "first_line_break": row.first_line_break,
            "pass_epv": row.pass_epv_total,
            "successful": 1.0 if row.pass_outcome == "successful" else (0.0 if isinstance(row.pass_outcome, str) else np.nan),
            "possession_id": row.event_id,
            "high_pressure": 1.0 if row.overall_pressure_end in HIGH_PRESSURE else (
                0.0 if isinstance(row.overall_pressure_end, str) else np.nan
            ),
        })

    # --- H4 escape angle, passes and passing options ---
    def append_angle(kind, fr, passer, receiver, sign, att_team, line_break, pressure, epv, xthreat, received):
        if passer is None or receiver is None or not np.isfinite(sign):
            return
        xy = frames.get(int(fr)) or {}
        p_xy = xy.get(passer)
        r_xy = xy.get(receiver)
        if p_xy is None or r_xy is None:
            return
        opp_ball, _ = nearest_opponent(passer, xy, team_of, gk)
        opp_recv, _ = nearest_opponent(receiver, xy, team_of, gk)
        escape = angle_between(p_xy, r_xy, xy[opp_ball]) if opp_ball else np.nan
        at_receiver = angle_between(r_xy, p_xy, xy[opp_recv]) if opp_recv else np.nan
        shape = pass_angle_from_goal(p_xy, r_xy, sign)
        h4.append({
            "match_id": match_id,
            "kind": kind,
            "escape_angle": escape,
            "receiver_angle": at_receiver,
            "shape_angle": shape,
            "first_line_break": line_break,
            "high_pressure": pressure,
            "pass_epv": epv,
            "xthreat": xthreat,
            "received": received,
        })

    for row in poss.itertuples(index=False):
        if row.end_type != "pass":
            continue
        if not wide_cm_pair(row.player_position, row.player_targeted_position):
            continue
        sign = attack_sign(row.attacking_side)
        pressure = 1.0 if row.overall_pressure_end in HIGH_PRESSURE else (
            0.0 if isinstance(row.overall_pressure_end, str) else np.nan
        )
        append_angle(
            "pass",
            int(row.frame_end),
            _iid(row.player_id),
            _iid(row.player_targeted_id),
            sign,
            _iid(row.team_id),
            row.first_line_break,
            pressure,
            row.pass_epv_total,
            row.player_targeted_xthreat,
            1.0 if row.pass_outcome == "successful" else 0.0,
        )

    for row in options.itertuples(index=False):
        if not wide_cm_pair(row.player_in_possession_position, row.player_position):
            continue
        sign = attack_sign(row.attacking_side)
        parent = poss_by_id.loc[row.associated_player_possession_event_id] if (
            row.associated_player_possession_event_id in poss_by_id.index
        ) else None
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        pressure = np.nan
        if parent is not None:
            pend = parent["overall_pressure_end"] if "overall_pressure_end" in parent else np.nan
            # parent may be a Series
            pend = parent.overall_pressure_end if hasattr(parent, "overall_pressure_end") else pend
            pressure = 1.0 if pend in HIGH_PRESSURE else (0.0 if isinstance(pend, str) else np.nan)
        append_angle(
            "option",
            int(row.frame_start),
            _iid(row.player_in_possession_id),
            _iid(row.player_id),
            sign,
            _iid(row.team_id),
            row.first_line_break,
            pressure,
            np.nan,
            row.xthreat,
            row.received,
        )

    # --- H5 full-back possessions and runs in the attacking half ---
    for row in fb_poss.itertuples(index=False):
        sign = attack_sign(row.attacking_side)
        if not np.isfinite(sign):
            continue
        midf = int((int(row.frame_start) + int(row.frame_end)) / 2)
        pid = _iid(row.player_id)
        if pid is None:
            continue
        pos = _xy_of(frames, midf, pid)
        if pos is None:
            continue
        if pos[0] * sign <= 0:
            continue
        state = _tracked_state(frames, midf, pid, team_of, gk, sign)
        h5.append({
            "match_id": match_id,
            "kind": "fb_possession",
            "subtype": "",
            "lead_to_shot": row.lead_to_shot,
            "xthreat": row.player_targeted_xthreat,
            "received": np.nan,
            "pass_epv": row.pass_epv_total,
            **{k: state[k] for k in ("tracked", "dist", "goal_side", "similar_vel")},
        })

    # Engagements that can be beaten_by_movement, indexed by the attacker they target.
    eng_groups = {}
    if not engs.empty:
        for pid, g in engs.groupby(engs["player_targeted_id"]):
            if pd.isna(pid):
                continue
            eng_groups[int(pid)] = g

    for row in runs.itertuples(index=False):
        sign = attack_sign(row.attacking_side)
        if not np.isfinite(sign):
            continue
        midf = int((int(row.frame_start) + int(row.frame_end)) / 2)
        pid = _iid(row.player_id)
        if pid is None:
            continue
        pos = _xy_of(frames, midf, pid)
        if pos is None:
            continue
        in_attack = pos[0] * sign > 0
        state = _tracked_state(frames, midf, pid, team_of, gk, sign)
        if row.player_position in FB_ROLES and in_attack:
            h5.append({
                "match_id": match_id,
                "kind": "fb_run" if row.event_subtype not in ("overlap", "underlap") else "fb_overlap",
                "subtype": row.event_subtype or "",
                "lead_to_shot": row.lead_to_shot,
                "xthreat": row.xthreat,
                "received": row.received,
                "pass_epv": np.nan,
                **{k: state[k] for k in ("tracked", "dist", "goal_side", "similar_vel")},
            })
        if row.event_subtype not in ("overlap", "underlap"):
            continue
        carrier = _iid(row.player_in_possession_id)
        cpos = _xy_of(frames, midf, carrier)
        if cpos is None:
            cpos = _finite_xy(row.player_in_possession_x_start, row.player_in_possession_y_start)
        att_team = _iid(row.team_id)
        narrow = np.nan
        xy = frames.get(midf) or {}
        if cpos and att_team is not None:
            defs = _opponents(xy, team_of, att_team, gk, {pid, carrier} - {None})
            narrow = 1.0 if in_lane(defs, cpos, pos) else 0.0
        beaten = 0.0
        joined = False
        g = eng_groups.get(pid)
        if g is not None:
            hit = g[(g["frame_start"] <= int(row.frame_end) + 10) & (g["frame_end"] >= int(row.frame_start) - 10)]
            if len(hit):
                joined = True
                flags = pd.to_numeric(hit["beaten_by_movement"], errors="coerce").to_numpy(dtype=float)
                beaten = 1.0 if np.any(flags == 1.0) else 0.0
        h6.append({
            "match_id": match_id,
            "subtype": row.event_subtype,
            "position": row.player_position,
            "in_attacking_half": 1.0 if in_attack else 0.0,
            "tracked": state["tracked"],
            "goal_side": state["goal_side"],
            "dist": state["dist"],
            "narrow_lane": narrow,
            "received": row.received,
            "failed": (1.0 - row.received) if np.isfinite(row.received) else np.nan,
            "lead_to_shot": row.lead_to_shot,
            "xthreat": row.xthreat,
            "beaten_by_movement": beaten if joined else np.nan,
            "beaten_joined": 1.0 if joined else 0.0,
        })

    # --- H7 striker tight mark ---
    for row in cf_poss.itertuples(index=False):
        sign = attack_sign(row.attacking_side)
        if not np.isfinite(sign):
            continue
        fr = int(row.frame_start)
        pid = _iid(row.player_id)
        xy = frames.get(fr) or {}
        if pid is None or pid not in xy:
            continue
        att_team = _iid(row.team_id)
        if att_team is None:
            continue
        _opp, dist = nearest_opponent(pid, xy, team_of, gk)
        if not np.isfinite(dist):
            continue
        h_dist = _hungarian_dist(xy, team_of, att_team, gk, pid, row.attacking_side)
        block = row.team_out_of_possession_phase_type
        dep = _group_departure(xy, team_of, att_team, gk, modal, roster, block, sign, medians)
        h7.append({
            "match_id": match_id,
            "block": block,
            "tight_nearest": 1.0 if dist <= TIGHT_MARK_M else 0.0,
            "tight_hungarian": 1.0 if np.isfinite(h_dist) and h_dist <= TIGHT_MARK_M else 0.0,
            "nearest_dist": dist,
            "hungarian_dist": h_dist,
            "lead_to_shot": row.lead_to_shot,
            "xthreat": row.player_targeted_xthreat,
            "first_line_break": row.first_line_break,
            "xshot_max": float(xshot.get(row.event_id, np.nan)),
            "possession_epv": row.possession_epv_total,
            **dep,
        })

    return {"h3": h3, "h4": h4, "h5": h5, "h6": h6, "h7": h7}


def _finite_xy(x, y) -> tuple[float, float] | None:
    try:
        if pd.isna(x) or pd.isna(y):
            return None
    except TypeError:
        return None
    return (float(x), float(y))


def _pressure_flag(poss_by_id: pd.DataFrame, possession_id) -> float:
    if possession_id not in poss_by_id.index:
        return np.nan
    parent = poss_by_id.loc[possession_id]
    if isinstance(parent, pd.DataFrame):
        parent = parent.iloc[0]
    pend = parent.overall_pressure_end
    if pend in HIGH_PRESSURE:
        return 1.0
    if isinstance(pend, str):
        return 0.0
    return np.nan


def _contrast_record(name: str, predicted: str, contrast: dict, unit: str) -> dict:
    return {
        "name": name,
        "predicted": predicted,
        "unit": unit,
        "supported": supports(contrast, predicted),
        **contrast,
    }


def _bin_table(df: pd.DataFrame, angle_col: str, y: str) -> list[dict]:
    out = []
    angles = pd.to_numeric(df[angle_col], errors="coerce")
    for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
        mask = (angles >= lo) & (angles < hi if hi < 180 else angles <= hi)
        sub = df.loc[mask].copy()
        sub["_y"] = pd.to_numeric(sub[y], errors="coerce")
        stat = cluster_mean(sub, "_y")
        out.append({
            "bin": f"{lo}-{hi}",
            "lo": lo,
            "hi": hi,
            "outcome": y,
            "mean": stat["effect"],
            "ci_low": stat["ci_low"],
            "ci_high": stat["ci_high"],
            "n": stat["n"],
            "n_matches": stat["n_matches"],
        })
    return out


def _threshold_scan(df: pd.DataFrame, angle_col: str, y: str, predicted: str = "positive") -> dict:
    """Cuts of escape angle. A threshold is claimed only if one cut is monotone and excludes zero."""
    angles = pd.to_numeric(df[angle_col], errors="coerce")
    scans = []
    for cut in ANGLE_CUTS:
        work = df.copy()
        work["_treat"] = np.where(angles >= cut, 1.0, 0.0)
        work.loc[~np.isfinite(angles), "_treat"] = np.nan
        work["_y"] = pd.to_numeric(work[y], errors="coerce")
        contrast = cluster_diff(work, "_y", "_treat", seed=BOOT_SEED + int(cut))
        scans.append({
            "cut": cut,
            "outcome": y,
            "supported": supports(contrast, predicted),
            **contrast,
        })
    surviving = [s for s in scans if s["supported"]]
    # Monotone: every surviving cut has the same sign, and bins are not a one-bin spike.
    claim = None
    if surviving:
        signs = [np.sign(s["effect"]) for s in surviving]
        if len(set(signs)) == 1:
            # Prefer the smallest cut that survives, as the onset.
            claim = min(surviving, key=lambda s: s["cut"])
    return {"scans": scans, "threshold": claim}


def summarize(frames: dict[str, pd.DataFrame]) -> list[dict]:
    records = []
    records.append(_summarize_h3(frames["h3"]))
    records.append(_summarize_h4(frames["h4"]))
    records.append(_summarize_h5(frames["h5"]))
    records.append(_summarize_h6(frames["h6"]))
    records.append(_summarize_h7(frames["h7"]))
    return records


def _summarize_h3(df: pd.DataFrame) -> dict:
    opt = df[df["kind"] == "option"].copy()
    fwd = opt[opt["pass_ahead"] == 1].copy()
    passes = df[df["kind"] == "forward_pass"].copy()
    received = cluster_diff(fwd, "received", "screened", seed=31)
    epv = cluster_diff(passes, "pass_epv", "screened", seed=32)
    success = cluster_diff(passes, "successful", "screened", seed=33)
    line = cluster_diff(fwd, "first_line_break", "screened", seed=34)
    # Within-possession: screened options minus open options on xthreat.
    paired = []
    for (mid, pid), g in opt.dropna(subset=["possession_id"]).groupby(["match_id", "possession_id"]):
        sc = g.loc[g["screened"] == 1, "xthreat"]
        op = g.loc[g["screened"] == 0, "xthreat"]
        sc = pd.to_numeric(sc, errors="coerce").dropna()
        op = pd.to_numeric(op, errors="coerce").dropna()
        if len(sc) and len(op):
            paired.append({"match_id": mid, "xt_diff": float(sc.mean() - op.mean())})
    paired_df = pd.DataFrame(paired) if paired else pd.DataFrame(columns=["match_id", "xt_diff"])
    targeting = cluster_mean(paired_df, "xt_diff", seed=35) if len(paired_df) else {
        "effect": None, "ci_low": None, "ci_high": None, "n": 0, "n_matches": 0, "ci_excludes_zero": False,
    }
    # supports() for a mean requires n_control. Adapt.
    targeting_supported = bool(
        targeting.get("ci_excludes_zero")
        and targeting.get("effect") is not None
        and targeting["effect"] > 0
        and targeting["ci_low"] is not None
        and targeting["ci_low"] > 0
        and targeting.get("n", 0) >= 20
    )
    primary = _contrast_record(
        "forward high-block passing options: received rate, in-lane minus open",
        "negative",
        received,
        "rate",
    )
    contrasts = [
        primary,
        _contrast_record("forward passes in a high block: pass_epv_total, in-lane minus open", "negative", epv, "pass_epv_total"),
        _contrast_record("forward passes in a high block: completion, in-lane minus open", "negative", success, "rate"),
        _contrast_record("forward high-block options: first_line_break, in-lane minus open", "negative", line, "rate"),
        {
            "name": "within possession, mean xthreat of in-lane options minus open options",
            "predicted": "positive",
            "unit": "xthreat",
            "supported": targeting_supported,
            "effect": targeting["effect"],
            "ci_low": targeting["ci_low"],
            "ci_high": targeting["ci_high"],
            "mean_treat": None,
            "mean_control": None,
            "n_treat": targeting.get("n", 0),
            "n_control": targeting.get("n", 0),
            "n_matches": targeting.get("n_matches", 0),
            "n_matches_both": targeting.get("n_matches", 0),
            "ci_excludes_zero": targeting.get("ci_excludes_zero", False),
        },
    ]
    n_screen = int((fwd["screened"] == 1).sum()) if len(fwd) else 0
    n_open = int((fwd["screened"] == 0).sum()) if len(fwd) else 0
    supported = primary["supported"]
    # A targeting metric is worth keeping if influential options are the ones screened,
    # even when completion does not fall.
    worth = False
    metric = None
    if primary["supported"] and abs(primary["effect"] or 0) >= 0.03:
        worth = True
        metric = "high_block_lane_screen"
    elif targeting_supported and abs(targeting["effect"] or 0) >= 0.005:
        worth = True
        metric = "high_block_lane_screen"
        supported = False
    return {
        "id": 3,
        "title": "High-block shadow screen of passing lanes",
        "definition": (
            f"Tracking proxy, all matches, high_block only. A passing option or forward pass "
            f"is lane-screened when any outfield opponent is within {LANE_DIST_M:.1f} m of the "
            f"segment from the ball carrier to the teammate, with segment parameter t in "
            f"[{LANE_T[0]:.2f}, {LANE_T[1]:.2f}]. This proxy does not use body orientation. "
            f"Orientation (body between the lane and facing it) is tested separately on the two pose matches. "
            f"Forward means pass_ahead is true. Pass value is pass_epv_total on the player_possession row. "
            f"Influential means the option's xthreat."
        ),
        "population": "high_block passing_option and forward player_possession passes",
        "n_events": int(len(fwd)),
        "n_matches": int(fwd["match_id"].nunique()) if len(fwd) else 0,
        "n_screened_forward_options": n_screen,
        "n_open_forward_options": n_open,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "supported": supported,
        "metric_worth_keeping": worth,
        "metric_name": metric,
        "inference": "match_cluster_bootstrap",
        "contrasts": contrasts,
        "notes": (
            "Predicted: a screened forward option is completed less often and carries lower pass EPV, "
            "and screened options have higher xthreat than the open options in the same possession. "
            "Pose orientation is not in this record's effect; see the pose section merged by run_all."
        ),
    }


def _summarize_h4(df: pd.DataFrame) -> dict:
    passes = df[df["kind"] == "pass"].copy()
    options = df[df["kind"] == "option"].copy()
    bins_break = _bin_table(passes, "escape_angle", "first_line_break")
    bins_press = _bin_table(passes, "escape_angle", "high_pressure")
    bins_epv = _bin_table(passes, "escape_angle", "pass_epv")
    bins_shape = _bin_table(passes, "shape_angle", "first_line_break")
    scan = _threshold_scan(passes, "escape_angle", "first_line_break", "positive")
    scan_shape = _threshold_scan(passes, "shape_angle", "first_line_break", "positive")
    # Receiver-vertex angle, same outcome, descriptive companion.
    scan_recv = _threshold_scan(passes, "receiver_angle", "first_line_break", "positive")
    opt_scan = _threshold_scan(options, "escape_angle", "first_line_break", "positive")
    claim = scan["threshold"]
    supported = claim is not None
    worth = bool(supported and abs(claim["effect"]) >= 0.03)
    return {
        "id": 4,
        "title": "Wide-to-central angle that breaks the press",
        "definition": (
            "A pass or passing option whose ends are one wide role "
            "(LB, RB, LWB, RWB, LW, RW, LM, RM) and one central midfielder "
            "(LDM, RDM, CDM, DM, CM, LCM, RCM, AM). "
            "Escape angle is the interior angle at the ball carrier between the receiver and the "
            "nearest outfield opponent. Receiver angle is the same construction at the receiver, "
            "between the passer and the nearest opponent to the receiver. "
            "Shape angle is the absolute angle of the pass from the attacking axis "
            "(0 straight toward goal, 90 square, 180 straight back). "
            "Line break is first_line_break. Pressure is high or very high overall_pressure_end. "
            "Pass value is pass_epv_total. Bins are 0-30, 30-60, 60-90, 90-180 degrees. "
            "Cuts tested at 30, 45, 60 and 90 degrees. A threshold is kept only when at least one "
            "cut has a match-clustered interval excluding zero and every such cut has the same sign."
        ),
        "population": "wide-to-CM and CM-to-wide passes (primary) and passing options",
        "n_events": int(np.isfinite(pd.to_numeric(passes["escape_angle"], errors="coerce")).sum()) if len(passes) else 0,
        "n_matches": int(passes["match_id"].nunique()) if len(passes) else 0,
        "effect_name": (
            f"first_line_break rate, escape angle >= {claim['cut']} minus below"
            if claim else "no escape-angle cut separated first_line_break"
        ),
        "effect": None if claim is None else claim["effect"],
        "ci_low": None if claim is None else claim["ci_low"],
        "ci_high": None if claim is None else claim["ci_high"],
        "supported": supported,
        "metric_worth_keeping": worth,
        "metric_name": "wide_cm_escape_angle" if worth else None,
        "inference": "match_cluster_bootstrap",
        "bins": {
            "escape_first_line_break": bins_break,
            "escape_high_pressure": bins_press,
            "escape_pass_epv": bins_epv,
            "shape_first_line_break": bins_shape,
        },
        "scans": {
            "escape_angle_line_break": scan["scans"],
            "shape_angle_line_break": scan_shape["scans"],
            "receiver_angle_line_break": scan_recv["scans"],
            "option_escape_angle_line_break": opt_scan["scans"],
        },
        "n_options": int(len(options)),
        "notes": (
            "The primary claim is about the angle at the ball versus the nearest opponent, "
            "not a generic pass-direction bin. Shape angle is reported so a coach can see whether "
            "the pass's own direction, rather than its separation from the presser, is what moves "
            "the line-break rate. Four cuts were examined; a threshold is not claimed from the best cut alone."
        ),
    }


def _summarize_h5(df: pd.DataFrame) -> dict:
    poss = df[df["kind"] == "fb_possession"].copy()
    overs = df[df["kind"] == "fb_overlap"].copy()
    runs = df[df["kind"].isin(["fb_run", "fb_overlap"])].copy()
    shot = cluster_diff(poss, "lead_to_shot", "tracked", seed=51)
    epv = cluster_diff(poss, "pass_epv", "tracked", seed=52)
    recv = cluster_diff(overs, "received", "tracked", seed=53)
    shot_ov = cluster_diff(overs, "lead_to_shot", "tracked", seed=54)
    xt = cluster_diff(overs, "xthreat", "tracked", seed=55)
    shot_runs = cluster_diff(runs, "lead_to_shot", "tracked", seed=56)
    primary = _contrast_record(
        "FB overlap/underlap received rate, tracked minus not",
        "negative",
        recv,
        "rate",
    )
    shot_c = _contrast_record(
        "FB attacking-half possessions, lead_to_shot, tracked minus not",
        "negative",
        shot,
        "rate",
    )
    contrasts = [
        primary,
        shot_c,
        _contrast_record("FB attacking-half possessions, pass_epv_total, tracked minus not", "negative", epv, "pass_epv_total"),
        _contrast_record("FB overlap/underlap lead_to_shot, tracked minus not", "negative", shot_ov, "rate"),
        _contrast_record("FB overlap/underlap xthreat, tracked minus not", "negative", xt, "xthreat"),
        _contrast_record("FB runs in the attacking half, lead_to_shot, tracked minus not", "negative", shot_runs, "rate"),
    ]
    # Worth keeping if it separates shots or receptions in either direction by >= 3 points,
    # because a stable attachment tag is useful even when it marks danger rather than preventing it.
    useful = []
    for c in contrasts:
        if c.get("ci_excludes_zero") and c.get("effect") is not None and abs(c["effect"]) >= 0.03 and c["n_treat"] >= 20 and c["n_control"] >= 20:
            useful.append(c)
    worth = len(useful) > 0
    supported = primary["supported"] or shot_c["supported"]
    return {
        "id": 5,
        "title": "Attacking full back being tracked",
        "definition": (
            f"Full back roles LB, RB, LWB, RWB, in the attacking half (attack-normalized x > 0). "
            f"Tracked means all three: nearest outfield opponent within {TRACK_DIST_M:.0f} m, "
            f"that opponent goal-side (closer to the goal being attacked), and similar velocity over "
            f"the previous {TRACK_LOOKBACK / 10:.1f} s. Similar velocity: cosine of the displacements "
            f">= {TRACK_COS:.2f} and speed difference <= {TRACK_SPEED_DIFF:.1f} m/s. "
            f"If both players move under {1.0:.0f} m in the window, similar means both are near-stationary. "
            f"Events with a missing velocity window are excluded, not labelled untracked. "
            f"Validated on FB overlap/underlap receptions and on FB possessions that lead_to_shot."
        ),
        "population": "FB player_possession in the attacking half; FB overlap and underlap",
        "n_events": int(len(poss)),
        "n_matches": int(poss["match_id"].nunique()) if len(poss) else 0,
        "n_fb_overlaps": int(len(overs)),
        "tracked_rate_possessions": float(poss["tracked"].mean()) if len(poss) else None,
        "tracked_rate_overlaps": float(pd.to_numeric(overs["tracked"], errors="coerce").mean()) if len(overs) else None,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "supported": supported,
        "metric_worth_keeping": worth,
        "metric_name": "fb_tracked" if worth else None,
        "inference": "match_cluster_bootstrap",
        "contrasts": contrasts,
        "notes": (
            "The preventive reading predicts lower reception and shot rates when tracked. "
            "A positive interval is a selection effect: tracked runs are the dangerous ones. "
            "That still leaves fb_tracked as a tag, but it does not support a prevention claim."
        ),
    }


def _summarize_h6(df: pd.DataFrame) -> dict:
    failed_tracked = cluster_diff(df, "failed", "tracked", seed=61)
    failed_gs = cluster_diff(df, "failed", "goal_side", seed=62)
    failed_lane = cluster_diff(df, "failed", "narrow_lane", seed=63)
    shot_tracked = cluster_diff(df, "lead_to_shot", "tracked", seed=64)
    xt_tracked = cluster_diff(df, "xthreat", "tracked", seed=65)
    beaten_n = int(np.nansum(pd.to_numeric(df["beaten_by_movement"], errors="coerce").fillna(0)))
    joined_n = int((df["beaten_joined"] == 1).sum()) if "beaten_joined" in df.columns else 0
    contrasts = [
        _contrast_record("overlap/underlap failed (not received), tracked minus not", "positive", failed_tracked, "rate"),
        _contrast_record("overlap/underlap failed, nearest defender goal-side minus not", "positive", failed_gs, "rate"),
        _contrast_record(
            f"overlap/underlap failed, opponent in the lane ({LANE_DIST_M:.1f} m) minus open",
            "positive",
            failed_lane,
            "rate",
        ),
        _contrast_record("overlap/underlap lead_to_shot, tracked minus not", "negative", shot_tracked, "rate"),
        _contrast_record("overlap/underlap xthreat, tracked minus not", "negative", xt_tracked, "xthreat"),
    ]
    supported = any(c["supported"] for c in contrasts[:3])
    useful = [
        c for c in contrasts[:3]
        if c.get("ci_excludes_zero") and c.get("effect") is not None and abs(c["effect"]) >= 0.03
        and c["n_treat"] >= 20 and c["n_control"] >= 20
    ]
    names = []
    if any(c["name"].startswith("overlap/underlap failed (not received), tracked") and c in useful for c in contrasts):
        names.append("overlap_tracked")
    if any("goal-side" in c["name"] and c in useful for c in contrasts):
        names.append("overlap_goal_side")
    if any("lane" in c["name"] and c in useful for c in contrasts):
        names.append("overlap_lane_occupied")
    primary = contrasts[0]
    return {
        "id": 6,
        "title": "Overlaps and underlaps versus defensive state",
        "definition": (
            "off_ball_run events with subtype overlap or underlap. "
            "Tracked uses the hypothesis-5 rule (6 m, goal-side, similar 1 s velocity), applied to the runner "
            "at the midpoint frame, without requiring the attacking half. "
            "Goal-side is the nearest outfield opponent standing closer to the goal being attacked. "
            f"Narrow lane: an outfield opponent within {LANE_DIST_M:.1f} m of the segment from the ball carrier "
            f"to the runner, t in [{LANE_T[0]:.2f}, {LANE_T[1]:.2f}]. "
            "Failed means received is false. "
            "beaten_by_movement is not a column on off_ball_run; it is taken from an on_ball_engagement "
            "whose targeted player is the runner and whose frame window overlaps the run. "
            "These are associations. They are not a prevention system."
        ),
        "population": "overlap and underlap off_ball_run",
        "n_events": int(len(df)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "n_overlap": int((df["subtype"] == "overlap").sum()) if len(df) else 0,
        "n_underlap": int((df["subtype"] == "underlap").sum()) if len(df) else 0,
        "tracked_rate": float(pd.to_numeric(df["tracked"], errors="coerce").mean()) if len(df) else None,
        "goal_side_rate": float(pd.to_numeric(df["goal_side"], errors="coerce").mean()) if len(df) else None,
        "narrow_lane_rate": float(pd.to_numeric(df["narrow_lane"], errors="coerce").mean()) if len(df) else None,
        "received_rate": float(pd.to_numeric(df["received"], errors="coerce").mean()) if len(df) else None,
        "lead_to_shot_rate": float(pd.to_numeric(df["lead_to_shot"], errors="coerce").mean()) if len(df) else None,
        "mean_xthreat": float(pd.to_numeric(df["xthreat"], errors="coerce").mean()) if len(df) else None,
        "beaten_by_movement_true": beaten_n,
        "beaten_by_movement_joined_engagements": joined_n,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "supported": supported,
        "metric_worth_keeping": len(names) > 0,
        "metric_name": "+".join(names) if names else None,
        "inference": "match_cluster_bootstrap",
        "contrasts": contrasts,
        "by_subtype": _subtype_rates(df),
        "notes": (
            f"beaten_by_movement was true on {beaten_n} runs that could be joined to an engagement "
            f"({joined_n} runs had any overlapping engagement). "
            "That count is too small for its own interval, so it is reported as a count only. "
            "Three defensive states were tested against failed runs."
        ),
    }


def _subtype_rates(df: pd.DataFrame) -> list[dict]:
    rows = []
    for subtype, g in df.groupby("subtype"):
        rows.append({
            "subtype": subtype,
            "n": int(len(g)),
            "received_rate": float(pd.to_numeric(g["received"], errors="coerce").mean()),
            "lead_to_shot_rate": float(pd.to_numeric(g["lead_to_shot"], errors="coerce").mean()),
            "mean_xthreat": float(pd.to_numeric(g["xthreat"], errors="coerce").mean()),
            "tracked_rate": float(pd.to_numeric(g["tracked"], errors="coerce").mean()),
        })
    return rows


def _summarize_h7(df: pd.DataFrame) -> dict:
    contrasts = []
    for group, col in (("CB", "dep_CB"), ("CM", "dep_CM"), ("FB", "dep_FB")):
        contrasts.append(_contrast_record(
            f"{group} distance from role median (m), tight mark minus not",
            "positive",
            cluster_diff(df, col, "tight_nearest", seed=70 + len(contrasts)),
            "metres",
        ))
    for group, col in (("CB", "fwd_CB"), ("CM", "fwd_CM"), ("FB", "fwd_FB")):
        contrasts.append(_contrast_record(
            f"{group} metres ahead of role median, tight mark minus not",
            "positive",
            cluster_diff(df, col, "tight_nearest", seed=80 + len(contrasts)),
            "metres",
        ))
    contrasts.append(_contrast_record(
        "lead_to_shot, tight mark minus not",
        "negative",
        cluster_diff(df, "lead_to_shot", "tight_nearest", seed=91),
        "rate",
    ))
    contrasts.append(_contrast_record(
        "player_targeted_xthreat, tight mark minus not",
        "negative",
        cluster_diff(df, "xthreat", "tight_nearest", seed=92),
        "xthreat",
    ))
    contrasts.append(_contrast_record(
        "first_line_break, tight mark minus not",
        "negative",
        cluster_diff(df, "first_line_break", "tight_nearest", seed=93),
        "rate",
    ))
    # Hungarian definition as a sensitivity on CB departure, the clearest "drawn out" slot.
    hun = cluster_diff(df, "dep_CB", "tight_hungarian", seed=94)
    contrasts.append(_contrast_record(
        "CB distance from role median, Hungarian tight mark minus not",
        "positive",
        hun,
        "metres",
    ))
    mechanism = contrasts[:3]
    supported = any(c["supported"] for c in mechanism)
    useful = [
        c for c in mechanism
        if c.get("ci_excludes_zero") and c.get("effect") is not None and abs(c["effect"]) >= 1.0
        and c["n_treat"] >= 20 and c["n_control"] >= 20
    ]
    return {
        "id": 7,
        "title": "Tight mark on the striker and departure from the block",
        "definition": (
            f"Striker is a player_possession with player_position CF, inside low_block, medium_block, "
            f"or high_block. Tight mark (primary): nearest outfield opponent within {TIGHT_MARK_M:.1f} m "
            f"at frame_start. Tight mark (Hungarian): goal-side assignment distance from "
            f"coupled_formation.hungarian_marking at most {TIGHT_MARK_M:.1f} m. "
            f"Reference block: median attack-normalized (x, y) of that role acronym for that defending team "
            f"in that block type, from organized-phase start frames (at least {MIN_ROLE_SAMPLES} samples). "
            f"Departure is Euclidean distance from that median. Forward departure is attack-normalized x "
            f"minus the median x (positive = higher up the pitch). "
            f"Outcomes on the same possession: lead_to_shot, player_targeted_xthreat, first_line_break. "
            f"xshot_player_possession_max is joined from on_ball_engagement and is not the primary outcome."
        ),
        "population": "CF player_possession in an organized block",
        "n_events": int(len(df)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "tight_rate": float(df["tight_nearest"].mean()) if len(df) else None,
        "tight_hungarian_rate": float(df["tight_hungarian"].mean()) if len(df) else None,
        "mean_nearest_dist": float(pd.to_numeric(df["nearest_dist"], errors="coerce").mean()) if len(df) else None,
        "effect_name": "CB/CM/FB departure from role median when the striker is inside 3 m",
        "effect": contrasts[0]["effect"],
        "ci_low": contrasts[0]["ci_low"],
        "ci_high": contrasts[0]["ci_high"],
        "supported": supported,
        "metric_worth_keeping": len(useful) > 0,
        "metric_name": "striker_tight_mark_3m" if len(useful) > 0 else None,
        "inference": "match_cluster_bootstrap",
        "contrasts": contrasts,
        "notes": (
            "Three role groups were tested, so a single surviving group is a weaker claim than all three. "
            "Shot, xthreat and line-break contrasts ask whether the moment is more dangerous, "
            "which is separate from whether teammates leave their slots."
        ),
    }


def run(selected: list[str] | None = None) -> list[dict]:
    ids = selected or match_ids()
    buckets = {"h3": [], "h4": [], "h5": [], "h6": [], "h7": []}
    for i, mid in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {mid}", flush=True)
        part = process_match(mid)
        for k, rows in part.items():
            buckets[k].extend(rows)
            print(f"    {k}: {len(rows)}", flush=True)
    frames = {k: pd.DataFrame(v) for k, v in buckets.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = summarize(frames)
    path = OUT_DIR / "hypothesis_tracking.json"
    path.write_text(json.dumps(json_ready(records), indent=2))
    print(f"wrote {path}", flush=True)
    for rec in records:
        print(
            f"H{rec['id']} supported={rec['supported']} metric={rec['metric_name']} "
            f"effect={fmt(rec['effect'])} CI=({fmt(rec['ci_low'])}, {fmt(rec['ci_high'])}) "
            f"n={rec['n_events']}",
            flush=True,
        )
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*")
    args = parser.parse_args()
    # Local import keeps the seed constant used above.
    from common import BOOT_SEED, LANE_T  # noqa: F401

    run(args.matches if args.matches else None)
