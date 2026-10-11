"""Hypothesis 6: how often overlaps and underlaps succeed, and which
defensive states associate with the ones that do not.

Population: off_ball_run events with subtype overlap or underlap, all matches.
Outcomes on the run: beaten_by_movement, lead_to_shot, xthreat.
Defensive state at the midpoint frame, from tracking:
  nearest outfield opponent distance,
  whether that opponent is goal-side,
  whether any outfield opponent occupies the lane from the ball carrier to the runner.

beaten_by_movement is not filled on off_ball_run. It is taken from an
on_ball_engagement whose targeted player is the runner and whose frame window
meets the run, padded by 1.0 s. Runs with no such engagement are missing on
that outcome, not coded as a failure.

These are associations. They are not a prevention system.
An association is supported only when its match-clustered interval excludes zero
and both arms have at least 20 events in at least two matches.

Run: opendata/.venv/bin/python analysis/hypotheses/h6_overlap_underlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BOOT_SEED,
    LANE_DIST_M,
    LANE_T,
    OUT_DIR,
    TRACK_DIST_M,
    attack_sign,
    cluster_diff,
    cluster_mean,
    in_lane,
    is_goal_side,
    json_ready,
    load_events,
    load_roster,
    match_ids,
    nearest_opponent,
    stream_frames,
)

SUMMARY_PATH = OUT_DIR / "h6_summary.json"
MIN_ARM = 20
ENGAGEMENT_PAD = 10  # frames at 10 fps = 1.0 s
SUBTYPES = ("overlap", "underlap")


def _iid(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _finite_xy(x, y) -> tuple[float, float] | None:
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf) or not np.isfinite(yf):
        return None
    return (xf, yf)


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


def _engagement_index(engs: pd.DataFrame) -> dict[int, list[tuple[int, int, float]]]:
    index: dict[int, list[tuple[int, int, float]]] = {}
    if engs.empty:
        return index
    targeted = engs["player_targeted_id"]
    starts = engs["frame_start"].to_numpy()
    ends = engs["frame_end"].to_numpy()
    flags = pd.to_numeric(engs["beaten_by_movement"], errors="coerce").to_numpy(dtype=float)
    for i, tid in enumerate(targeted):
        pid = _iid(tid)
        if pid is None:
            continue
        index.setdefault(pid, []).append((int(starts[i]), int(ends[i]), float(flags[i])))
    return index


def _beaten(index: dict[int, list[tuple[int, int, float]]], pid: int, frame_start: int, frame_end: int) -> tuple[float, float]:
    """Return (beaten_by_movement, joined). Unjoined runs stay missing."""
    hits = index.get(pid)
    if not hits:
        return np.nan, 0.0
    lo = frame_start - ENGAGEMENT_PAD
    hi = frame_end + ENGAGEMENT_PAD
    flags = [flag for a, b, flag in hits if a <= hi and b >= lo]
    if not flags:
        return np.nan, 0.0
    arr = np.asarray(flags, dtype=float)
    if np.any(arr == 1.0):
        return 1.0, 1.0
    if np.any(np.isfinite(arr)):
        return 0.0, 1.0
    return np.nan, 1.0


def process_match(match_id: str) -> list[dict]:
    events = load_events(match_id)
    team_of, _roster, gk, _plen, _pwid = load_roster(match_id)
    runs = events[
        (events["event_type"] == "off_ball_run") & (events["event_subtype"].isin(SUBTYPES))
    ]
    engs = events[events["event_type"] == "on_ball_engagement"]
    eng_index = _engagement_index(engs)

    needed: set[int] = set()
    prepared = []
    for row in runs.itertuples(index=False):
        pid = _iid(row.player_id)
        if pid is None:
            continue
        frame_start = int(row.frame_start)
        frame_end = int(row.frame_end)
        mid = int((frame_start + frame_end) / 2)
        if mid >= 0:
            needed.add(mid)
        prepared.append((row, pid, frame_start, frame_end, mid))

    frames = stream_frames(match_id, needed)
    rows = []
    for row, pid, frame_start, frame_end, mid in prepared:
        sign = attack_sign(row.attacking_side)
        xy = frames.get(mid) or {}
        runner = xy.get(pid)
        carrier_id = _iid(row.player_in_possession_id)
        carrier = xy.get(carrier_id) if carrier_id is not None else None
        if carrier is None:
            carrier = _finite_xy(row.player_in_possession_x_start, row.player_in_possession_y_start)
        att_team = _iid(row.team_id)
        if att_team is None and pid in team_of:
            att_team = team_of[pid]

        dist = np.nan
        goal_side = np.nan
        lane = np.nan
        if runner is not None:
            opp_id, dist = nearest_opponent(pid, xy, team_of, gk)
            if opp_id is not None and np.isfinite(sign):
                goal_side = 1.0 if is_goal_side(runner, xy[opp_id], sign) else 0.0
            if carrier is not None and att_team is not None:
                skip = {pid}
                if carrier_id is not None:
                    skip.add(carrier_id)
                defs = _opponents(xy, team_of, att_team, gk, skip)
                lane = 1.0 if in_lane(defs, carrier, runner) else 0.0

        beaten, joined = _beaten(eng_index, pid, frame_start, frame_end)
        shot = float(row.lead_to_shot) if np.isfinite(row.lead_to_shot) else np.nan
        xt = float(row.xthreat) if np.isfinite(row.xthreat) else np.nan
        rows.append({
            "match_id": match_id,
            "subtype": row.event_subtype,
            "nearest_dist": dist,
            "goal_side": goal_side,
            "lane_occupied": lane,
            "close_defender": (1.0 if dist <= TRACK_DIST_M else 0.0) if np.isfinite(dist) else np.nan,
            "lead_to_shot": shot,
            "failed_shot": (1.0 - shot) if np.isfinite(shot) else np.nan,
            "xthreat": xt,
            "beaten_by_movement": beaten,
            "failed_movement": (1.0 - beaten) if np.isfinite(beaten) else np.nan,
            "beaten_joined": joined,
        })
    return rows


def _rate(df: pd.DataFrame, y: str, seed: int) -> dict:
    stat = cluster_mean(df, y, seed=seed)
    return {
        "mean": stat["effect"],
        "ci_low": stat["ci_low"],
        "ci_high": stat["ci_high"],
        "n": stat["n"],
        "n_matches": stat["n_matches"],
        "ci_excludes_zero": stat["ci_excludes_zero"],
    }


def _holds(contrast: dict) -> bool:
    """Supported only when the interval excludes zero, on a real two-arm sample."""
    if not contrast.get("ci_excludes_zero"):
        return False
    effect = contrast.get("effect")
    if effect is None or not np.isfinite(effect):
        return False
    if contrast.get("ci_low") is None or contrast.get("ci_high") is None:
        return False
    if int(contrast.get("n_treat") or 0) < MIN_ARM or int(contrast.get("n_control") or 0) < MIN_ARM:
        return False
    if int(contrast.get("n_matches_both") or 0) < 2:
        return False
    return True


def _contrast(name: str, df: pd.DataFrame, y: str, treat: str, unit: str, seed: int, reading_rule: str) -> dict:
    stat = cluster_diff(df, y, treat, seed=seed)
    rec = {
        "name": name,
        "unit": unit,
        "reading_rule": reading_rule,
        "supported": False,
        **stat,
    }
    rec["supported"] = _holds(rec)
    return rec


def _reading(contrast: dict) -> str:
    effect = contrast["effect"]
    rule = contrast["reading_rule"]
    if rule == "failure_rate":
        if effect > 0:
            return "This state has a higher failure rate than the comparison state."
        return "This state has a lower failure rate than the comparison state."
    if rule == "distance_on_failure":
        if effect < 0:
            return "Failed runs have a nearer defender than runs that succeeded on this outcome."
        return "Failed runs have a farther defender than runs that succeeded on this outcome."
    if effect < 0:
        return "This state has lower xthreat than the comparison state."
    return "This state has higher xthreat than the comparison state."


def summarize(df: pd.DataFrame) -> dict:
    joined = df[df["beaten_joined"] == 1].copy()
    rates = {
        "lead_to_shot": _rate(df, "lead_to_shot", seed=BOOT_SEED + 1),
        "xthreat": _rate(df, "xthreat", seed=BOOT_SEED + 2),
        "beaten_by_movement_among_joined_engagements": _rate(joined, "beaten_by_movement", seed=BOOT_SEED + 3),
        "failed_shot": _rate(df, "failed_shot", seed=BOOT_SEED + 4),
        "failed_movement_among_joined_engagements": _rate(joined, "failed_movement", seed=BOOT_SEED + 5),
    }
    state_rates = {
        "nearest_defender_distance_m": _rate(df, "nearest_dist", seed=BOOT_SEED + 6),
        "goal_side": _rate(df, "goal_side", seed=BOOT_SEED + 7),
        "lane_occupied": _rate(df, "lane_occupied", seed=BOOT_SEED + 8),
        "nearest_defender_within_6m": _rate(df, "close_defender", seed=BOOT_SEED + 9),
    }
    by_subtype = []
    for subtype in SUBTYPES:
        sub = df[df["subtype"] == subtype]
        sub_joined = sub[sub["beaten_joined"] == 1]
        by_subtype.append({
            "subtype": subtype,
            "n": int(len(sub)),
            "n_matches": int(sub["match_id"].nunique()) if len(sub) else 0,
            "lead_to_shot": _rate(sub, "lead_to_shot", seed=BOOT_SEED + 10),
            "xthreat": _rate(sub, "xthreat", seed=BOOT_SEED + 11),
            "beaten_by_movement_among_joined_engagements": _rate(
                sub_joined, "beaten_by_movement", seed=BOOT_SEED + 12
            ),
            "n_beaten_joined": int(len(sub_joined)),
        })

    contrasts = [
        _contrast(
            "no lead_to_shot, nearest defender goal-side minus not",
            df, "failed_shot", "goal_side", "rate", BOOT_SEED + 21, "failure_rate",
        ),
        _contrast(
            f"no lead_to_shot, defender in the lane ({LANE_DIST_M:.1f} m) minus open",
            df, "failed_shot", "lane_occupied", "rate", BOOT_SEED + 22, "failure_rate",
        ),
        _contrast(
            "nearest defender distance (m), no lead_to_shot minus lead_to_shot",
            df, "nearest_dist", "failed_shot", "metres", BOOT_SEED + 23, "distance_on_failure",
        ),
        _contrast(
            "xthreat, nearest defender goal-side minus not",
            df, "xthreat", "goal_side", "xthreat", BOOT_SEED + 24, "xthreat",
        ),
        _contrast(
            f"xthreat, defender in the lane ({LANE_DIST_M:.1f} m) minus open",
            df, "xthreat", "lane_occupied", "xthreat", BOOT_SEED + 25, "xthreat",
        ),
        _contrast(
            f"xthreat, nearest defender within {TRACK_DIST_M:.0f} m minus farther",
            df, "xthreat", "close_defender", "xthreat", BOOT_SEED + 26, "xthreat",
        ),
        _contrast(
            "not beaten_by_movement, nearest defender goal-side minus not",
            joined, "failed_movement", "goal_side", "rate", BOOT_SEED + 27, "failure_rate",
        ),
        _contrast(
            f"not beaten_by_movement, defender in the lane ({LANE_DIST_M:.1f} m) minus open",
            joined, "failed_movement", "lane_occupied", "rate", BOOT_SEED + 28, "failure_rate",
        ),
        _contrast(
            "nearest defender distance (m), not beaten_by_movement minus beaten_by_movement",
            joined, "nearest_dist", "failed_movement", "metres", BOOT_SEED + 29, "distance_on_failure",
        ),
    ]
    held = []
    for c in contrasts:
        if not c["supported"]:
            continue
        held.append({
            "name": c["name"],
            "effect": c["effect"],
            "ci_low": c["ci_low"],
            "ci_high": c["ci_high"],
            "unit": c["unit"],
            "n_treat": c["n_treat"],
            "n_control": c["n_control"],
            "n_matches_both": c["n_matches_both"],
            "reading": _reading(c),
        })

    beaten_true = int(np.nansum(pd.to_numeric(df["beaten_by_movement"], errors="coerce").fillna(0)))
    return {
        "id": 6,
        "title": "Overlaps and underlaps versus defensive state",
        "definition": (
            "off_ball_run events with subtype overlap or underlap. "
            "Success is measured by three outcomes already on the run or joined to it: "
            "lead_to_shot, xthreat, and beaten_by_movement. "
            "A run fails a binary outcome when that outcome is false. xthreat stays continuous. "
            "Defensive state is measured from tracking at the midpoint frame of the run, "
            "not from the hypothesis-5 tracked flag. "
            "Nearest defender distance is the distance to the nearest outfield opponent. "
            "Goal-side means that opponent is closer to the goal being attacked than the runner is. "
            f"The lane is occupied when any outfield opponent is within {LANE_DIST_M:.1f} m of the "
            f"segment from the ball carrier to the runner, with segment parameter t in "
            f"[{LANE_T[0]:.2f}, {LANE_T[1]:.2f}]. "
            f"The {TRACK_DIST_M:.0f} m split is only a distance band for the xthreat contrast. "
            "It is not a tracking rule and it is not a prevention system. "
            "beaten_by_movement comes from an on_ball_engagement whose targeted player is the runner "
            f"and whose frame window meets the run within {ENGAGEMENT_PAD / 10:.1f} s. "
            "Runs with no such engagement are left missing on that outcome."
        ),
        "population": "overlap and underlap off_ball_run",
        "n_events": int(len(df)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "n_overlap": int((df["subtype"] == "overlap").sum()) if len(df) else 0,
        "n_underlap": int((df["subtype"] == "underlap").sum()) if len(df) else 0,
        "n_with_nearest_defender": int(np.isfinite(pd.to_numeric(df["nearest_dist"], errors="coerce")).sum()),
        "n_with_lane": int(np.isfinite(pd.to_numeric(df["lane_occupied"], errors="coerce")).sum()),
        "beaten_by_movement_true": beaten_true,
        "beaten_by_movement_joined_engagements": int((df["beaten_joined"] == 1).sum()),
        "inference": "match_cluster_bootstrap",
        "n_boot": 2000,
        "support_rule": (
            "An association is supported only when the match-clustered 95% interval excludes zero, "
            f"both arms have at least {MIN_ARM} events, and both arms appear in at least two matches. "
            "A supported association is not a prevention system."
        ),
        "prevention_claim": False,
        "rates": rates,
        "by_subtype": by_subtype,
        "defensive_state_rates": state_rates,
        "contrasts": contrasts,
        "associations_that_hold": held,
        "supported": len(held) > 0,
        "notes": (
            "Failure on lead_to_shot is the complement of lead_to_shot. "
            "Failure on beaten_by_movement is the complement among runs joined to an engagement. "
            "A higher failure rate, a nearer defender on failed runs, or a lower xthreat "
            "describes an association at the moment of the run. "
            "It does not show that the defensive state prevented the overlap or underlap."
        ),
    }


def run(selected: list[str] | None = None) -> dict:
    ids = selected or match_ids()
    rows: list[dict] = []
    for i, mid in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {mid}", flush=True)
        part = process_match(mid)
        rows.extend(part)
        print(f"    runs: {len(part)}", flush=True)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH}", flush=True)
    print(
        f"n={summary['n_events']} matches={summary['n_matches']} "
        f"overlap={summary['n_overlap']} underlap={summary['n_underlap']} "
        f"joined={summary['beaten_by_movement_joined_engagements']} "
        f"beaten={summary['beaten_by_movement_true']}",
        flush=True,
    )
    if summary["associations_that_hold"]:
        print("associations that hold:", flush=True)
        for item in summary["associations_that_hold"]:
            print(
                f"  {item['name']}: {item['effect']:.4f} "
                f"({item['ci_low']:.4f}, {item['ci_high']:.4f})",
                flush=True,
            )
    else:
        print("associations that hold: none", flush=True)
    return summary


if __name__ == "__main__":
    run()
