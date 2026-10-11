"""Hypothesis 5: is an attacking full back being tracked?

Population is off_ball_run events with subtype overlap or underlap, restricted
to full backs by the match-json role. Tracking is streamed one match at a time.

Run: opendata/.venv/bin/python analysis/hypotheses/h5_fullback_tracked.py
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
    FB_ROLES,
    N_BOOT,
    OUT_DIR,
    STATIONARY_DISP_M,
    TRACK_COS,
    TRACK_DIST_M,
    TRACK_LOOKBACK,
    TRACK_SPEED_DIFF,
    attack_sign,
    cluster_diff,
    displacement,
    is_goal_side,
    json_ready,
    load_events,
    load_roster,
    match_ids,
    nearest_opponent,
    similar_velocity,
    stream_frames,
)

# Grid used to find an attacking-half frame. The label itself is one frame.
SAMPLE_STEP = 5

OUTCOMES = (
    ("received", "overlap/underlap received, tracked minus untracked", "rate", 51),
    ("lead_to_shot", "overlap/underlap lead_to_shot, tracked minus untracked", "rate", 52),
    ("xthreat", "overlap/underlap xthreat, tracked minus untracked", "xthreat", 53),
)


def _iid(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _sample_frames(start: int, end: int) -> list[int]:
    """Grid plus the temporal midpoint, so the measurement can sit on the midpoint."""
    if end < start:
        return []
    mid = (start + end) // 2
    frames = list(range(start, end + 1, SAMPLE_STEP))
    frames.extend((end, mid))
    return sorted({fr for fr in frames if start <= fr <= end})


def _state_at(frames, frame: int, pid: int, team_of, gk, sign: float) -> dict:
    xy = frames.get(frame) or {}
    if pid not in xy:
        return {"tracked": np.nan, "dist": np.nan, "goal_side": np.nan, "similar_vel": np.nan, "in_half": 0}
    in_half = 1 if xy[pid][0] * sign > 0.0 else 0
    if not in_half:
        return {"tracked": np.nan, "dist": np.nan, "goal_side": np.nan, "similar_vel": np.nan, "in_half": 0}
    opp_id, dist = nearest_opponent(pid, xy, team_of, gk)
    if opp_id is None:
        return {"tracked": np.nan, "dist": np.nan, "goal_side": np.nan, "similar_vel": np.nan, "in_half": 1}
    goal_side = 1.0 if is_goal_side(xy[pid], xy[opp_id], sign) else 0.0
    sim = similar_velocity(
        displacement(frames, pid, frame),
        displacement(frames, opp_id, frame),
    )
    if not np.isfinite(sim):
        tracked = np.nan
    else:
        tracked = 1.0 if (dist <= TRACK_DIST_M and goal_side == 1.0 and sim == 1.0) else 0.0
    return {
        "tracked": tracked,
        "dist": dist,
        "goal_side": goal_side,
        "similar_vel": sim,
        "in_half": 1,
    }


def process_match(match_id: str) -> tuple[list[dict], dict]:
    """Event rows for one match. Frames are discarded before return."""
    events = load_events(match_id)
    team_of, roster, gk, _plen, _pwid = load_roster(match_id)
    runs = events[
        (events["event_type"] == "off_ball_run")
        & (events["event_subtype"].isin(["overlap", "underlap"]))
    ]
    counts = {
        "overlap_underlap": int(len(runs)),
        "not_full_back": 0,
        "bad_side": 0,
        "no_attacking_half": 0,
        "missing_velocity": 0,
        "eligible": 0,
    }
    candidates = []
    needed: set[int] = set()
    for row in runs.itertuples(index=False):
        pid = _iid(row.player_id)
        if pid is None or roster.get(pid) not in FB_ROLES:
            counts["not_full_back"] += 1
            continue
        sign = attack_sign(row.attacking_side)
        if not np.isfinite(sign):
            counts["bad_side"] += 1
            continue
        start, end = int(row.frame_start), int(row.frame_end)
        samples = _sample_frames(start, end)
        if not samples:
            counts["bad_side"] += 1
            continue
        for fr in samples:
            if fr >= 0:
                needed.add(fr)
            prev = fr - TRACK_LOOKBACK
            if prev >= 0:
                needed.add(prev)
        candidates.append((row, pid, sign, samples))

    frames = stream_frames(match_id, needed)
    out = []
    for row, pid, sign, samples in candidates:
        mid = (int(row.frame_start) + int(row.frame_end)) // 2
        chosen = None
        chosen_state = None
        saw_half = False
        for fr in samples:
            state = _state_at(frames, fr, pid, team_of, gk, sign)
            if not state["in_half"]:
                continue
            saw_half = True
            if not np.isfinite(state["tracked"]):
                continue
            if chosen is None or (abs(fr - mid), fr) < (abs(chosen - mid), chosen):
                chosen = fr
                chosen_state = state
        if not saw_half:
            counts["no_attacking_half"] += 1
            continue
        if chosen_state is None:
            counts["missing_velocity"] += 1
            continue
        counts["eligible"] += 1
        out.append({
            "match_id": match_id,
            "subtype": row.event_subtype,
            "role": roster.get(pid),
            "tracked": chosen_state["tracked"],
            "dist": chosen_state["dist"],
            "goal_side": chosen_state["goal_side"],
            "similar_vel": chosen_state["similar_vel"],
            "close": 1.0 if chosen_state["dist"] <= TRACK_DIST_M else 0.0,
            "received": row.received,
            "lead_to_shot": row.lead_to_shot,
            "xthreat": row.xthreat,
        })
    del frames
    return out, counts


def _contrast(df: pd.DataFrame, y: str, name: str, unit: str, seed: int) -> dict:
    stat = cluster_diff(df, y, "tracked", n_boot=N_BOOT, seed=seed)
    return {
        "name": name,
        "outcome": y,
        "unit": unit,
        "predicted_if_prevention": "negative",
        "supported": bool(stat["ci_excludes_zero"]),
        "n": int(stat["n_treat"] + stat["n_control"]),
        "n_tracked": int(stat["n_treat"]),
        "n_untracked": int(stat["n_control"]),
        "n_matches": int(stat["n_matches"]),
        "n_matches_both": int(stat["n_matches_both"]),
        "mean_tracked": stat["mean_treat"],
        "mean_untracked": stat["mean_control"],
        "effect": stat["effect"],
        "ci_low": stat["ci_low"],
        "ci_high": stat["ci_high"],
        "ci_excludes_zero": bool(stat["ci_excludes_zero"]),
    }


def summarize(df: pd.DataFrame, counts: dict) -> dict:
    contrasts = [_contrast(df, y, name, unit, BOOT_SEED + seed) for y, name, unit, seed in OUTCOMES]
    separating = [c["outcome"] for c in contrasts if c["supported"]]
    primary = contrasts[0]
    definition = (
        f"Full back roles are LB, RB, LWB, and RWB from the match json, not from the event position. "
        f"The run is an off_ball_run with subtype overlap or underlap. "
        f"Candidate frames are every {SAMPLE_STEP / 10:.1f} s from frame_start through frame_end, plus the temporal midpoint. "
        f"The measurement frame is the candidate in the attacking half (attack-normalized tracking x > 0) "
        f"closest to that midpoint, among candidates with a velocity window. "
        f"Tracked means all three at that frame: the nearest outfield opponent is within {TRACK_DIST_M:.0f} m, "
        f"that opponent is goal-side (closer to the goal being attacked), and the two velocities are similar "
        f"over the previous {TRACK_LOOKBACK / 10:.1f} s. Similar velocity: cosine of the displacements "
        f">= {TRACK_COS:.2f} and speed difference <= {TRACK_SPEED_DIFF:.1f} m/s. "
        f"If both players move under {STATIONARY_DISP_M:.0f} m in the window, similar means the speeds differ by at most 1.5 m/s. "
        f"A missing velocity window excludes the run. It is not labelled untracked. "
        f"Runs that never enter the attacking half are excluded. "
        f"Outcomes on the same event are received, lead_to_shot, and xthreat. xG is not used. "
        f"The effect is the pooled mean for tracked runs minus untracked runs. "
        f"The interval resamples matches ({N_BOOT} draws, percentile 2.5 to 97.5). "
        f"A contrast is supported only when that interval excludes zero."
    )
    return {
        "id": 5,
        "title": "Attacking full back being tracked",
        "metric_definition": definition,
        "population": "full-back overlap and underlap off_ball_run while in the attacking half",
        "separates_tracked_from_untracked": bool(separating),
        "outcomes_that_separate": separating,
        "supported": bool(separating),
        "n": int(len(df)),
        "n_tracked": int((df["tracked"] == 1).sum()) if len(df) else 0,
        "n_untracked": int((df["tracked"] == 0).sum()) if len(df) else 0,
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "tracked_rate": float(df["tracked"].mean()) if len(df) else None,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "inference": "match_cluster_bootstrap",
        "n_boot": N_BOOT,
        "counts": counts,
        "measurement_rates": {
            "close_rate": float(df["close"].mean()) if len(df) else None,
            "goal_side_rate": float(df["goal_side"].mean()) if len(df) else None,
            "similar_velocity_rate": float(df["similar_vel"].mean()) if len(df) else None,
            "mean_nearest_defender_m": float(pd.to_numeric(df["dist"], errors="coerce").mean()) if len(df) else None,
        },
        "by_subtype": _by_subtype(df),
        "contrasts": contrasts,
        "notes": (
            "Prevention would show a negative effect: tracked runs received less often, "
            "lead to a shot less often, and carry lower xthreat. "
            "An interval above zero is still a separation. It means the tracked runs are the more dangerous ones. "
            "Support uses only whether the match-clustered interval excludes zero."
        ),
    }


def _by_subtype(df: pd.DataFrame) -> list[dict]:
    rows = []
    if df.empty:
        return rows
    for subtype, g in df.groupby("subtype"):
        rows.append({
            "subtype": subtype,
            "n": int(len(g)),
            "n_tracked": int((g["tracked"] == 1).sum()),
            "tracked_rate": float(g["tracked"].mean()),
            "received_rate": float(pd.to_numeric(g["received"], errors="coerce").mean()),
            "lead_to_shot_rate": float(pd.to_numeric(g["lead_to_shot"], errors="coerce").mean()),
            "mean_xthreat": float(pd.to_numeric(g["xthreat"], errors="coerce").mean()),
        })
    return rows


def run(selected: list[str] | None = None) -> dict:
    ids = selected or match_ids()
    rows: list[dict] = []
    totals = {
        "overlap_underlap": 0,
        "not_full_back": 0,
        "bad_side": 0,
        "no_attacking_half": 0,
        "missing_velocity": 0,
        "eligible": 0,
    }
    for i, mid in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {mid}", flush=True)
        part, counts = process_match(mid)
        rows.extend(part)
        for k, v in counts.items():
            totals[k] += v
        print(f"    eligible {counts['eligible']} of {counts['overlap_underlap']} overlap/underlap", flush=True)
    df = pd.DataFrame(rows)
    summary = summarize(df, totals)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "h5_summary.json"
    path.write_text(json.dumps(json_ready(summary), indent=2))
    print(f"wrote {path}", flush=True)
    print(
        f"separates={summary['separates_tracked_from_untracked']} "
        f"outcomes={summary['outcomes_that_separate']} "
        f"n={summary['n']} tracked={summary['n_tracked']}",
        flush=True,
    )
    for c in summary["contrasts"]:
        print(
            f"  {c['outcome']}: effect={c['effect']} "
            f"CI=({c['ci_low']}, {c['ci_high']}) "
            f"n={c['n_tracked']}/{c['n_untracked']} supported={c['supported']}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*")
    args = parser.parse_args()
    run(args.matches if args.matches else None)
