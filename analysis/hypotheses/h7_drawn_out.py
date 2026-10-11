"""Hypothesis 7: a tightly marked striker and who leaves the block.

Population: player possessions by a CF or ST inside a low, medium, or high block.
Tight mark: nearest outfield opponent within TIGHT_MARK_M at the possession's
first frame. A Hungarian goal-side assignment at the same distance is reported
as a sensitivity; it is not used to decide the claim.
Drawn out: mean distance of that role from its median position for the same
defending team and the same block in this match. The median is taken on a 2 Hz
sample of organized-block frames, in attack-normalized coordinates.
A role is drawn out only when the match-clustered interval for
(tight minus not) lies entirely above zero.

Run: opendata/.venv/bin/python analysis/hypotheses/h7_drawn_out.py
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
    CB_ROLES,
    CM_ROLES,
    FB_ROLES,
    MIN_ROLE_SAMPLES,
    ORGANIZED,
    OUT_DIR,
    STRIKER_ROLES,
    TIGHT_MARK_M,
    attack_sign,
    cluster_diff,
    json_ready,
    load_events,
    load_phases,
    load_roster,
    match_ids,
    modal_positions,
    nearest_opponent,
    stream_frames,
    supports,
)

SAMPLE_STRIDE = 5  # 10 fps tracking -> 2 Hz
GROUPS = ("CB", "CM", "FB")
GROUP_ROLES = {"CB": CB_ROLES, "CM": CM_ROLES, "FB": FB_ROLES}


def _iid(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _role(pid: int, roster: dict[int, str], modal: dict[int, str]) -> str | None:
    """Squad acronym from the match json. Event position only if that acronym is missing or SUB."""
    role = roster.get(pid) or ""
    if not role or role == "SUB":
        role = modal.get(pid) or ""
    if not role or role == "SUB":
        return None
    return role


def _group(role: str | None) -> str | None:
    if role in CB_ROLES:
        return "CB"
    if role in CM_ROLES:
        return "CM"
    if role in FB_ROLES:
        return "FB"
    return None


def _sample_index(phases: pd.DataFrame) -> dict[int, tuple[str, str, int]]:
    """2 Hz frames inside organized blocks -> (block, attacking_side, attacking team)."""
    index: dict[int, tuple[str, str, int]] = {}
    for row in phases.itertuples(index=False):
        block = row.team_out_of_possession_phase_type
        if block not in ORGANIZED:
            continue
        sign = attack_sign(row.attacking_side)
        att = _iid(row.team_in_possession_id)
        if not np.isfinite(sign) or att is None:
            continue
        start = int(row.frame_start)
        end = int(row.frame_end)
        if end < start:
            continue
        for fr in range(start, end + 1, SAMPLE_STRIDE):
            index[fr] = (block, row.attacking_side, att)
    return index


def _role_medians(
    frames: dict[int, dict[int, tuple[float, float]]],
    sample_index: dict[int, tuple[str, str, int]],
    team_of: dict[int, int],
    gk: set[int],
    modal: dict[int, str],
    roster: dict[int, str],
) -> dict[tuple, tuple[float, float]]:
    samples: dict[tuple, list[tuple[float, float]]] = {}
    for fr, (block, side, att_team) in sample_index.items():
        sign = attack_sign(side)
        xy = frames.get(fr)
        if not xy or not np.isfinite(sign):
            continue
        for pid, (x, y) in xy.items():
            if pid in gk or team_of.get(pid) in (None, att_team):
                continue
            role = _role(pid, roster, modal)
            if _group(role) is None:
                continue
            key = (team_of[pid], block, role)
            samples.setdefault(key, []).append((x * sign, y * sign))
    medians = {}
    for key, pts in samples.items():
        if len(pts) < MIN_ROLE_SAMPLES:
            continue
        arr = np.asarray(pts, dtype=float)
        medians[key] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    return medians


def _departure(
    xy: dict[int, tuple[float, float]],
    team_of: dict[int, int],
    att_team: int,
    gk: set[int],
    modal: dict[int, str],
    roster: dict[int, str],
    block: str,
    sign: float,
    medians: dict[tuple, tuple[float, float]],
) -> dict[str, float]:
    dist = {g: [] for g in GROUPS}
    for pid, (x, y) in xy.items():
        if pid in gk or team_of.get(pid) in (None, att_team):
            continue
        role = _role(pid, roster, modal)
        group = _group(role)
        if group is None:
            continue
        ref = medians.get((team_of[pid], block, role))
        if ref is None:
            continue
        dist[group].append(float(np.hypot(x * sign - ref[0], y * sign - ref[1])))
    return {g: (float(np.mean(dist[g])) if dist[g] else np.nan) for g in GROUPS}


def _hungarian_dist(
    xy: dict[int, tuple[float, float]],
    team_of: dict[int, int],
    att_team: int,
    gk: set[int],
    striker_id: int,
    attacking_side: str,
) -> float:
    attackers: list[tuple[float, float]] = []
    defenders: list[tuple[float, float]] = []
    att_index = None
    for pid, pos in xy.items():
        if pid in gk or team_of.get(pid) is None:
            continue
        if team_of[pid] == att_team:
            if pid == striker_id:
                att_index = len(attackers)
            attackers.append(pos)
        else:
            defenders.append(pos)
    if att_index is None or not attackers or not defenders:
        return float("nan")
    marking = hungarian_marking(
        np.asarray(defenders, dtype=float),
        np.asarray(attackers, dtype=float),
        attack_direction(attacking_side),
    )
    for pair in marking["pairs"]:
        if pair["att_idx"] == att_index and pair["goal_side"]:
            return float(pair["dist"])
    return float("nan")


def process_match(match_id: str) -> list[dict]:
    events = load_events(match_id)
    phases = load_phases(match_id)
    team_of, roster, gk, _plen, _pwid = load_roster(match_id)
    modal = modal_positions(events)
    poss = events[
        (events["event_type"] == "player_possession")
        & events["player_position"].isin(STRIKER_ROLES)
        & events["team_out_of_possession_phase_type"].isin(ORGANIZED)
    ]
    sample_index = _sample_index(phases)
    needed = set(sample_index)
    for fr in poss["frame_start"]:
        fid = _iid(fr)
        if fid is not None and fid >= 0:
            needed.add(fid)
    frames = stream_frames(match_id, needed)
    medians = _role_medians(frames, sample_index, team_of, gk, modal, roster)

    rows = []
    for row in poss.itertuples(index=False):
        sign = attack_sign(row.attacking_side)
        pid = _iid(row.player_id)
        att_team = _iid(row.team_id)
        fr = _iid(row.frame_start)
        if not np.isfinite(sign) or pid is None or att_team is None or fr is None:
            continue
        xy = frames.get(fr) or {}
        if pid not in xy:
            continue
        _opp, dist = nearest_opponent(pid, xy, team_of, gk)
        if not np.isfinite(dist):
            continue
        block = row.team_out_of_possession_phase_type
        dep = _departure(xy, team_of, att_team, gk, modal, roster, block, sign, medians)
        h_dist = _hungarian_dist(xy, team_of, att_team, gk, pid, row.attacking_side)
        role = _role(pid, roster, modal)
        rows.append({
            "match_id": match_id,
            "block": block,
            "striker_role": role,
            "tight": 1.0 if dist <= TIGHT_MARK_M else 0.0,
            "tight_hungarian": 1.0 if np.isfinite(h_dist) and h_dist <= TIGHT_MARK_M else 0.0,
            "nearest_dist": dist,
            "hungarian_dist": h_dist,
            "lead_to_shot": row.lead_to_shot,
            "xthreat": row.player_targeted_xthreat,
            "line_break": row.first_line_break,
            "dep_CB": dep["CB"],
            "dep_CM": dep["CM"],
            "dep_FB": dep["FB"],
        })
    return rows


def _arm(contrast: dict, *, positive: bool) -> dict:
    supported = supports(contrast, "positive") if positive else bool(contrast.get("ci_excludes_zero"))
    return {
        "n": int(contrast["n_treat"] + contrast["n_control"]),
        "n_tight": contrast["n_treat"],
        "n_not": contrast["n_control"],
        "n_matches": contrast["n_matches"],
        "n_matches_both": contrast["n_matches_both"],
        "effect": contrast["effect"],
        "ci_low": contrast["ci_low"],
        "ci_high": contrast["ci_high"],
        "mean_tight": contrast["mean_treat"],
        "mean_not": contrast["mean_control"],
        "supported": supported,
    }


def summarize(df: pd.DataFrame) -> dict:
    roles = []
    drawn = []
    for i, group in enumerate(GROUPS):
        contrast = cluster_diff(df, f"dep_{group}", "tight", seed=70 + i)
        arm = _arm(contrast, positive=True)
        arm["role"] = group
        arm["roles"] = sorted(GROUP_ROLES[group])
        arm["unit"] = "metres"
        roles.append(arm)
        if arm["supported"]:
            drawn.append(group)

    outcomes = {}
    for i, (name, col) in enumerate((
        ("lead_to_shot", "lead_to_shot"),
        ("xthreat", "xthreat"),
        ("line_break", "line_break"),
    )):
        contrast = cluster_diff(df, col, "tight", seed=91 + i)
        outcomes[name] = _arm(contrast, positive=False)

    hungarian = []
    for i, group in enumerate(GROUPS):
        contrast = cluster_diff(df, f"dep_{group}", "tight_hungarian", seed=94 + i)
        arm = _arm(contrast, positive=True)
        arm["role"] = group
        arm["unit"] = "metres"
        hungarian.append(arm)

    return {
        "id": 7,
        "title": "Tight mark on the striker and departure from the block",
        "population": "CF or ST player possession in a low, medium, or high block",
        "n_events": int(len(df)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "tight_mark_m": TIGHT_MARK_M,
        "tight_rate": float(df["tight"].mean()) if len(df) else None,
        "tight_hungarian_rate": float(df["tight_hungarian"].mean()) if len(df) else None,
        "mean_nearest_dist": float(pd.to_numeric(df["nearest_dist"], errors="coerce").mean()) if len(df) else None,
        "n_by_block": {k: int(v) for k, v in df["block"].value_counts().to_dict().items()} if len(df) else {},
        "definition": (
            f"Tight mark: nearest outfield opponent within {TIGHT_MARK_M:.1f} m at frame_start. "
            f"Goalkeepers are excluded. Drawn out: mean Euclidean distance, in metres, of the "
            f"centre-backs, central midfielders, or full backs from that role acronym's median "
            f"position for the same defending team and the same block in the match. "
            f"The median uses a {10 // SAMPLE_STRIDE} Hz sample of organized-block frames "
            f"in attack-normalized coordinates and needs at least {MIN_ROLE_SAMPLES} samples. "
            f"Role acronyms come from the match json, with the modal event position used only "
            f"when the squad role is missing or SUB. "
            f"The interval resamples matches. A role is supported only when that interval "
            f"lies entirely above zero. "
            f"For lead_to_shot, xthreat, and line_break, supported means the interval excludes zero in either direction. "
            f"xthreat is player_targeted_xthreat on the possession. "
            f"line_break is first_line_break. "
            f"Hungarian sensitivity: goal-side assignment from coupled_formation.hungarian_marking "
            f"within {TIGHT_MARK_M:.1f} m. It does not decide drawn_out_roles."
        ),
        "inference": "match_cluster_bootstrap",
        "roles": roles,
        "drawn_out_roles": drawn,
        "outcomes": outcomes,
        "hungarian_sensitivity": hungarian,
    }


def run(selected: list[str] | None = None) -> dict:
    ids = selected or match_ids()
    rows: list[dict] = []
    for i, mid in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {mid}", flush=True)
        part = process_match(mid)
        rows.extend(part)
        print(f"    possessions {len(part)}", flush=True)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "h7_summary.json"
    path.write_text(json.dumps(json_ready(summary), indent=2))
    print(f"wrote {path}", flush=True)
    print(f"drawn_out_roles={summary['drawn_out_roles']}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*")
    args = parser.parse_args()
    run(args.matches if args.matches else None)
