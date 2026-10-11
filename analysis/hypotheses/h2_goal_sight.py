"""Hypothesis 2: how much of the goal mouth the on-ball attacker can see.

Question
--------
What fraction of the goal mouth is visible to the on-ball attacker, and does
a visibility threshold separate big chances?

The quantity is the share of the 7.32 m goal mouth that is inside the
attacker's forward half-plane and not blocked by an opponent. It is not an
angle, and it is not xG, xGOT, or xT.

Definitions locked before the contrast
--------------------------------------
Population
    One row per on-ball player possession that the dynamic events place in
    the attacking third, measured on the last tracking frame of that
    possession that still sits in the attacking third. Two matches only:
    1925299 and 1996435, the matches with body pose.

Big chance
    lead_to_shot is true, OR the maximum xshot_player_possession_max on
    on-ball engagements linked by associated_player_possession_event_id is
    at least 0.20. xshot is SkillCorner's probability that the player
    possession produces a shot. 0.20 is a pre-specified "high" cut: engagement
    xshot is typically near zero (median about 0.003), while possessions that
    end in a shot sit much higher (median about 0.7).

Visible fraction
    Goal mouth is the segment of width 7.32 m centered at y = 0 on the goal
    line x = ±pitch_length/2. Pitch length and width come from the match
    json. The origin is the pitch center. attacking_side left_to_right uses
    the goal at +pitch_length/2; right_to_left uses −pitch_length/2.

    The viewpoint is the attacker's tracking location. The heading is the
    shoulder orientation from src.features.pose_orientation (0° along +x,
    counter-clockwise positive): the left-to-right shoulder axis rotated
    +90°. A goal-mouth point counts as visible when

    1. its direction lies within ±90° of that heading (the forward
       half-plane), and
    2. the ground-plane segment from the attacker to that point stays more
       than the opponent occlusion radius away from every opponent.

Opponent occlusion radius
    0.25 m. That is half of a 0.50 m shoulder width, inside the pose
    helper's plausible shoulder-width band of 0.15–0.60 m. Applied to
    opponent tracking coordinates. Teammates and the attacker are not
    occluders. Pose z is not used: the pose notes say z is relative to the
    player's centroid, not a pitch height.

Primary threshold
    0.50 of the goal mouth. Pre-specified. The effect is the risk
    difference P(big chance | visible fraction >= 0.50) minus
    P(big chance | visible fraction < 0.50). A 95% percentile bootstrap
    interval is taken over possessions. The hypothesis is supported only
    when that interval excludes zero. Thresholds 0.25 and 0.75 are reported
    the same way and do not change the verdict.

This contrast uses two matches, so it is underpowered whether or not the
interval excludes zero.
"""

from __future__ import annotations

import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
OPENDATA = WORKSPACE / "opendata"
if str(OPENDATA) not in sys.path:
    sys.path.insert(0, str(OPENDATA))

from src.data.pose_loading import download_match, tracking_to_pose_frame
from src.features.pose_orientation import (
    DEFAULT_MAX_ERR_CM,
    fold_to_tracking_frame,
    shoulder_table,
)

MATCH_IDS = (1925299, 1996435)
GOAL_WIDTH_M = 7.32
HALF_GOAL_M = GOAL_WIDTH_M / 2.0
OCCLUSION_RADIUS_M = 0.25
FORWARD_HALF_ANGLE_DEG = 90.0
HIGH_XSHOT = 0.20
PRIMARY_THRESHOLD = 0.50
THRESHOLD_GRID = (0.25, 0.50, 0.75)
MIN_GOAL_DEPTH_M = 1.0
POSE_FRAME_TOLERANCE = 2  # tracking frames, 0.2 s at 10 fps
ALIGN_TOLERANCE_M = 2.0
MOUTH_SAMPLES = 733  # 1 cm steps across 7.32 m, endpoints included
N_BOOT = 4000
BOOT_SEED = 20261011
POSE_DIR = Path(os.environ.get("H2_POSE_DIR", "/tmp/pose_data"))
OUT_PATH = WORKSPACE / "analysis" / "output" / "h2_summary.json"

EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "frame_start",
    "frame_end",
    "period",
    "attacking_side",
    "player_id",
    "team_id",
    "third_start",
    "third_end",
    "lead_to_shot",
    "end_type",
    "phase_index",
    "xshot_player_possession_max",
    "associated_player_possession_event_id",
]


def goal_x_for(attacking_side: str, pitch_length: float) -> float:
    if attacking_side == "left_to_right":
        return pitch_length / 2.0
    if attacking_side == "right_to_left":
        return -pitch_length / 2.0
    raise ValueError(f"unknown attacking_side {attacking_side}")


def in_attacking_third(x: float, attacking_side: str, pitch_length: float) -> bool:
    boundary = pitch_length / 6.0
    if attacking_side == "left_to_right":
        return x >= boundary
    return x <= -boundary


def pose_frames_for_tracking(tracking_frame: int) -> list[int]:
    """Pose frames that fold onto one 10 fps tracking frame.

    The window is filtered with fold_to_tracking_frame so the helper, not a
    re-derived formula, decides the match.
    """
    approx = int(tracking_to_pose_frame(tracking_frame))
    return [
        pose_frame
        for pose_frame in range(approx - 3, approx + 5)
        if fold_to_tracking_frame(pose_frame) == tracking_frame
    ]


def _segment_blocked(
    ax: float,
    ay: float,
    goal_x: float,
    ys: np.ndarray,
    opponents: np.ndarray,
    radius: float,
) -> np.ndarray:
    """True where an opponent circle intersects the open segment to the goal."""
    n = ys.shape[0]
    blocked = np.zeros(n, dtype=bool)
    if opponents.size == 0:
        return blocked
    dx = goal_x - ax
    dy = ys - ay
    length_sq = dx * dx + dy * dy
    radius_sq = radius * radius
    for ox, oy in opponents:
        if (ox - ax) ** 2 + (oy - ay) ** 2 <= radius_sq:
            blocked[:] = True
            return blocked
        t = ((ox - ax) * dx + (oy - ay) * dy) / length_sq
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        dist_sq = (ox - closest_x) ** 2 + (oy - closest_y) ** 2
        blocked |= (t > 0.0) & (t < 1.0) & (dist_sq <= radius_sq)
    return blocked


def mouth_fractions(
    ax: float,
    ay: float,
    heading_deg: float,
    opponents: np.ndarray,
    goal_x: float,
    radius: float = OCCLUSION_RADIUS_M,
) -> tuple[float, float, float]:
    """Return (visible, forward-only, unoccluded) fractions of the goal mouth.

    visible is the primary quantity: in the forward half-plane and not
    occluded. The other two are the separate pieces of that definition.
    """
    ys = np.linspace(-HALF_GOAL_M, HALF_GOAL_M, MOUTH_SAMPLES)
    dx = goal_x - ax
    dy = ys - ay
    bearing = np.degrees(np.arctan2(dy, dx))
    delta = (bearing - heading_deg + 180.0) % 360.0 - 180.0
    forward = np.abs(delta) <= FORWARD_HALF_ANGLE_DEG
    blocked = _segment_blocked(ax, ay, goal_x, ys, opponents, radius)
    open_mouth = ~blocked
    visible = forward & open_mouth
    scale = float(MOUTH_SAMPLES)
    return (
        float(visible.sum() / scale),
        float(forward.sum() / scale),
        float(open_mouth.sum() / scale),
    )


def _self_check() -> None:
    for tracking_frame in (0, 1, 10, 49795, 61300):
        frames = pose_frames_for_tracking(tracking_frame)
        assert frames, tracking_frame
        assert all(fold_to_tracking_frame(p) == tracking_frame for p in frames)

    clear, forward, open_mouth = mouth_fractions(40.0, 0.0, 0.0, np.zeros((0, 2)), 52.5)
    assert clear == 1.0 and forward == 1.0 and open_mouth == 1.0

    away, _, _ = mouth_fractions(40.0, 0.0, 180.0, np.zeros((0, 2)), 52.5)
    assert away == 0.0

    other_goal, _, _ = mouth_fractions(-40.0, 0.0, 180.0, np.zeros((0, 2)), -52.5)
    assert other_goal == 1.0

    blocked, _, open_blocked = mouth_fractions(
        40.0, 0.0, 0.0, np.array([[46.0, 0.0]]), 52.5
    )
    # Tangent shadow of a 0.25 m circle 6 m in front, goal 12.5 m away.
    assert 0.84 < blocked < 0.88, blocked
    assert abs(blocked - open_blocked) < 1e-9

    swallowed, _, _ = mouth_fractions(40.0, 0.0, 0.0, np.array([[40.1, 0.0]]), 52.5)
    assert swallowed == 0.0
    assert goal_x_for("left_to_right", 105) == 52.5
    assert goal_x_for("right_to_left", 105) == -52.5
    assert in_attacking_third(17.5, "left_to_right", 105)
    assert not in_attacking_third(17.0, "left_to_right", 105)
    assert in_attacking_third(-17.5, "right_to_left", 105)


def _frame_number(line: bytes | str) -> int:
    if isinstance(line, str):
        comma = line.find(",")
        return int(line[9:comma])
    comma = line.find(b",")
    return int(line[9:comma])


def load_match_meta(match_id: int) -> dict:
    path = OPENDATA / "data" / "matches" / str(match_id) / f"{match_id}_match.json"
    meta = json.loads(path.read_text())
    return {
        "match_id": match_id,
        "pitch_length_m": float(meta["pitch_length"]),
        "pitch_width_m": float(meta["pitch_width"]),
        "home_team": meta["home_team"]["name"],
        "away_team": meta["away_team"]["name"],
        "home_team_side": list(meta["home_team_side"]),
    }


def load_events(match_id: int) -> pd.DataFrame:
    path = (
        OPENDATA / "data" / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    )
    return pd.read_csv(path, usecols=EVENT_COLUMNS, low_memory=False)


def candidate_possessions(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    possessions = events.loc[events["event_type"] == "player_possession"].copy()
    in_third = (possessions["third_start"] == "attacking_third") | (
        possessions["third_end"] == "attacking_third"
    )
    possessions = possessions.loc[in_third].copy()
    engagements = events.loc[events["event_type"] == "on_ball_engagement"]
    xshot = engagements.groupby("associated_player_possession_event_id")[
        "xshot_player_possession_max"
    ].max()
    possessions["xshot_max"] = possessions["event_id"].map(xshot)
    possessions["lead_to_shot"] = possessions["lead_to_shot"].fillna(False).astype(bool)
    possessions["high_xshot"] = possessions["xshot_max"] >= HIGH_XSHOT
    possessions["big_chance"] = possessions["lead_to_shot"] | possessions["high_xshot"].fillna(
        False
    )
    possessions["match_id"] = match_id
    possessions["frame_start"] = possessions["frame_start"].astype(int)
    possessions["frame_end"] = possessions["frame_end"].astype(int)
    possessions["player_id"] = possessions["player_id"].astype(int)
    possessions["team_id"] = possessions["team_id"].astype(int)
    return possessions.reset_index(drop=True)


def player_teams(events: pd.DataFrame) -> dict[int, int]:
    known = events.dropna(subset=["player_id", "team_id"])
    teams: dict[int, int] = {}
    for player_id, team_id in zip(known["player_id"], known["team_id"]):
        teams[int(player_id)] = int(team_id)
    return teams


def load_tracking(match_id: int, frames: set[int]) -> dict[int, dict[int, tuple[float, float]]]:
    path = (
        OPENDATA
        / "data"
        / "matches"
        / str(match_id)
        / f"{match_id}_tracking_extrapolated.jsonl"
    )
    kept: dict[int, dict[int, tuple[float, float]]] = {}
    with path.open() as handle:
        for line in handle:
            frame = _frame_number(line)
            if frame not in frames:
                continue
            record = json.loads(line)
            players: dict[int, tuple[float, float]] = {}
            for player in record.get("player_data") or []:
                x, y = player.get("x"), player.get("y")
                if x is None or y is None:
                    continue
                players[int(player["player_id"])] = (float(x), float(y))
            kept[frame] = players
            if len(kept) == len(frames):
                break
    return kept


def _geometric_ok(
    x: float,
    y: float,
    attacking_side: str,
    pitch_length: float,
    pitch_width: float,
) -> bool:
    goal_x = goal_x_for(attacking_side, pitch_length)
    if not in_attacking_third(x, attacking_side, pitch_length):
        return False
    depth = (goal_x - x) if attacking_side == "left_to_right" else (x - goal_x)
    if depth < MIN_GOAL_DEPTH_M:
        return False
    if abs(y) > pitch_width / 2.0 + 2.0:
        return False
    if abs(x) > pitch_length / 2.0 + 2.0:
        return False
    return True


def preferred_frame(
    row: pd.Series,
    tracking: dict[int, dict[int, tuple[float, float]]],
    pitch_length: float,
    pitch_width: float,
) -> int | None:
    player_id = int(row.player_id)
    for frame in range(int(row.frame_end), int(row.frame_start) - 1, -1):
        players = tracking.get(frame)
        if not players or player_id not in players:
            continue
        if len(players) < 10:
            continue
        x, y = players[player_id]
        if _geometric_ok(x, y, row.attacking_side, pitch_length, pitch_width):
            return frame
    return None


def load_shoulders(match_id: int, pose_frames: set[int]) -> pd.DataFrame:
    archive = download_match(match_id, dest_dir=POSE_DIR)
    member = f"{match_id}.jsonl"
    batch: list[dict] = []
    tables: list[pd.DataFrame] = []
    scanned = 0

    def flush() -> None:
        if not batch:
            return
        tables.append(shoulder_table(batch, max_err_cm=DEFAULT_MAX_ERR_CM))
        batch.clear()

    with zipfile.ZipFile(archive) as zf, zf.open(member) as handle:
        for line in handle:
            scanned += 1
            if scanned % 25000 == 0:
                print(f"  pose {match_id}: scanned {scanned} frames", flush=True)
            if _frame_number(line) not in pose_frames:
                continue
            batch.append(json.loads(line))
            if len(batch) >= 200:
                flush()
    flush()
    print(f"  pose {match_id}: scanned {scanned} frames, kept {len(pose_frames)}", flush=True)
    if not tables:
        return pd.DataFrame()
    shoulders = pd.concat(tables, ignore_index=True)
    shoulders = shoulders.loc[shoulders["is_reliable"]].copy()
    return shoulders


def _shoulder_index(shoulders: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if shoulders.empty:
        return {}
    return {
        int(player_id): group.reset_index(drop=True)
        for player_id, group in shoulders.groupby("player_id")
    }


def choose_shoulder(
    player_rows: pd.DataFrame,
    preferred: int,
    frame_start: int,
    frame_end: int,
    tracking: dict[int, dict[int, tuple[float, float]]],
    attacking_side: str,
    pitch_length: float,
    pitch_width: float,
) -> pd.Series | None:
    low = max(frame_start - POSE_FRAME_TOLERANCE, preferred - POSE_FRAME_TOLERANCE)
    high = min(frame_end + POSE_FRAME_TOLERANCE, preferred + POSE_FRAME_TOLERANCE)
    # The tolerance is around the preferred frame. Also keep the row inside
    # the possession expanded by the same 0.2 s, which `low`/`high` already
    # intersect with the possession window.
    window = player_rows.loc[
        (player_rows["frame_10fps"] >= low) & (player_rows["frame_10fps"] <= high)
    ]
    if window.empty:
        return None
    target_pose = tracking_to_pose_frame(preferred)
    best: tuple[tuple[int, float], pd.Series] | None = None
    for row in window.itertuples(index=False):
        frame = int(row.frame_10fps)
        players = tracking.get(frame)
        if not players or int(row.player_id) not in players:
            continue
        track_x, track_y = players[int(row.player_id)]
        if not _geometric_ok(track_x, track_y, attacking_side, pitch_length, pitch_width):
            continue
        mid_x = (row.l_shoulder_x + row.r_shoulder_x) / 2.0
        mid_y = (row.l_shoulder_y + row.r_shoulder_y) / 2.0
        if math.hypot(mid_x - track_x, mid_y - track_y) > ALIGN_TOLERANCE_M:
            continue
        if row.shoulder_deg is None or (isinstance(row.shoulder_deg, float) and math.isnan(row.shoulder_deg)):
            continue
        rank = (abs(frame - preferred), abs(row.frame - target_pose))
        if best is None or rank < best[0]:
            best = (rank, row)
    if best is None:
        return None
    return best[1]


def measure_possession(
    row: pd.Series,
    tracking: dict[int, dict[int, tuple[float, float]]],
    shoulder_rows: pd.DataFrame | None,
    teams: dict[int, int],
    pitch_length: float,
    pitch_width: float,
) -> dict | None:
    if shoulder_rows is None or shoulder_rows.empty:
        return None
    preferred = preferred_frame(row, tracking, pitch_length, pitch_width)
    if preferred is None:
        return None
    chosen = choose_shoulder(
        shoulder_rows,
        preferred,
        int(row.frame_start),
        int(row.frame_end),
        tracking,
        row.attacking_side,
        pitch_length,
        pitch_width,
    )
    if chosen is None:
        return None
    frame = int(chosen.frame_10fps)
    players = tracking[frame]
    attacker_id = int(row.player_id)
    ax, ay = players[attacker_id]
    attacker_team = int(row.team_id)
    opponents: list[tuple[float, float]] = []
    unknown = 0
    for player_id, (x, y) in players.items():
        if player_id == attacker_id:
            continue
        team_id = teams.get(player_id)
        if team_id is None:
            unknown += 1
            continue
        if team_id == attacker_team:
            continue
        opponents.append((x, y))
    visible, forward, open_mouth = mouth_fractions(
        ax,
        ay,
        float(chosen.shoulder_deg),
        np.asarray(opponents, dtype=float).reshape(-1, 2),
        goal_x_for(row.attacking_side, pitch_length),
    )
    to_goal = math.degrees(
        math.atan2(0.0 - ay, goal_x_for(row.attacking_side, pitch_length) - ax)
    )
    heading_gap = (to_goal - float(chosen.shoulder_deg) + 180.0) % 360.0 - 180.0
    return {
        "match_id": int(row.match_id),
        "event_id": row.event_id,
        "phase_index": None if pd.isna(row.phase_index) else int(row.phase_index),
        "big_chance": bool(row.big_chance),
        "lead_to_shot": bool(row.lead_to_shot),
        "high_xshot": bool(row.high_xshot) if not pd.isna(row.high_xshot) else False,
        "xshot_max": None if pd.isna(row.xshot_max) else float(row.xshot_max),
        "end_type": None if pd.isna(row.end_type) else str(row.end_type),
        "visible_fraction": visible,
        "forward_fraction": forward,
        "unoccluded_fraction": open_mouth,
        "abs_heading_gap_deg": abs(heading_gap),
        "n_opponents": len(opponents),
        "n_unknown_team": unknown,
        "frame": frame,
    }


def needed_tracking_frames(possessions: pd.DataFrame) -> set[int]:
    frames: set[int] = set()
    for start, end in zip(possessions["frame_start"], possessions["frame_end"]):
        low = max(0, int(start) - POSE_FRAME_TOLERANCE)
        high = int(end) + POSE_FRAME_TOLERANCE
        frames.update(range(low, high + 1))
    return frames


def needed_pose_frames(measurement_frames: list[int]) -> set[int]:
    pose_frames: set[int] = set()
    for frame in measurement_frames:
        for delta in range(-POSE_FRAME_TOLERANCE, POSE_FRAME_TOLERANCE + 1):
            if frame + delta < 0:
                continue
            pose_frames.update(pose_frames_for_tracking(frame + delta))
    return pose_frames


def analyze_match(match_id: int) -> tuple[list[dict], dict]:
    meta = load_match_meta(match_id)
    events = load_events(match_id)
    possessions = candidate_possessions(events, match_id)
    teams = player_teams(events)
    print(
        f"match {match_id}: {len(possessions)} attacking-third possessions, "
        f"{int(possessions.big_chance.sum())} big chances before pose",
        flush=True,
    )
    tracking_frames = needed_tracking_frames(possessions)
    tracking = load_tracking(match_id, tracking_frames)
    preferred = [
        preferred_frame(row, tracking, meta["pitch_length_m"], meta["pitch_width_m"])
        for row in possessions.itertuples(index=False)
    ]
    # itertuples renames columns; preferred_frame expects a Series-like with attrs.
    # Rebuild via iterrows so attribute access matches the helper. The list
    # above used itertuples, which also exposes the columns. Keep it.
    measurable = [frame for frame in preferred if frame is not None]
    pose_frames = needed_pose_frames(measurable)
    shoulders = load_shoulders(match_id, pose_frames)
    by_player = _shoulder_index(shoulders)

    rows: list[dict] = []
    dropped_geometry = 0
    dropped_pose = 0
    for record, frame in zip(possessions.to_dict("records"), preferred):
        if frame is None:
            dropped_geometry += 1
            continue
        series = pd.Series(record)
        measured = measure_possession(
            series,
            tracking,
            by_player.get(int(series.player_id)),
            teams,
            meta["pitch_length_m"],
            meta["pitch_width_m"],
        )
        if measured is None:
            dropped_pose += 1
            continue
        rows.append(measured)
    coverage = {
        "match_id": match_id,
        "fixture": f"{meta['home_team']} vs {meta['away_team']}",
        "pitch_length_m": meta["pitch_length_m"],
        "pitch_width_m": meta["pitch_width_m"],
        "pitch_origin": "center",
        "home_team_side_by_period": meta["home_team_side"],
        "candidate_possessions": int(len(possessions)),
        "candidate_big_chances": int(possessions["big_chance"].sum()),
        "dropped_not_in_attacking_third_at_a_tracked_frame": dropped_geometry,
        "dropped_no_reliable_shoulder": dropped_pose,
        "n": len(rows),
        "n_big_chances": int(sum(row["big_chance"] for row in rows)),
    }
    print(
        f"match {match_id}: kept {coverage['n']} "
        f"({coverage['n_big_chances']} big chances)",
        flush=True,
    )
    return rows, coverage


def _risk_difference(visible: np.ndarray, outcome: np.ndarray, threshold: float) -> dict:
    above = visible >= threshold
    n_above = int(above.sum())
    n_below = int((~above).sum())
    if n_above == 0 or n_below == 0:
        return {
            "threshold": threshold,
            "estimate": None,
            "ci95": [None, None],
            "excludes_zero": False,
            "separates": False,
            "n_at_or_above": n_above,
            "n_below": n_below,
            "n_big_at_or_above": int(outcome[above].sum()) if n_above else 0,
            "n_big_below": int(outcome[~above].sum()) if n_below else 0,
            "rate_at_or_above": None,
            "rate_below": None,
            "undefined_split": True,
        }
    rate_above = float(outcome[above].mean())
    rate_below = float(outcome[~above].mean())
    return {
        "threshold": threshold,
        "estimate": rate_above - rate_below,
        "n_at_or_above": n_above,
        "n_below": n_below,
        "n_big_at_or_above": int(outcome[above].sum()),
        "n_big_below": int(outcome[~above].sum()),
        "rate_at_or_above": rate_above,
        "rate_below": rate_below,
        "undefined_split": False,
    }


def bootstrap(visible: np.ndarray, outcome: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    n = visible.shape[0]
    point = {threshold: _risk_difference(visible, outcome, threshold) for threshold in THRESHOLD_GRID}
    mean_big = float(visible[outcome == 1].mean()) if outcome.any() else None
    mean_other = float(visible[outcome == 0].mean()) if (outcome == 0).any() else None
    mean_diff = None if mean_big is None or mean_other is None else mean_big - mean_other

    draws = rng.integers(0, n, size=(N_BOOT, n))
    vis_b = visible[draws]
    out_b = outcome[draws]
    threshold_draws: dict[float, np.ndarray] = {}
    for threshold in THRESHOLD_GRID:
        above = vis_b >= threshold
        n_above = above.sum(axis=1)
        n_below = (~above).sum(axis=1)
        valid = (n_above > 0) & (n_below > 0)
        diff = np.full(N_BOOT, np.nan)
        sum_above = (out_b * above).sum(axis=1)
        sum_below = (out_b * ~above).sum(axis=1)
        diff[valid] = sum_above[valid] / n_above[valid] - sum_below[valid] / n_below[valid]
        threshold_draws[threshold] = diff

    n_big = out_b.sum(axis=1)
    n_other = n - n_big
    mean_valid = (n_big > 0) & (n_other > 0)
    mean_draw = np.full(N_BOOT, np.nan)
    mean_draw[mean_valid] = (
        (vis_b * out_b).sum(axis=1)[mean_valid] / n_big[mean_valid]
        - (vis_b * (1.0 - out_b)).sum(axis=1)[mean_valid] / n_other[mean_valid]
    )

    def ci(samples: np.ndarray) -> tuple[float | None, float | None, int]:
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return None, None, 0
        low, high = np.percentile(finite, [2.5, 97.5])
        return float(low), float(high), int(finite.size)

    results = []
    for threshold in THRESHOLD_GRID:
        low, high, n_valid = ci(threshold_draws[threshold])
        item = dict(point[threshold])
        item["ci95"] = [low, high]
        item["n_boot_valid"] = n_valid
        excludes = (
            low is not None
            and high is not None
            and item["estimate"] is not None
            and (high < 0.0 or low > 0.0)
        )
        item["excludes_zero"] = bool(excludes)
        item["separates"] = bool(excludes)
        results.append(item)

    mean_low, mean_high, mean_valid_n = ci(mean_draw)
    mean_excludes = (
        mean_low is not None
        and mean_high is not None
        and mean_diff is not None
        and (mean_high < 0.0 or mean_low > 0.0)
    )
    return {
        "thresholds": results,
        "mean_visible_big_chance": mean_big,
        "mean_visible_other": mean_other,
        "mean_difference_big_minus_other": mean_diff,
        "mean_difference_ci95": [mean_low, mean_high],
        "mean_difference_excludes_zero": bool(mean_excludes),
        "mean_difference_n_boot_valid": mean_valid_n,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_summary(rows: list[dict], coverage: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    visible = frame["visible_fraction"].to_numpy(dtype=float)
    outcome = frame["big_chance"].to_numpy(dtype=float)
    sampled = bootstrap(visible, outcome)
    primary = next(item for item in sampled["thresholds"] if item["threshold"] == PRIMARY_THRESHOLD)
    separating = [
        item["threshold"]
        for item in sampled["thresholds"]
        if item["separates"]
    ]
    supported = bool(primary["excludes_zero"])
    phases = frame.dropna(subset=["phase_index"])
    n_phases = int(phases.groupby(["match_id", "phase_index"]).ngroups) if len(phases) else 0
    return {
        "hypothesis_id": "h2",
        "hypothesis": (
            "What fraction of the goal mouth is visible to the on-ball "
            "attacker, and does a visibility threshold separate big chances?"
        ),
        "verdict": "supported" if supported else "not_supported",
        "supported": supported,
        "underpowered": True,
        "underpowered_reason": (
            "Only two matches have body pose (1925299 and 1996435). "
            f"n = {len(frame)} on-ball possessions, not matches, and those "
            "possessions cluster inside team possessions "
            f"({n_phases} phase indexes). The 95% interval resamples "
            "possessions and does not recover match-level variation. Two "
            "matches cannot pin down the association."
        ),
        "support_rule": (
            "Supported only if the 95% bootstrap interval for the risk "
            "difference at visible fraction 0.50 excludes zero. The grid "
            "0.25 and 0.75 is reported and does not decide the verdict."
        ),
        "n": int(len(frame)),
        "n_big_chances": int(outcome.sum()),
        "n_other": int((outcome == 0).sum()),
        "n_matches": len(MATCH_IDS),
        "matches": list(MATCH_IDS),
        "effect": {
            "name": "risk_difference",
            "description": (
                "P(big chance | visible fraction >= 0.50) - "
                "P(big chance | visible fraction < 0.50)"
            ),
            "estimate": primary["estimate"],
            "ci95": primary["ci95"],
            "ci_level": 0.95,
            "interval_method": (
                f"percentile bootstrap of possessions, {N_BOOT} draws, "
                f"seed {BOOT_SEED}"
            ),
            "excludes_zero": primary["excludes_zero"],
            "n_at_or_above": primary["n_at_or_above"],
            "n_below": primary["n_below"],
            "n_big_at_or_above": primary["n_big_at_or_above"],
            "n_big_below": primary["n_big_below"],
            "rate_at_or_above": primary["rate_at_or_above"],
            "rate_below": primary["rate_below"],
            "undefined_split": primary["undefined_split"],
        },
        "primary_visibility_threshold": PRIMARY_THRESHOLD,
        "thresholds": sampled["thresholds"],
        "thresholds_that_separate": separating,
        "visible_fraction": {
            "mean": float(visible.mean()),
            "median": float(np.median(visible)),
            "p25": float(np.percentile(visible, 25)),
            "p75": float(np.percentile(visible, 75)),
            "mean_big_chance": sampled["mean_visible_big_chance"],
            "mean_other": sampled["mean_visible_other"],
            "difference_big_minus_other": sampled["mean_difference_big_minus_other"],
            "difference_ci95": sampled["mean_difference_ci95"],
            "difference_excludes_zero": sampled["mean_difference_excludes_zero"],
            "mean_forward_fraction": float(frame["forward_fraction"].mean()),
            "mean_unoccluded_fraction": float(frame["unoccluded_fraction"].mean()),
            "median_abs_heading_gap_deg": float(frame["abs_heading_gap_deg"].median()),
        },
        "big_chance_definition": (
            "An on-ball player possession is a big chance when lead_to_shot "
            f"is true OR xshot_player_possession_max >= {HIGH_XSHOT:.2f}. "
            "xshot is the maximum xshot_player_possession_max across on-ball "
            "engagements whose associated_player_possession_event_id is that "
            "possession. xshot is SkillCorner's shot probability for the "
            "possession, not xG, xGOT, or xT. Possessions with no linked "
            "engagement can still be big chances through lead_to_shot."
        ),
        "high_xshot_threshold": HIGH_XSHOT,
        "big_chance_composition": {
            "n_lead_to_shot": int(frame["lead_to_shot"].sum()),
            "n_high_xshot": int(frame["high_xshot"].sum()),
            "n_both": int((frame["lead_to_shot"] & frame["high_xshot"]).sum()),
            "n_end_type_shot": int((frame["end_type"] == "shot").sum()),
        },
        "visibility_definition": (
            "Fraction of the 7.32 m goal mouth (posts at y = ±3.66 m on the "
            "goal line) for which the ground-plane segment from the attacker's "
            "tracking location lies within ±90° of the shoulder heading and "
            "passes no closer than the opponent occlusion radius to an "
            "opponent. Shoulder heading is src.features.pose_orientation."
            "shoulder_orientation_deg, kept only when shoulder_table marks "
            f"the row reliable (max shoulder error {DEFAULT_MAX_ERR_CM:.0f} cm, "
            "plausible shoulder width). Heading may come from a pose frame "
            f"within ±{POSE_FRAME_TOLERANCE} tracking frames (0.2 s) of the "
            "last attacking-third frame of the possession. The shoulder "
            f"midpoint must lie within {ALIGN_TOLERANCE_M:.0f} m of that "
            "tracking location. Pose z is not used."
        ),
        "opponent_occlusion_radius_m": OCCLUSION_RADIUS_M,
        "opponent_occlusion_radius_basis": (
            "0.25 m is half of a 0.50 m shoulder width, inside the pose "
            "helper's plausible shoulder-width band of 0.15–0.60 m. The "
            "circle is placed on each opponent's extrapolated tracking "
            "coordinate. An opponent whose center is within 0.25 m of the "
            "attacker blocks the whole mouth. Teammates are not occluders."
        ),
        "goal_width_m": GOAL_WIDTH_M,
        "forward_half_angle_deg": FORWARD_HALF_ANGLE_DEG,
        "goal_line_rule": (
            "Pitch origin is the center. Goal x is +pitch_length/2 when "
            "attacking_side is left_to_right and -pitch_length/2 when it is "
            "right_to_left. The attacking third is the third of the pitch "
            "nearest that goal (boundary at pitch_length/6)."
        ),
        "pitch": coverage,
        "coordinate_sources": {
            "positions": "tracking_extrapolated.jsonl at 10 fps, including extrapolated players",
            "heading": "body pose shoulders at 25 fps via src.features.pose_orientation",
            "events": "dynamic_events player_possession and on_ball_engagement",
        },
        "not_used": ["xG", "xGOT", "xT"],
    }


def main() -> None:
    _self_check()
    rows: list[dict] = []
    coverage: list[dict] = []
    for match_id in MATCH_IDS:
        match_rows, match_coverage = analyze_match(match_id)
        rows.extend(match_rows)
        coverage.append(match_coverage)
    if not rows:
        raise SystemExit("no possessions had a reliable shoulder and a tracking fix")
    summary = _jsonable(build_summary(rows, coverage))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    effect = summary["effect"]
    print(
        f"n={summary['n']} big={summary['n_big_chances']} "
        f"effect={effect['estimate']} ci95={effect['ci95']} "
        f"excludes_zero={effect['excludes_zero']} verdict={summary['verdict']}",
        flush=True,
    )
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
