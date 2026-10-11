"""Hypothesis 1: nearest-defender body shape at a pass reception.

At each pass reception, label the nearest outfield opponent's shoulder heading
relative to the vector from that defender to the receiver, then relate the
label to SkillCorner xshot, xthreat, and possession/pass EPV.

There is no xG, xGOT, or xT in this data. The three outcomes keep their
SkillCorner names:

- xshot: ``xshot_player_possession_start`` on the on-ball engagement whose
  window contains the reception. It is the expected-shot value of that player
  possession, not a shot-quality model.
- xthreat: ``xthreat`` on the passing option that was received. It is the
  expected threat of that option, not a generic possession-value surface.
- possession EPV: ``possession_epv_for_start`` on the reception possession
  (attacking-team expected possession value at the reception).
- pass EPV: ``pass_epv_delta_for`` on the passer possession that delivered the
  ball (attacking-team EPV change of that pass), plus
  ``pass_reception_epv_for`` on the receiver possession.

Pose is 25 fps and tracking is 10 fps, so the pose frame is 2.5 times the
tracking frame. Shoulders use x, y only. One label per reception, not per frame.

The pre-specified effect is the OLS slope of ``possession_epv_for_start`` on an
orientation score (0 square/in front, 1 side-on, 2 back-to). ``supported`` is
true only when that slope's 95% bootstrap interval excludes zero. Two matches
have body pose, so the comparison is underpowered either way.
"""

from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
OPENDATA = WORKSPACE / "opendata"
OUT_PATH = WORKSPACE / "analysis" / "output" / "h1_summary.json"

MATCH_IDS = (1925299, 1996435)

# Heading is absolute angular distance in [0, 180].
# Front and back are +/-45 degrees; side-on is the remaining 180 degrees of the circle.
SQUARE_MAX_DEG = 45.0
BACK_MIN_DEG = 135.0
# Two pose frames that map to one odd tracking frame must agree within this.
HEADING_AGREE_DEG = 45.0
# Bearings from a defender standing on the receiver are undefined.
MIN_DEFENDER_DISTANCE_M = 0.5
# Received passing option must end within this many tracking frames (10 fps).
MAX_RECEPTION_GAP_FRAMES = 30
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 1925299

SQUARE = "square_in_front"
SIDE = "side_on"
BACK = "back_to"
LABELS = (SQUARE, SIDE, BACK)
SCORE = {SQUARE: 0, SIDE: 1, BACK: 2}

# Random heading lands in these shares: front 90/360, side 180/360, back 90/360.
NULL_SHARE = {SQUARE: 0.25, SIDE: 0.50, BACK: 0.25}

PRIMARY_OUTCOME = "possession_epv_for_start"
PRIMARY_EFFECT = "orientation_score_slope"

OUTCOMES = (
    (
        "possession_epv_for_start",
        "Attacking-team possession EPV at the reception "
        "(player_possession.possession_epv_for_start). Not xG.",
    ),
    (
        "pass_reception_epv_for",
        "Pass-reception EPV for the attacking team on the receiver possession "
        "(pass_reception_epv_for). Not xT.",
    ),
    (
        "pass_epv_delta_for",
        "Attacking-team EPV change of the pass that was received "
        "(pass_epv_delta_for on the passer possession). Not xT.",
    ),
    (
        "xthreat",
        "SkillCorner xthreat of the received passing option. "
        "Expected threat of that option, not xT.",
    ),
    (
        "xshot_player_possession_start",
        "SkillCorner xshot_player_possession_start on an on-ball engagement "
        "whose frame window contains the reception. Expected-shot value of the "
        "player possession, not xG and not xGOT. Missing when no engagement covers the reception.",
    ),
)

FRAME_RE = re.compile(br'"frame":\s*(\d+)')


def _import_pose_helpers():
    sys.path.insert(0, str(OPENDATA))
    from src.features.pose_orientation import (  # noqa: WPS433
        DEFAULT_MAX_ERR_CM,
        LEFT_SHOULDER,
        PLAUSIBLE_WIDTH_M,
        RIGHT_SHOULDER,
        shoulder_orientation_deg,
    )

    return (
        shoulder_orientation_deg,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        PLAUSIBLE_WIDTH_M,
        DEFAULT_MAX_ERR_CM,
    )


def angular_distance_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, in [0, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def circular_mean_deg(angles: list[float]) -> float | None:
    if not angles:
        return None
    rad = np.deg2rad(np.asarray(angles, dtype=float))
    s = float(np.sin(rad).mean())
    c = float(np.cos(rad).mean())
    if math.hypot(s, c) < 1e-8:
        return None
    return math.degrees(math.atan2(s, c))


def label_from_delta(delta_deg: float) -> str:
    if delta_deg < SQUARE_MAX_DEG:
        return SQUARE
    if delta_deg < BACK_MIN_DEG:
        return SIDE
    return BACK


def classify_heading(shoulder_deg: float, dx: float, dy: float) -> tuple[str, float, float]:
    """Label shoulder heading against the defender-to-receiver vector (dx, dy)."""
    bearing = math.degrees(math.atan2(dy, dx))
    delta = angular_distance_deg(shoulder_deg, bearing)
    return label_from_delta(delta), delta, bearing


def pose_frames_for_tracking(tracking_frame: int) -> list[int]:
    """Pose frame is 2.5 times the tracking frame. Odd frames sit between two pose frames."""
    target = 2.5 * tracking_frame
    lo = math.floor(target)
    hi = math.ceil(target)
    if lo == hi:
        return [int(lo)]
    return [int(lo), int(hi)]


def _self_check() -> None:
    shoulder_orientation_deg, *_ = _import_pose_helpers()
    # Left shoulder at +y, right at -y: left-to-right is -y, +90 CCW faces +x.
    facing_pos_x = shoulder_orientation_deg(0.0, 0.2, 0.0, -0.2)
    assert facing_pos_x is not None and abs(facing_pos_x) < 1e-6
    label, delta, _ = classify_heading(facing_pos_x, 1.0, 0.0)
    assert label == SQUARE and delta < 1e-6
    label, delta, _ = classify_heading(facing_pos_x, -1.0, 0.0)
    assert label == BACK and abs(delta - 180.0) < 1e-6
    label, delta, _ = classify_heading(facing_pos_x, 0.0, 1.0)
    assert label == SIDE and abs(delta - 90.0) < 1e-6
    assert label_from_delta(44.999) == SQUARE
    assert label_from_delta(45.0) == SIDE
    assert label_from_delta(135.0) == BACK
    assert abs(angular_distance_deg(170.0, -170.0) - 20.0) < 1e-6
    mean = circular_mean_deg([170.0, -170.0])
    assert mean is not None and abs(angular_distance_deg(mean, 180.0)) < 1e-6


def find_pose_zip(match_id: int) -> Path:
    roots = (
        OPENDATA / "data" / "bodypose",
        WORKSPACE / "data" / "bodypose",
        Path("/tmp/bodypose"),
    )
    for root in roots:
        path = root / f"{match_id}.jsonl.zip"
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    raise FileNotFoundError(
        f"No body-pose zip for match {match_id}. Download "
        f"raw/{match_id}.jsonl.zip from SkillCorner/opendata-bodypose."
    )


def load_match_context(match_id: int) -> dict:
    path = OPENDATA / "data" / "matches" / str(match_id) / f"{match_id}_match.json"
    match = json.loads(path.read_text())
    team_of: dict[int, int] = {}
    gk_ids: set[int] = set()
    for player in match["players"]:
        pid = int(player["id"])
        team_of[pid] = int(player["team_id"])
        role = player.get("player_role") or {}
        if role.get("name") == "Goalkeeper":
            gk_ids.add(pid)
    return {
        "match_id": match_id,
        "home": match["home_team"]["name"],
        "away": match["away_team"]["name"],
        "team_of": team_of,
        "gk_ids": gk_ids,
    }


def load_events(match_id: int) -> pd.DataFrame:
    path = OPENDATA / "data" / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    usecols = [
        "match_id",
        "event_id",
        "event_type",
        "start_type",
        "frame_start",
        "frame_end",
        "period",
        "player_id",
        "team_id",
        "x_start",
        "y_start",
        "received",
        "associated_player_possession_event_id",
        "xshot_player_possession_start",
        "xthreat",
        "possession_epv_for_start",
        "pass_reception_epv_for",
        "pass_epv_delta_for",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["match_id"] = int(match_id)
    return df


def reception_table(events: pd.DataFrame) -> pd.DataFrame:
    rec = events[
        (events["event_type"] == "player_possession")
        & (events["start_type"] == "pass_reception")
    ].copy()
    rec["frame_start"] = rec["frame_start"].astype(int)
    rec["player_id"] = rec["player_id"].astype(int)
    rec["team_id"] = rec["team_id"].astype(int)
    return rec


def attach_xshot(receptions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """xshot lives only on on-ball engagements, not on the reception row."""
    eng = events.loc[
        events["event_type"] == "on_ball_engagement",
        [
            "match_id",
            "associated_player_possession_event_id",
            "frame_start",
            "frame_end",
            "xshot_player_possession_start",
        ],
    ].rename(
        columns={
            "associated_player_possession_event_id": "event_id",
            "frame_start": "eng_frame_start",
            "frame_end": "eng_frame_end",
        }
    )
    eng = eng.dropna(subset=["event_id", "xshot_player_possession_start"])
    base = receptions.drop(columns=["xshot_player_possession_start"], errors="ignore")
    merged = base.merge(eng, on=["match_id", "event_id"], how="left")
    covers = (merged["eng_frame_start"] <= merged["frame_start"]) & (
        merged["eng_frame_end"] >= merged["frame_start"]
    )
    covered = merged.loc[covers].copy()
    covered["xshot_lead_frames"] = covered["frame_start"] - covered["eng_frame_start"]
    # Engagements of one possession agree on xshot; average is a no-op when they do.
    agg = covered.groupby(["match_id", "event_id"], as_index=False).agg(
        xshot_player_possession_start=("xshot_player_possession_start", "mean"),
        xshot_lead_frames=("xshot_lead_frames", "median"),
        n_covering_engagements=("xshot_player_possession_start", "size"),
    )
    return base.merge(agg, on=["match_id", "event_id"], how="left")


def attach_received_pass(receptions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """xthreat is on the received passing option; pass EPV is on the passer possession."""
    options = events.loc[
        (events["event_type"] == "passing_option") & (events["received"] == True),  # noqa: E712
        [
            "match_id",
            "player_id",
            "team_id",
            "frame_end",
            "xthreat",
            "associated_player_possession_event_id",
        ],
    ].rename(
        columns={
            "player_id": "receiver_id",
            "team_id": "receiver_team_id",
            "frame_end": "option_frame_end",
            "associated_player_possession_event_id": "passer_possession_id",
        }
    )
    options = options.dropna(subset=["receiver_id", "xthreat", "passer_possession_id"])
    options["receiver_id"] = options["receiver_id"].astype(int)
    options["receiver_team_id"] = options["receiver_team_id"].astype(int)
    options["option_frame_end"] = options["option_frame_end"].astype(int)

    base_for_link = receptions.drop(columns=["xthreat"], errors="ignore")
    merged = base_for_link.merge(
        options,
        left_on=["match_id", "player_id", "team_id"],
        right_on=["match_id", "receiver_id", "receiver_team_id"],
        how="left",
    )
    gap = merged["frame_start"] - merged["option_frame_end"]
    ok = gap.between(0, MAX_RECEPTION_GAP_FRAMES)
    linked = merged.loc[ok].copy()
    linked["reception_gap_frames"] = gap.loc[ok]
    linked = linked.sort_values("reception_gap_frames").drop_duplicates(
        ["match_id", "event_id"], keep="first"
    )

    passer = events.loc[
        events["event_type"] == "player_possession",
        ["match_id", "event_id", "pass_epv_delta_for"],
    ].rename(
        columns={
            "event_id": "passer_possession_id",
            "pass_epv_delta_for": "pass_epv_delta_for",
        }
    )
    linked = linked.drop(columns=["pass_epv_delta_for"], errors="ignore").merge(
        passer, on=["match_id", "passer_possession_id"], how="left"
    )
    keep = [
        "match_id",
        "event_id",
        "xthreat",
        "pass_epv_delta_for",
        "reception_gap_frames",
    ]
    base_cols = [c for c in receptions.columns if c not in ("xthreat", "pass_epv_delta_for")]
    base = receptions[base_cols]
    return base.merge(linked[keep], on=["match_id", "event_id"], how="left")


def load_tracking_frames(match_id: int, frames: set[int]) -> dict[int, list[tuple[int, float, float, bool]]]:
    path = (
        OPENDATA
        / "data"
        / "matches"
        / str(match_id)
        / f"{match_id}_tracking_extrapolated.jsonl"
    )
    found: dict[int, list[tuple[int, float, float, bool]]] = {}
    with path.open("rb") as fh:
        for line in fh:
            m = FRAME_RE.search(line, 0, 48)
            if m is None or int(m.group(1)) not in frames:
                continue
            rec = json.loads(line)
            rows = []
            for player in rec.get("player_data") or []:
                x, y = player.get("x"), player.get("y")
                if x is None or y is None:
                    continue
                rows.append(
                    (
                        int(player["player_id"]),
                        float(x),
                        float(y),
                        bool(player.get("is_detected")),
                    )
                )
            found[int(rec["frame"])] = rows
    return found


def nearest_outfield_opponent(
    players: list[tuple[int, float, float, bool]],
    receiver_id: int,
    receiver_team: int,
    team_of: dict[int, int],
    gk_ids: set[int],
    fallback_xy: tuple[float, float],
) -> dict | None:
    receiver = next((p for p in players if p[0] == receiver_id), None)
    if receiver is None:
        rx, ry = fallback_xy
        receiver_detected = False
        if not math.isfinite(rx) or not math.isfinite(ry):
            return None
    else:
        _, rx, ry, receiver_detected = receiver
    best = None
    for pid, x, y, detected in players:
        if pid == receiver_id or pid in gk_ids:
            continue
        team = team_of.get(pid)
        if team is None or team == receiver_team:
            continue
        dist = math.hypot(x - rx, y - ry)
        if not math.isfinite(dist):
            continue
        if best is None or dist < best["distance_m"] or (
            dist == best["distance_m"] and pid < best["defender_id"]
        ):
            best = {
                "defender_id": pid,
                "defender_x": x,
                "defender_y": y,
                "defender_detected": detected,
                "distance_m": dist,
                "receiver_x": rx,
                "receiver_y": ry,
                "receiver_detected": receiver_detected,
            }
    if best is None or best["distance_m"] < MIN_DEFENDER_DISTANCE_M:
        return None
    return best


def reliable_heading(player: dict, helpers) -> float | None:
    shoulder_orientation_deg, left_name, right_name, width_ok, max_err = helpers
    joints = player.get("joints")
    if not joints:
        return None
    left, right = joints.get(left_name), joints.get(right_name)
    if not left or not right:
        return None
    lx, ly = left["xyz"][0], left["xyz"][1]
    rx, ry = right["xyz"][0], right["xyz"][1]
    angle = shoulder_orientation_deg(float(lx), float(ly), float(rx), float(ry))
    if angle is None:
        return None
    width = math.hypot(float(rx) - float(lx), float(ry) - float(ly))
    if not (width_ok[0] <= width <= width_ok[1]):
        return None
    err = max(left.get("p90_mae_cm") or math.inf, right.get("p90_mae_cm") or math.inf)
    if err > max_err:
        return None
    return float(angle)


def load_headings(
    match_id: int,
    needed: dict[int, set[int]],
    helpers,
) -> dict[tuple[int, int], float]:
    """Stream the pose zip and keep reliable shoulder headings for needed players."""
    archive = find_pose_zip(match_id)
    wanted_frames = set(needed)
    out: dict[tuple[int, int], float] = {}
    member = f"{match_id}.jsonl"
    seen = 0
    with zipfile.ZipFile(archive) as zf, zf.open(member) as fh:
        for line in fh:
            seen += 1
            if seen % 25000 == 0:
                print(f"  match {match_id}: scanned {seen:,} pose frames", flush=True)
            m = FRAME_RE.search(line, 0, 48)
            if m is None:
                continue
            frame = int(m.group(1))
            if frame not in wanted_frames:
                continue
            rec = json.loads(line)
            want_players = needed[frame]
            for player in rec.get("player_data") or []:
                pid = int(player["player_id"])
                if pid not in want_players:
                    continue
                heading = reliable_heading(player, helpers)
                if heading is not None:
                    out[(pid, frame)] = heading
    print(f"  match {match_id}: kept {len(out):,} reliable headings", flush=True)
    return out


def heading_at_reception(
    defender_id: int,
    tracking_frame: int,
    headings: dict[tuple[int, int], float],
) -> tuple[float, int] | None:
    frames = pose_frames_for_tracking(tracking_frame)
    angles = [headings[(defender_id, fr)] for fr in frames if (defender_id, fr) in headings]
    if not angles:
        return None
    if len(frames) == 2 and len(angles) == 2:
        if angular_distance_deg(angles[0], angles[1]) > HEADING_AGREE_DEG:
            return None
    if len(frames) == 2 and len(angles) == 1:
        # One of the two bracketing pose frames failed the shoulder gate.
        return None
    mean = circular_mean_deg(angles)
    if mean is None:
        return None
    return mean, len(angles)


def label_match(match_id: int, receptions: pd.DataFrame, helpers) -> tuple[pd.DataFrame, dict]:
    ctx = load_match_context(match_id)
    rec = receptions.loc[receptions["match_id"] == match_id].copy()
    frames = set(rec["frame_start"].tolist())
    print(f"match {match_id}: loading {len(frames)} tracking frames", flush=True)
    tracking = load_tracking_frames(match_id, frames)
    funnel = {
        "pass_receptions": int(len(rec)),
        "tracking_frame_found": 0,
        "nearest_outfield_opponent": 0,
        "reliable_shoulder": 0,
        "labelled": 0,
    }
    pending = []
    needed: dict[int, set[int]] = {}
    for row in rec.itertuples(index=False):
        players = tracking.get(int(row.frame_start))
        if not players:
            continue
        funnel["tracking_frame_found"] += 1
        nearest = nearest_outfield_opponent(
            players,
            int(row.player_id),
            int(row.team_id),
            ctx["team_of"],
            ctx["gk_ids"],
            (float(row.x_start), float(row.y_start)),
        )
        if nearest is None:
            continue
        funnel["nearest_outfield_opponent"] += 1
        pose_frames = pose_frames_for_tracking(int(row.frame_start))
        for fr in pose_frames:
            needed.setdefault(fr, set()).add(nearest["defender_id"])
        pending.append((row, nearest, pose_frames))

    print(f"match {match_id}: scanning pose for {len(needed)} frames", flush=True)
    headings = load_headings(match_id, needed, helpers)
    labelled = []
    for row, nearest, pose_frames in pending:
        got = heading_at_reception(nearest["defender_id"], int(row.frame_start), headings)
        if got is None:
            continue
        funnel["reliable_shoulder"] += 1
        shoulder, n_pose = got
        dx = nearest["receiver_x"] - nearest["defender_x"]
        dy = nearest["receiver_y"] - nearest["defender_y"]
        label, delta, bearing = classify_heading(shoulder, dx, dy)
        funnel["labelled"] += 1
        labelled.append(
            {
                "match_id": match_id,
                "event_id": row.event_id,
                "frame_start": int(row.frame_start),
                "period": None if pd.isna(row.period) else int(row.period),
                "receiver_id": int(row.player_id),
                "defender_id": nearest["defender_id"],
                "defender_distance_m": nearest["distance_m"],
                "receiver_detected": nearest["receiver_detected"],
                "defender_detected": nearest["defender_detected"],
                "shoulder_deg": shoulder,
                "bearing_deg": bearing,
                "delta_deg": delta,
                "label": label,
                "orientation_score": SCORE[label],
                "n_pose_frames": n_pose,
                "possession_epv_for_start": _finite(row.possession_epv_for_start),
                "pass_reception_epv_for": _finite(row.pass_reception_epv_for),
                "pass_epv_delta_for": _finite(row.pass_epv_delta_for),
                "xthreat": _finite(row.xthreat),
                "xshot_player_possession_start": _finite(row.xshot_player_possession_start),
                "xshot_lead_frames": _finite(row.xshot_lead_frames),
                "reception_gap_frames": _finite(row.reception_gap_frames),
            }
        )
    print(f"match {match_id} funnel {funnel}", flush=True)
    return pd.DataFrame(labelled), funnel


def _finite(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return float(value)


def ols_slope(score: np.ndarray, y: np.ndarray) -> float:
    x = score - score.mean()
    yy = y - y.mean()
    den = float(np.dot(x, x))
    if den <= 0.0 or len(y) < 3:
        return float("nan")
    return float(np.dot(x, yy) / den)


def bootstrap_slope(score: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    n = int(len(y))
    estimate = ols_slope(score, y)
    if n < 3 or not math.isfinite(estimate):
        return {
            "estimate": None,
            "ci95": [None, None],
            "excludes_zero": False,
            "n": n,
            "n_bootstrap": 0,
        }
    idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    s = score[idx]
    yy = y[idx]
    s = s - s.mean(axis=1, keepdims=True)
    yy = yy - yy.mean(axis=1, keepdims=True)
    den = np.sum(s * s, axis=1)
    slopes = np.full(BOOTSTRAP_DRAWS, np.nan)
    ok = den > 0
    slopes[ok] = np.sum(s[ok] * yy[ok], axis=1) / den[ok]
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) < 100:
        return {
            "estimate": _round(estimate),
            "ci95": [None, None],
            "excludes_zero": False,
            "n": n,
            "n_bootstrap": int(len(slopes)),
        }
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    lo_r, hi_r = _round(float(lo)), _round(float(hi))
    est_r = _round(estimate)
    return {
        "estimate": est_r,
        "ci95": [lo_r, hi_r],
        "excludes_zero": bool(lo_r > 0.0 or hi_r < 0.0),
        "n": n,
        "n_bootstrap": int(len(slopes)),
    }


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 8)


def group_stats(df: pd.DataFrame, outcome: str) -> dict:
    stats = {}
    for label in LABELS:
        values = df.loc[df["label"] == label, outcome].dropna().to_numpy(dtype=float)
        stats[label] = {
            "n": int(len(values)),
            "mean": _round(float(values.mean())) if len(values) else None,
            "std": _round(float(values.std(ddof=1))) if len(values) > 1 else None,
        }
    return stats


def mean_contrast(
    df: pd.DataFrame,
    outcome: str,
    label_a: str,
    label_b: str,
    rng: np.random.Generator,
) -> dict:
    """Mean(outcome | label_b) minus mean(outcome | label_a), percentile bootstrap."""
    a = df.loc[df["label"] == label_a, outcome].dropna().to_numpy(dtype=float)
    b = df.loc[df["label"] == label_b, outcome].dropna().to_numpy(dtype=float)
    name = f"{label_b} - {label_a}"
    counts = {f"n_{label_a}": int(len(a)), f"n_{label_b}": int(len(b))}
    if len(a) < 2 or len(b) < 2:
        return {
            "contrast": name,
            "estimate": None,
            "ci95": [None, None],
            "excludes_zero": False,
            **counts,
        }
    estimate = float(b.mean() - a.mean())
    ia = rng.integers(0, len(a), size=(BOOTSTRAP_DRAWS, len(a)))
    ib = rng.integers(0, len(b), size=(BOOTSTRAP_DRAWS, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    lo_r, hi_r = _round(float(lo)), _round(float(hi))
    return {
        "contrast": name,
        "estimate": _round(estimate),
        "ci95": [lo_r, hi_r],
        "excludes_zero": bool(lo_r > 0.0 or hi_r < 0.0),
        **counts,
    }


def outcome_block(df: pd.DataFrame, outcome: str, description: str, rng: np.random.Generator) -> dict:
    sub = df.loc[df[outcome].notna(), ["match_id", "label", "orientation_score", outcome]].copy()
    score = sub["orientation_score"].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    slope = bootstrap_slope(score, y, rng)
    by_match = []
    for match_id, part in sub.groupby("match_id"):
        by_match.append(
            {
                "match_id": int(match_id),
                "n": int(len(part)),
                "slope": _round(
                    ols_slope(
                        part["orientation_score"].to_numpy(dtype=float),
                        part[outcome].to_numpy(dtype=float),
                    )
                ),
            }
        )
    return {
        "outcome": outcome,
        "description": description,
        "n": int(len(sub)),
        "n_by_label": {label: int((sub["label"] == label).sum()) for label in LABELS},
        "mean_by_label": group_stats(sub, outcome),
        "orientation_score_slope": slope,
        "supported": bool(slope["excludes_zero"]),
        "side_on_minus_square_in_front": mean_contrast(sub, outcome, SQUARE, SIDE, rng),
        "back_to_minus_square_in_front": mean_contrast(sub, outcome, SQUARE, BACK, rng),
        "by_match": by_match,
    }


def _delta_quantiles(df: pd.DataFrame) -> dict:
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    vals = np.quantile(df["delta_deg"].to_numpy(dtype=float), qs)
    return {f"p{int(q * 100)}": _round(float(v)) for q, v in zip(qs, vals)}


def label_summary(df: pd.DataFrame) -> dict:
    n = int(len(df))
    counts = {label: int((df["label"] == label).sum()) for label in LABELS}
    return {
        "n": n,
        "counts": counts,
        "share": {label: _round(counts[label] / n) if n else None for label in LABELS},
        "null_share_if_heading_random": NULL_SHARE,
        "mean_delta_deg": {
            label: _round(float(df.loc[df["label"] == label, "delta_deg"].mean()))
            if counts[label]
            else None
            for label in LABELS
        },
        "median_defender_distance_m": _round(float(df["defender_distance_m"].median())) if n else None,
        "share_receiver_detected": _round(float(df["receiver_detected"].mean())) if n else None,
        "share_defender_detected": _round(float(df["defender_detected"].mean())) if n else None,
        "delta_deg_quantiles": _delta_quantiles(df) if n else None,
    }


def build_summary(df: pd.DataFrame, funnels: dict, fixtures: dict) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = [outcome_block(df, name, desc, rng) for name, desc in OUTCOMES]
    primary = next(b for b in blocks if b["outcome"] == PRIMARY_OUTCOME)
    slope = primary["orientation_score_slope"]
    supported = bool(slope["excludes_zero"])
    labels = label_summary(df)
    n = labels["n"]
    share_txt = ", ".join(
        f"{label} {labels['counts'][label]} ({100 * labels['share'][label]:.1f}%)"
        for label in LABELS
    )
    est, (lo, hi) = slope["estimate"], slope["ci95"]
    if supported:
        interval_txt = f"The 95% interval [{lo}, {hi}] excludes zero, so supported is true"
    else:
        interval_txt = f"The 95% interval [{lo}, {hi}] includes zero, so supported is false"
    means = primary["mean_by_label"]
    back_n = primary["n_by_label"][BACK]
    side_contrast = primary["side_on_minus_square_in_front"]
    back_contrast = primary["back_to_minus_square_in_front"]
    back_zero = "excludes zero" if back_contrast["excludes_zero"] else "includes zero"
    other_supported = [
        b["outcome"] for b in blocks if b["outcome"] != PRIMARY_OUTCOME and b["supported"]
    ]
    verdict = (
        f"Across {len(MATCH_IDS)} matches and {n} labelled pass receptions, the nearest "
        f"outfield defender was {share_txt}. A random heading would be about 25% square/in front, "
        f"50% side-on, and 25% back-to, so these defenders usually face the receiver. "
        f"The pre-specified effect is the slope of {PRIMARY_OUTCOME} on orientation score "
        f"(0 {SQUARE}, 1 {SIDE}, 2 {BACK}): estimate {est} EPV per step. {interval_txt}. "
        f"Group means are square/in front {means[SQUARE]['mean']} (n={means[SQUARE]['n']}), "
        f"side-on {means[SIDE]['mean']} (n={means[SIDE]['n']}), "
        f"back-to {means[BACK]['mean']} (n={means[BACK]['n']}). "
        f"Side-on minus square/in front is {side_contrast['estimate']} "
        f"with interval {side_contrast['ci95']}. "
        f"Back-to has n={back_n}, and back-to minus square/in front is {back_contrast['estimate']} "
        f"with interval {back_contrast['ci95']}, which {back_zero}. "
        f"The slope is the square/in front versus side-on association. "
        f"Other outcomes whose slope interval excludes zero: {', '.join(other_supported) or 'none'}. "
        f"With only {len(MATCH_IDS)} matches this comparison is underpowered; receptions "
        f"are nested in matches, and the interval does not correct for that."
    )
    return {
        "hypothesis_id": "h1",
        "question": (
            "At a pass reception, is the nearest defender side-on, square-on/in front, "
            "or back-to the receiver, and do those labels differ on xshot, xthreat, "
            "and possession/pass EPV?"
        ),
        "supported": supported,
        "underpowered": True,
        "n_matches": len(MATCH_IDS),
        "n": n,
        "effect": est,
        "effect_name": PRIMARY_EFFECT,
        "effect_outcome": PRIMARY_OUTCOME,
        "effect_unit": "change in possession_epv_for_start per step from square_in_front toward back_to",
        "interval": [lo, hi],
        "interval_level": 0.95,
        "interval_method": (
            f"percentile bootstrap of the OLS slope, {BOOTSTRAP_DRAWS} resamples of receptions, "
            f"seed {BOOTSTRAP_SEED}. Receptions are treated as independent."
        ),
        "interval_excludes_zero": bool(slope["excludes_zero"]),
        "verdict": verdict,
        "matches": [
            {"match_id": mid, "fixture": fixtures[mid], "funnel": funnels[mid]} for mid in MATCH_IDS
        ],
        "unit": (
            "One pass reception: a player_possession with start_type pass_reception. "
            "Not a pose frame."
        ),
        "label_definition": {
            "vector": "pitch x,y from the nearest outfield opponent to the receiver at the reception tracking frame",
            "heading": (
                "shoulder facing from pose x,y only: left-to-right shoulder axis rotated +90 degrees. "
                "Goalkeepers are excluded. z is not used."
            ),
            "square_in_front": f"absolute heading error in [0, {SQUARE_MAX_DEG})",
            "side_on": f"absolute heading error in [{SQUARE_MAX_DEG}, {BACK_MIN_DEG})",
            "back_to": f"absolute heading error in [{BACK_MIN_DEG}, 180]",
            "pose_frame": "2.5 * tracking_frame. Even tracking frames use that pose frame. Odd frames use the circular mean of the two bracketing pose frames when both shoulders are reliable and agree within 45 degrees.",
            "orientation_score": {SQUARE: 0, SIDE: 1, BACK: 2},
            "null_share_if_heading_random": NULL_SHARE,
        },
        "labels": labels,
        "primary_test": {
            "outcome": PRIMARY_OUTCOME,
            "estimate": est,
            "interval": [lo, hi],
            "excludes_zero": bool(slope["excludes_zero"]),
            "supported": supported,
            "n": primary["n"],
            "n_by_label": primary["n_by_label"],
            "why_this_outcome": (
                "possession_epv_for_start is on the reception event itself. "
                "xshot is published only on on-ball engagements and is missing when "
                "no engagement covers the reception, so it is a selected subset."
            ),
        },
        "outcomes": blocks,
        "metric_names": {
            "xshot": "xshot_player_possession_start. Not xG. Not xGOT.",
            "xthreat": "xthreat on the received passing option. Not xT.",
            "possession_epv": "possession_epv_for_start. Attacking-team EPV at the reception.",
            "pass_epv": "pass_epv_delta_for of the received pass, and pass_reception_epv_for on the receiver.",
        },
        "caveats": [
            "Body pose exists for two matches only (1925299 Brisbane Roar v Perth Glory, 1996435 Sydney FC v Adelaide United). The comparison is underpowered.",
            "The bootstrap treats receptions as independent. Players and phases repeat inside a match, so the interval is narrower than a match-level interval.",
            "Back-to has only a handful of events. The orientation-score slope is identified by square/in front versus side-on, not by back-to. The association is unadjusted for reception location, pressure, or phase.",
            "Shoulder heading and tracking positions are generated separately and can be slightly misaligned.",
            "xshot is observed only when an on-ball engagement window contains the reception frame.",
            "A random heading would be side-on about half the time because that bin covers 180 degrees of the circle.",
        ],
    }


def main() -> None:
    _self_check()
    helpers = _import_pose_helpers()
    frames = []
    fixtures = {}
    for match_id in MATCH_IDS:
        events = load_events(match_id)
        rec = reception_table(events)
        rec = attach_xshot(rec, events)
        rec = attach_received_pass(rec, events)
        frames.append(rec)
        ctx = load_match_context(match_id)
        fixtures[match_id] = f"{ctx['home']} vs {ctx['away']}"
    receptions = pd.concat(frames, ignore_index=True)
    labelled_frames = []
    funnels = {}
    for match_id in MATCH_IDS:
        labelled, funnel = label_match(match_id, receptions, helpers)
        labelled_frames.append(labelled)
        funnels[match_id] = funnel
    labelled = pd.concat(labelled_frames, ignore_index=True)
    if labelled.empty:
        raise RuntimeError("No receptions received a body-shape label.")
    # Bin means must fall inside the bins. A failure here is a coding error.
    for label, lo, hi in ((SQUARE, 0.0, SQUARE_MAX_DEG), (SIDE, SQUARE_MAX_DEG, BACK_MIN_DEG), (BACK, BACK_MIN_DEG, 180.01)):
        part = labelled.loc[labelled["label"] == label, "delta_deg"]
        if part.empty:
            continue
        if not (lo - 1e-6 <= float(part.min()) and float(part.max()) <= hi + 1e-6):
            raise RuntimeError(f"{label} delta out of range: {part.min()}..{part.max()}")
    summary = build_summary(labelled, funnels, fixtures)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary["verdict"], flush=True)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
