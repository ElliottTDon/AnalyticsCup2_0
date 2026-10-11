"""Hypothesis 3: high-block shadow screens of passing options.

Orientation is shoulder heading on matches 1925299 and 1996435 only.
The 20-match result is a tracking lane-occupation proxy, not body orientation.

Run: opendata/.venv/bin/python analysis/hypotheses/h3_shadow_screen.py
"""

from __future__ import annotations

import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "opendata"))

from common import (  # noqa: E402
    BODY_SQUARE_DEG,
    BOOT_SEED,
    LANE_DIST_M,
    LANE_T,
    OUT_DIR,
    cluster_diff,
    cluster_mean,
    heading_delta_deg,
    in_lane,
    json_ready,
    load_events,
    load_roster,
    match_ids,
    nearest_opponent,
    segment_params,
    stream_frames,
    supports,
)
from src.features.pose_orientation import (  # noqa: E402
    DEFAULT_MAX_ERR_CM,
    LEFT_SHOULDER,
    PLAUSIBLE_WIDTH_M,
    RIGHT_SHOULDER,
    fold_to_tracking_frame,
    shoulder_orientation_deg,
)

POSE_MATCHES = ("1925299", "1996435")
MANIFEST = ROOT / "opendata" / "data" / "bodypose" / "MANIFEST.json"
SUMMARY_PATH = OUT_DIR / "h3_summary.json"
MIN_SEGMENT_M = 1.0


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


def _frame_of(line: bytes) -> int | None:
    if not line.startswith(b'{"frame":'):
        return None
    i = 9
    n = len(line)
    while i < n and line[i] == 32:
        i += 1
    j = i
    while j < n and 48 <= line[j] <= 57:
        j += 1
    if j == i:
        return None
    return int(line[i:j])


def _expected_bytes(match_id: str) -> int | None:
    if not MANIFEST.is_file():
        return None
    meta = json.loads(MANIFEST.read_text())
    entry = meta.get("files", {}).get(f"raw/{match_id}.jsonl.zip")
    if not entry:
        return None
    return int(entry["size_bytes"])


def _zip_ok(path: Path, match_id: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            zf.getinfo(f"{match_id}.jsonl")
        return True
    except (zipfile.BadZipFile, KeyError, OSError):
        return False


def resolve_pose_zip(match_id: str) -> Path:
    """Use a finished zip if one is already on disk. Wait while another download grows."""
    name = f"{match_id}.jsonl.zip"
    paths = [ROOT / "pose_data" / name, ROOT / "opendata" / "pose_data" / name]
    expected = _expected_bytes(match_id)
    deadline = time.time() + 40 * 60
    last_size: dict[Path, int] = {}
    while True:
        existing = [p for p in paths if p.exists()]
        for path in existing:
            size = path.stat().st_size
            complete = expected is not None and size == expected and _zip_ok(path, match_id)
            stable = last_size.get(path) == size and _zip_ok(path, match_id)
            if complete or stable:
                return path
            last_size[path] = size
        if existing and time.time() < deadline:
            size = existing[0].stat().st_size
            print(f"waiting for {match_id} pose zip ({size} bytes)", flush=True)
            time.sleep(15)
            continue
        if not existing:
            from src.data.pose_loading import download_match

            dest = ROOT / "opendata" / "pose_data"
            dest.mkdir(parents=True, exist_ok=True)
            print(f"downloading pose {match_id}", flush=True)
            return download_match(int(match_id), dest_dir=dest)
        raise TimeoutError(f"pose zip for {match_id} did not become readable")


def _pose_frame_map(tracking_frames: set[int]) -> dict[int, int]:
    """Pose frames (25 fps) that fold onto the requested tracking frames (10 fps)."""
    out: dict[int, int] = {}
    for tf in tracking_frames:
        ideal = tf * 2.5
        lo = max(0, math.floor(ideal - 1.25))
        hi = math.ceil(ideal + 1.25)
        for pose_frame in range(lo, hi + 1):
            if fold_to_tracking_frame(pose_frame) == tf:
                out[pose_frame] = tf
    return out


def _shoulders(player: dict) -> tuple[float, bool] | None:
    joints = player.get("joints")
    if not joints:
        return None
    left, right = joints.get(LEFT_SHOULDER), joints.get(RIGHT_SHOULDER)
    if not left or not right:
        return None
    lx, ly, _ = left["xyz"]
    rx, ry, _ = right["xyz"]
    heading = shoulder_orientation_deg(lx, ly, rx, ry)
    if heading is None:
        return None
    width = math.hypot(rx - lx, ry - ly)
    err = max(left.get("p90_mae_cm") or math.inf, right.get("p90_mae_cm") or math.inf)
    reliable = PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1] and err <= DEFAULT_MAX_ERR_CM
    return float(heading), bool(reliable)


def stream_headings(zip_path: Path, tracking_frames: set[int]) -> dict[tuple[int, int], tuple[float, bool]]:
    """(tracking frame, player) -> (shoulder heading degrees, reliable).

    When several pose frames fold onto one tracking frame, keep the reliable
    heading closest to tracking_frame * 2.5. If none is reliable, keep the
    closest unreliable heading so coverage can be counted.
    """
    pose_to_track = _pose_frame_map(tracking_frames)
    if not pose_to_track:
        return {}
    max_pose = max(pose_to_track)
    ideal = {tf: tf * 2.5 for tf in set(pose_to_track.values())}
    best: dict[tuple[int, int], tuple[float, int, float, bool]] = {}
    member = f"{zip_path.name.replace('.jsonl.zip', '')}.jsonl"
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        for line in fh:
            pose_frame = _frame_of(line)
            if pose_frame is None:
                continue
            if pose_frame > max_pose:
                break
            tf = pose_to_track.get(pose_frame)
            if tf is None:
                continue
            dist = abs(pose_frame - ideal[tf])
            rec = json.loads(line)
            for player in rec.get("player_data") or []:
                parsed = _shoulders(player)
                if parsed is None:
                    continue
                heading, reliable = parsed
                key = (tf, int(player["player_id"]))
                prev = best.get(key)
                if prev is None:
                    best[key] = (dist, pose_frame, heading, reliable)
                    continue
                prev_dist, _, _, prev_ok = prev
                if reliable and not prev_ok:
                    best[key] = (dist, pose_frame, heading, True)
                elif reliable == prev_ok and dist < prev_dist:
                    best[key] = (dist, pose_frame, heading, reliable)
    return {key: (heading, ok) for key, (_, _, heading, ok) in best.items()}


def _closest_in_window(
    xy: dict[int, tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
    att_team: int,
    team_of: dict[int, int],
    gk: set[int],
    skip: set[int],
) -> tuple[int, float, float] | None:
    """Outfield opponent closest to segment ab with t inside the lane window."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if float(np.linalg.norm(bb - aa)) < MIN_SEGMENT_M:
        return None
    best_id = None
    best_dist = 1e9
    best_t = float("nan")
    for pid, pos in xy.items():
        if pid in skip or pid in gk:
            continue
        tm = team_of.get(pid)
        if tm is None or tm == att_team:
            continue
        t, dist = segment_params(np.asarray(pos, dtype=float), aa, bb)
        if LANE_T[0] <= t <= LANE_T[1] and dist < best_dist:
            best_dist = dist
            best_t = t
            best_id = pid
    if best_id is None:
        return None
    return best_id, float(best_dist), float(best_t)


def _facing_ball(heading: float, defender: tuple[float, float], ball: tuple[float, float]) -> float:
    """Degrees between the shoulder heading and the direction from defender to ball."""
    dx = ball[0] - defender[0]
    dy = ball[1] - defender[1]
    if math.hypot(dx, dy) < 0.5 or not np.isfinite(heading):
        return float("nan")
    target = math.degrees(math.atan2(dy, dx))
    return heading_delta_deg(heading, target)


def _success(outcome) -> float:
    if outcome == "successful":
        return 1.0
    if isinstance(outcome, str):
        return 0.0
    return float("nan")


def _pass_volume(events: pd.DataFrame) -> dict[int, int]:
    poss = events[events["event_type"] == "player_possession"]
    passes = poss[poss["end_type"] == "pass"]
    counts: dict[int, int] = {}
    for pid in passes["player_id"]:
        iid = _iid(pid)
        if iid is not None:
            counts[iid] = counts.get(iid, 0) + 1
    for pid in passes["player_targeted_id"]:
        iid = _iid(pid)
        if iid is not None:
            counts[iid] = counts.get(iid, 0) + 1
    return counts


def _xy_at(frames, frame, pid, fallback) -> tuple[float, float] | None:
    if pid is not None:
        pos = frames.get(int(frame), {}).get(int(pid))
        if pos is not None:
            return pos
    return fallback


def _needed_frames(options: pd.DataFrame, forwards: pd.DataFrame) -> set[int]:
    needed: set[int] = set()
    for fr in list(options["frame_start"]) + list(forwards["frame_end"]):
        if pd.notna(fr) and int(fr) >= 0:
            needed.add(int(fr))
    return needed


def process_match(match_id: str) -> list[dict]:
    events = load_events(match_id)
    team_of, _roster, gk, _plen, _pwid = load_roster(match_id)
    volume = _pass_volume(events)
    pose_match = match_id in POSE_MATCHES

    options = events[
        (events["event_type"] == "passing_option")
        & (events["team_out_of_possession_phase_type"] == "high_block")
    ]
    poss = events[events["event_type"] == "player_possession"]
    forwards = poss[
        (poss["end_type"] == "pass")
        & (poss["team_out_of_possession_phase_type"] == "high_block")
        & (poss["pass_ahead"] == 1)
    ]

    needed = _needed_frames(options, forwards)
    frames = stream_frames(match_id, needed)
    headings = _load_headings(match_id, needed) if pose_match else None

    rows: list[dict] = []

    def append(kind, frame, passer, teammate, a_fb, b_fb, received, ahead, xthreat, epv, successful, possession):
        if passer is None or teammate is None or passer == teammate:
            return
        if passer in gk or teammate in gk:
            return
        att = team_of.get(teammate)
        if att is None:
            return
        a = _xy_at(frames, frame, passer, a_fb)
        b = _xy_at(frames, frame, teammate, b_fb)
        xy = frames.get(int(frame)) or {}
        proxy = float("nan")
        shadow = float("nan")
        facing = float("nan")
        lane_dist = float("nan")
        screener = None
        if a is not None and b is not None:
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg >= MIN_SEGMENT_M:
                defs = []
                for pid, pos in xy.items():
                    if pid in gk or pid in (passer, teammate):
                        continue
                    tm = team_of.get(pid)
                    if tm is None or tm == att:
                        continue
                    defs.append(pos)
                proxy = 1.0 if in_lane(defs, a, b) else 0.0
                hit = _closest_in_window(xy, a, b, att, team_of, gk, {passer, teammate})
                if hit is None:
                    lane_dist = float("nan")
                    if pose_match:
                        shadow = 0.0
                else:
                    screener, lane_dist, _t = hit
                    if pose_match and proxy == 0.0:
                        shadow = 0.0
                    elif pose_match and proxy == 1.0:
                        info = None if headings is None else headings.get((int(frame), screener))
                        if info is None or not info[1] or screener not in xy:
                            shadow = float("nan")
                        else:
                            delta = _facing_ball(info[0], xy[screener], a)
                            if not np.isfinite(delta):
                                shadow = float("nan")
                            else:
                                facing = 1.0 if delta <= BODY_SQUARE_DEG else 0.0
                                shadow = facing
            elif pose_match:
                shadow = float("nan")
        _nearest_id, nearest_dist = nearest_opponent(teammate, xy, team_of, gk)
        del _nearest_id
        rows.append({
            "match_id": match_id,
            "kind": kind,
            "possession_id": possession,
            "player_id": teammate,
            "pass_volume": int(volume.get(teammate, 0)),
            "lane_proxy": proxy,
            "shadow_screen": shadow,
            "facing_lane": facing,
            "lane_dist_m": lane_dist,
            "nearest_to_teammate_m": nearest_dist,
            "received": received,
            "pass_ahead": ahead,
            "xthreat": xthreat,
            "pass_epv": epv,
            "successful": successful,
            "pose_match": 1.0 if pose_match else 0.0,
        })

    for row in options.itertuples(index=False):
        append(
            "option",
            int(row.frame_start),
            _iid(row.player_in_possession_id),
            _iid(row.player_id),
            _finite_xy(row.player_in_possession_x_start, row.player_in_possession_y_start),
            _finite_xy(row.x_start, row.y_start),
            row.received,
            row.pass_ahead,
            row.xthreat,
            float("nan"),
            float("nan"),
            row.associated_player_possession_event_id,
        )
    for row in forwards.itertuples(index=False):
        append(
            "forward_pass",
            int(row.frame_end),
            _iid(row.player_id),
            _iid(row.player_targeted_id),
            _finite_xy(row.x_end, row.y_end),
            _finite_xy(row.player_targeted_x_reception, row.player_targeted_y_reception),
            _success(row.pass_outcome),
            row.pass_ahead,
            row.player_targeted_xthreat,
            row.pass_epv_total,
            _success(row.pass_outcome),
            row.event_id,
        )
    return rows


def _usable(contrast: dict) -> dict:
    """A one-match bootstrap interval has zero width. Do not treat it as evidence."""
    out = dict(contrast)
    both = int(out.get("n_matches_both") or 0)
    if both < 2:
        out["ci_low"] = None
        out["ci_high"] = None
        out["ci_excludes_zero"] = False
        out["supported"] = False
        out["interval_note"] = "fewer than two matches contain both arms"
    return out


def _record(name: str, predicted: str, contrast: dict, unit: str) -> dict:
    base = {
        "name": name,
        "predicted": predicted,
        "unit": unit,
        "supported": supports(contrast, predicted),
        **contrast,
    }
    return _usable(base)


def _paired_xthreat(df: pd.DataFrame, treat: str, seed: int) -> dict:
    paired = []
    work = df.dropna(subset=["possession_id"]).copy()
    work["xthreat"] = pd.to_numeric(work["xthreat"], errors="coerce")
    work[treat] = pd.to_numeric(work[treat], errors="coerce")
    for (mid, pid), g in work.groupby(["match_id", "possession_id"], sort=False):
        sc = g.loc[g[treat] == 1, "xthreat"].dropna()
        op = g.loc[g[treat] == 0, "xthreat"].dropna()
        if len(sc) and len(op):
            paired.append({"match_id": mid, "possession_id": pid, "xt_diff": float(sc.mean() - op.mean())})
    paired_df = pd.DataFrame(paired) if paired else pd.DataFrame(columns=["match_id", "xt_diff"])
    stat = cluster_mean(paired_df, "xt_diff", seed=seed) if len(paired_df) else {
        "effect": None, "ci_low": None, "ci_high": None, "n": 0, "n_matches": 0, "ci_excludes_zero": False,
    }
    if stat.get("n_matches", 0) < 2:
        stat = {**stat, "ci_low": None, "ci_high": None, "ci_excludes_zero": False}
    supported = bool(
        stat.get("ci_excludes_zero")
        and stat.get("effect") is not None
        and stat["effect"] > 0
        and stat.get("ci_low") is not None
        and stat["ci_low"] > 0
        and stat.get("n", 0) >= 20
        and stat.get("n_matches", 0) >= 2
    )
    return {
        "name": "within possession, mean xthreat of screened options minus open options",
        "predicted": "positive",
        "unit": "xthreat",
        "supported": supported,
        "effect": stat.get("effect"),
        "ci_low": stat.get("ci_low"),
        "ci_high": stat.get("ci_high"),
        "mean_treat": None,
        "mean_control": None,
        "n_treat": int(stat.get("n") or 0),
        "n_control": int(stat.get("n") or 0),
        "n_matches": int(stat.get("n_matches") or 0),
        "n_matches_both": int(stat.get("n_matches") or 0),
        "ci_excludes_zero": bool(stat.get("ci_excludes_zero")),
    }


def _volume_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if the teammate's match pass volume is above the within-match player median."""
    out = df.copy()
    out["high_volume"] = np.nan
    for mid, g in out.groupby("match_id", sort=False):
        vol = g.groupby("player_id")["pass_volume"].first()
        if vol.nunique() < 2:
            continue
        med = float(vol.median())
        high = set(vol[vol > med].index)
        low = set(vol[vol < med].index)
        flag = np.full(len(g), np.nan)
        pids = g["player_id"].to_numpy()
        for i, pid in enumerate(pids):
            if pid in high:
                flag[i] = 1.0
            elif pid in low:
                flag[i] = 0.0
        out.loc[g.index, "high_volume"] = flag
    return out


def _rate(df: pd.DataFrame, col: str, seed: int) -> dict:
    stat = cluster_mean(df, col, seed=seed)
    if stat.get("n_matches", 0) < 2:
        stat = {**stat, "ci_low": None, "ci_high": None, "ci_excludes_zero": False}
    return stat


def _arm_by_match(df: pd.DataFrame, y: str, treat: str) -> list[dict]:
    rows = []
    work = df[["match_id", y, treat]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work[treat] = pd.to_numeric(work[treat], errors="coerce")
    work = work[np.isfinite(work[y]) & np.isfinite(work[treat])]
    for mid, g in work.groupby("match_id", sort=False):
        yt = g.loc[g[treat] == 1, y]
        yc = g.loc[g[treat] == 0, y]
        effect = None
        if len(yt) and len(yc):
            effect = float(yt.mean() - yc.mean())
        rows.append({
            "match_id": str(mid),
            "mean_treat": None if not len(yt) else float(yt.mean()),
            "mean_control": None if not len(yc) else float(yc.mean()),
            "n_treat": int(len(yt)),
            "n_control": int(len(yc)),
            "effect": effect,
        })
    return rows


def _contrasts(df: pd.DataFrame, treat: str, label: str, seed0: int) -> list[dict]:
    opt = df[df["kind"] == "option"].copy()
    fwd_opt = opt[opt["pass_ahead"] == 1].copy()
    passes = df[df["kind"] == "forward_pass"].copy()
    vol = _volume_flag(opt)
    completion = _record(
        f"high-block forward passes: completion, {label} minus not",
        "negative",
        cluster_diff(passes, "successful", treat, seed=seed0),
        "rate",
    )
    epv = _record(
        f"high-block forward passes: pass_epv_total, {label} minus not",
        "negative",
        cluster_diff(passes, "pass_epv", treat, seed=seed0 + 1),
        "pass_epv_total",
    )
    received = _record(
        f"high-block forward passing options: received, {label} minus not",
        "negative",
        cluster_diff(fwd_opt, "received", treat, seed=seed0 + 2),
        "rate",
    )
    targeted = _paired_xthreat(opt, treat, seed0 + 3)
    targeted["name"] = f"within possession, mean xthreat of {label} options minus open options"
    volume = _record(
        f"high-block passing options: {label} rate, above-median pass volume minus below",
        "positive",
        cluster_diff(vol, treat, "high_volume", seed=seed0 + 4),
        "rate",
    )
    return [completion, epv, received, targeted, volume]


def _empty_contrast(name: str, predicted: str, unit: str) -> dict:
    return {
        "name": name,
        "predicted": predicted,
        "unit": unit,
        "supported": False,
        "effect": None,
        "ci_low": None,
        "ci_high": None,
        "mean_treat": None,
        "mean_control": None,
        "n_treat": 0,
        "n_control": 0,
        "n_matches": 0,
        "n_matches_both": 0,
        "ci_excludes_zero": False,
    }


def summarize_pose(df: pd.DataFrame) -> dict:
    label = "shadow-screen"
    if df.empty:
        contrasts = [
            _empty_contrast("high-block forward passes: completion, shadow-screen minus not", "negative", "rate"),
        ]
    else:
        contrasts = _contrasts(df, "shadow_screen", label, BOOT_SEED + 300)
    opt = df[df["kind"] == "option"] if len(df) else df
    fwd = opt[opt["pass_ahead"] == 1] if len(opt) else opt
    passes = df[df["kind"] == "forward_pass"] if len(df) else df
    occupied = opt[pd.to_numeric(opt["lane_proxy"], errors="coerce") == 1] if len(opt) else opt
    facing_known = occupied[np.isfinite(pd.to_numeric(occupied["facing_lane"], errors="coerce"))] if len(occupied) else occupied
    primary = contrasts[0]
    disruption = any(c["supported"] for c in contrasts[:3])
    targeting = any(c["supported"] for c in contrasts[3:])
    return {
        "label": "body_orientation",
        "underpowered": True,
        "matches": list(POSE_MATCHES),
        "definition": (
            "High-block passing options and forward passes on matches 1925299 and 1996435. "
            "The screening defender is the outfield opponent closest to the segment from the "
            f"ball carrier to the teammate, with segment parameter t in [{LANE_T[0]:.2f}, {LANE_T[1]:.2f}]. "
            f"Body between ball and teammate: that defender is within {LANE_DIST_M:.1f} m of the segment. "
            f"Facing the lane: a reliable shoulder heading is within {BODY_SQUARE_DEG:.0f} degrees of the "
            "direction from that defender to the ball, so the chest looks back along the lane toward the passer. "
            "Shoulders are reliable when both are present, width is between "
            f"{PLAUSIBLE_WIDTH_M[0]:.2f} and {PLAUSIBLE_WIDTH_M[1]:.2f} m, and the worse p90 error is at most "
            f"{DEFAULT_MAX_ERR_CM:.0f} cm. Heading is the left-to-right shoulder axis rotated +90 degrees. "
            "Shadow-screen requires both the lane occupation and the facing. "
            "If the lane is occupied but the shoulders are missing or unreliable, the event is excluded. "
            "If nobody is in the lane window, the event is not a shadow-screen. "
            "Tracking picks the defender. Pose supplies only the heading. "
            "Forward means pass_ahead. Completion is pass_outcome successful versus any other recorded outcome. "
            "Pass value is pass_epv_total. Influential is the option xthreat, compared inside the same possession, "
            "and separately a within-match split of players above versus below the median number of passes played or received. "
            "Goalkeepers are excluded. Segments shorter than 1 m are excluded. No xG."
        ),
        "population": "high_block passing_option and pass_ahead player_possession passes",
        "n_options": int(len(opt)),
        "n_forward_options": int(len(fwd)),
        "n_forward_passes": int(len(passes)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "lane_occupation_rate_options": _rate(opt, "lane_proxy", BOOT_SEED + 320) if len(opt) else None,
        "shadow_rate_options": _rate(opt, "shadow_screen", BOOT_SEED + 321) if len(opt) else None,
        "facing_rate_when_lane_occupied": _rate(facing_known, "facing_lane", BOOT_SEED + 322) if len(facing_known) else None,
        "n_lane_occupied_options": int((pd.to_numeric(opt["lane_proxy"], errors="coerce") == 1).sum()) if len(opt) else 0,
        "n_lane_occupied_with_reliable_shoulders": int(len(facing_known)),
        "n_shadow_options": int((pd.to_numeric(opt["shadow_screen"], errors="coerce") == 1).sum()) if len(opt) else 0,
        "n_open_options": int((pd.to_numeric(opt["shadow_screen"], errors="coerce") == 0).sum()) if len(opt) else 0,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "supported": bool(primary["supported"]),
        "disruption_supported": disruption,
        "targeting_supported": targeting,
        "metric_worth_keeping": False,
        "metric_name": None,
        "inference": "match_cluster_bootstrap",
        "by_match_forward_pass_completion": _arm_by_match(passes, "successful", "shadow_screen") if len(passes) else [],
        "contrasts": contrasts,
        "notes": (
            "Two matches. The cluster interval is reported, and the test is underpowered. "
            "supported on the primary completion contrast is true only when that interval excludes zero "
            "in the predicted direction, both arms have at least 20 events, and at least two matches "
            "contain both arms. A rate whose interval excludes zero is not itself a supported claim. "
            "The orientation metric is not worth keeping from two matches."
        ),
    }


def summarize_proxy(df: pd.DataFrame) -> dict:
    label = "lane-occupation proxy"
    contrasts = _contrasts(df, "lane_proxy", label, BOOT_SEED + 310) if len(df) else []
    opt = df[df["kind"] == "option"] if len(df) else df
    fwd = opt[opt["pass_ahead"] == 1] if len(opt) else opt
    passes = df[df["kind"] == "forward_pass"] if len(df) else df
    primary = contrasts[0] if contrasts else _empty_contrast(
        "high-block forward passes: completion, lane-occupation proxy minus not", "negative", "rate",
    )
    disruption = any(c["supported"] for c in contrasts[:3]) if contrasts else False
    targeting = any(c["supported"] for c in contrasts[3:]) if contrasts else False
    worth = bool(primary["supported"] and primary.get("effect") is not None and abs(primary["effect"]) >= 0.03)
    if not worth:
        for c in contrasts[1:3]:
            if c["supported"] and c.get("effect") is not None and abs(c["effect"]) >= 0.03 and c["unit"] == "rate":
                worth = True
                break
    return {
        "label": "tracking_lane_occupation_proxy",
        "not_body_orientation": True,
        "definition": (
            "Same high-block events on every match. Lane occupation means any outfield opponent "
            f"is within {LANE_DIST_M:.1f} m of the segment from the ball carrier to the teammate, "
            f"with t in [{LANE_T[0]:.2f}, {LANE_T[1]:.2f}]. This proxy does not use body orientation, "
            "shoulder heading, or which way the defender is facing. "
            "Forward means pass_ahead. Completion is pass_outcome. Pass value is pass_epv_total. "
            "Influential is option xthreat inside the possession, and a within-match median split of "
            "passes played or received. Goalkeepers excluded. Segments shorter than 1 m excluded. No xG."
        ),
        "population": "high_block passing_option and pass_ahead player_possession passes, all matches",
        "n_options": int(len(opt)),
        "n_forward_options": int(len(fwd)),
        "n_forward_passes": int(len(passes)),
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "lane_occupation_rate_options": _rate(opt, "lane_proxy", BOOT_SEED + 330) if len(opt) else None,
        "n_occupied_forward_options": int((pd.to_numeric(fwd["lane_proxy"], errors="coerce") == 1).sum()) if len(fwd) else 0,
        "n_open_forward_options": int((pd.to_numeric(fwd["lane_proxy"], errors="coerce") == 0).sum()) if len(fwd) else 0,
        "effect_name": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "supported": bool(primary["supported"]),
        "disruption_supported": disruption,
        "targeting_supported": targeting,
        "metric_worth_keeping": worth,
        "metric_name": "high_block_lane_occupation" if worth else None,
        "inference": "match_cluster_bootstrap",
        "by_match_forward_pass_completion": _arm_by_match(passes, "successful", "lane_proxy") if len(passes) else [],
        "contrasts": contrasts,
        "notes": (
            "This is a tracking proxy for a body in the passing lane. It is not a shadow screen "
            "and it does not measure facing. supported on the primary completion contrast is true "
            "only when the match-clustered interval excludes zero in the predicted direction."
        ),
    }


def _load_headings(match_id: str, events_needed: set[int]) -> dict:
    path = resolve_pose_zip(match_id)
    print(f"  pose zip {path} ({path.stat().st_size} bytes), frames {len(events_needed)}", flush=True)
    t0 = time.time()
    headings = stream_headings(path, events_needed)
    print(f"  headings {len(headings)} in {time.time() - t0:.1f}s", flush=True)
    return headings


def run() -> dict:
    ids = match_ids()
    rows: list[dict] = []
    for i, mid in enumerate(ids, start=1):
        t0 = time.time()
        print(f"[{i}/{len(ids)}] {mid}", flush=True)
        part = process_match(mid)
        rows.extend(part)
        n_proxy = sum(1 for r in part if r["lane_proxy"] == 1)
        n_shadow = sum(1 for r in part if r["shadow_screen"] == 1)
        print(f"    rows {len(part)} proxy {n_proxy} shadow {n_shadow} in {time.time() - t0:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    pose_df = frame[frame["match_id"].isin(POSE_MATCHES)].copy() if len(frame) else frame
    summary = {
        "id": 3,
        "title": "High-block shadow screen of passing lanes",
        "supported": False,
        "proxy_supported": False,
        "pose": summarize_pose(pose_df),
        "proxy": summarize_proxy(frame),
    }
    summary["supported"] = bool(summary["pose"]["supported"])
    summary["proxy_supported"] = bool(summary["proxy"]["supported"])
    summary["notes"] = (
        "Pose supported refers only to body orientation on two matches and is underpowered. "
        "proxy_supported refers only to the tracking lane-occupation proxy on all matches. "
        "The proxy is not body orientation."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(json_ready(summary), indent=2))
    print(f"wrote {SUMMARY_PATH}", flush=True)
    return summary


if __name__ == "__main__":
    run()
