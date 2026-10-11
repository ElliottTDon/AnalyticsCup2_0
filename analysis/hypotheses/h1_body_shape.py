"""Hypothesis 1: nearest-defender body shape at a pass reception.

At each pass reception, label the nearest outfield defender from the angle
between their shoulder heading and the vector from that defender to the
receiver. Relate the label to xshot, the receiver's xthreat, possession EPV,
and pass EPV.

Pose exists for two matches only. The sample is the reception event, not the
frame. Run:

    opendata/.venv/bin/python analysis/hypotheses/h1_body_shape.py
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
from common import (  # noqa: E402
    BODY_SIDE_DEG,
    BODY_SQUARE_DEG,
    BOOT_SEED,
    MATCHES_DIR,
    N_BOOT,
    body_label,
    cluster_diff,
    heading_delta_deg,
    json_ready,
    load_roster,
    nearest_opponent,
    stream_frames,
)

ROOT = Path(__file__).resolve().parents[2]
OPENDATA = ROOT / "opendata"
sys.path.insert(0, str(OPENDATA))
from src.data.pose_loading import tracking_to_pose_frame  # noqa: E402
from src.features.pose_orientation import (  # noqa: E402
    DEFAULT_MAX_ERR_CM,
    LEFT_SHOULDER,
    PLAUSIBLE_WIDTH_M,
    RIGHT_SHOULDER,
    shoulder_orientation_deg,
)

POSE_MATCHES = ("1925299", "1996435")
# Incoming pass must end within this many 10 fps frames before the reception.
PASS_LOOKBACK = 40
MIN_VECTOR_M = 0.5
# A contrast with fewer events than this in either arm is not called supported.
MIN_ARM = 20
OUT_PATH = ROOT / "analysis" / "output" / "h1_summary.json"
MANIFEST_PATH = OPENDATA / "data" / "bodypose" / "MANIFEST.json"
POSE_CANDIDATES = (ROOT / "pose_data", OPENDATA / "pose_data")

EVENT_COLS = [
    "event_id",
    "match_id",
    "event_type",
    "start_type",
    "end_type",
    "player_id",
    "player_targeted_id",
    "team_id",
    "frame_start",
    "frame_end",
    "x_start",
    "y_start",
    "pass_outcome",
    "player_targeted_xthreat",
    "possession_epv_total",
    "pass_epv_total",
    "xshot_player_possession_max",
    "associated_player_possession_event_id",
]

OUTCOMES = (
    "xshot_player_possession_max",
    "receiver_xthreat",
    "possession_epv_total",
    "pass_epv_total",
)
LABELS = ("square_in_front", "side_on", "back_to")


def _as_int(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_sizes() -> dict[str, int]:
    meta = json.loads(MANIFEST_PATH.read_text())
    return {
        match_id: int(meta["files"][f"raw/{match_id}.jsonl.zip"]["size_bytes"])
        for match_id in POSE_MATCHES
    }


def _archive_path(match_id: str, expected: int) -> Path | None:
    for folder in POSE_CANDIDATES:
        path = folder / f"{match_id}.jsonl.zip"
        if path.is_file() and path.stat().st_size == expected:
            return path
    return None


def _wait_for_archive(match_id: str, expected: int, timeout_s: float = 900.0) -> Path | None:
    """Reuse a zip another process is still writing. Return None if it stalls incomplete."""
    deadline = time.time() + timeout_s
    last: dict[Path, int] = {}
    still: dict[Path, int] = {}
    while time.time() < deadline:
        found = _archive_path(match_id, expected)
        if found is not None:
            return found
        growing = False
        for folder in POSE_CANDIDATES:
            path = folder / f"{match_id}.jsonl.zip"
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size != last.get(path):
                last[path] = size
                still[path] = 0
                growing = True
            else:
                still[path] = still.get(path, 0) + 1
        if not growing and last and all(n >= 2 for n in still.values()):
            return None
        time.sleep(5.0)
    return _archive_path(match_id, expected)


def _download(match_id: str, expected: int) -> Path:
    import os
    import urllib.request

    meta = json.loads(MANIFEST_PATH.read_text())
    dest_dir = ROOT / "pose_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{match_id}.jsonl.zip"
    url = f"{meta['base_url']}/raw/{match_id}.jsonl.zip"
    token = os.environ.get("HF_TOKEN")
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    tmp = dest.with_suffix(".zip.part")
    print(f"downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(request) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size != expected:
        raise RuntimeError(
            f"{match_id} download is {tmp.stat().st_size} bytes, expected {expected}"
        )
    tmp.replace(dest)
    return dest


def ensure_archive(match_id: str) -> Path:
    expected = _expected_sizes()[match_id]
    found = _archive_path(match_id, expected)
    if found is not None:
        return found
    found = _wait_for_archive(match_id, expected)
    if found is not None:
        return found
    return _download(match_id, expected)


def _frame_prefix(line: bytes) -> int | None:
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


def pose_frames_for(tracking_frame: int) -> list[int]:
    """Pose frames aligned to one tracking frame. Halfway frames use both neighbors."""
    target = float(tracking_to_pose_frame(int(tracking_frame)))
    if abs(target - round(target)) < 1e-6:
        return [int(round(target))]
    return [math.floor(target), math.ceil(target)]


def _heading_from_player(player: dict) -> tuple[str, float | None]:
    joints = player.get("joints")
    if not joints:
        return "no_pose", None
    left, right = joints.get(LEFT_SHOULDER), joints.get(RIGHT_SHOULDER)
    if not left or not right:
        return "unreliable", None
    lx, ly = float(left["xyz"][0]), float(left["xyz"][1])
    rx, ry = float(right["xyz"][0]), float(right["xyz"][1])
    angle = shoulder_orientation_deg(lx, ly, rx, ry)
    width = math.hypot(rx - lx, ry - ly)
    err = max(left.get("p90_mae_cm") or math.inf, right.get("p90_mae_cm") or math.inf)
    if (
        angle is None
        or not (PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1])
        or err > DEFAULT_MAX_ERR_CM
    ):
        return "unreliable", None
    return "ok", float(angle)


def _circular_mean(degrees: list[float]) -> float | None:
    if not degrees:
        return None
    radians = np.radians(np.asarray(degrees, dtype=float))
    s = float(np.sin(radians).sum())
    c = float(np.cos(radians).sum())
    if math.hypot(s, c) / len(degrees) < 0.5:
        return None
    return float(np.degrees(np.arctan2(s, c)))


def stream_headings(
    archive: Path, needed: dict[int, set[int]]
) -> dict[int, dict[int, tuple[str, float | None]]]:
    """Shoulder status for selected players at selected pose frames."""
    out: dict[int, dict[int, tuple[str, float | None]]] = {}
    if not needed:
        return out
    pending = set(needed)
    max_frame = max(pending)
    member = None
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.endswith(".jsonl") and "MACOSX" not in name:
                member = name
                break
        if member is None:
            raise RuntimeError(f"no jsonl member in {archive}")
        scanned = 0
        with zf.open(member) as fh:
            for line in fh:
                scanned += 1
                if scanned % 20000 == 0:
                    print(f"  {archive.name} scanned {scanned} pose frames, {len(pending)} left", flush=True)
                frame = _frame_prefix(line)
                if frame is None:
                    continue
                if frame not in pending:
                    if frame > max_frame:
                        break
                    continue
                record = json.loads(line)
                wanted = needed[frame]
                found: dict[int, tuple[str, float | None]] = {}
                for player in record.get("player_data") or []:
                    pid = _as_int(player.get("player_id"))
                    if pid is None or pid not in wanted:
                        continue
                    found[pid] = _heading_from_player(player)
                out[frame] = found
                pending.discard(frame)
                if not pending:
                    break
    print(f"  {archive.name} done, missing pose frames {len(pending)}", flush=True)
    return out


def _load_match_events(match_id: str) -> pd.DataFrame:
    path = MATCHES_DIR / match_id / f"{match_id}_dynamic_events.csv"
    df = pd.read_csv(path, usecols=EVENT_COLS)
    df["match_id"] = df["match_id"].astype(str)
    return df


def _receiver_xthreat(poss: pd.DataFrame, receptions: pd.DataFrame) -> pd.Series:
    """player_targeted_xthreat on the successful pass that this reception completed."""
    passes = poss[poss["end_type"] == "pass"]
    by_target: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row in passes.itertuples(index=False):
        team = _as_int(row.team_id)
        target = _as_int(row.player_targeted_id)
        if team is None or target is None:
            continue
        if row.pass_outcome != "successful":
            continue
        threat = row.player_targeted_xthreat
        threat_v = float(threat) if pd.notna(threat) else float("nan")
        by_target.setdefault((team, target), []).append((int(row.frame_end), threat_v))
    for key in by_target:
        by_target[key].sort()

    values = []
    for row in receptions.itertuples(index=False):
        team = _as_int(row.team_id)
        receiver = _as_int(row.player_id)
        cands = by_target.get((team, receiver), []) if team is not None and receiver is not None else []
        best = None
        start = int(row.frame_start)
        for frame_end, threat in cands:
            if frame_end <= start:
                best = (frame_end, threat)
            else:
                break
        if best is None or start - best[0] > PASS_LOOKBACK:
            values.append(float("nan"))
        else:
            values.append(best[1])
    return pd.Series(values, index=receptions.index, dtype=float)


def reception_table(match_id: str) -> pd.DataFrame:
    events = _load_match_events(match_id)
    poss = events[events["event_type"] == "player_possession"].copy()
    receptions = poss[poss["start_type"] == "pass_reception"].copy()
    engagements = events[events["event_type"] == "on_ball_engagement"]
    xshot = (
        engagements.dropna(subset=["associated_player_possession_event_id"])
        .groupby("associated_player_possession_event_id")["xshot_player_possession_max"]
        .max()
    )
    receptions["xshot_player_possession_max"] = receptions["event_id"].map(xshot)
    receptions["receiver_xthreat"] = _receiver_xthreat(poss, receptions)
    receptions["possession_epv_total"] = pd.to_numeric(receptions["possession_epv_total"], errors="coerce")
    receptions["pass_epv_total"] = pd.to_numeric(receptions["pass_epv_total"], errors="coerce")
    return receptions


def attach_defender(match_id: str, receptions: pd.DataFrame) -> pd.DataFrame:
    team_of, _roster, gk, _length, _width = load_roster(match_id)
    needed = {int(fr) for fr in receptions["frame_start"]}
    frames = stream_frames(match_id, needed)
    defender_ids = []
    distances = []
    receiver_xy = []
    defender_xy = []
    for row in receptions.itertuples(index=False):
        pid = _as_int(row.player_id)
        frame = int(row.frame_start)
        xy = dict(frames.get(frame) or {})
        if pid is None:
            defender_ids.append(None)
            distances.append(float("nan"))
            receiver_xy.append(None)
            defender_xy.append(None)
            continue
        if pid not in xy and pd.notna(row.x_start) and pd.notna(row.y_start):
            xy[pid] = (float(row.x_start), float(row.y_start))
        oid, dist = nearest_opponent(pid, xy, team_of, gk)
        defender_ids.append(oid)
        distances.append(dist)
        receiver_xy.append(xy.get(pid))
        defender_xy.append(xy.get(oid) if oid is not None else None)
    out = receptions.copy()
    out["defender_id"] = defender_ids
    out["defender_distance_m"] = distances
    out["receiver_xy"] = receiver_xy
    out["defender_xy"] = defender_xy
    out["gk_receiver"] = [_as_int(pid) in gk for pid in out["player_id"]]
    return out


def label_match(match_id: str, archive: Path) -> pd.DataFrame:
    print(f"{match_id}: events and tracking", flush=True)
    table = attach_defender(match_id, reception_table(match_id))
    needed: dict[int, set[int]] = {}
    query_frames: list[list[int]] = []
    for row in table.itertuples(index=False):
        frames: list[int] = []
        dist = row.defender_distance_m
        if (
            row.defender_id is not None
            and not row.gk_receiver
            and np.isfinite(dist)
            and dist >= MIN_VECTOR_M
        ):
            frames = pose_frames_for(int(row.frame_start))
            for fr in frames:
                needed.setdefault(fr, set()).add(int(row.defender_id))
        query_frames.append(frames)
    print(f"{match_id}: pose frames to read {len(needed)}", flush=True)
    headings = stream_headings(archive, needed)

    labels = []
    bearings = []
    reasons = []
    for row, frames in zip(table.itertuples(index=False), query_frames):
        if row.gk_receiver:
            labels.append(None)
            bearings.append(float("nan"))
            reasons.append("goalkeeper_receiver")
            continue
        if row.defender_id is None or not np.isfinite(row.defender_distance_m):
            labels.append(None)
            bearings.append(float("nan"))
            reasons.append("no_defender")
            continue
        if row.defender_distance_m < MIN_VECTOR_M or row.receiver_xy is None or row.defender_xy is None:
            labels.append(None)
            bearings.append(float("nan"))
            reasons.append("undefined_vector")
            continue
        statuses = []
        degrees = []
        for fr in frames:
            status, deg = headings.get(fr, {}).get(int(row.defender_id), ("no_pose", None))
            statuses.append(status)
            if deg is not None:
                degrees.append(deg)
        heading = _circular_mean(degrees)
        if heading is None:
            labels.append(None)
            bearings.append(float("nan"))
            reasons.append("unreliable_shoulders" if "unreliable" in statuses else "no_pose")
            continue
        rx, ry = row.receiver_xy
        dx, dy = row.defender_xy
        target = math.degrees(math.atan2(ry - dy, rx - dx))
        bearing = heading_delta_deg(heading, target)
        label = body_label(bearing)
        labels.append(label)
        bearings.append(bearing)
        reasons.append("labeled" if label else "undefined_bearing")
    table["body_label"] = labels
    table["bearing_deg"] = bearings
    table["label_reason"] = reasons
    return table


def _match_effects(df: pd.DataFrame, outcome: str, treat_label: str) -> list[dict]:
    rows = []
    work = df[df["body_label"].isin([treat_label, "square_in_front"])]
    for match_id, group in work.groupby("match_id", sort=False):
        treat = pd.to_numeric(group.loc[group["body_label"] == treat_label, outcome], errors="coerce")
        control = pd.to_numeric(group.loc[group["body_label"] == "square_in_front", outcome], errors="coerce")
        treat = treat[np.isfinite(treat)]
        control = control[np.isfinite(control)]
        effect = None
        if len(treat) and len(control):
            effect = float(treat.mean() - control.mean())
        rows.append({
            "match_id": str(match_id),
            "n_treat": int(len(treat)),
            "n_control": int(len(control)),
            "effect": effect,
        })
    return rows


def _contrast_record(name: str, df: pd.DataFrame, outcome: str, treat_label: str) -> dict:
    work = df[df["body_label"].isin([treat_label, "square_in_front"])].copy()
    work["treat"] = (work["body_label"] == treat_label).astype(float)
    contrast = cluster_diff(work, outcome, "treat", n_boot=N_BOOT, seed=BOOT_SEED)
    # The interval has to exclude zero. An arm with a handful of events does not.
    supported = bool(
        contrast.get("ci_excludes_zero")
        and contrast.get("n_treat", 0) >= MIN_ARM
        and contrast.get("n_control", 0) >= MIN_ARM
    )
    return {
        "name": name,
        "outcome": outcome,
        "comparison": f"{treat_label} minus square_in_front",
        "predicted": "positive",
        "unit": outcome,
        "supported": supported,
        "by_match": _match_effects(df, outcome, treat_label),
        **contrast,
    }


def _label_block(df: pd.DataFrame) -> dict:
    block = {"n": int(len(df))}
    for outcome in OUTCOMES:
        finite = pd.to_numeric(df[outcome], errors="coerce")
        finite = finite[np.isfinite(finite)]
        block[outcome] = {
            "n": int(len(finite)),
            "mean": float(finite.mean()) if len(finite) else None,
        }
    return block


def summarize(tables: list[pd.DataFrame]) -> dict:
    df = pd.concat(tables, ignore_index=True)
    labeled = df[df["body_label"].isin(LABELS)].copy()
    reasons = df["label_reason"].value_counts(dropna=False).to_dict()
    by_label = {label: _label_block(labeled[labeled["body_label"] == label]) for label in LABELS}
    contrasts = []
    for treat_label, treat_name in (("back_to", "back-to"), ("side_on", "side-on")):
        for outcome in OUTCOMES:
            contrasts.append(
                _contrast_record(
                    f"{outcome}, {treat_name} minus square/in front",
                    labeled,
                    outcome,
                    treat_label,
                )
            )
    side = [c for c in contrasts if c["comparison"].startswith("side_on")]
    primary = next(c for c in side if c["outcome"] == "xshot_player_possession_max")
    supported = any(c["supported"] for c in side)
    n_by_match = {}
    for match_id, g in df.groupby("match_id"):
        lab = g[g["body_label"].isin(LABELS)]
        n_by_match[str(match_id)] = {
            "n_receptions": int(len(g)),
            "n_labeled": int(len(lab)),
            **{label: int((lab["body_label"] == label).sum()) for label in LABELS},
        }
    counts = {label: int((labeled["body_label"] == label).sum()) for label in LABELS}
    n_labeled = int(len(labeled))
    shares = {
        label: (counts[label] / n_labeled if n_labeled else None)
        for label in LABELS
    }
    bearing = _bearing_summary(labeled)
    verdict = _verdict(n_labeled, counts, shares, contrasts, reasons, bearing)
    return {
        "id": 1,
        "title": "Defender body shape at pass reception",
        "hypothesis": (
            "At a pass reception, is the nearest defender side-on, square-on/in front, "
            "or back-to the receiver, and do those labels relate to xshot, xthreat, "
            "possession EPV, and pass EPV?"
        ),
        "definition": (
            "Sample is a player_possession with start_type pass_reception in matches "
            "1925299 and 1996435. One row per reception. The nearest outfield opponent "
            "is taken from tracking x,y at frame_start (goalkeepers excluded; no distance cap). "
            "Shoulder heading is the ground-plane facing direction "
            "from the left and right shoulders (x,y only; z unused), on the pose frame "
            "pose_frame = 2.5 * tracking_frame. Odd tracking frames use the circular mean "
            "of the two neighboring pose frames when both shoulders are reliable. "
            f"Reliable means shoulder width in {PLAUSIBLE_WIDTH_M[0]:.2f}-{PLAUSIBLE_WIDTH_M[1]:.2f} m "
            f"and p90 error at most {DEFAULT_MAX_ERR_CM:.0f} cm. "
            "The label is the absolute angle between that heading and the tracking vector "
            f"from defender to receiver: square_in_front at most {BODY_SQUARE_DEG:.0f} degrees, "
            f"side_on up to {BODY_SIDE_DEG:.0f} degrees, otherwise back_to. "
            "xshot_player_possession_max is the maximum on on_ball_engagement rows linked by "
            "associated_player_possession_event_id; receptions with no engagement have no xshot. "
            "receiver_xthreat is player_targeted_xthreat on the successful pass that targeted "
            f"this receiver and ended within {PASS_LOOKBACK} tracking frames before the reception. "
            "possession_epv_total and pass_epv_total are the fields on the reception possession "
            "(pass EPV is the pass played from that possession). "
            "Back-to is reported, but the primary comparison is side-on minus square/in front "
            "because that is the split with events in both arms. "
            "Intervals resample the two matches. supported is true only when that interval "
            f"excludes zero and both arms have at least {MIN_ARM} events."
        ),
        "population": (
            "pass receptions in the two pose matches whose nearest outfield defender "
            "has a reliable shoulder heading"
        ),
        "n_receptions": int(len(df)),
        "n_labeled": n_labeled,
        "n_matches": int(df["match_id"].nunique()),
        "underpowered": True,
        "label_counts": counts,
        "label_share": shares,
        "n_by_match": n_by_match,
        "unlabeled_reasons": {str(k): int(v) for k, v in reasons.items()},
        "median_defender_distance_m_labeled": _mean_distance(labeled),
        "bearing_deg": bearing,
        "by_label": by_label,
        "primary_contrast": primary["name"],
        "effect": primary["effect"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "mean_side_on": primary["mean_treat"],
        "mean_square_in_front": primary["mean_control"],
        "n_side_on": primary["n_treat"],
        "n_square_in_front": primary["n_control"],
        "supported": supported,
        "inference": "match_cluster_bootstrap",
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "contrasts": contrasts,
        "verdict": verdict,
        "notes": (
            "Two matches only, so every interval is underpowered and this is not a practical "
            "metric. A narrow interval means the two matches agree, not that the league-wide "
            "effect is precise. Back-to has too few events to support a contrast. "
            "Outcomes are xshot_player_possession_max, player_targeted_xthreat on the received "
            "pass, possession_epv_total, and pass_epv_total."
        ),
    }


def _mean_distance(labeled: pd.DataFrame) -> float | None:
    if labeled.empty:
        return None
    return float(np.nanmedian(labeled["defender_distance_m"].to_numpy(dtype=float)))


def _fmt(value, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _interval(contrast: dict) -> str:
    return f"{_fmt(contrast['ci_low'])} to {_fmt(contrast['ci_high'])}"


def _bearing_summary(labeled: pd.DataFrame) -> dict:
    if labeled.empty:
        return {"n": 0}
    vals = labeled["bearing_deg"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    bins = {
        "0_45": int((vals <= 45.0).sum()),
        "45_90": int(((vals > 45.0) & (vals <= 90.0)).sum()),
        "90_135": int(((vals > 90.0) & (vals <= 135.0)).sum()),
        "135_180": int((vals > 135.0).sum()),
    }
    return {
        "n": int(len(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "bins": bins,
    }


def _verdict(
    n_labeled: int,
    counts: dict,
    shares: dict,
    contrasts: list[dict],
    reasons: dict,
    bearing: dict,
) -> str:
    def pct(label: str) -> str:
        share = shares.get(label)
        if share is None:
            return "n/a"
        return f"{100.0 * share:.0f}%"

    lines = [
        (
            f"Underpowered: pose covers 2 matches and each interval resamples those matches. "
            f"Of {n_labeled} labeled receptions, the nearest defender was square/in front in "
            f"{counts['square_in_front']} ({pct('square_in_front')}), side-on in "
            f"{counts['side_on']} ({pct('side_on')}), and back-to in "
            f"{counts['back_to']} ({pct('back_to')}). "
            f"Median heading-to-receiver angle was {_fmt(bearing.get('median'), 1)} degrees "
            f"(IQR {_fmt(bearing.get('p25'), 1)} to {_fmt(bearing.get('p75'), 1)}). "
            f"Of the side-on labels, {bearing.get('bins', {}).get('45_90', 0)} were 45 to 90 degrees "
            f"and {bearing.get('bins', {}).get('90_135', 0)} were 90 to 135."
        )
    ]
    side = [c for c in contrasts if c["comparison"].startswith("side_on")]
    bits = []
    for contrast in side:
        bits.append(
            f"{contrast['outcome']}: effect {_fmt(contrast['effect'])} "
            f"(interval {_interval(contrast)}, n_side={contrast['n_treat']}, "
            f"n_square={contrast['n_control']}, supported={str(contrast['supported']).lower()})"
        )
    lines.append("Side-on minus square/in front: " + "; ".join(bits) + ".")
    lines.append(
        f"Back-to has n={counts['back_to']}, below {MIN_ARM}, so those intervals are not used for support."
    )
    held = [c["outcome"] for c in side if c["supported"]]
    if held:
        lines.append(
            "supported=true because the side-on minus square/in front interval excludes zero for: "
            + ", ".join(held)
            + "."
        )
    else:
        lines.append(
            "supported=false because no side-on minus square/in front interval excludes zero."
        )
    lines.append(
        "Unlabeled receptions: "
        + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
        + "."
    )
    return " ".join(lines)


def main() -> None:
    tables = []
    for match_id in POSE_MATCHES:
        archive = ensure_archive(match_id)
        print(f"{match_id}: using {archive} ({archive.stat().st_size} bytes)", flush=True)
        tables.append(label_match(match_id, archive))
    summary = json_ready(summarize(tables))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary["verdict"], flush=True)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
