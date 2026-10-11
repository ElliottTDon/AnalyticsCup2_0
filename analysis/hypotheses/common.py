"""Shared geometry, loading, and match-clustered intervals for the hypothesis tests.

Events and phases are the sample. Frames are only used to measure positions
inside an event. Intervals resample whole matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MATCHES_DIR = ROOT / "opendata" / "data" / "matches"
POSE_DIR = ROOT / "opendata" / "pose_data"
OUT_DIR = ROOT / "analysis" / "output"
SUMMARY_PATH = OUT_DIR / "hypothesis_summary.json"

ORGANIZED = ("low_block", "medium_block", "high_block")
CB_ROLES = frozenset({"LCB", "RCB", "CB", "CCB"})
FB_ROLES = frozenset({"LB", "RB", "LWB", "RWB"})
CM_ROLES = frozenset({"LDM", "RDM", "CDM", "DM", "CM", "LCM", "RCM", "AM", "CAM"})
WIDE_ROLES = frozenset({"LB", "RB", "LWB", "RWB", "LW", "RW", "LM", "RM"})
STRIKER_ROLES = frozenset({"CF", "ST"})
ROLE_GROUP = {}
for _r in CB_ROLES:
    ROLE_GROUP[_r] = "CB"
for _r in FB_ROLES:
    ROLE_GROUP[_r] = "FB"
for _r in CM_ROLES:
    ROLE_GROUP[_r] = "CM"

# Frozen definitions. Change these only if the hypothesis statement changes.
LANE_DIST_M = 1.5
LANE_T = (0.10, 0.90)
TRACK_DIST_M = 6.0
TRACK_COS = 0.60
TRACK_SPEED_DIFF = 2.0
TRACK_LOOKBACK = 10  # frames at 10 fps = 1.0 s
STATIONARY_DISP_M = 1.0
TIGHT_MARK_M = 3.0
OCCLUSION_RADIUS_M = 0.40
GOAL_WIDTH_M = 7.32
GOAL_HALF_M = GOAL_WIDTH_M / 2.0
VISIBILITY_THRESHOLD = 0.50
BIG_CHANCE_XSHOT = 0.15
BODY_SQUARE_DEG = 45.0
BODY_SIDE_DEG = 135.0
MIN_ROLE_SAMPLES = 8
N_BOOT = 2000
BOOT_SEED = 11

HIGH_PRESSURE = frozenset({"high_pressure", "very_high_pressure"})
OPEN_PLAY_STARTS = frozenset(
    {"pass_reception", "recovery", "keep_possession", "pass_interception", "unknown"}
)

EVENT_COLS = [
    "event_id",
    "match_id",
    "frame_start",
    "frame_end",
    "event_type",
    "event_subtype",
    "player_id",
    "player_position",
    "player_in_possession_id",
    "player_in_possession_position",
    "team_id",
    "attacking_side",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "third_start",
    "player_targeted_id",
    "player_targeted_position",
    "player_targeted_x_pass",
    "player_targeted_y_pass",
    "player_targeted_x_reception",
    "player_targeted_y_reception",
    "player_in_possession_x_start",
    "player_in_possession_y_start",
    "player_in_possession_x_end",
    "player_in_possession_y_end",
    "associated_player_possession_event_id",
    "start_type",
    "end_type",
    "pass_outcome",
    "pass_ahead",
    "pass_angle",
    "lead_to_shot",
    "lead_to_goal",
    "xthreat",
    "player_targeted_xthreat",
    "xshot_player_possession_max",
    "possession_epv_total",
    "pass_epv_total",
    "first_line_break",
    "overall_pressure_start",
    "overall_pressure_end",
    "received",
    "beaten_by_movement",
    "team_out_of_possession_phase_type",
]


def match_ids() -> list[str]:
    ids = []
    for d in sorted(MATCHES_DIR.iterdir()):
        if d.is_dir() and (d / f"{d.name}_dynamic_events.csv").is_file():
            ids.append(d.name)
    return ids


def bool01(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    out = np.full(len(s), np.nan)
    vals = s.to_numpy()
    for i, x in enumerate(vals):
        if x is True or x == "True" or x == "true" or x == 1 or x == 1.0:
            out[i] = 1.0
        elif x is False or x == "False" or x == "false" or x == 0 or x == 0.0:
            out[i] = 0.0
    return pd.Series(out, index=s.index)


def load_events(match_id: str) -> pd.DataFrame:
    path = MATCHES_DIR / match_id / f"{match_id}_dynamic_events.csv"
    df = pd.read_csv(path, usecols=EVENT_COLS)
    for col in (
        "lead_to_shot",
        "lead_to_goal",
        "pass_ahead",
        "received",
        "first_line_break",
        "beaten_by_movement",
    ):
        if col in df.columns:
            df[col] = bool01(df[col])
    df["match_id"] = df["match_id"].astype(str)
    return df


def load_phases(match_id: str) -> pd.DataFrame:
    path = MATCHES_DIR / match_id / f"{match_id}_phases_of_play.csv"
    cols = [
        "index",
        "frame_start",
        "frame_end",
        "team_in_possession_id",
        "attacking_side",
        "team_out_of_possession_phase_type",
    ]
    return pd.read_csv(path, usecols=cols)


def load_roster(match_id: str) -> tuple[dict[int, int], dict[int, str], set[int], float, float]:
    """team id, roster acronym, goalkeeper ids, pitch length, pitch width."""
    raw = json.loads((MATCHES_DIR / match_id / f"{match_id}_match.json").read_text())
    team: dict[int, int] = {}
    acronym: dict[int, str] = {}
    gk: set[int] = set()
    for p in raw["players"]:
        pid = int(p["id"])
        role = p.get("player_role") or {}
        acr = role.get("acronym") or ""
        team[pid] = int(p["team_id"])
        acronym[pid] = acr
        if role.get("name") == "Goalkeeper" or acr == "GK":
            gk.add(pid)
    return team, acronym, gk, float(raw["pitch_length"]), float(raw["pitch_width"])


def modal_positions(events: pd.DataFrame) -> dict[int, str]:
    sub = events.dropna(subset=["player_id", "player_position"])
    out: dict[int, str] = {}
    if sub.empty:
        return out
    for pid, g in sub.groupby(sub["player_id"].astype(int)):
        out[int(pid)] = str(g["player_position"].value_counts().index[0])
    return out


def player_role(pid: int, modal: dict[int, str], roster: dict[int, str]) -> str | None:
    role = modal.get(pid) or roster.get(pid) or ""
    if not role or role == "SUB":
        role = roster.get(pid) or ""
    if not role or role == "SUB":
        return None
    return role


def attack_sign(attacking_side: str) -> float:
    if attacking_side == "left_to_right":
        return 1.0
    if attacking_side == "right_to_left":
        return -1.0
    return float("nan")


def parse_frame_prefix(line: str) -> int | None:
    if not line.startswith('{"frame":'):
        return None
    i = 9
    n = len(line)
    while i < n and line[i] == " ":
        i += 1
    j = i
    while j < n and line[j].isdigit():
        j += 1
    if j == i:
        return None
    return int(line[i:j])


def stream_frames(match_id: str, needed: set[int]) -> dict[int, dict[int, tuple[float, float]]]:
    """Player (x, y) at the requested 10 fps frames. One pass, no full-match store."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    if not needed:
        return out
    pending = set(needed)
    max_f = max(pending)
    path = MATCHES_DIR / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    with path.open() as fh:
        for line in fh:
            fr = parse_frame_prefix(line)
            if fr is None:
                continue
            if fr not in pending:
                if fr > max_f:
                    break
                continue
            rec = json.loads(line)
            xy: dict[int, tuple[float, float]] = {}
            for p in rec.get("player_data") or []:
                pid = p.get("player_id")
                x, y = p.get("x"), p.get("y")
                if pid is None or x is None or y is None:
                    continue
                xy[int(pid)] = (float(x), float(y))
            out[fr] = xy
            pending.discard(fr)
            if not pending:
                break
    return out


def hypot(x: float, y: float) -> float:
    return float(np.hypot(x, y))


def segment_params(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (t clamped to [0, 1], distance from p to segment ab)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return 0.0, float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return t, float(np.linalg.norm(p - proj))


def in_lane(
    defenders: list[tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
    dist_max: float = LANE_DIST_M,
) -> bool:
    if not defenders:
        return False
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if float(np.linalg.norm(bb - aa)) < 1.0:
        return False
    lo, hi = LANE_T
    for d in defenders:
        t, dist = segment_params(np.asarray(d, dtype=float), aa, bb)
        if lo <= t <= hi and dist <= dist_max:
            return True
    return False


def nearest_opponent(
    pid: int,
    xy: dict[int, tuple[float, float]],
    team_of: dict[int, int],
    gk: set[int],
) -> tuple[int | None, float]:
    if pid not in xy or pid not in team_of:
        return None, float("nan")
    x, y = xy[pid]
    my = team_of[pid]
    best_id = None
    best_d = 1e9
    for oid, (ox, oy) in xy.items():
        if oid == pid or oid in gk:
            continue
        if team_of.get(oid) is None or team_of[oid] == my:
            continue
        d = hypot(ox - x, oy - y)
        if d < best_d:
            best_d = d
            best_id = oid
    if best_id is None:
        return None, float("nan")
    return best_id, float(best_d)


def is_goal_side(
    player: tuple[float, float],
    opponent: tuple[float, float],
    sign: float,
) -> bool:
    return (opponent[0] - player[0]) * sign > 0.0


def displacement(
    frames: dict[int, dict[int, tuple[float, float]]],
    pid: int,
    frame: int,
    lookback: int = TRACK_LOOKBACK,
) -> tuple[float, float] | None:
    now = frames.get(frame, {}).get(pid)
    prev = frames.get(frame - lookback, {}).get(pid)
    if now is None or prev is None:
        return None
    return (now[0] - prev[0], now[1] - prev[1])


def similar_velocity(v_att: tuple[float, float] | None, v_def: tuple[float, float] | None) -> float:
    """1 similar, 0 not, nan if either displacement is missing."""
    if v_att is None or v_def is None:
        return float("nan")
    s1 = hypot(*v_att)
    s2 = hypot(*v_def)
    if s1 < STATIONARY_DISP_M and s2 < STATIONARY_DISP_M:
        return 1.0 if abs(s1 - s2) <= 1.5 else 0.0
    if s1 < 0.5 or s2 < 0.5:
        return 0.0
    cos = (v_att[0] * v_def[0] + v_att[1] * v_def[1]) / (s1 * s2)
    if cos >= TRACK_COS and abs(s1 - s2) <= TRACK_SPEED_DIFF:
        return 1.0
    return 0.0


def angle_between(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Interior angle at origin between points a and b, in degrees [0, 180]."""
    v1 = np.array([a[0] - origin[0], a[1] - origin[1]], dtype=float)
    v2 = np.array([b[0] - origin[0], b[1] - origin[1]], dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 0.5 or n2 < 0.5:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def pass_angle_from_goal(
    passer: tuple[float, float],
    receiver: tuple[float, float],
    sign: float,
) -> float:
    """Absolute pass angle from the attacking axis. 0 = straight at goal, 180 = straight back."""
    dx = (receiver[0] - passer[0]) * sign
    dy = (receiver[1] - passer[1]) * sign
    if hypot(dx, dy) < 0.5 or not np.isfinite(sign):
        return float("nan")
    return float(abs(np.degrees(np.arctan2(dy, dx))))


def heading_delta_deg(heading: float, target_deg: float) -> float:
    """Absolute difference in degrees, in [0, 180]."""
    return float(abs((heading - target_deg + 180.0) % 360.0 - 180.0))


def body_label(bearing_deg: float) -> str | None:
    if not np.isfinite(bearing_deg):
        return None
    if bearing_deg <= BODY_SQUARE_DEG:
        return "square_in_front"
    if bearing_deg <= BODY_SIDE_DEG:
        return "side_on"
    return "back_to"


def wide_cm_pair(pos_a: str | None, pos_b: str | None) -> bool:
    if not pos_a or not pos_b:
        return False
    a_wide, b_wide = pos_a in WIDE_ROLES, pos_b in WIDE_ROLES
    a_cm, b_cm = pos_a in CM_ROLES, pos_b in CM_ROLES
    return (a_wide and b_cm) or (a_cm and b_wide)


def _arm_stats(df: pd.DataFrame, y: str, treat: str, match: str) -> np.ndarray:
    """Per-match (sum_t, n_t, sum_c, n_c) for a 0/1 treatment column."""
    rows = []
    work = df[[match, y, treat]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work[treat] = pd.to_numeric(work[treat], errors="coerce")
    work = work[np.isfinite(work[y]) & np.isfinite(work[treat])]
    for _, g in work.groupby(match, sort=False):
        yt = g.loc[g[treat] == 1, y].to_numpy(dtype=float)
        yc = g.loc[g[treat] == 0, y].to_numpy(dtype=float)
        rows.append((float(yt.sum()) if len(yt) else 0.0, float(len(yt)), float(yc.sum()) if len(yc) else 0.0, float(len(yc))))
    if not rows:
        return np.zeros((0, 4))
    return np.asarray(rows, dtype=float)


def cluster_diff(
    df: pd.DataFrame,
    y: str,
    treat: str,
    match: str = "match_id",
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Pooled mean(y|treat=1) - mean(y|treat=0), CI by resampling matches."""
    arr = _arm_stats(df, y, treat, match)
    empty = {
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
    if len(arr) == 0:
        return empty
    n_t = arr[:, 1].sum()
    n_c = arr[:, 3].sum()
    n_both = int(((arr[:, 1] > 0) & (arr[:, 3] > 0)).sum())
    if n_t < 1 or n_c < 1:
        empty.update({
            "n_treat": int(n_t),
            "n_control": int(n_c),
            "n_matches": int(len(arr)),
            "n_matches_both": n_both,
        })
        return empty
    mean_t = float(arr[:, 0].sum() / n_t)
    mean_c = float(arr[:, 2].sum() / n_c)
    effect = mean_t - mean_c
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    samp = arr[idx]
    st = samp[:, :, 0].sum(axis=1)
    nt = samp[:, :, 1].sum(axis=1)
    sc = samp[:, :, 2].sum(axis=1)
    nc = samp[:, :, 3].sum(axis=1)
    ok = (nt > 0) & (nc > 0)
    diffs = np.full(n_boot, np.nan)
    diffs[ok] = st[ok] / nt[ok] - sc[ok] / nc[ok]
    finite = diffs[np.isfinite(diffs)]
    if len(finite) < max(100, int(0.5 * n_boot)):
        ci_low = ci_high = None
        excludes = False
    else:
        ci_low = float(np.percentile(finite, 2.5))
        ci_high = float(np.percentile(finite, 97.5))
        excludes = bool(ci_low > 0 or ci_high < 0)
    return {
        "effect": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_treat": mean_t,
        "mean_control": mean_c,
        "n_treat": int(n_t),
        "n_control": int(n_c),
        "n_matches": int((arr[:, 1] + arr[:, 3] > 0).sum()),
        "n_matches_both": n_both,
        "ci_excludes_zero": excludes,
    }


def cluster_mean(
    df: pd.DataFrame,
    y: str,
    match: str = "match_id",
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Pooled mean of y, CI by resampling matches. Used for paired within-event diffs."""
    work = df[[match, y]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work[np.isfinite(work[y])]
    empty = {
        "effect": None,
        "ci_low": None,
        "ci_high": None,
        "n": 0,
        "n_matches": 0,
        "ci_excludes_zero": False,
    }
    if work.empty:
        return empty
    rows = []
    for _, g in work.groupby(match, sort=False):
        vals = g[y].to_numpy(dtype=float)
        rows.append((float(vals.sum()), float(len(vals))))
    arr = np.asarray(rows, dtype=float)
    n = float(arr[:, 1].sum())
    effect = float(arr[:, 0].sum() / n)
    rng = np.random.default_rng(seed)
    m = len(arr)
    if m < 2:
        return {
            "effect": effect,
            "ci_low": None,
            "ci_high": None,
            "n": int(n),
            "n_matches": m,
            "ci_excludes_zero": False,
        }
    idx = rng.integers(0, m, size=(n_boot, m))
    samp = arr[idx]
    sm = samp[:, :, 0].sum(axis=1)
    nm = samp[:, :, 1].sum(axis=1)
    ok = nm > 0
    boots = np.full(n_boot, np.nan)
    boots[ok] = sm[ok] / nm[ok]
    finite = boots[np.isfinite(boots)]
    ci_low = float(np.percentile(finite, 2.5))
    ci_high = float(np.percentile(finite, 97.5))
    return {
        "effect": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(n),
        "n_matches": m,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def supports(contrast: dict, predicted: str, min_n: int = 20) -> bool:
    """True only when the interval excludes zero and the sign matches the claim."""
    if not contrast.get("ci_excludes_zero"):
        return False
    effect = contrast.get("effect")
    if effect is None or not np.isfinite(effect):
        return False
    n_t = contrast.get("n_treat", contrast.get("n", 0))
    n_c = contrast.get("n_control", contrast.get("n", 0))
    if n_t < min_n or n_c < min_n:
        return False
    if predicted == "positive":
        return effect > 0 and contrast["ci_low"] > 0
    if predicted == "negative":
        return effect < 0 and contrast["ci_high"] < 0
    return False


def fmt(x, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
