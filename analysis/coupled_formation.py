"""Coupled defense/attack formation graphs on SkillCorner tracking.

Question: in an organized block, does the defense cover the attack's passing
graph, or only the ball?

State constraint: ball in play, OOP phase in {low_block, medium_block,
high_block}, positions on the measured pitch (origin at center).

Sampling: 2 Hz (every 5th tracking frame at 10 fps) inside those phases only.
Streams JSONL; never holds all matches in memory.

Laplacian convention
--------------------
Defense connectivity uses the *combinatorial* Laplacian L = D - W.  Algebraic
connectivity is lambda2 = second-smallest eigenvalue of L.  The Fiedler vector
is the corresponding eigenvector; its sign split names the two subgroups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.distance import cdist

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_DIR = WORKSPACE / "analysis" / "output"

ORGANIZED_BLOCKS = ("low_block", "medium_block", "high_block")
FPS = 10
SAMPLE_STRIDE = 5  # 2 Hz
SIGMA_M = 8.0
SUPPORT_RADIUS_M = 18.0
LANE_OCCLUSION_M = 2.0
FREE_ASSIGN_M = 8.0
GOAL_SIDE_PENALTY_M = 12.0
HIGH_DANGER_QUANTILE = 0.75
PRESSURE_ORDER = [
    "no_pressure",
    "low_pressure",
    "medium_pressure",
    "high_pressure",
    "very_high_pressure",
]
PRESSURE_SCORE = {p: i for i, p in enumerate(PRESSURE_ORDER)}


# ---------------------------------------------------------------------------
# Match / roster helpers
# ---------------------------------------------------------------------------


@dataclass
class MatchMeta:
    match_id: str
    pitch_length: float
    pitch_width: float
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    # player_id (match players[].id) -> info
    players: dict[int, dict[str, Any]] = field(default_factory=dict)
    gk_ids: set[int] = field(default_factory=set)
    player_id_field: str = "id"  # confirmed against tracking

    def team_id_for_group(self, group: str | None) -> int | None:
        if group is None:
            return None
        g = group.strip().lower()
        if g == "home team":
            return self.home_team_id
        if g == "away team":
            return self.away_team_id
        return None


def load_match_meta(match_dir: Path, match_id: str) -> MatchMeta:
    with open(match_dir / f"{match_id}_match.json") as f:
        raw = json.load(f)
    players: dict[int, dict[str, Any]] = {}
    gk_ids: set[int] = set()
    for p in raw["players"]:
        pid = int(p["id"])
        role = p.get("player_role") or {}
        role_name = role.get("name") or ""
        info = {
            "id": pid,
            "team_id": int(p["team_id"]),
            "team_player_id": p.get("team_player_id"),
            "short_name": p.get("short_name") or f"p{pid}",
            "role_name": role_name,
            "position_group": role.get("position_group"),
            "is_gk": role_name == "Goalkeeper",
        }
        players[pid] = info
        if info["is_gk"]:
            gk_ids.add(pid)
    return MatchMeta(
        match_id=match_id,
        pitch_length=float(raw["pitch_length"]),
        pitch_width=float(raw["pitch_width"]),
        home_team_id=int(raw["home_team"]["id"]),
        away_team_id=int(raw["away_team"]["id"]),
        home_name=raw["home_team"].get("short_name") or raw["home_team"]["name"],
        away_name=raw["away_team"].get("short_name") or raw["away_team"]["name"],
        players=players,
        gk_ids=gk_ids,
        player_id_field="id",
    )


def confirm_player_id_field(meta: MatchMeta, tracking_path: Path) -> str:
    """Confirm tracking player_id matches match players[].id on one frame."""
    id_set = set(meta.players)
    tp_set = {p["team_player_id"] for p in meta.players.values() if p.get("team_player_id") is not None}
    with open(tracking_path) as f:
        for line in f:
            fr = json.loads(line)
            pdata = fr.get("player_data") or []
            if len(pdata) < 10:
                continue
            pids = [p.get("player_id") for p in pdata if p.get("player_id") is not None]
            if not pids:
                continue
            n_id = sum(1 for x in pids if x in id_set)
            n_tp = sum(1 for x in pids if x in tp_set)
            if n_id >= n_tp and n_id >= 8:
                meta.player_id_field = "id"
                return "id"
            if n_tp > n_id:
                # Remap roster keys to team_player_id
                remapped = {}
                for p in meta.players.values():
                    remapped[int(p["team_player_id"])] = p
                meta.players = remapped
                meta.gk_ids = {int(p["team_player_id"]) for p in remapped.values() if p["is_gk"]}
                meta.player_id_field = "team_player_id"
                return "team_player_id"
            break
    return meta.player_id_field


def discover_match_ids(data_dir: Path = DATA_DIR) -> list[str]:
    ids: list[str] = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        mid = d.name
        tracking = d / f"{mid}_tracking_extrapolated.jsonl"
        phases = d / f"{mid}_phases_of_play.csv"
        match_json = d / f"{mid}_match.json"
        if tracking.is_file() and phases.is_file() and match_json.is_file():
            # skip LFS pointer stubs
            if tracking.stat().st_size < 10_000:
                continue
            ids.append(mid)
    return ids


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


def gaussian_weights(xy: np.ndarray, sigma: float = SIGMA_M, radius: float = SUPPORT_RADIUS_M) -> np.ndarray:
    """Symmetric proximity weights; zero beyond radius."""
    n = len(xy)
    if n == 0:
        return np.zeros((0, 0))
    d = distance_matrix(xy, xy)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    w[d > radius] = 0.0
    np.fill_diagonal(w, 0.0)
    return w


def combinatorial_laplacian(w: np.ndarray) -> np.ndarray:
    deg = w.sum(axis=1)
    return np.diag(deg) - w


def algebraic_connectivity(w: np.ndarray) -> tuple[float, np.ndarray | None]:
    """lambda2 and Fiedler vector of the combinatorial Laplacian."""
    n = w.shape[0]
    if n < 2:
        return float("nan"), None
    L = combinatorial_laplacian(w)
    # Symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(L)
    # Numerical floor for disconnected components
    lambda2 = float(evals[1]) if n >= 2 else float("nan")
    fiedler = evecs[:, 1]
    return lambda2, fiedler


def attack_direction(attacking_side: str) -> float:
    """+1 if attack progresses toward +x, -1 toward -x."""
    if attacking_side == "left_to_right":
        return 1.0
    if attacking_side == "right_to_left":
        return -1.0
    return 1.0


def defended_goal_x(attacking_side: str, pitch_length: float) -> float:
    return attack_direction(attacking_side) * (pitch_length / 2.0)


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to segment ab (clamped to segment)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def lane_occluded(def_xy: np.ndarray, a: np.ndarray, b: np.ndarray, thresh: float = LANE_OCCLUSION_M) -> bool:
    if len(def_xy) == 0:
        return False
    for i in range(len(def_xy)):
        if point_to_segment_distance(def_xy[i], a, b) <= thresh:
            return True
    return False


def delaunay_edges(xy: np.ndarray) -> list[tuple[int, int]]:
    n = len(xy)
    if n < 3:
        # complete graph fallback for tiny sets
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(xy)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = int(simplex[a]), int(simplex[b])
            if i > j:
                i, j = j, i
            edges.add((i, j))
    return sorted(edges)


def attack_edge_dangers(
    att_xy: np.ndarray,
    def_xy: np.ndarray,
    attack_dir: float,
    pitch_length: float,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Return edges, danger weights, and occlusion flags."""
    edges = delaunay_edges(att_xy)
    if not edges:
        return [], np.zeros(0), np.zeros(0, dtype=bool)
    dangers = np.zeros(len(edges))
    occluded = np.zeros(len(edges), dtype=bool)
    for k, (i, j) in enumerate(edges):
        a, b = att_xy[i], att_xy[j]
        # Orient so receiver is the more progressive of the two endpoints
        prog_i = float(a[0] * attack_dir)
        prog_j = float(b[0] * attack_dir)
        if prog_j >= prog_i:
            passer, recv = a, b
            progress = prog_j - prog_i
        else:
            passer, recv = b, a
            progress = prog_i - prog_j
        length = float(np.linalg.norm(recv - passer))
        # Shorter + more progressive = more dangerous
        length_factor = float(np.exp(-length / 20.0))
        progress_factor = float(np.clip(progress / max(pitch_length, 1.0), 0.0, 1.0))
        occ = lane_occluded(def_xy, passer, recv)
        occluded[k] = occ
        open_factor = 0.15 if occ else 1.0
        dangers[k] = (0.35 + 0.65 * progress_factor) * length_factor * open_factor
    return edges, dangers, occluded


def hungarian_marking(
    def_xy: np.ndarray,
    att_xy: np.ndarray,
    attack_dir: float,
) -> dict[str, Any]:
    """Minimum-cost assignment with goal-side penalty. Cost in metres."""
    n_d, n_a = len(def_xy), len(att_xy)
    if n_d == 0 or n_a == 0:
        return {
            "pairs": [],
            "n_free": int(n_a),
            "free_idx": list(range(n_a)),
            "mean_assign_dist": float("nan"),
            "assign_dists": np.zeros(0),
            "goal_side": np.zeros(0, dtype=bool),
        }
    dist = cdist(def_xy, att_xy)
    # Goal-side: defender further toward defended goal than attacker
    # (def_x - att_x) * attack_dir > 0
    cost = dist.copy()
    goal_side_mat = np.zeros_like(dist, dtype=bool)
    for i in range(n_d):
        for j in range(n_a):
            gs = (def_xy[i, 0] - att_xy[j, 0]) * attack_dir > 0.0
            goal_side_mat[i, j] = gs
            if not gs:
                cost[i, j] += GOAL_SIDE_PENALTY_M
    # Pad rectangular: if more attackers, some will be unmatched via large cost rows
    # linear_sum_assignment handles rectangular: assigns min(n_d, n_a) pairs
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    assign_dists = []
    goal_sides = []
    assigned_att: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        d = float(dist[r, c])
        gs = bool(goal_side_mat[r, c])
        pairs.append({"def_idx": int(r), "att_idx": int(c), "dist": d, "goal_side": gs})
        assign_dists.append(d)
        goal_sides.append(gs)
        assigned_att.add(int(c))
    free_idx = []
    for j in range(n_a):
        if j not in assigned_att:
            free_idx.append(j)
            continue
        # Find assignment for this attacker
        pair = next(p for p in pairs if p["att_idx"] == j)
        if pair["dist"] > FREE_ASSIGN_M or not pair["goal_side"]:
            free_idx.append(j)
    return {
        "pairs": pairs,
        "n_free": len(free_idx),
        "free_idx": free_idx,
        "mean_assign_dist": float(np.mean(assign_dists)) if assign_dists else float("nan"),
        "assign_dists": np.asarray(assign_dists, dtype=float),
        "goal_side": np.asarray(goal_sides, dtype=bool),
    }


def bipartite_gaussian(
    def_xy: np.ndarray,
    att_xy: np.ndarray,
    sigma: float = SIGMA_M,
    radius: float = SUPPORT_RADIUS_M,
) -> np.ndarray:
    """n_d x n_a interlayer weight matrix."""
    if len(def_xy) == 0 or len(att_xy) == 0:
        return np.zeros((len(def_xy), len(att_xy)))
    d = cdist(def_xy, att_xy)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    w[d > radius] = 0.0
    return w


def supra_structural_pressure(
    w_def: np.ndarray,
    w_att_adj: np.ndarray,
    w_inter: np.ndarray,
    high_danger_mask: np.ndarray,
) -> dict[str, float]:
    """Build supra-Laplacian blocks; report coupling onto high-danger attack nodes.

    Does not rely only on lambda2 of the full supra-Laplacian: the primary
    structural-pressure score is the mean interlayer weight incident on
    high-danger attack nodes (and the fraction of their degree that is interlayer).
    """
    n_d = w_def.shape[0]
    n_a = w_att_adj.shape[0]
    if n_d == 0 or n_a == 0:
        return {
            "structural_pressure": float("nan"),
            "structural_pressure_frac": float("nan"),
            "lambda2_supra": float("nan"),
        }
    L_d = combinatorial_laplacian(w_def)
    L_a = combinatorial_laplacian(w_att_adj)
    # Degree from interlayer
    d_inter_d = w_inter.sum(axis=1)
    d_inter_a = w_inter.sum(axis=0)
    # Supra Laplacian (n_d + n_a)
    n = n_d + n_a
    L = np.zeros((n, n))
    L[:n_d, :n_d] = L_d + np.diag(d_inter_d)
    L[n_d:, n_d:] = L_a + np.diag(d_inter_a)
    L[:n_d, n_d:] = -w_inter
    L[n_d:, :n_d] = -w_inter.T
    evals = np.linalg.eigvalsh(L)
    lambda2_supra = float(evals[1]) if n >= 2 else float("nan")

    if high_danger_mask.any():
        # Mean interlayer weight onto high-danger attack nodes
        pressure_per_node = d_inter_a  # shape (n_a,)
        # Also attack intra degree for fraction
        att_intra = w_att_adj.sum(axis=1)
        hd = high_danger_mask
        sp = float(pressure_per_node[hd].mean())
        tot = pressure_per_node[hd] + att_intra[hd]
        frac = float(np.mean(np.where(tot > 1e-9, pressure_per_node[hd] / tot, 0.0)))
    else:
        sp = float(d_inter_a.mean()) if n_a else float("nan")
        frac = float("nan")
    return {
        "structural_pressure": sp,
        "structural_pressure_frac": frac,
        "lambda2_supra": lambda2_supra,
    }


def on_pitch(xy: np.ndarray, pitch_length: float, pitch_width: float, margin: float = 0.5) -> np.ndarray:
    half_l = pitch_length / 2.0 + margin
    half_w = pitch_width / 2.0 + margin
    return (np.abs(xy[:, 0]) <= half_l) & (np.abs(xy[:, 1]) <= half_w)


# ---------------------------------------------------------------------------
# Frame-level computation
# ---------------------------------------------------------------------------


def compute_frame_features(
    def_xy: np.ndarray,
    def_ids: list[int],
    att_xy: np.ndarray,
    att_ids: list[int],
    ball_xy: np.ndarray | None,
    attacking_side: str,
    pitch_length: float,
) -> dict[str, Any] | None:
    if len(def_xy) < 3 or len(att_xy) < 3:
        return None
    attack_dir = attack_direction(attacking_side)
    w_def = gaussian_weights(def_xy)
    lambda2, fiedler = algebraic_connectivity(w_def)
    fiedler_pos = []
    fiedler_neg = []
    if fiedler is not None:
        for i, pid in enumerate(def_ids):
            if fiedler[i] >= 0:
                fiedler_pos.append(pid)
            else:
                fiedler_neg.append(pid)

    edges, dangers, occluded = attack_edge_dangers(att_xy, def_xy, attack_dir, pitch_length)
    # Attack adjacency from danger weights (symmetric)
    n_a = len(att_xy)
    w_att = np.zeros((n_a, n_a))
    for k, (i, j) in enumerate(edges):
        w_att[i, j] = dangers[k]
        w_att[j, i] = dangers[k]
    node_danger = w_att.sum(axis=1)
    if node_danger.max() > 0:
        thresh = float(np.quantile(node_danger, HIGH_DANGER_QUANTILE))
        high_danger_mask = node_danger >= max(thresh, 1e-12)
    else:
        high_danger_mask = np.zeros(n_a, dtype=bool)

    uncovered = float(dangers[~occluded].sum()) if len(dangers) else 0.0
    total_danger = float(dangers.sum()) if len(dangers) else 0.0
    n_open_edges = int((~occluded).sum()) if len(occluded) else 0

    marking = hungarian_marking(def_xy, att_xy, attack_dir)
    w_inter = bipartite_gaussian(def_xy, att_xy)
    supra = supra_structural_pressure(w_def, w_att, w_inter, high_danger_mask)

    nearest_ball = float("nan")
    if ball_xy is not None and len(def_xy):
        nearest_ball = float(np.min(np.linalg.norm(def_xy - ball_xy.reshape(1, 2), axis=1)))

    return {
        "lambda2_defense": lambda2,
        "fiedler_pos_ids": fiedler_pos,
        "fiedler_neg_ids": fiedler_neg,
        "n_free_attackers": marking["n_free"],
        "free_att_ids": [att_ids[i] for i in marking["free_idx"]],
        "mean_assign_dist": marking["mean_assign_dist"],
        "uncovered_danger": uncovered,
        "total_attack_danger": total_danger,
        "n_open_edges": n_open_edges,
        "n_attack_edges": len(edges),
        "structural_pressure": supra["structural_pressure"],
        "structural_pressure_frac": supra["structural_pressure_frac"],
        "lambda2_supra": supra["lambda2_supra"],
        "nearest_defender_to_ball": nearest_ball,
        "n_def": len(def_xy),
        "n_att": len(att_xy),
        # raw geometry for plotting
        "_def_xy": def_xy,
        "_att_xy": att_xy,
        "_def_ids": def_ids,
        "_att_ids": att_ids,
        "_w_def": w_def,
        "_edges": edges,
        "_dangers": dangers,
        "_occluded": occluded,
        "_marking": marking,
        "_high_danger_mask": high_danger_mask,
        "_attack_dir": attack_dir,
    }


# ---------------------------------------------------------------------------
# Phase / pressure indexing
# ---------------------------------------------------------------------------


def load_organized_phases(match_dir: Path, match_id: str) -> pd.DataFrame:
    ph = pd.read_csv(match_dir / f"{match_id}_phases_of_play.csv")
    ph = ph[ph["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS)].copy()
    ph["match_id"] = match_id
    return ph.reset_index(drop=True)


def build_frame_phase_map(phases: pd.DataFrame) -> dict[int, int]:
    """Map sampled frame -> phase row index (last overlapping phase wins)."""
    mapping: dict[int, int] = {}
    for i, row in phases.iterrows():
        fs, fe = int(row["frame_start"]), int(row["frame_end"])
        # Align to SAMPLE_STRIDE
        start = fs if fs % SAMPLE_STRIDE == 0 else fs + (SAMPLE_STRIDE - fs % SAMPLE_STRIDE)
        for fr in range(start, fe + 1, SAMPLE_STRIDE):
            mapping[fr] = int(i)
    return mapping


def load_pressure_intervals(match_dir: Path, match_id: str) -> list[tuple[int, int, str]]:
    """player_possession intervals with overall_pressure_start."""
    path = match_dir / f"{match_id}_dynamic_events.csv"
    usecols = [
        "event_type",
        "frame_start",
        "frame_end",
        "overall_pressure_start",
    ]
    ev = pd.read_csv(path, usecols=usecols, low_memory=False)
    poss = ev[ev["event_type"] == "player_possession"].dropna(subset=["overall_pressure_start"])
    out: list[tuple[int, int, str]] = []
    for _, r in poss.iterrows():
        out.append((int(r["frame_start"]), int(r["frame_end"]), str(r["overall_pressure_start"])))
    out.sort(key=lambda x: x[0])
    return out


def pressure_at_frame(intervals: list[tuple[int, int, str]], frame: int) -> str | None:
    # Linear scan is fine for sparse lookups if we advance a pointer externally;
    # for random access, binary-ish scan:
    for fs, fe, lab in intervals:
        if fs <= frame <= fe:
            return lab
        if fs > frame:
            break
    return None


class PressureCursor:
    """Advance-only lookup for sorted possession intervals."""

    def __init__(self, intervals: list[tuple[int, int, str]]):
        self.intervals = intervals
        self.i = 0

    def at(self, frame: int) -> str | None:
        while self.i < len(self.intervals) and self.intervals[self.i][1] < frame:
            self.i += 1
        j = self.i
        while j < len(self.intervals) and self.intervals[j][0] <= frame:
            fs, fe, lab = self.intervals[j]
            if fs <= frame <= fe:
                return lab
            j += 1
        return None


# ---------------------------------------------------------------------------
# Streaming match processor
# ---------------------------------------------------------------------------


def iter_match_frame_rows(
    match_id: str,
    data_dir: Path = DATA_DIR,
    keep_geometry: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream one match; yield per-sampled-frame feature dicts."""
    match_dir = data_dir / match_id
    tracking_path = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    meta = load_match_meta(match_dir, match_id)
    confirm_player_id_field(meta, tracking_path)
    phases = load_organized_phases(match_dir, match_id)
    if phases.empty:
        return
    frame_to_phase = build_frame_phase_map(phases)
    needed = set(frame_to_phase)
    if not needed:
        return
    pressure_iv = load_pressure_intervals(match_dir, match_id)
    press_cur = PressureCursor(pressure_iv)
    half_l = meta.pitch_length / 2.0
    half_w = meta.pitch_width / 2.0

    with open(tracking_path) as f:
        for line in f:
            # Cheap frame peek: "frame": N near start
            # Full parse only for needed frames
            # Find frame number without full json when possible
            try:
                # Fast path: frames are sequential; parse fully only if needed
                # Minimal parse
                fr_num_start = line.find('"frame":')
                if fr_num_start < 0:
                    continue
                fr_num_end = line.find(",", fr_num_start)
                frame = int(line[fr_num_start + 8 : fr_num_end].strip())
            except Exception:
                fr = json.loads(line)
                frame = int(fr["frame"])
            else:
                if frame not in needed:
                    continue
                fr = json.loads(line)

            if frame not in needed:
                continue
            possession = fr.get("possession") or {}
            group = possession.get("group")
            if group not in ("home team", "away team"):
                continue
            ball = fr.get("ball_data") or {}
            if ball.get("x") is None or ball.get("y") is None:
                continue
            # Ball roughly on pitch
            if abs(float(ball["x"])) > half_l + 2 or abs(float(ball["y"])) > half_w + 2:
                continue

            phase_idx = frame_to_phase[frame]
            phase = phases.iloc[phase_idx]
            attacking_side = str(phase["attacking_side"])
            tip_id = int(phase["team_in_possession_id"])
            # Attack = team in possession; defense = the other
            att_team = tip_id
            def_team = meta.home_team_id if tip_id == meta.away_team_id else meta.away_team_id

            def_xy_list = []
            def_ids = []
            att_xy_list = []
            att_ids = []
            for p in fr.get("player_data") or []:
                pid = p.get("player_id")
                if pid is None or pid not in meta.players:
                    continue
                if pid in meta.gk_ids:
                    continue
                x, y = p.get("x"), p.get("y")
                if x is None or y is None:
                    continue
                xy = np.array([float(x), float(y)])
                if abs(xy[0]) > half_l + 1 or abs(xy[1]) > half_w + 1:
                    continue
                info = meta.players[pid]
                if info["team_id"] == def_team:
                    def_xy_list.append(xy)
                    def_ids.append(pid)
                elif info["team_id"] == att_team:
                    att_xy_list.append(xy)
                    att_ids.append(pid)

            if len(def_xy_list) < 3 or len(att_xy_list) < 3:
                continue
            def_xy = np.vstack(def_xy_list)
            att_xy = np.vstack(att_xy_list)
            ball_xy = np.array([float(ball["x"]), float(ball["y"])])

            feats = compute_frame_features(
                def_xy, def_ids, att_xy, att_ids, ball_xy, attacking_side, meta.pitch_length
            )
            if feats is None:
                continue

            # Baselines from phase row (interpolate start/end by frame position)
            fs, fe = int(phase["frame_start"]), int(phase["frame_end"])
            t = 0.0 if fe == fs else (frame - fs) / (fe - fs)
            oop_w = (1 - t) * float(phase["team_out_of_possession_width_start"]) + t * float(
                phase["team_out_of_possession_width_end"]
            )
            oop_l = (1 - t) * float(phase["team_out_of_possession_length_start"]) + t * float(
                phase["team_out_of_possession_length_end"]
            )
            pressure = press_cur.at(frame)
            pressure_score = PRESSURE_SCORE.get(pressure, np.nan) if pressure else np.nan

            row = {
                "match_id": match_id,
                "frame": frame,
                "phase_index": int(phase["index"]) if "index" in phase else phase_idx,
                "phase_row": phase_idx,
                "block_type": phase["team_out_of_possession_phase_type"],
                "attacking_side": attacking_side,
                "team_in_possession_id": tip_id,
                "team_in_possession": phase.get("team_in_possession_shortname"),
                "defending_team_id": def_team,
                "lead_to_shot": bool(phase["team_possession_lead_to_shot"]),
                "lead_to_goal": bool(phase["team_possession_lead_to_goal"]),
                "phase_duration": float(phase["duration"]),
                "frame_start": fs,
                "frame_end": fe,
                "oop_width": oop_w,
                "oop_length": oop_l,
                "oop_area": oop_w * oop_l,
                "overall_pressure": pressure,
                "overall_pressure_score": pressure_score,
                "lambda2_defense": feats["lambda2_defense"],
                "n_free_attackers": feats["n_free_attackers"],
                "mean_assign_dist": feats["mean_assign_dist"],
                "uncovered_danger": feats["uncovered_danger"],
                "total_attack_danger": feats["total_attack_danger"],
                "n_open_edges": feats["n_open_edges"],
                "structural_pressure": feats["structural_pressure"],
                "structural_pressure_frac": feats["structural_pressure_frac"],
                "lambda2_supra": feats["lambda2_supra"],
                "nearest_defender_to_ball": feats["nearest_defender_to_ball"],
                "n_def": feats["n_def"],
                "n_att": feats["n_att"],
                "home_name": meta.home_name,
                "away_name": meta.away_name,
            }
            if keep_geometry:
                row["_geometry"] = {k: feats[k] for k in feats if k.startswith("_")}
                row["_meta"] = meta
            yield row


def aggregate_phase_features(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Mean (and last-2s) aggregation of frame features to phase level."""
    if frame_df.empty:
        return frame_df
    keys = ["match_id", "phase_index", "phase_row"]
    agg_cols = [
        "lambda2_defense",
        "n_free_attackers",
        "mean_assign_dist",
        "uncovered_danger",
        "total_attack_danger",
        "n_open_edges",
        "structural_pressure",
        "structural_pressure_frac",
        "lambda2_supra",
        "nearest_defender_to_ball",
        "oop_width",
        "oop_length",
        "oop_area",
        "overall_pressure_score",
    ]

    rows = []
    for (mid, pidx, prow), g in frame_df.groupby(keys, sort=False):
        g = g.sort_values("frame")
        fe = int(g["frame_end"].iloc[0])
        pre_shot_mask = g["frame"] >= fe - 2 * FPS  # last 2 seconds
        pre = g.loc[pre_shot_mask]
        row = {
            "match_id": mid,
            "phase_index": pidx,
            "phase_row": prow,
            "block_type": g["block_type"].iloc[0],
            "team_in_possession_id": g["team_in_possession_id"].iloc[0],
            "team_in_possession": g["team_in_possession"].iloc[0],
            "defending_team_id": g["defending_team_id"].iloc[0],
            "lead_to_shot": bool(g["lead_to_shot"].iloc[0]),
            "lead_to_goal": bool(g["lead_to_goal"].iloc[0]),
            "phase_duration": float(g["phase_duration"].iloc[0]),
            "frame_start": int(g["frame_start"].iloc[0]),
            "frame_end": fe,
            "n_frames": len(g),
            "home_name": g["home_name"].iloc[0],
            "away_name": g["away_name"].iloc[0],
            "overall_pressure_mode": (
                g["overall_pressure"].dropna().mode().iloc[0]
                if g["overall_pressure"].notna().any()
                else None
            ),
        }
        for c in agg_cols:
            row[f"mean_{c}"] = float(g[c].mean(skipna=True))
            if len(pre):
                row[f"pre2s_{c}"] = float(pre[c].mean(skipna=True))
            else:
                row[f"pre2s_{c}"] = float("nan")
            # Delta into last 2s vs earlier part of phase
            early = g.loc[~pre_shot_mask]
            if len(early) and len(pre) and c == "lambda2_defense":
                row["lambda2_drop_pre2s"] = float(early[c].mean()) - float(pre[c].mean())
            elif c == "lambda2_defense":
                row["lambda2_drop_pre2s"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def process_all_matches(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    match_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all matches; write frame and phase tables. Streams one match at a time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = match_ids or discover_match_ids(data_dir)
    phase_parts: list[pd.DataFrame] = []
    frame_path = out_dir / "frame_features.csv"
    if frame_path.exists():
        frame_path.unlink()
    # Write frame features incrementally
    wrote_header = False
    n_frames_total = 0
    for mid in ids:
        print(f"processing {mid} ...", flush=True)
        rows = list(iter_match_frame_rows(mid, data_dir=data_dir, keep_geometry=False))
        if not rows:
            print(f"  no frames", flush=True)
            continue
        fdf = pd.DataFrame(rows)
        # Drop any non-scalar leftovers
        fdf = fdf[[c for c in fdf.columns if not c.startswith("_")]]
        fdf.to_csv(frame_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        n_frames_total += len(fdf)
        pdf = aggregate_phase_features(fdf)
        phase_parts.append(pdf)
        print(f"  frames={len(fdf)} phases={len(pdf)}", flush=True)
        del rows, fdf

    if not phase_parts:
        phase_df = pd.DataFrame()
    else:
        phase_df = pd.concat(phase_parts, ignore_index=True)
    phase_path = out_dir / "phase_features.csv"
    phase_df.to_csv(phase_path, index=False)
    print(f"wrote {phase_path} ({len(phase_df)} phases); frames={n_frames_total}", flush=True)
    frame_df = pd.read_csv(frame_path) if frame_path.exists() else pd.DataFrame()
    return frame_df, phase_df


def load_frame_geometry(
    match_id: str,
    frame: int,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any] | None:
    """Recompute one frame with geometry for plotting."""
    target = int(frame)
    # Snap to sample grid
    if target % SAMPLE_STRIDE != 0:
        target = target - (target % SAMPLE_STRIDE)
    for row in iter_match_frame_rows(match_id, data_dir=data_dir, keep_geometry=True):
        fr = int(row["frame"])
        if fr == target:
            return row
        if fr > target + SAMPLE_STRIDE * 2:
            break
    return None


def possession_timeseries(
    match_id: str,
    frame_start: int,
    frame_end: int,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Frame features for one phase window (geometry stripped)."""
    rows = []
    fs, fe = int(frame_start), int(frame_end)
    for row in iter_match_frame_rows(match_id, data_dir=data_dir, keep_geometry=False):
        fr = int(row["frame"])
        if fr > fe:
            break
        if fr < fs:
            continue
        if int(row["frame_start"]) == fs and int(row["frame_end"]) == fe:
            rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    process_all_matches()
