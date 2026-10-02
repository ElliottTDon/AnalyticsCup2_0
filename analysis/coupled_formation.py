"""Minimal frame geometry for organized-block marking analysis.

Hungarian assignment between defenders and attackers; flags free attackers
(more than 8 m from assigned marker or not goal-side). Used on tracking frames
at goals and shot-ending possessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_DIR = WORKSPACE / "analysis" / "output"

MARK_FREE_DIST_M = 8.0
DEF_EDGE_THRESH = 0.05


@dataclass
class MatchMeta:
    match_id: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    pitch_length: float
    pitch_width: float
    player_acronym: dict[int, str]
    player_team: dict[int, int]

    @property
    def half_length(self) -> float:
        return self.pitch_length / 2

    @property
    def half_width(self) -> float:
        return self.pitch_width / 2


def load_match_meta(data_dir: Path, match_id: str) -> MatchMeta:
    raw = json.loads((data_dir / match_id / f"{match_id}_match.json").read_text())
    player_acronym: dict[int, str] = {}
    player_team: dict[int, int] = {}
    for p in raw["players"]:
        pid = int(p["id"])
        player_acronym[pid] = p["player_role"]["acronym"]
        player_team[pid] = int(p["team_id"])
    home = raw["home_team"]
    away = raw["away_team"]
    return MatchMeta(
        match_id=match_id,
        home_id=int(home["id"]),
        away_id=int(away["id"]),
        home_name=home["short_name"],
        away_name=away["short_name"],
        pitch_length=float(raw.get("pitch_length", 105.0)),
        pitch_width=float(raw.get("pitch_width", 68.0)),
        player_acronym=player_acronym,
        player_team=player_team,
    )


def _tracking_path(data_dir: Path, match_id: str) -> Path:
    return data_dir / match_id / f"{match_id}_tracking_extrapolated.jsonl"


def load_frame_players(data_dir: Path, match_id: str, frame: int) -> dict | None:
    """Return one tracking frame with player_data list, or None."""
    path = _tracking_path(data_dir, match_id)
    if not path.is_file() or path.stat().st_size < 1000:
        return None
    best = None
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("frame") == frame:
                return row
            if row.get("frame", -1) > frame:
                break
    return best


def _proximity_weights(xy: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    n = len(xy)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(xy[i] - xy[j])
            val = np.exp(-(d / sigma) ** 2)
            w[i, j] = w[j, i] = val
    return w


def _laplacian_lambda2(w: np.ndarray) -> float:
    if w.size == 0:
        return float("nan")
    deg = w.sum(axis=1)
    lap = np.diag(deg) - w
    vals = np.linalg.eigvalsh(lap)
    vals = np.sort(vals)
    return float(vals[1]) if len(vals) > 1 else float("nan")


def _goal_side(attacker: np.ndarray, defender: np.ndarray, defending_left: bool) -> bool:
    """Defender is goal-side of attacker (between attacker and own goal)."""
    if defending_left:
        return defender[0] <= attacker[0]
    return defender[0] >= attacker[0]


def hungarian_marking(
    def_xy: np.ndarray,
    att_xy: np.ndarray,
    *,
    defending_left: bool,
) -> dict:
    """Assign defenders to attackers; list free attacker indices."""
    n_def, n_att = len(def_xy), len(att_xy)
    if n_def == 0 or n_att == 0:
        return {
            "assign_att_to_def": {},
            "distances": {},
            "free_idx": list(range(n_att)),
        }
    cost = np.linalg.norm(def_xy[:, None, :] - att_xy[None, :, :], axis=2)
    if n_def >= n_att:
        r_def, c_att = linear_sum_assignment(cost)
        pairs = list(zip(c_att, r_def))
    else:
        r_att, c_def = linear_sum_assignment(cost.T)
        pairs = list(zip(r_att, c_def))

    assign: dict[int, int] = {}
    dists: dict[int, float] = {}
    free: list[int] = []
    matched_att = set()
    for a_idx, d_idx in pairs:
        matched_att.add(a_idx)
        d = float(cost[d_idx, a_idx])
        assign[a_idx] = d_idx
        dists[a_idx] = d
        if d > MARK_FREE_DIST_M or not _goal_side(att_xy[a_idx], def_xy[d_idx], defending_left):
            free.append(int(a_idx))
    for a_idx in range(n_att):
        if a_idx not in matched_att:
            free.append(a_idx)
    return {"assign_att_to_def": assign, "distances": dists, "free_idx": free}


def load_frame_geometry(
    data_dir: Path,
    meta: MatchMeta,
    frame: int,
    *,
    defending_team_id: int,
) -> dict | None:
    row = load_frame_players(data_dir, meta.match_id, frame)
    if not row or not row.get("player_data"):
        return None
    att_id = int(meta.away_id if defending_team_id == meta.home_id else meta.home_id)
    def_xy, att_xy, att_pids = [], [], []
    for pl in row["player_data"]:
        if pl.get("x") is None or pl.get("y") is None:
            continue
        pid = int(pl["player_id"])
        team = meta.player_team.get(pid)
        if team is None:
            continue
        xy = np.array([float(pl["x"]), float(pl["y"])], dtype=float)
        if team == defending_team_id:
            def_xy.append(xy)
        elif team == att_id:
            att_xy.append(xy)
            att_pids.append(pid)
    if len(att_xy) < 2 or len(def_xy) < 2:
        return None
    def_xy_a = np.asarray(def_xy)
    att_xy_a = np.asarray(att_xy)
    # defending team protects x = -half_length (left) or +half_length
    mean_att_x = float(att_xy_a[:, 0].mean())
    defending_left = mean_att_x > 0
    w_def = _proximity_weights(def_xy_a)
    try:
        tri = Delaunay(att_xy_a)
        edges = {tuple(sorted(e)) for s in tri.simplices for e in [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]}
    except Exception:
        edges = set()
    marking = hungarian_marking(def_xy_a, att_xy_a, defending_left=defending_left)
    free_roles = [
        meta.player_acronym.get(att_pids[i], "?")
        for i in marking["free_idx"]
        if i < len(att_pids)
    ]
    return {
        "frame": frame,
        "lambda2_defense": _laplacian_lambda2(w_def),
        "n_free_attackers": len(marking["free_idx"]),
        "free_roles": free_roles,
        "marking": marking,
        "_def_xy": def_xy_a,
        "_att_xy": att_xy_a,
        "_att_pids": att_pids,
    }
