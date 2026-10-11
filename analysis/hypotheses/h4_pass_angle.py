"""Hypothesis 4: passing angle between a wide player and a central midfielder.

Question: when is that angle large enough to break the press?

Sample is wide-to-CM and CM-to-wide passes in an organised high or medium
block (the press). The angle is the absolute degrees between the attacking
direction and the passer-to-teammate vector at the pass frame, computed from
tracking. Each match is streamed and discarded before the next match is read.

The pre-specified effect is the change in first-line-break probability per
10 degrees of absolute angle, with match-clustered standard errors and a
Student-t interval on G-1 degrees of freedom. The hypothesis is supported
only if that interval excludes zero and the estimate is positive (a larger
angle raises the line-break rate). A threshold is reported only when the
binned rates show one dominant cut.

Pass EPV is pass_epv_total (net pass value). Expected goals are not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_PATH = WORKSPACE / "analysis" / "output" / "h4_summary.json"

# These 20 matches have no CM acronym. Central midfield is AM/DM/LDM/RDM.
CM_ROLES = frozenset({"AM", "DM", "LDM", "RDM"})
WIDE_ATTACKER = frozenset({"LW", "RW", "LF", "RF"})
WIDE_MID = frozenset({"LM", "RM"})
FULL_BACK = frozenset({"LB", "RB", "LWB", "RWB"})
WIDE_ROLES = WIDE_ATTACKER | WIDE_MID | FULL_BACK
PRESS_PHASES = frozenset({"high_block", "medium_block"})

BIN_WIDTH = 15
MIN_BIN_N = 20
MIN_BIN_MATCHES = 5
# A cut must be at least this many times a typical adjacent step, and this
# many times the next step that also clears that bar.
CUT_DOMINANCE = 2.0
# Tracking angle and the event interplayer_angle are the same quantity.
# A larger gap means the pass frame did not land on those two players.
MAX_VENDOR_GAP_DEG = 15.0

PASS_COLUMNS = [
    "event_type",
    "end_type",
    "player_id",
    "player_in_possession_id",
    "player_targeted_id",
    "frame_end",
    "peak_passing_option_frame",
    "attacking_side",
    "organised_defense",
    "team_out_of_possession_phase_type",
    "first_line_break",
    "interplayer_angle",
    "overall_pressure_end",
    "overall_pressure_end_id",
    "pass_epv_total",
    "pass_outcome",
]


def discover_match_ids(data_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_dir():
            continue
        mid = path.name
        needed = [
            path / f"{mid}_match.json",
            path / f"{mid}_dynamic_events.csv",
            path / f"{mid}_tracking_extrapolated.jsonl",
        ]
        if all(p.is_file() for p in needed):
            ids.append(mid)
    return ids


def load_roles(match_dir: Path, match_id: str) -> dict[int, str]:
    with open(match_dir / f"{match_id}_match.json") as handle:
        raw = json.load(handle)
    roles: dict[int, str] = {}
    for player in raw["players"]:
        role = player.get("player_role") or {}
        roles[int(player["id"])] = role.get("acronym") or ""
    return roles


def wide_group(role: str) -> str | None:
    if role in WIDE_ATTACKER:
        return "wide_attacker"
    if role in WIDE_MID:
        return "wide_midfield"
    if role in FULL_BACK:
        return "full_back"
    return None


def pair_direction(passer_role: str, receiver_role: str) -> str | None:
    passer_wide = passer_role in WIDE_ROLES
    receiver_wide = receiver_role in WIDE_ROLES
    passer_cm = passer_role in CM_ROLES
    receiver_cm = receiver_role in CM_ROLES
    if passer_wide and receiver_cm:
        return "wide_to_cm"
    if passer_cm and receiver_wide:
        return "cm_to_wide"
    return None


def parse_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_candidates(match_dir: Path, match_id: str, roles: dict[int, str]) -> list[dict]:
    """Wide-CM passes and passing options. No tracking is read here."""
    path = match_dir / f"{match_id}_dynamic_events.csv"
    frame = pd.read_csv(path, usecols=PASS_COLUMNS, low_memory=False)
    rows: list[dict] = []
    for record in frame.itertuples(index=False):
        kind = None
        if record.event_type == "player_possession" and record.end_type == "pass":
            kind = "pass"
            passer_id = parse_int(str(record.player_id) if pd.notna(record.player_id) else "")
            receiver_id = parse_int(
                str(record.player_targeted_id) if pd.notna(record.player_targeted_id) else ""
            )
            frame_id = parse_int(str(record.frame_end) if pd.notna(record.frame_end) else "")
        elif record.event_type == "passing_option":
            kind = "passing_option"
            passer_id = parse_int(
                str(record.player_in_possession_id)
                if pd.notna(record.player_in_possession_id)
                else ""
            )
            receiver_id = parse_int(str(record.player_id) if pd.notna(record.player_id) else "")
            frame_id = parse_int(
                str(record.peak_passing_option_frame)
                if pd.notna(record.peak_passing_option_frame)
                else ""
            )
        else:
            continue
        if kind is None or passer_id is None or receiver_id is None or frame_id is None:
            continue
        if passer_id == receiver_id:
            continue
        passer_role = roles.get(passer_id)
        receiver_role = roles.get(receiver_id)
        if not passer_role or not receiver_role:
            continue
        direction = pair_direction(passer_role, receiver_role)
        if direction is None:
            continue
        if record.organised_defense is not True and str(record.organised_defense) != "True":
            continue
        line_raw = record.first_line_break
        if line_raw is True or str(line_raw) == "True":
            line_break = 1
        elif line_raw is False or str(line_raw) == "False":
            line_break = 0
        else:
            continue
        wide_role = passer_role if direction == "wide_to_cm" else receiver_role
        vendor = parse_float(
            "" if pd.isna(record.interplayer_angle) else str(record.interplayer_angle)
        )
        pressure_id = None
        pressure_label = None
        pass_epv = None
        if kind == "pass":
            pressure_id = parse_float(
                ""
                if pd.isna(record.overall_pressure_end_id)
                else str(record.overall_pressure_end_id)
            )
            if pd.notna(record.overall_pressure_end) and str(record.overall_pressure_end) != "":
                pressure_label = str(record.overall_pressure_end)
            pass_epv = parse_float(
                "" if pd.isna(record.pass_epv_total) else str(record.pass_epv_total)
            )
        rows.append(
            {
                "match_id": match_id,
                "kind": kind,
                "frame": frame_id,
                "passer_id": passer_id,
                "receiver_id": receiver_id,
                "passer_role": passer_role,
                "receiver_role": receiver_role,
                "direction": direction,
                "wide_group": wide_group(wide_role),
                "attacking_side": str(record.attacking_side),
                "phase": str(record.team_out_of_possession_phase_type),
                "line_break": line_break,
                "pressure_id": pressure_id,
                "pressure_label": pressure_label,
                "pass_epv": pass_epv,
                "vendor_angle": vendor,
            }
        )
    return rows


def frame_number(line: str) -> int | None:
    prefix = '{"frame": '
    if not line.startswith(prefix):
        return None
    end = line.find(",", len(prefix))
    if end < 0:
        return None
    try:
        return int(line[len(prefix) : end])
    except ValueError:
        return None


def stream_positions(tracking_path: Path, wanted: dict[int, set[int]]) -> dict[int, dict[int, tuple[float, float]]]:
    """Read one tracking file. Keep coordinates only for requested frames and players."""
    found: dict[int, dict[int, tuple[float, float]]] = {}
    if not wanted:
        return found
    remaining = set(wanted)
    with open(tracking_path) as handle:
        first = handle.readline()
        if first.startswith("version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(f"Tracking file is a Git LFS pointer: {tracking_path}")
        handle.seek(0)
        for line in handle:
            frame = frame_number(line)
            if frame not in remaining:
                continue
            payload = json.loads(line)
            needed = wanted[frame]
            coords: dict[int, tuple[float, float]] = {}
            for player in payload.get("player_data") or []:
                player_id = player.get("player_id")
                if player_id not in needed:
                    continue
                x = player.get("x")
                y = player.get("y")
                if x is None or y is None:
                    continue
                coords[int(player_id)] = (float(x), float(y))
            found[frame] = coords
            remaining.discard(frame)
            if not remaining:
                break
    return found


def attacking_angle(dx: float, dy: float, attacking_side: str) -> float:
    """Degrees from the attacking direction, anticlockwise, in (-180, 180]."""
    if attacking_side == "right_to_left":
        dx = -dx
        dy = -dy
    elif attacking_side != "left_to_right":
        raise ValueError(f"Unknown attacking side: {attacking_side}")
    return math.degrees(math.atan2(dy, dx))


def circular_diff(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def attach_angles(candidates: list[dict], positions: dict[int, dict[int, tuple[float, float]]]) -> list[dict]:
    kept: list[dict] = []
    for row in candidates:
        coords = positions.get(row["frame"]) or {}
        passer = coords.get(row["passer_id"])
        receiver = coords.get(row["receiver_id"])
        if passer is None or receiver is None:
            continue
        signed = attacking_angle(
            receiver[0] - passer[0],
            receiver[1] - passer[1],
            row["attacking_side"],
        )
        row = dict(row)
        row["angle"] = signed
        row["abs_angle"] = min(180.0, abs(signed))
        kept.append(row)
    return kept


def collect_all(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Stream each match, then drop its tracking coordinates."""
    match_ids = discover_match_ids(data_dir)
    if len(match_ids) != 20:
        raise SystemExit(f"Expected 20 matches, found {len(match_ids)} in {data_dir}")
    records: list[dict] = []
    angle_gaps: list[float] = []
    n_candidates = 0
    n_missing_tracking = 0
    matches_without_cm: list[str] = []
    for match_id in match_ids:
        match_dir = data_dir / match_id
        roles = load_roles(match_dir, match_id)
        if not any(role in CM_ROLES for role in roles.values()):
            matches_without_cm.append(match_id)
        candidates = read_candidates(match_dir, match_id, roles)
        n_candidates += len(candidates)
        wanted: dict[int, set[int]] = {}
        for row in candidates:
            bucket = wanted.setdefault(row["frame"], set())
            bucket.add(row["passer_id"])
            bucket.add(row["receiver_id"])
        positions = stream_positions(
            match_dir / f"{match_id}_tracking_extrapolated.jsonl",
            wanted,
        )
        angled = attach_angles(candidates, positions)
        n_missing_tracking += len(candidates) - len(angled)
        for row in angled:
            if row["vendor_angle"] is not None:
                angle_gaps.append(circular_diff(row["angle"], row["vendor_angle"]))
            records.append(row)
        del positions
        del candidates
        print(f"{match_id}: kept {len(angled)} wide-CM lanes", flush=True)
    frame = pd.DataFrame.from_records(records)
    gaps = np.asarray(angle_gaps, dtype=float) if angle_gaps else np.asarray([])
    meta = {
        "n_matches_scanned": len(match_ids),
        "match_ids": match_ids,
        "n_candidates": n_candidates,
        "n_with_tracking_angle": int(len(frame)),
        "n_missing_tracking_position": n_missing_tracking,
        "vendor_angle_median_abs_diff_deg": (
            float(np.median(np.abs(gaps))) if len(gaps) else None
        ),
        "vendor_angle_p95_abs_diff_deg": (
            float(np.percentile(np.abs(gaps), 95)) if len(gaps) else None
        ),
        "matches_without_central_midfield": matches_without_cm,
    }
    if meta["vendor_angle_median_abs_diff_deg"] is None or meta["vendor_angle_median_abs_diff_deg"] > 1.0:
        raise SystemExit(
            "Tracking angle does not match SkillCorner interplayer_angle: "
            f"{meta['vendor_angle_median_abs_diff_deg']}"
        )
    return frame, meta


def cluster_fit(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    """OLS with match-clustered CR1 standard errors. x already includes a constant."""
    if len(y) < 3 or len(np.unique(groups)) < 2:
        return None
    result = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    return result


def t_interval(estimate: float, std_error: float, n_clusters: int) -> tuple[float, float] | None:
    if n_clusters < 2 or not math.isfinite(estimate) or not math.isfinite(std_error):
        return None
    if std_error < 0:
        return None
    critical = float(stats.t.ppf(0.975, n_clusters - 1))
    if not math.isfinite(critical):
        return None
    return estimate - critical * std_error, estimate + critical * std_error


def excludes_zero(interval: tuple[float, float] | None) -> bool:
    if interval is None:
        return False
    low, high = interval
    if not math.isfinite(low) or not math.isfinite(high):
        return False
    return high < 0 or low > 0


def slope_effect(frame: pd.DataFrame, outcome: str) -> dict:
    sub = frame.dropna(subset=[outcome, "abs_angle", "match_id"]).copy()
    n = int(len(sub))
    n_matches = int(sub["match_id"].nunique()) if n else 0
    empty = {
        "outcome": outcome,
        "estimate": None,
        "std_error": None,
        "interval": None,
        "interval_excludes_zero": False,
        "n": n,
        "n_matches": n_matches,
        "df": max(n_matches - 1, 0),
    }
    if n < 3 or n_matches < 2:
        return empty
    y = sub[outcome].to_numpy(dtype=float)
    slope_x = sub["abs_angle"].to_numpy(dtype=float) / 10.0
    design = np.column_stack([np.ones(n), slope_x])
    groups = sub["match_id"].to_numpy()
    fitted = cluster_fit(y, design, groups)
    if fitted is None:
        return empty
    estimate = float(fitted.params[1])
    std_error = float(fitted.bse[1])
    interval = t_interval(estimate, std_error, n_matches)
    return {
        "outcome": outcome,
        "estimand": "change per 10 degrees of absolute passing angle",
        "estimate": estimate,
        "std_error": std_error,
        "interval": list(interval) if interval else None,
        "interval_level": 0.95,
        "interval_excludes_zero": excludes_zero(interval),
        "n": n,
        "n_matches": n_matches,
        "df": n_matches - 1,
        "cluster": "match_id",
        "se": "cluster CR1, Student-t critical value with G-1 df",
    }


def cluster_mean(sub: pd.DataFrame, outcome: str) -> dict:
    y = sub[outcome].dropna()
    aligned = sub.loc[y.index]
    n = int(len(aligned))
    n_matches = int(aligned["match_id"].nunique()) if n else 0
    if n == 0:
        return {"mean": None, "interval": None, "n": 0, "n_matches": 0}
    estimate = float(y.mean())
    if n_matches < 2:
        return {"mean": estimate, "interval": None, "n": n, "n_matches": n_matches}
    design = np.ones((n, 1))
    fitted = cluster_fit(y.to_numpy(dtype=float), design, aligned["match_id"].to_numpy())
    if fitted is None:
        return {"mean": estimate, "interval": None, "n": n, "n_matches": n_matches}
    interval = t_interval(float(fitted.params[0]), float(fitted.bse[0]), n_matches)
    return {
        "mean": float(fitted.params[0]),
        "interval": list(interval) if interval else None,
        "n": n,
        "n_matches": n_matches,
    }


def bin_edges() -> list[int]:
    return list(range(0, 180 + BIN_WIDTH, BIN_WIDTH))


def slice_bin(frame: pd.DataFrame, low: int, high: int) -> pd.DataFrame:
    if high >= 180:
        mask = (frame["abs_angle"] >= low) & (frame["abs_angle"] <= high)
    else:
        mask = (frame["abs_angle"] >= low) & (frame["abs_angle"] < high)
    return frame.loc[mask]


def outcome_bins(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    edges = bin_edges()
    for low, high in zip(edges[:-1], edges[1:]):
        sub = slice_bin(frame, low, high)
        line = cluster_mean(sub, "line_break") if len(sub) else {
            "mean": None,
            "interval": None,
            "n": 0,
            "n_matches": 0,
        }
        pressure = cluster_mean(sub.dropna(subset=["pressure_id"]), "pressure_id") if len(sub) else {
            "mean": None,
            "interval": None,
            "n": 0,
            "n_matches": 0,
        }
        epv = cluster_mean(sub.dropna(subset=["pass_epv"]), "pass_epv") if len(sub) else {
            "mean": None,
            "interval": None,
            "n": 0,
            "n_matches": 0,
        }
        eligible = line["n"] >= MIN_BIN_N and line["n_matches"] >= MIN_BIN_MATCHES
        rows.append(
            {
                "abs_angle_lo": low,
                "abs_angle_hi": high,
                "n": int(len(sub)),
                "n_matches": int(sub["match_id"].nunique()) if len(sub) else 0,
                "line_break_rate": line["mean"],
                "line_break_interval": line["interval"],
                "mean_pressure_id": pressure["mean"],
                "pressure_interval": pressure["interval"],
                "pressure_n": pressure["n"],
                "mean_pass_epv": epv["mean"],
                "pass_epv_interval": epv["interval"],
                "pass_epv_n": epv["n"],
                "eligible_for_cut": eligible,
            }
        )
    return rows


def adjacent_contrast(left: pd.DataFrame, right: pd.DataFrame, outcome: str) -> dict | None:
    left_ok = left.dropna(subset=[outcome])
    right_ok = right.dropna(subset=[outcome])
    if len(left_ok) == 0 or len(right_ok) == 0:
        return None
    pooled = pd.concat([left_ok, right_ok], ignore_index=True)
    y = pooled[outcome].to_numpy(dtype=float)
    high = np.concatenate([np.zeros(len(left_ok)), np.ones(len(right_ok))])
    n_matches = int(pooled["match_id"].nunique())
    if len(pooled) < 3 or n_matches < 2:
        return None
    design = np.column_stack([np.ones(len(pooled)), high])
    fitted = cluster_fit(y, design, pooled["match_id"].to_numpy())
    if fitted is None:
        return None
    estimate = float(fitted.params[1])
    std_error = float(fitted.bse[1])
    interval = t_interval(estimate, std_error, n_matches)
    return {
        "estimate": estimate,
        "std_error": std_error,
        "interval": list(interval) if interval else None,
        "interval_excludes_zero": excludes_zero(interval),
        "n": int(len(pooled)),
        "n_matches": n_matches,
    }


def is_spike(bins: list[dict], index: int, jump: float) -> bool:
    """A one-bin spike rebounds past the midpoint of the step. A cut does not."""
    if index + 2 >= len(bins):
        return False
    beyond = bins[index + 2].get("line_break_rate")
    left = bins[index].get("line_break_rate")
    right = bins[index + 1].get("line_break_rate")
    if beyond is None or left is None or right is None:
        return False
    if bins[index + 2]["n"] < 10:
        return False
    midpoint = (left + right) / 2.0
    if jump < 0 and beyond > midpoint:
        return True
    if jump > 0 and beyond < midpoint:
        return True
    return False


def find_cut(frame: pd.DataFrame, bins: list[dict]) -> dict:
    edges = bin_edges()
    contrasts: list[dict] = []
    for index, ((low, high), left_bin, right_bin) in enumerate(
        zip(zip(edges[:-1], edges[1:]), bins, bins[1:])
    ):
        if not left_bin["eligible_for_cut"] or not right_bin["eligible_for_cut"]:
            continue
        contrast = adjacent_contrast(
            slice_bin(frame, low, high),
            slice_bin(frame, high, right_bin["abs_angle_hi"]),
            "line_break",
        )
        if contrast is None:
            continue
        contrast["boundary_deg"] = high
        contrast["bin_index"] = index
        contrast["spike"] = is_spike(bins, index, contrast["estimate"])
        contrasts.append(contrast)
    typical_step = float(np.median([abs(row["estimate"]) for row in contrasts])) if contrasts else 0.0
    for row in contrasts:
        row["above_typical_step"] = bool(
            typical_step > 0 and abs(row["estimate"]) >= CUT_DOMINANCE * typical_step
        )
    significant = [
        row
        for row in contrasts
        if row["interval_excludes_zero"] and not row["spike"] and row["above_typical_step"]
    ]
    significant.sort(key=lambda row: abs(row["estimate"]), reverse=True)
    proposed = None
    if not contrasts:
        reason = "No adjacent bins were large enough to compare."
    elif not any(row["interval_excludes_zero"] and not row["spike"] for row in contrasts):
        reason = (
            "No adjacent bin contrast has a cluster interval that excludes zero. "
            "The line-break rate changes gradually with angle."
        )
    elif not significant:
        reason = (
            "Any adjacent contrast whose interval excludes zero is no larger than a typical "
            "step along the gradient, so it is not a cut."
        )
    else:
        top = significant[0]
        dominant = len(significant) == 1 or abs(top["estimate"]) >= CUT_DOMINANCE * abs(
            significant[1]["estimate"]
        )
        if dominant:
            proposed = {
                "boundary_deg": top["boundary_deg"],
                "effect_high_minus_low": top["estimate"],
                "interval": top["interval"],
                "interval_excludes_zero": True,
                "n": top["n"],
                "n_matches": top["n_matches"],
                "meaning": (
                    "First-line break rate changes at this absolute angle. "
                    "The step is lower on the large-angle side."
                    if top["estimate"] < 0
                    else "First-line break rate is higher once absolute angle passes this boundary."
                ),
            }
            reason = (
                f"One dominant adjacent contrast at {top['boundary_deg']} degrees "
                "is at least twice a typical step and its cluster interval excludes zero."
            )
        else:
            reason = (
                "More than one step is large enough to be a cut and none dominates, "
                "so no single threshold is proposed."
            )
    return {
        "proposed": proposed is not None,
        "threshold_degrees": None if proposed is None else proposed["boundary_deg"],
        "rule": (
            f"{BIN_WIDTH}-degree bins; both sides need n>={MIN_BIN_N} and "
            f">={MIN_BIN_MATCHES} matches. Propose a boundary only if the high-minus-low "
            "cluster interval excludes zero, the step is not a one-bin spike, the step is "
            f"at least {CUT_DOMINANCE:.0f} times the median adjacent step, and it is the "
            f"only such step or at least {CUT_DOMINANCE:.0f} times the next."
        ),
        "reason": reason,
        "cut": proposed,
        "adjacent_contrasts": [
            {
                "boundary_deg": row["boundary_deg"],
                "effect_high_minus_low": row["estimate"],
                "interval": row["interval"],
                "interval_excludes_zero": row["interval_excludes_zero"],
                "spike": row["spike"],
                "above_typical_step": row["above_typical_step"],
                "n": row["n"],
                "n_matches": row["n_matches"],
            }
            for row in contrasts
        ],
    }


def pressure_label_map(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    sub = frame.dropna(subset=["pressure_id"])
    for label, group in sub.groupby(sub["pressure_label"].fillna("missing")):
        ids = sorted({int(value) for value in group["pressure_id"].dropna().unique()})
        mapping[str(label)] = ",".join(str(value) for value in ids)
    return mapping


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 8)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def reliable_angle(frame: pd.DataFrame) -> pd.Series:
    """Keep lanes whose tracking angle agrees with the event angle."""
    gap = circular_diff_series(frame["angle"], frame["vendor_angle"])
    missing_vendor = frame["vendor_angle"].isna()
    return missing_vendor | (gap.abs() <= MAX_VENDOR_GAP_DEG)


def circular_diff_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left - right + 180.0) % 360.0 - 180.0


def primary_sample(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (frame["kind"] == "pass")
        & (frame["phase"].isin(PRESS_PHASES))
        & frame["abs_angle"].notna()
        & frame["line_break"].notna()
        & reliable_angle(frame)
    )
    return frame.loc[mask].copy()


def build_summary(frame: pd.DataFrame, meta: dict) -> dict:
    sample = primary_sample(frame)
    line = slope_effect(sample, "line_break")
    pressure = slope_effect(sample, "pressure_id")
    epv = slope_effect(sample, "pass_epv")
    bins = outcome_bins(sample)
    cut = find_cut(sample, bins)
    positive = False
    if line["interval"] is not None and line["estimate"] is not None:
        positive = line["interval"][0] > 0
    supported = bool(line["interval_excludes_zero"] and positive)
    verdict = "supported" if supported else "not_supported"

    no_fullback = sample.loc[sample["wide_group"] != "full_back"]
    high_only = sample.loc[sample["phase"] == "high_block"]
    linked = frame.loc[frame["abs_angle"].notna() & frame["line_break"].notna() & reliable_angle(frame)]
    options = linked.loc[
        (linked["kind"] == "passing_option") & (linked["phase"].isin(PRESS_PHASES))
    ]
    all_organised_passes = linked.loc[linked["kind"] == "pass"]
    press_before_gap = frame.loc[
        (frame["kind"] == "pass")
        & (frame["phase"].isin(PRESS_PHASES))
        & frame["abs_angle"].notna()
        & frame["line_break"].notna()
    ]

    if line["interval"] is None:
        reason = "The line-break interval could not be estimated."
    elif not line["interval_excludes_zero"]:
        reason = "The match-clustered interval for the line-break slope includes zero."
    elif line["estimate"] < 0:
        reason = (
            "The interval excludes zero, but the slope is negative: a larger absolute "
            "angle lowers the first-line break rate. A larger angle is not what breaks the press."
        )
    else:
        reason = "The interval excludes zero and a larger absolute angle raises the first-line break rate."

    threshold_text = (
        "No angle threshold is proposed."
        if not cut["proposed"]
        else f"A cut is proposed at {cut['threshold_degrees']} degrees absolute angle."
    )
    estimate = line["estimate"]
    interval = line["interval"]
    if estimate is None or interval is None:
        effect_text = "Effect was not estimated."
    else:
        effect_text = (
            f"Effect {estimate:.4f} change in first-line-break probability per 10 degrees "
            f"(95% cluster interval [{interval[0]:.4f}, {interval[1]:.4f}]), "
            f"n={line['n']} passes, {line['n_matches']} matches."
        )

    summary = {
        "hypothesis_id": "h4",
        "question": (
            "When is the passing angle between a wide player and a central midfielder "
            "large enough to break the press?"
        ),
        "verdict": verdict,
        "supported": supported,
        "verdict_reason": reason,
        "n": line["n"],
        "n_matches": line["n_matches"],
        "effect": line["estimate"],
        "interval": line["interval"],
        "interval_level": 0.95,
        "interval_excludes_zero": line["interval_excludes_zero"],
        "effect_unit": "change in P(first line break) per 10 degrees of absolute passing angle",
        "cluster": "match_id",
        "angle_result": {
            "definition": (
                "Absolute degrees between the attacking direction and the vector from the "
                "passer to the wide or central-midfield teammate at the pass frame. "
                "0 is straight toward the opponent goal; 180 is straight back. "
                "Computed from streamed tracking."
            ),
            "hypothesis_sign": "positive",
            "effect_per_10_degrees": line["estimate"],
            "interval": line["interval"],
            "threshold_degrees": cut["threshold_degrees"],
            "threshold_proposed": cut["proposed"],
            "summary": f"{verdict}. {effect_text} {threshold_text}",
        },
        "line_break": line,
        "pressure": {
            **pressure,
            "field": "overall_pressure_end_id",
            "scale": "1 least pressure to 5 most pressure, on the passer at the pass",
            "label_to_id": pressure_label_map(sample),
        },
        "pass_epv": {
            **epv,
            "field": "pass_epv_total",
            "definition": "Net pass EPV, pass value for minus pass value against. Not xG.",
        },
        "bins": bins,
        "threshold": cut,
        "sample": {
            "unit": "player_possession pass between a wide player and a central midfielder",
            "press_phases": sorted(PRESS_PHASES),
            "organised_defense": True,
            "directions": {
                key: int(value)
                for key, value in sample["direction"].value_counts().to_dict().items()
            },
            "wide_groups": {
                key: int(value)
                for key, value in sample["wide_group"].value_counts().to_dict().items()
            },
            "phases": {
                key: int(value) for key, value in sample["phase"].value_counts().to_dict().items()
            },
            "roles": {
                "wide": sorted(WIDE_ROLES),
                "central_midfield": sorted(CM_ROLES),
                "note": (
                    "Match JSON has no CM acronym. Central midfield is AM, DM, LDM, and RDM. "
                    "Wide is wide attackers, LM/RM, and full backs or wing backs."
                ),
            },
            "n_matches_scanned": meta["n_matches_scanned"],
            "matches_without_central_midfield": meta["matches_without_central_midfield"],
            "n_press_passes_dropped_for_angle_mismatch": int(len(press_before_gap) - len(sample)),
            "tracking": {
                "streamed": True,
                "held_all_matches_in_memory": False,
                "vendor_angle_median_abs_diff_deg": meta["vendor_angle_median_abs_diff_deg"],
                "vendor_angle_p95_abs_diff_deg": meta["vendor_angle_p95_abs_diff_deg"],
                "max_vendor_gap_deg": MAX_VENDOR_GAP_DEG,
                "n_candidates": meta["n_candidates"],
                "n_missing_tracking_position": meta["n_missing_tracking_position"],
            },
        },
        "sensitivity": {
            "passes_excluding_full_backs": slope_effect(no_fullback, "line_break"),
            "high_block_only": slope_effect(high_only, "line_break"),
            "passing_options_same_filter": slope_effect(options, "line_break"),
            "all_organised_passes": slope_effect(all_organised_passes, "line_break"),
        },
        "notes": [
            "Supported only if the pre-specified line-break slope interval excludes zero and is positive.",
            "Pressure and pass EPV are reported on the same passes. They are not the verdict.",
            "xG is not used.",
            "The threshold search is descriptive. It is not the tested effect.",
        ],
    }
    return summary


def main() -> None:
    frame, meta = collect_all(DATA_DIR)
    summary = build_summary(frame, meta)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(json_ready(summary), handle, indent=2)
        handle.write("\n")
    print(summary["angle_result"]["summary"], flush=True)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
