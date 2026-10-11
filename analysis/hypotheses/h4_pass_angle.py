"""Hypothesis 4. Passing angle between a wide player and a central midfielder.

Question: when is that angle large enough to break the press?

The angle is measured from tracking, at the ball carrier, between the other
player in the wide-CM pair and the nearest outfield opponent. Roles come from
the match json. Passes and passing options are separate samples. A degree
threshold is proposed only when the line-break bins show one positive step
and the match-clustered interval for that cut excludes zero.

Run: opendata/.venv/bin/python analysis/hypotheses/h4_pass_angle.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    BOOT_SEED,
    CM_ROLES,
    HIGH_PRESSURE,
    N_BOOT,
    OUT_DIR,
    WIDE_ROLES,
    angle_between,
    attack_sign,
    cluster_diff,
    cluster_mean,
    json_ready,
    load_events,
    load_roster,
    match_ids,
    nearest_opponent,
    pass_angle_from_goal,
    stream_frames,
    supports,
    wide_cm_pair,
)

SUMMARY_PATH = OUT_DIR / "h4_summary.json"
ANGLE_BINS = (0, 30, 60, 90, 180)
ANGLE_CUTS = (30, 45, 60, 90)
PROPOSABLE_CUTS = (30, 60, 90)
MIN_ARM = 20


def _iid(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _role(pid: int | None, roster: dict[int, str]) -> str | None:
    if pid is None:
        return None
    role = roster.get(pid) or ""
    if not role or role == "SUB":
        return None
    return role


def _direction(passer_role: str, receiver_role: str) -> str | None:
    if passer_role in WIDE_ROLES and receiver_role in CM_ROLES:
        return "wide_to_cm"
    if passer_role in CM_ROLES and receiver_role in WIDE_ROLES:
        return "cm_to_wide"
    return None


def _pressure_flag(value) -> float:
    if value in HIGH_PRESSURE:
        return 1.0
    if isinstance(value, str):
        return 0.0
    return float("nan")


def _frame(value) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    fr = int(value)
    if fr < 0:
        return None
    return fr


def _parent_pressure(parent: pd.Series, frame: int) -> float:
    start = _frame(parent.frame_start)
    end = _frame(parent.frame_end)
    if start is None and end is None:
        return _pressure_flag(parent.overall_pressure_end)
    if start is None:
        return _pressure_flag(parent.overall_pressure_end)
    if end is None:
        return _pressure_flag(parent.overall_pressure_start)
    use_end = abs(frame - end) <= abs(frame - start)
    raw = parent.overall_pressure_end if use_end else parent.overall_pressure_start
    return _pressure_flag(raw)


def process_match(match_id: str) -> tuple[list[dict], dict]:
    """One match. Tracking frames for this match are dropped before return."""
    events = load_events(match_id)
    team_of, roster, gk, _plen, _pwid = load_roster(match_id)
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    options = events.loc[events["event_type"] == "passing_option"].copy()
    del events

    if poss["event_id"].duplicated().any():
        poss = poss.drop_duplicates("event_id", keep="first")
    parents = poss.set_index("event_id", drop=False)

    pending: list[dict] = []
    counts = {
        "pass_pairs": 0,
        "option_pairs": 0,
        "missing_frame": 0,
        "missing_xy": 0,
        "missing_opponent": 0,
        "angle_undefined": 0,
    }

    def consider(kind: str, row, passer_role: str, receiver_role: str, frame: int | None, pressure: float, epv):
        if frame is None:
            counts["missing_frame"] += 1
            return
        pending.append({
            "match_id": match_id,
            "kind": kind,
            "frame": frame,
            "passer": _iid(row.player_id) if kind == "pass" else _iid(row.player_in_possession_id),
            "receiver": _iid(row.player_targeted_id) if kind == "pass" else _iid(row.player_id),
            "sign": attack_sign(row.attacking_side),
            "direction": _direction(passer_role, receiver_role),
            "first_line_break": row.first_line_break,
            "high_pressure": pressure,
            "pass_epv": epv,
            "high_block": 1.0 if row.team_out_of_possession_phase_type == "high_block" else 0.0,
        })

    for row in poss.itertuples(index=False):
        if row.end_type != "pass":
            continue
        passer_role = _role(_iid(row.player_id), roster)
        receiver_role = _role(_iid(row.player_targeted_id), roster)
        if not wide_cm_pair(passer_role, receiver_role):
            continue
        counts["pass_pairs"] += 1
        consider(
            "pass",
            row,
            passer_role,
            receiver_role,
            _frame(row.frame_end),
            _pressure_flag(row.overall_pressure_end),
            row.pass_epv_total,
        )

    for row in options.itertuples(index=False):
        passer_role = _role(_iid(row.player_in_possession_id), roster)
        receiver_role = _role(_iid(row.player_id), roster)
        if not wide_cm_pair(passer_role, receiver_role):
            continue
        counts["option_pairs"] += 1
        frame = _frame(row.frame_start)
        pressure = float("nan")
        parent_id = _iid(row.associated_player_possession_event_id)
        if frame is not None and parent_id is not None and parent_id in parents.index:
            parent = parents.loc[parent_id]
            if isinstance(parent, pd.DataFrame):
                parent = parent.iloc[0]
            pressure = _parent_pressure(parent, frame)
        consider(
            "option",
            row,
            passer_role,
            receiver_role,
            frame,
            pressure,
            np.nan,
        )

    del poss, options, parents

    frames = stream_frames(match_id, {rec["frame"] for rec in pending})
    rows: list[dict] = []
    for rec in pending:
        xy = frames.get(rec["frame"]) or {}
        passer = rec["passer"]
        receiver = rec["receiver"]
        p_xy = xy.get(passer) if passer is not None else None
        r_xy = xy.get(receiver) if receiver is not None else None
        if p_xy is None or r_xy is None:
            counts["missing_xy"] += 1
            continue
        opp_id, opp_dist = nearest_opponent(passer, xy, team_of, gk)
        if opp_id is None or opp_id not in xy:
            counts["missing_opponent"] += 1
            continue
        escape = angle_between(p_xy, r_xy, xy[opp_id])
        if not np.isfinite(escape):
            counts["angle_undefined"] += 1
            continue
        sign = rec["sign"]
        shape = pass_angle_from_goal(p_xy, r_xy, sign) if np.isfinite(sign) else float("nan")
        rows.append({
            "match_id": match_id,
            "kind": rec["kind"],
            "direction": rec["direction"],
            "escape_angle": escape,
            "shape_angle": shape,
            "opp_dist": opp_dist,
            "first_line_break": rec["first_line_break"],
            "high_pressure": rec["high_pressure"],
            "pass_epv": rec["pass_epv"],
            "high_block": rec["high_block"],
        })
    del frames
    return rows, counts


def _finite_angle(df: pd.DataFrame, col: str = "escape_angle") -> pd.Series:
    angles = pd.to_numeric(df[col], errors="coerce")
    return angles.where((angles >= 0) & (angles <= 180))


def bin_table(df: pd.DataFrame, angle_col: str, y: str) -> list[dict]:
    out = []
    angles = _finite_angle(df, angle_col)
    for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
        if hi < 180:
            mask = (angles >= lo) & (angles < hi)
        else:
            mask = (angles >= lo) & (angles <= hi)
        sub = df.loc[mask].copy()
        sub["_y"] = pd.to_numeric(sub[y], errors="coerce")
        stat = cluster_mean(sub, "_y")
        out.append({
            "bin": f"{lo}-{hi}",
            "lo": lo,
            "hi": hi,
            "outcome": y,
            "mean": stat["effect"],
            "ci_low": stat["ci_low"],
            "ci_high": stat["ci_high"],
            "n": stat["n"],
            "n_matches": stat["n_matches"],
        })
    return out


def cut_contrast(df: pd.DataFrame, angle_col: str, y: str, cut: float, seed: int) -> dict:
    angles = _finite_angle(df, angle_col)
    work = df.copy()
    work["_treat"] = np.where(angles >= cut, 1.0, 0.0)
    work.loc[~np.isfinite(angles), "_treat"] = np.nan
    work["_y"] = pd.to_numeric(work[y], errors="coerce")
    contrast = cluster_diff(work, "_y", "_treat", seed=seed)
    return {
        "cut": int(cut),
        "outcome": y,
        "supported": supports(contrast, "positive", min_n=MIN_ARM),
        **contrast,
    }


def _bin_means(bins: list[dict]) -> list[float] | None:
    means = []
    for row in bins:
        mean = row["mean"]
        if mean is None or row["n"] < 1 or not np.isfinite(mean):
            return None
        means.append(float(mean))
    return means


def _step_candidate(means: list[float]) -> tuple[int | None, float | None, str]:
    """One positive step at 30, 60, or 90. Otherwise a reason and no boundary."""
    boundaries = list(PROPOSABLE_CUTS)
    jumps = [(boundaries[i], means[i + 1] - means[i]) for i in range(3)]
    ordered = sorted(jumps, key=lambda item: abs(item[1]), reverse=True)
    cut, jump = ordered[0]
    second = abs(ordered[1][1])
    if jump <= 0:
        return None, float(jump), "no_positive_step"
    if second > 0 and jump < 2.0 * second:
        return None, float(jump), "not_a_single_cut"
    idx = boundaries.index(cut)
    pre_max = max(means[: idx + 1])
    post_min = min(means[idx + 1 :])
    if post_min < pre_max + 0.5 * jump:
        return None, float(jump), "reversal"
    return cut, float(jump), "step"


def assess_line_break(df: pd.DataFrame, angle_col: str = "escape_angle") -> dict:
    """Threshold only if the bins step once and that cut's interval excludes zero."""
    bins = bin_table(df, angle_col, "first_line_break")
    scans = [
        cut_contrast(df, angle_col, "first_line_break", cut, BOOT_SEED + int(cut))
        for cut in ANGLE_CUTS
    ]
    by_cut = {row["cut"]: row for row in scans}
    means = _bin_means(bins)
    if means is None:
        return {
            "bins": bins,
            "scans": scans,
            "threshold": None,
            "boundary_deg": None,
            "jump": None,
            "reason": "incomplete_bins",
        }
    boundary, jump, shape = _step_candidate(means)
    if boundary is None:
        return {
            "bins": bins,
            "scans": scans,
            "threshold": None,
            "boundary_deg": None,
            "jump": jump,
            "reason": shape,
        }
    contrast = by_cut[boundary]
    reason = "proposed"
    threshold = contrast
    if not contrast.get("ci_excludes_zero"):
        reason = "interval_includes_zero"
        threshold = None
    elif contrast.get("effect") is None or contrast["effect"] <= 0:
        reason = "effect_not_positive"
        threshold = None
    elif contrast.get("n_treat", 0) < MIN_ARM or contrast.get("n_control", 0) < MIN_ARM:
        reason = "arm_below_min_n"
        threshold = None
    elif not supports(contrast, "positive", min_n=MIN_ARM):
        reason = "interval_includes_zero"
        threshold = None
    else:
        conflict = _direction_conflict(df, angle_col, boundary)
        if conflict:
            reason = "direction_conflict"
            threshold = None
    return {
        "bins": bins,
        "scans": scans,
        "threshold": threshold,
        "boundary_deg": boundary,
        "jump": jump,
        "reason": reason,
    }


def _direction_conflict(df: pd.DataFrame, angle_col: str, cut: int) -> bool:
    """True when wide-to-CM and CM-to-wide both exclude zero with opposite signs."""
    signs = []
    for direction in ("wide_to_cm", "cm_to_wide"):
        sub = df[df["direction"] == direction]
        contrast = cut_contrast(
            sub,
            angle_col,
            "first_line_break",
            cut,
            BOOT_SEED + 17 + int(cut),
        )
        if not contrast.get("ci_excludes_zero"):
            continue
        effect = contrast.get("effect")
        if effect is None or not np.isfinite(effect) or effect == 0:
            continue
        if contrast.get("n_treat", 0) < MIN_ARM or contrast.get("n_control", 0) < MIN_ARM:
            continue
        signs.append(np.sign(effect))
    return len(signs) == 2 and signs[0] != signs[1]


def _at_cut(df: pd.DataFrame, cut: int, y: str, predicted: str, seed: int) -> dict:
    contrast = cut_contrast(df, "escape_angle", y, cut, seed)
    # cut_contrast stores support for a positive effect only. Recompute the flag.
    contrast["supported"] = supports(contrast, predicted, min_n=MIN_ARM)
    contrast["predicted"] = predicted
    return contrast


def _match_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    if df.empty:
        return rows
    for match_id, g in df.groupby("match_id", sort=True):
        passes = g[g["kind"] == "pass"]
        options = g[g["kind"] == "option"]
        rows.append({
            "match_id": str(match_id),
            "n_pass_angle": int(len(passes)),
            "n_pass_line_break": int(pd.to_numeric(passes["first_line_break"], errors="coerce").notna().sum()),
            "n_option_angle": int(len(options)),
            "n_option_line_break": int(pd.to_numeric(options["first_line_break"], errors="coerce").notna().sum()),
        })
    return rows


def _sample_counts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "n_matches": 0}
    line = pd.to_numeric(df["first_line_break"], errors="coerce")
    angled = _finite_angle(df)
    used = df[np.isfinite(angled) & np.isfinite(line)]
    return {
        "n": int(len(used)),
        "n_with_angle": int(np.isfinite(angled).sum()),
        "n_matches": int(used["match_id"].nunique()) if len(used) else 0,
        "n_matches_with_angle": int(df.loc[np.isfinite(angled), "match_id"].nunique()),
    }


def summarize(df: pd.DataFrame, coverage: dict) -> dict:
    passes = df[df["kind"] == "pass"].copy()
    options = df[df["kind"] == "option"].copy()
    assessed = assess_line_break(passes, "escape_angle")
    claim = assessed["threshold"]
    supported = claim is not None
    pass_counts = _sample_counts(passes)
    option_counts = _sample_counts(options)

    at_threshold = None
    if claim is not None:
        cut = int(claim["cut"])
        at_threshold = {
            "cut": cut,
            "first_line_break": claim,
            "high_pressure": _at_cut(passes, cut, "high_pressure", "negative", BOOT_SEED + 301),
            "pass_epv": _at_cut(passes, cut, "pass_epv", "positive", BOOT_SEED + 302),
            "options_first_line_break": _at_cut(options, cut, "first_line_break", "positive", BOOT_SEED + 303),
        }

    option_assessed = assess_line_break(options, "escape_angle")
    shape_bins = bin_table(passes, "shape_angle", "first_line_break")
    pressure_bins = bin_table(passes, "escape_angle", "high_pressure")
    epv_bins = bin_table(passes, "escape_angle", "pass_epv")
    option_pressure_bins = bin_table(options, "escape_angle", "high_pressure")

    if supported:
        effect_name = (
            f"first_line_break rate, escape angle >= {claim['cut']} minus below"
        )
        effect = claim["effect"]
        ci_low = claim["ci_low"]
        ci_high = claim["ci_high"]
        ci_excludes = True
        n_treat = claim["n_treat"]
        n_control = claim["n_control"]
    else:
        effect_name = "no escape-angle cut separated first_line_break"
        effect = None
        ci_low = None
        ci_high = None
        ci_excludes = False
        n_treat = None
        n_control = None

    wide = passes[passes["direction"] == "wide_to_cm"]
    inward = passes[passes["direction"] == "cm_to_wide"]
    return {
        "id": 4,
        "title": "Wide-to-central angle that breaks the press",
        "verdict": "supported" if supported else "not_supported",
        "supported": supported,
        "hypothesis": (
            "When is the passing angle between a wide player and a central midfielder "
            "large enough to break the press?"
        ),
        "angle": {
            "name": "escape_angle",
            "unit": "degrees",
            "definition": (
                "Interior angle at the ball carrier between the other player in the "
                "wide-CM pair and the nearest outfield opponent, excluding goalkeepers. "
                "Zero means the receiver is in the same direction as that opponent. "
                "Larger means the pass points farther away from that opponent. "
                "Positions are taken from the tracking frame at the pass (frame_end) "
                "or at the passing option (frame_start)."
            ),
            "bins_deg": ["0-30", "30-60", "60-90", "90-180"],
            "threshold_deg": None if claim is None else int(claim["cut"]),
            "examined_boundary_deg": assessed["boundary_deg"],
            "bin_jump": assessed["jump"],
            "threshold_reason": assessed["reason"],
        },
        "population": (
            "Wide-to-CM and CM-to-wide player_possession passes. "
            "Passing options with the same role pair are a separate sample and are not pooled."
        ),
        "roles": (
            "player_role.acronym on the match json. "
            "Wide: LB, RB, LWB, RWB, LW, RW, LM, RM. "
            "Central midfield: LDM, RDM, CDM, DM, CM, LCM, RCM, AM, CAM."
        ),
        "n": pass_counts["n"],
        "n_with_angle": pass_counts["n_with_angle"],
        "n_matches": pass_counts["n_matches"],
        "n_treat": n_treat,
        "n_control": n_control,
        "effect_name": effect_name,
        "effect": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": ci_excludes,
        "inference": "match_cluster_bootstrap",
        "n_boot": N_BOOT,
        "coverage": {
            **coverage,
            "n_pass_pairs": int(coverage["pass_pairs"]),
            "n_option_pairs": int(coverage["option_pairs"]),
            "n_passes_wide_to_cm": int(len(wide)),
            "n_passes_cm_to_wide": int(len(inward)),
            "share_high_pressure_passes": _share(passes, "high_pressure"),
            "share_high_block_passes": _share(passes, "high_block"),
        },
        "bins": {
            "passes_first_line_break": assessed["bins"],
            "passes_high_pressure": pressure_bins,
            "passes_pass_epv": epv_bins,
            "options_first_line_break": option_assessed["bins"],
            "options_high_pressure": option_pressure_bins,
            "passes_shape_angle_first_line_break": shape_bins,
        },
        "scans": {
            "passes_escape_angle_line_break": assessed["scans"],
            "options_escape_angle_line_break": option_assessed["scans"],
        },
        "at_threshold": at_threshold,
        "options": {
            "n": option_counts["n"],
            "n_with_angle": option_counts["n_with_angle"],
            "n_matches": option_counts["n_matches"],
            "supported": option_assessed["threshold"] is not None,
            "threshold_deg": None if option_assessed["threshold"] is None else int(option_assessed["threshold"]["cut"]),
            "threshold_reason": option_assessed["reason"],
            "examined_boundary_deg": option_assessed["boundary_deg"],
            "bin_jump": option_assessed["jump"],
            "effect": None if option_assessed["threshold"] is None else option_assessed["threshold"]["effect"],
            "ci_low": None if option_assessed["threshold"] is None else option_assessed["threshold"]["ci_low"],
            "ci_high": None if option_assessed["threshold"] is None else option_assessed["threshold"]["ci_high"],
            "n_treat": None if option_assessed["threshold"] is None else option_assessed["threshold"]["n_treat"],
            "n_control": None if option_assessed["threshold"] is None else option_assessed["threshold"]["n_control"],
        },
        "matches": _match_rows(df),
        "definition": (
            "A threshold is proposed only when the four line-break bins "
            "(0-30, 30-60, 60-90, 90-180 degrees) show a single positive step "
            "at 30, 60, or 90 degrees: the largest adjacent jump is positive, "
            "at least twice the next jump, and later bins stay in the upper half of that jump. "
            "The contrast is the pooled line-break rate at or above that cut minus the rate below it. "
            "The interval resamples matches. Supported only when that interval excludes zero, "
            "the effect is positive, and both arms have at least 20 events. "
            "A smooth gradient, a spike that reverses, or an interval that includes zero "
            "does not produce a threshold. "
            "45 degrees is reported in the scans and is not eligible as the threshold. "
            "Shape angle is the pass direction from the attacking axis and is not used for the verdict. "
            "Pass value is pass_epv_total. Pressure is high or very high. "
            "Line break is first_line_break. No xG."
        ),
        "notes": (
            "Passes are the sample for the verdict because pass EPV is defined on the pass. "
            "Passing options are not pooled with passes: a received option is the same action. "
            "Option pressure is the parent possession's start or end pressure, whichever frame is closer. "
            "The cut is chosen from the same line-break bins it is tested on, so a proposed interval "
            "is the clustered interval at that cut, not a pre-registered contrast."
        ),
    }


def _share(df: pd.DataFrame, col: str) -> float | None:
    if df.empty:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return float(vals.mean())


def main() -> None:
    ids = match_ids()
    if len(ids) != 20:
        raise SystemExit(f"expected 20 matches, found {len(ids)}")
    rows: list[dict] = []
    coverage = {
        "n_matches_read": 0,
        "pass_pairs": 0,
        "option_pairs": 0,
        "missing_frame": 0,
        "missing_xy": 0,
        "missing_opponent": 0,
        "angle_undefined": 0,
    }
    for match_id in ids:
        match_rows, counts = process_match(match_id)
        rows.extend(match_rows)
        coverage["n_matches_read"] += 1
        for key in (
            "pass_pairs",
            "option_pairs",
            "missing_frame",
            "missing_xy",
            "missing_opponent",
            "angle_undefined",
        ):
            coverage[key] += counts[key]
        print(
            f"{match_id} passes={counts['pass_pairs']} options={counts['option_pairs']} "
            f"kept={len(match_rows)}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    summary = summarize(df, coverage)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    print(
        f"verdict={summary['verdict']} n={summary['n']} "
        f"threshold={summary['angle']['threshold_deg']} "
        f"effect={summary['effect']} ci=({summary['ci_low']}, {summary['ci_high']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
