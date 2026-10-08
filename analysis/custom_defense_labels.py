"""High-value custom labels from defensive positioning (Analytics Cup).

Event labels across all matches + optional pose attention labels on the
two SkillCorner body-pose matches when archives are present.

Labels (from prior research / cup notes):
  1. coverage_debt          — free attackers + press/far-side structure at danger
  2. compact_but_broken     — tight OOP box + inside shape / beaten / free attackers
  3. weak_side_press_leak   — pressing chain on one flank + far-side runner
  4. step_vs_hold           — line push/break vs force-back / reduce-danger engagements
  5. phase_outcome_dense    — regain / disruption / progression / shot (denser than binary shot)
  6. proximity_intent_debt  — close marker beaten by movement (pose-free proxy)
  7. ball_aware / runner_aware / scan_lag — shoulder facing (pose matches only)

Writes analysis/output/labels/* and docs/custom-labels*.html.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_DIR = WORKSPACE / "analysis" / "output"
LABEL_DIR = OUT_DIR / "labels"
FIG_DIR = OUT_DIR / "figures" / "labels"
DOCS = WORKSPACE / "docs"
DOCS_FIG = DOCS / "eda" / "figures" / "labels"
HUN_PATH = OUT_DIR / "hungarian_free_roles.csv"
DEBT_PATH = OUT_DIR / "coverage_debt_events.csv"
POSE_DIR = WORKSPACE / "data" / "bodypose"
POSE_MATCHES = (1925299, 1996435)

ORGANIZED = ("low_block", "medium_block", "high_block")
ATTACK_ROLES = {"CF", "RW", "LW", "AM", "RM", "LM", "RDM", "LDM", "LF", "RF", "SS"}
WIDE_LEFT = {"wide_left", "half_space_left", "left_halfspace"}
WIDE_RIGHT = {"wide_right", "half_space_right", "right_halfspace"}
PROGRESS_PHASES = {"create", "finish", "build_up", "direct"}

# Pose gates (SkillCorner tutorial defaults)
MAX_ERR_CM = 15.0
MIN_SHOULDER_M, MAX_SHOULDER_M = 0.25, 0.55
BALL_AWARE_DEG = 60.0  # |heading − bearing_to_ball| < this ⇒ ball-aware

CSS = """
:root { color-scheme: light dark; }
body { margin: 0 auto; max-width: 44rem; padding: 1.1rem 0.9rem 3rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55; font-size: 1.05rem; }
h1 { font-size: 1.45rem; margin: 0 0 0.4rem; }
h2 { font-size: 1.12rem; margin: 1.6rem 0 0.55rem; padding-top: 0.45rem; border-top: 1px solid #ddd; }
@media (prefers-color-scheme: dark) { h2 { border-top-color: #444; } }
.mission { margin: 0.7rem 0 1rem; padding: 0.75rem 0.85rem; border-left: 3px solid #888;
  background: rgba(127,127,127,0.08); }
.meta { color: #666; font-size: 0.92rem; }
@media (prefers-color-scheme: dark) { .meta { color: #aaa; } }
figure { margin: 0 0 1.25rem; }
img { width: 100%; height: auto; border-radius: 4px; background: #f3f3f3; display: block; }
figcaption { margin-top: 0.4rem; font-size: 0.95rem; color: #444; }
@media (prefers-color-scheme: dark) { figcaption { color: #bbb; } img { background: #222; } }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 0.75rem 0 1.25rem; }
th, td { text-align: left; padding: 0.4rem 0.3rem; border-bottom: 1px solid #ddd; vertical-align: top; }
@media (prefers-color-scheme: dark) { th, td { border-bottom-color: #444; } }
.nav a { display: inline-block; margin: 0.35rem 1rem 0.35rem 0; color: inherit; }
ul { padding-left: 1.2rem; } li { margin: 0 0 0.45rem; }
code { font-size: 0.9em; }
.card-link { display: block; padding: 0.85rem 0.2rem; border-bottom: 1px solid #eee;
  text-decoration: none; color: inherit; }
@media (prefers-color-scheme: dark) { .card-link { border-bottom-color: #333; } }
.card-link .title { font-weight: 600; display: block; }
.card-link .path { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
"""


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "axes.titlesize": 12})


def savefig(fig: plt.Figure, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    shutil.copy2(path, DOCS_FIG / name)
    plt.close(fig)
    return name


def write_page(name: str, title: str, body: str) -> Path:
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style>
</head><body>
{body}
<p class="nav">
  <a href="custom-labels.html">Custom labels hub</a>
  <a href="interpretable-graphs.html">Interpretable graphs</a>
  <a href="cup-ideas.html">Cup ideas</a>
  <a href="index.html">Index</a>
</p>
</body></html>
"""
    out = DOCS / name
    out.write_text(html)
    return out


def discover_ids() -> list[str]:
    return sorted(
        d.name
        for d in DATA_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}_phases_of_play.csv").is_file()
        and (d / f"{d.name}_dynamic_events.csv").is_file()
    )


def _side(channel) -> str:
    if pd.isna(channel):
        return "center"
    c = str(channel)
    if c in WIDE_LEFT or "left" in c:
        return "left"
    if c in WIDE_RIGHT or "right" in c:
        return "right"
    return "center"


# ---------- loaders ----------


def load_phases(ids: list[str]) -> pd.DataFrame:
    frames = []
    for mid in ids:
        df = pd.read_csv(DATA_DIR / mid / f"{mid}_phases_of_play.csv")
        df["match_id"] = int(mid)
        frames.append(df)
    ph = pd.concat(frames, ignore_index=True)
    ph["width"] = ph[
        ["team_out_of_possession_width_start", "team_out_of_possession_width_end"]
    ].mean(axis=1)
    ph["length"] = ph[
        ["team_out_of_possession_length_start", "team_out_of_possession_length_end"]
    ].mean(axis=1)
    ph["box_area"] = ph["width"] * ph["length"]
    ph["shot"] = ph["team_possession_lead_to_shot"].astype(bool)
    ph["goal"] = ph["team_possession_lead_to_goal"].astype(bool)
    ph["oop"] = ph["team_out_of_possession_phase_type"]
    ph["phase_index"] = ph["index"]
    return ph


EVENT_COLS = [
    "match_id",
    "event_type",
    "event_subtype",
    "end_type",
    "lead_to_shot",
    "lead_to_goal",
    "team_shortname",
    "player_position",
    "player_in_possession_position",
    "channel_start",
    "frame_start",
    "frame_end",
    "phase_index",
    "team_out_of_possession_phase_type",
    "current_team_out_of_possession_next_phase_type",
    "overall_pressure_start",
    "pressing_chain",
    "pressing_chain_index",
    "pressing_chain_end_type",
    "inside_defensive_shape_start",
    "x_start",
    "y_start",
    "player_id",
    "organised_defense",
    "beaten_by_movement",
    "beaten_by_possession",
    "force_backward",
    "push_defensive_line",
    "break_defensive_line",
    "reduce_possession_danger",
    "stop_possession_danger",
    "interplayer_distance_start",
    "distance_to_player_in_possession_start",
]


def load_events(ids: list[str]) -> pd.DataFrame:
    frames = []
    for mid in ids:
        path = DATA_DIR / mid / f"{mid}_dynamic_events.csv"
        cols = [c for c in EVENT_COLS if c in pd.read_csv(path, nrows=0).columns]
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        df["match_id"] = int(mid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------- label builders ----------


def label_coverage_debt() -> pd.DataFrame:
    if DEBT_PATH.is_file():
        d = pd.read_csv(DEBT_PATH)
        d["label"] = "coverage_debt"
        # High debt: free CF/wide pressure or far-side leak — median danger debt is already elevated
        d["label_fire"] = (
            (d["debt_score"] >= d["debt_score"].quantile(0.67))
            | (d["far_runner"].astype(bool) & (d["n_free"] >= 2))
        ).astype(int)
        if "shot" not in d.columns and "danger" in d.columns:
            d["shot"] = d["danger"].isin(["shot", "goal"])
            d["goal"] = d["danger"].eq("goal")
        return d
    return pd.DataFrame()


def label_compact_but_broken(phases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    org = phases.loc[phases["oop"].isin(ORGANIZED)].copy()
    org["area_tercile"] = org.groupby("oop")["box_area"].transform(
        lambda s: pd.qcut(s, 3, labels=["tight", "mid", "open"], duplicates="drop")
    )
    tight = org.loc[org["area_tercile"] == "tight"].copy()

    poss = events.loc[events["event_type"] == "player_possession"].copy()
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    for col in ("beaten_by_movement", "beaten_by_possession"):
        if col not in eng.columns:
            eng[col] = False

    hun = pd.read_csv(HUN_PATH) if HUN_PATH.is_file() else pd.DataFrame()
    if not hun.empty:
        free_agg = (
            hun.loc[hun["free_role"].isin(ATTACK_ROLES)]
            .groupby(["match_id", "frame"])
            .agg(n_free=("free_role", "nunique"))
            .reset_index()
        )
    else:
        free_agg = pd.DataFrame()

    rows = []
    for _, ph in tight.iterrows():
        mid = int(ph["match_id"])
        f0, f1 = int(ph["frame_start"]), int(ph["frame_end"])
        reasons = []
        n_free = 0
        if not free_agg.empty and (ph["shot"] or ph["goal"]):
            sub = free_agg.loc[
                (free_agg["match_id"] == mid) & free_agg["frame"].between(f0, f1 + 25)
            ]
            if not sub.empty:
                n_free = int(sub["n_free"].max())
                if n_free >= 2:
                    reasons.append(f"{n_free}_free_attackers")
        ins = poss.loc[
            (poss["match_id"] == mid)
            & (poss["frame_start"] <= f1)
            & (poss["frame_end"] >= f0)
            & (poss["inside_defensive_shape_start"] == True)  # noqa: E712
        ]
        if len(ins):
            reasons.append("ball_inside_shape")
        beat = eng.loc[
            (eng["match_id"] == mid)
            & (eng["frame_start"] <= f1)
            & (eng["frame_end"] >= f0)
            & (eng["beaten_by_movement"].fillna(False) | eng["beaten_by_possession"].fillna(False))
        ]
        if len(beat):
            reasons.append("beaten_engagement")
        fire = int(bool(reasons))
        rows.append(
            {
                "match_id": mid,
                "phase_index": ph["phase_index"],
                "frame_start": f0,
                "frame_end": f1,
                "oop": ph["oop"],
                "box_area": float(ph["box_area"]),
                "shot": bool(ph["shot"]),
                "goal": bool(ph["goal"]),
                "label": "compact_but_broken",
                "label_fire": fire,
                "n_free": n_free,
                "reasons": ";".join(reasons) if reasons else "tight_only",
            }
        )
    return pd.DataFrame(rows)


def label_weak_side_press(events: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["in_chain"] = eng["pressing_chain"].fillna(False).astype(bool)
    chains = eng.loc[eng["in_chain"]].copy()
    runs = events.loc[events["event_type"] == "off_ball_run"].copy()

    # one row per pressing_chain_index × match
    if "pressing_chain_index" not in chains.columns:
        return pd.DataFrame()
    rows = []
    for (mid, cidx), g in chains.groupby(["match_id", "pressing_chain_index"]):
        f0, f1 = int(g["frame_start"].min()), int(g["frame_end"].max())
        press_sides = {_side(c) for c in g["channel_start"].dropna()}
        far = False
        far_role = ""
        r_win = runs.loc[
            (runs["match_id"] == mid) & (runs["frame_start"] <= f1 + 25) & (runs["frame_end"] >= f0 - 10)
        ]
        for _, rr in r_win.iterrows():
            rs = _side(rr.get("channel_start"))
            if (rs == "left" and "right" in press_sides) or (rs == "right" and "left" in press_sides):
                far = True
                far_role = str(rr.get("player_position") or "")
                break
        end = g["pressing_chain_end_type"].dropna().mode()
        chain_end = end.iloc[0] if len(end) else "unknown"
        # phase shot context
        ph = phases.loc[
            (phases["match_id"] == mid)
            & (phases["frame_start"] <= f1)
            & (phases["frame_end"] >= f0)
            & phases["oop"].isin(ORGANIZED)
        ]
        shot = bool(ph["shot"].any()) if len(ph) else False
        goal = bool(ph["goal"].any()) if len(ph) else False
        rows.append(
            {
                "match_id": int(mid),
                "pressing_chain_index": cidx,
                "frame_start": f0,
                "frame_end": f1,
                "chain_end": chain_end,
                "press_sides": ",".join(sorted(press_sides)),
                "far_runner": int(far),
                "far_role": far_role,
                "shot": shot,
                "goal": goal,
                "label": "weak_side_press_leak",
                "label_fire": int(far and chain_end != "regain"),
            }
        )
    return pd.DataFrame(rows)


def label_step_vs_hold(events: pd.DataFrame) -> pd.DataFrame:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    for col in (
        "push_defensive_line",
        "break_defensive_line",
        "force_backward",
        "reduce_possession_danger",
        "stop_possession_danger",
        "beaten_by_movement",
        "organised_defense",
    ):
        if col not in eng.columns:
            eng[col] = False
    eng = eng.loc[eng["team_out_of_possession_phase_type"].isin(ORGANIZED)].copy()
    step = eng["push_defensive_line"].fillna(False) | eng["break_defensive_line"].fillna(False)
    hold = (
        eng["force_backward"].fillna(False)
        | eng["reduce_possession_danger"].fillna(False)
        | eng["stop_possession_danger"].fillna(False)
    )
    eng["decision"] = np.where(step, "step", np.where(hold, "hold", "neutral"))
    eng["label"] = "step_vs_hold"
    eng["label_fire"] = (eng["decision"] != "neutral").astype(int)
    eng["shot"] = eng["lead_to_shot"].fillna(False).astype(bool)
    eng["goal"] = eng["lead_to_goal"].fillna(False).astype(bool)
    keep = [
        "match_id",
        "frame_start",
        "frame_end",
        "event_subtype",
        "decision",
        "label",
        "label_fire",
        "shot",
        "goal",
        "beaten_by_movement",
        "team_out_of_possession_phase_type",
        "player_position",
    ]
    return eng[[c for c in keep if c in eng.columns]].copy()


def label_phase_outcome_dense(phases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Denser outcome than shot/goal: regain / disruption / progression / shot / other."""
    org = phases.loc[phases["oop"].isin(ORGANIZED)].copy()
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["in_chain"] = eng["pressing_chain"].fillna(False).astype(bool)
    poss = events.loc[events["event_type"] == "player_possession"].copy()

    rows = []
    for _, ph in org.iterrows():
        mid = int(ph["match_id"])
        f0, f1 = int(ph["frame_start"]), int(ph["frame_end"])
        outcome = "other"
        if ph["goal"]:
            outcome = "goal"
        elif ph["shot"]:
            outcome = "shot"
        else:
            chain = eng.loc[
                (eng["match_id"] == mid)
                & eng["in_chain"]
                & (eng["frame_start"] <= f1)
                & (eng["frame_end"] >= f0)
            ]
            ends = chain["pressing_chain_end_type"].dropna()
            if len(ends) and (ends == "regain").any():
                outcome = "regain"
            elif len(ends) and (ends == "disruption").any():
                outcome = "disruption"
            else:
                p_win = poss.loc[
                    (poss["match_id"] == mid)
                    & (poss["frame_start"] <= f1)
                    & (poss["frame_end"] >= f0)
                ]
                next_oop = p_win["current_team_out_of_possession_next_phase_type"].dropna()
                if len(next_oop) and next_oop.isin(PROGRESS_PHASES).any():
                    outcome = "progression"
                else:
                    et = p_win["end_type"].dropna()
                    if len(et) and et.isin(["indirect_regain", "direct_regain"]).any():
                        outcome = "regain"
                    elif len(et) and et.isin(["indirect_disruption", "direct_disruption"]).any():
                        outcome = "disruption"
        rows.append(
            {
                "match_id": mid,
                "phase_index": ph["phase_index"],
                "frame_start": f0,
                "frame_end": f1,
                "oop": ph["oop"],
                "box_area": float(ph["box_area"]),
                "shot": bool(ph["shot"]),
                "goal": bool(ph["goal"]),
                "dense_outcome": outcome,
                "label": "phase_outcome_dense",
                "label_fire": int(outcome in ("shot", "goal", "progression")),
            }
        )
    return pd.DataFrame(rows)


def label_proximity_intent_debt(events: pd.DataFrame) -> pd.DataFrame:
    """Close engagement + beaten by movement = proximity said marked, intent failed."""
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    if "beaten_by_movement" not in eng.columns:
        return pd.DataFrame()
    dist = eng.get("interplayer_distance_start")
    if dist is None or dist.isna().all():
        dist = eng.get("distance_to_player_in_possession_start")
    eng = eng.copy()
    eng["dist"] = dist
    close = eng["dist"].notna() & (eng["dist"] <= 3.0)
    beaten = eng["beaten_by_movement"].fillna(False).astype(bool)
    eng["label"] = "proximity_intent_debt"
    eng["label_fire"] = (close & beaten).astype(int)
    eng["shot"] = eng["lead_to_shot"].fillna(False).astype(bool)
    eng["goal"] = eng["lead_to_goal"].fillna(False).astype(bool)
    keep = [
        "match_id",
        "frame_start",
        "frame_end",
        "dist",
        "beaten_by_movement",
        "label",
        "label_fire",
        "shot",
        "goal",
        "player_position",
        "event_subtype",
        "team_out_of_possession_phase_type",
    ]
    return eng.loc[close | beaten, [c for c in keep if c in eng.columns]].copy()


# ---------- pose attention labels ----------


def _pose_archive(match_id: int) -> Path | None:
    candidates = [
        POSE_DIR / "raw" / f"{match_id}.jsonl.zip",
        POSE_DIR / f"{match_id}.jsonl.zip",
        POSE_DIR / f"raw_{match_id}.jsonl.zip",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # huggingface_hub local_dir layout
    for p in POSE_DIR.rglob(f"{match_id}.jsonl.zip"):
        return p
    return None


def _shoulder_heading(joints: dict) -> tuple[float | None, float | None]:
    """Return (heading_deg toward chest-forward, shoulder_width_m) using SkillCorner joint names."""
    ls, rs = joints.get("lShoulder"), joints.get("rShoulder")
    if not ls or not rs:
        return None, None
    if ls.get("p90_mae_cm", 99) > MAX_ERR_CM or rs.get("p90_mae_cm", 99) > MAX_ERR_CM:
        return None, None
    lx, ly = ls["xyz"][0], ls["xyz"][1]
    rx, ry = rs["xyz"][0], rs["xyz"][1]
    sx, sy = rx - lx, ry - ly
    width = math.hypot(sx, sy)
    if not (MIN_SHOULDER_M <= width <= MAX_SHOULDER_M):
        return None, None
    f1 = (-sy, sx)
    f2 = (sy, -sx)
    neck, hip = joints.get("neck"), joints.get("midHip")
    if neck and hip and neck.get("xyz") and hip.get("xyz"):
        tx = neck["xyz"][0] - hip["xyz"][0]
        ty = neck["xyz"][1] - hip["xyz"][1]
        f = f1 if (f1[0] * tx + f1[1] * ty) >= (f2[0] * tx + f2[1] * ty) else f2
    else:
        f = f1
    heading = math.degrees(math.atan2(f[1], f[0]))
    return heading, width


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def label_pose_attention(events: pd.DataFrame, hun: pd.DataFrame) -> pd.DataFrame:
    """Ball-aware / runner-aware shoulder labels at pressing-chain starts (pose matches)."""
    rows: list[dict] = []
    eng = events.loc[
        (events["event_type"] == "on_ball_engagement")
        & events["pressing_chain"].fillna(False).astype(bool)
        & events["match_id"].isin(POSE_MATCHES)
    ].copy()
    if eng.empty:
        return pd.DataFrame()

    for mid in POSE_MATCHES:
        archive = _pose_archive(mid)
        if archive is None:
            continue
        need = eng.loc[eng["match_id"] == mid]
        if need.empty:
            continue
        windows = []
        track_need: set[int] = set()
        for _, r in need.iterrows():
            t0 = int(r["frame_start"])
            a, b = t0 - 5, t0 + 15
            windows.append((a, b, t0, r))
            track_need.update(range(a, b + 1))

        pose_by_track: dict[int, dict] = {}
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(name) as fh:
                for line in fh:
                    fr = json.loads(line)
                    pose_f = int(fr["frame"])
                    if pose_f % 5 != 0:
                        continue
                    tframe = int(round(pose_f / 2.5))
                    if tframe not in track_need:
                        continue
                    ball = fr.get("ball_data") or {}
                    ball_xy = None
                    if ball.get("x") is not None and ball.get("y") is not None:
                        ball_xy = (float(ball["x"]), float(ball["y"]))
                    players = []
                    for p in fr.get("player_data") or []:
                        joints = p.get("joints")
                        if not joints or p.get("x") is None:
                            continue
                        heading, width = _shoulder_heading(joints)
                        if heading is None:
                            continue
                        players.append(
                            {
                                "player_id": p.get("player_id"),
                                "x": float(p["x"]),
                                "y": float(p["y"]),
                                "heading": heading,
                                "shoulder_w": width,
                            }
                        )
                    if players:
                        pose_by_track[tframe] = {"players": players, "ball": ball_xy}

        runs = events.loc[
            (events["event_type"] == "off_ball_run") & (events["match_id"] == mid)
        ]

        for a, b, t0, r in windows:
            snap_t = next((t for t in range(t0, b + 1) if t in pose_by_track), None)
            if snap_t is None:
                snap_t = next((t for t in range(t0, a - 1, -1) if t in pose_by_track), None)
            if snap_t is None:
                continue
            snap = pose_by_track[snap_t]
            players = snap["players"]
            ball_xy = snap["ball"]
            if ball_xy is None:
                if pd.notna(r.get("x_start")) and pd.notna(r.get("y_start")):
                    ball_xy = (float(r["x_start"]), float(r["y_start"]))
                else:
                    continue

            def d2(p: dict) -> float:
                return (p["x"] - ball_xy[0]) ** 2 + (p["y"] - ball_xy[1]) ** 2

            pid = r.get("player_id")
            marker = None
            if pd.notna(pid):
                marker = next((p for p in players if p["player_id"] == pid), None)
            if marker is None:
                marker = min(players, key=d2)

            bearing = math.degrees(math.atan2(ball_xy[1] - marker["y"], ball_xy[0] - marker["x"]))
            ball_err = _angle_diff(marker["heading"], bearing)
            ball_aware = int(ball_err <= BALL_AWARE_DEG)

            runner_aware = None
            scan_lag_frames = None
            press_side = _side(r.get("channel_start"))
            opp = {"left": "right", "right": "left"}.get(press_side)
            far_run = False
            if opp and not runs.empty:
                r_win = runs.loc[(runs["frame_start"] <= b) & (runs["frame_end"] >= a)]
                for _, rr in r_win.iterrows():
                    if _side(rr.get("channel_start")) == opp:
                        far_run = True
                        break
            if opp and far_run:
                face_right = math.cos(math.radians(marker["heading"])) > 0.25
                face_left = math.cos(math.radians(marker["heading"])) < -0.25
                runner_aware = int(face_right if opp == "right" else face_left)
                lag = None
                for t in range(t0, b + 1):
                    if t not in pose_by_track:
                        continue
                    for p in pose_by_track[t]["players"]:
                        if p["player_id"] != marker["player_id"]:
                            continue
                        fr_ok = math.cos(math.radians(p["heading"])) > 0.25
                        fl_ok = math.cos(math.radians(p["heading"])) < -0.25
                        if fr_ok if opp == "right" else fl_ok:
                            lag = t - t0
                            break
                    if lag is not None:
                        break
                scan_lag_frames = lag

            fire = int(ball_aware == 0 or (runner_aware is not None and runner_aware == 0))
            rows.append(
                {
                    "match_id": mid,
                    "frame_start": t0,
                    "presser_role": str(r.get("player_position") or ""),
                    "channel": r.get("channel_start"),
                    "ball_aware": ball_aware,
                    "ball_heading_err_deg": round(ball_err, 1),
                    "runner_aware": runner_aware,
                    "scan_lag_frames": scan_lag_frames,
                    "shot": bool(r.get("lead_to_shot")),
                    "goal": bool(r.get("lead_to_goal")),
                    "label": "pose_attention",
                    "label_fire": fire,
                }
            )
    return pd.DataFrame(rows)


# ---------- charts + rates ----------


def shot_rate_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        if df is None or df.empty or "label_fire" not in df.columns:
            continue
        for fire, g in df.groupby(df["label_fire"].astype(int)):
            if "shot" not in g.columns:
                continue
            rows.append(
                {
                    "label": name,
                    "fire": int(fire),
                    "n": int(len(g)),
                    "shot_rate": float(g["shot"].mean()),
                    "goal_rate": float(g["goal"].mean()) if "goal" in g.columns else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def fig_label_lift(rates: pd.DataFrame) -> str:
    if rates.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No rates", ha="center")
        return savefig(fig, "label_shot_lift.png")
    # pivot fire 0 vs 1
    labels = []
    offs, ons = [], []
    for lab, g in rates.groupby("label"):
        z = g.loc[g["fire"] == 0, "shot_rate"]
        o = g.loc[g["fire"] == 1, "shot_rate"]
        if z.empty or o.empty:
            continue
        # Skip danger-only samples (e.g. coverage_debt at shots/goals → both ~100%)
        if float(z.iloc[0]) > 0.95 and float(o.iloc[0]) > 0.95:
            continue
        labels.append(lab.replace("_", "\n"))
        offs.append(float(z.iloc[0]) * 100)
        ons.append(float(o.iloc[0]) * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, offs, 0.35, label="label off", color="#85929e")
    ax.bar(x + 0.18, ons, 0.35, label="label on", color="#1a5276")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Shot rate (%)")
    ax.set_title("Custom labels — shot rate when the label fires")
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, "label_shot_lift.png")


def fig_outcome_mix(dense: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vc = dense["dense_outcome"].value_counts()
    order = [o for o in ["regain", "disruption", "other", "progression", "shot", "goal"] if o in vc.index]
    ax.bar([o.replace("_", "\n") for o in order], [vc[o] for o in order], color="#1a5276")
    ax.set_ylabel("Organised phases")
    ax.set_title("5. Phase outcomes denser than shot/goal")
    fig.tight_layout()
    return savefig(fig, "phase_outcome_dense.png")


def fig_step_hold(step: pd.DataFrame) -> str:
    g = (
        step.groupby("decision")
        .agg(n=("shot", "size"), shot_rate=("shot", "mean"))
        .reindex(["hold", "neutral", "step"])
        .dropna(how="all")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(g.index.astype(str), g["shot_rate"] * 100, color=["#117a65", "#85929e", "#c0392b"][: len(g)])
    for i, (idx, r) in enumerate(g.iterrows()):
        ax.text(i, r["shot_rate"] * 100 + 0.3, f"n={int(r['n'])}", ha="center", fontsize=9)
    ax.set_ylabel("Shot rate after engagement (%)")
    ax.set_title("4. Step vs hold — engagement decisions in organised blocks")
    fig.tight_layout()
    return savefig(fig, "step_vs_hold.png")


def fig_pose(pose: pd.DataFrame) -> list[str]:
    if pose.empty:
        return []
    figs = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label, color in [
        ("ball_aware", "Ball-aware", "#1a5276"),
        ("runner_aware", "Runner-aware", "#e67e22"),
    ]:
        if col not in pose.columns or pose[col].isna().all():
            continue
        rate = pose[col].dropna().mean() * 100
        ax.bar([label], [rate], color=color)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of pressing-chain samples")
    ax.set_title("Pose attention — head/shoulder proxy (not eye gaze)")
    fig.tight_layout()
    figs.append(savefig(fig, "pose_attention_rates.png"))
    if pose["ball_heading_err_deg"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(pose["ball_heading_err_deg"].dropna(), bins=20, color="#5dade2", edgecolor="white")
        ax.axvline(BALL_AWARE_DEG, color="#c0392b", lw=2, label=f"aware ≤ {BALL_AWARE_DEG:.0f}°")
        ax.set_xlabel("|Shoulder heading − ball bearing| (deg)")
        ax.set_title("Ball-awareness error distribution")
        ax.legend(frameon=False)
        fig.tight_layout()
        figs.append(savefig(fig, "pose_ball_err_hist.png"))
    return figs


# ---------- pages ----------


def page_hub(summary: dict, rates: pd.DataFrame) -> Path:
    rate_rows = ""
    if not rates.empty:
        for _, r in rates.sort_values(["label", "fire"]).iterrows():
            rate_rows += (
                f"<tr><td>{r['label']}</td><td>{'on' if r['fire'] else 'off'}</td>"
                f"<td>{int(r['n'])}</td><td>{100*r['shot_rate']:.1f}%</td></tr>"
            )
    pose_note = summary.get("pose_note", "Pose archives not loaded.")
    body = f"""
  <h1>Custom labels — defensive positioning</h1>
  <p class="meta">High-value labels SkillCorner does not ship: coverage, compact-broken,
  weak-side press, step/hold, dense outcomes, proximity-intent, pose attention.</p>
  <div class="mission">
    <strong>Bet.</strong> Labels you can film and graph beat another residual on λ₂.
    Event labels run on all <strong>{summary['n_matches']}</strong> matches;
    pose attention uses shoulder facing on matches with body-pose archives
    ({', '.join(str(m) for m in POSE_MATCHES)}).
  </div>
  <ul>
    <li>Organised phases scored: <strong>{summary.get('n_phases', 0):,}</strong></li>
    <li>Pose samples labeled: <strong>{summary.get('n_pose', 0):,}</strong> — {pose_note}</li>
  </ul>
  <a class="card-link" href="custom-labels-catalog.html">
    <span class="title">Label catalog + shot lift</span>
    <span class="path">Rates, definitions, clip counts</span>
  </a>
  <a class="card-link" href="custom-labels-clips.html">
    <span class="title">Film lists</span>
    <span class="path">Match id + frame for each label fire</span>
  </a>
  <a class="card-link" href="custom-labels-pose.html">
    <span class="title">Pose attention (ball / runner / scan lag)</span>
    <span class="path">Head–shoulder proxy — not eye gaze</span>
  </a>
  <a class="card-link" href="coverage-debt-graph.html">
    <span class="title">Coverage-debt event graph</span>
    <span class="path">Interpretable structure feeding these labels</span>
  </a>
  <h2>Shot rate when label fires</h2>
  <table><thead><tr><th>Label</th><th>State</th><th>n</th><th>Shot rate</th></tr></thead>
  <tbody>{rate_rows or '<tr><td colspan="4">No rates</td></tr>'}</tbody></table>
"""
    return write_page("custom-labels.html", "Custom defense labels", body)


def page_catalog(rates: pd.DataFrame, figs: list[str], definitions: list[tuple[str, str]]) -> Path:
    imgs = "".join(
        f'<figure><img src="eda/figures/labels/{f}" alt="{f}" loading="lazy"></figure>' for f in figs
    )
    defs = "".join(f"<li><code>{n}</code> — {d}</li>" for n, d in definitions)
    body = f"""
  <h1>Label catalog</h1>
  <p class="meta">Definitions + lift charts. Tunable rules, not trained weights.</p>
  <div class="mission">
    Each label is a boolean (or small enum) you can put on a phase/event and then
    retrieve film. Prefer denser outcomes than shot/goal when the sample is small.
  </div>
  {imgs}
  <h2>Definitions</h2>
  <ul>{defs}</ul>
"""
    return write_page("custom-labels-catalog.html", "Label catalog", body)


def page_clips(clips: dict[str, pd.DataFrame]) -> Path:
    sections = []
    for name, df in clips.items():
        if df is None or df.empty:
            continue
        fire = df.loc[df["label_fire"] == 1] if "label_fire" in df.columns else df
        fire = fire.head(25)
        rows = ""
        for _, r in fire.iterrows():
            fr = r["frame_end"] if "frame_end" in r.index and pd.notna(r["frame_end"]) else r.get("frame_start")
            try:
                fr_s = str(int(fr)) if fr is not None and pd.notna(fr) else ""
            except (TypeError, ValueError):
                fr_s = str(fr) if fr is not None else ""
            extra = r.get("reasons") or r.get("decision") or r.get("dense_outcome") or r.get("far_role") or ""
            rows += (
                f"<tr><td>{int(r['match_id'])}</td><td>{fr_s}</td>"
                f"<td>{'goal' if r.get('goal') else ('shot' if r.get('shot') else '—')}</td>"
                f"<td>{extra}</td></tr>"
            )
        sections.append(
            f"<h2>{name}</h2>"
            f"<table><thead><tr><th>Match</th><th>Frame</th><th>Danger</th><th>Detail</th></tr></thead>"
            f"<tbody>{rows or '<tr><td colspan=4>None</td></tr>'}</tbody></table>"
        )
    body = f"""
  <h1>Film lists</h1>
  <p class="meta">Match id + frame — pull in SkillCorner / your video tool.</p>
  {''.join(sections)}
"""
    return write_page("custom-labels-clips.html", "Custom label clips", body)


def page_pose(pose: pd.DataFrame, figs: list[str], pose_note: str) -> Path:
    imgs = "".join(
        f'<figure><img src="eda/figures/labels/{f}" alt="{f}" loading="lazy"></figure>' for f in figs
    )
    rows = ""
    if not pose.empty:
        for _, r in pose.head(30).iterrows():
            rows += (
                f"<tr><td>{int(r['match_id'])}</td><td>{int(r['frame_start'])}</td>"
                f"<td>{r.get('ball_aware')}</td><td>{r.get('runner_aware')}</td>"
                f"<td>{r.get('scan_lag_frames')}</td>"
                f"<td>{r.get('ball_heading_err_deg')}</td></tr>"
            )
    body = f"""
  <h1>Pose attention labels</h1>
  <p class="meta">Ball-aware · runner-aware · scan lag — shoulder / head proxy, not pupils.</p>
  <div class="mission">
    {pose_note}<br>
    Gate: joint p90 MAE ≤ {MAX_ERR_CM:.0f} cm and shoulder width
    {MIN_SHOULDER_M}–{MAX_SHOULDER_M} m (SkillCorner tutorial).
    Ball-aware if |heading − ball bearing| ≤ {BALL_AWARE_DEG:.0f}°.
  </div>
  {imgs if imgs else '<p class="meta">No pose figures yet — place '
                     f'<code>data/bodypose/raw/&lt;match&gt;.jsonl.zip</code> and re-run.</p>'}
  <h2>Samples</h2>
  <table><thead><tr>
    <th>Match</th><th>Frame</th><th>Ball-aware</th><th>Runner-aware</th>
    <th>Scan lag (frames)</th><th>Ball err°</th>
  </tr></thead>
  <tbody>{rows or '<tr><td colspan="6">No pose samples</td></tr>'}</tbody></table>
  <h2>How to fetch pose</h2>
  <p><code>huggingface-cli download SkillCorner/opendata-bodypose raw/1925299.jsonl.zip
  --repo-type dataset --local-dir data/bodypose</code></p>
"""
    return write_page("custom-labels-pose.html", "Pose attention labels", body)


# ---------- main ----------


def main() -> None:
    style()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    ids = discover_ids()
    print(f"Loading {len(ids)} matches…")
    phases = load_phases(ids)
    events = load_events(ids)
    hun = pd.read_csv(HUN_PATH) if HUN_PATH.is_file() else pd.DataFrame()
    print(f"  phases={len(phases):,} events={len(events):,}")

    print("Building labels…")
    debt = label_coverage_debt()
    compact = label_compact_but_broken(phases, events)
    weak = label_weak_side_press(events, phases)
    step = label_step_vs_hold(events)
    dense = label_phase_outcome_dense(phases, events)
    intent = label_proximity_intent_debt(events)

    for name, df in [
        ("coverage_debt", debt),
        ("compact_but_broken", compact),
        ("weak_side_press_leak", weak),
        ("step_vs_hold", step),
        ("phase_outcome_dense", dense),
        ("proximity_intent_debt", intent),
    ]:
        if df is not None and not df.empty:
            df.to_csv(LABEL_DIR / f"{name}.csv", index=False)
            print(f"  {name}: {len(df):,} rows, fire={int(df['label_fire'].sum()) if 'label_fire' in df else '?'}")

    print("Pose attention…")
    pose = label_pose_attention(events, hun)
    archives = [m for m in POSE_MATCHES if _pose_archive(m)]
    if archives:
        pose_note = f"Using pose archives for {archives}."
    else:
        pose_note = (
            "No pose zip under data/bodypose yet — recipe page ready; "
            "download SkillCorner/opendata-bodypose raw/{match}.jsonl.zip to fill."
        )
    if not pose.empty:
        pose.to_csv(LABEL_DIR / "pose_attention.csv", index=False)
        print(f"  pose_attention: {len(pose):,} rows")
    else:
        print(f"  pose_attention: 0 rows ({pose_note})")

    label_dfs = {
        "coverage_debt": debt,
        "compact_but_broken": compact,
        "weak_side_press_leak": weak,
        "step_vs_hold": step,
        "phase_outcome_dense": dense,
        "proximity_intent_debt": intent,
    }
    rates = shot_rate_table(label_dfs)
    rates.to_csv(LABEL_DIR / "label_shot_rates.csv", index=False)

    figs = [fig_label_lift(rates), fig_outcome_mix(dense), fig_step_hold(step)]
    if not compact.empty:
        g = compact.groupby("label_fire").agg(n=("shot", "size"), shot_rate=("shot", "mean"))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(
            ["Tight box\nonly", "Compact but\nbroken"],
            [g.loc[0, "shot_rate"] * 100 if 0 in g.index else 0,
             g.loc[1, "shot_rate"] * 100 if 1 in g.index else 0],
            color=["#85929e", "#c0392b"],
        )
        ax.set_ylabel("Shot rate (%)")
        ax.set_title("2. Compact-but-broken")
        fig.tight_layout()
        figs.append(savefig(fig, "compact_but_broken.png"))
    if not weak.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        g = weak.groupby("label_fire")["shot"].mean() * 100
        ax.bar(
            ["Chain, no\nfar leak", "Weak-side\npress leak"],
            [g.get(0, 0), g.get(1, 0)],
            color=["#117a65", "#e67e22"],
        )
        ax.set_ylabel("Shot rate in chain window (%)")
        ax.set_title("3. Weak-side press leak")
        fig.tight_layout()
        figs.append(savefig(fig, "weak_side_press_leak.png"))

    pose_figs = fig_pose(pose)

    definitions = [
        ("coverage_debt", "Danger graph debt_score ≥ 3 (free attackers + far runner + failed chain)."),
        ("compact_but_broken", "Bottom-tercile OOP box + (ball inside shape | beaten engagement | ≥2 free attackers at danger)."),
        ("weak_side_press_leak", "Pressing chain with far-side off-ball runner and chain end ≠ regain."),
        ("step_vs_hold", "Engagement push/break line (step) vs force-back / reduce-danger (hold) in organised blocks."),
        ("phase_outcome_dense", "Organised phase → regain | disruption | progression | shot | goal | other."),
        ("proximity_intent_debt", "Engagement ≤3 m and beaten_by_movement (pose-free intent proxy)."),
        ("pose_attention", "Shoulder facing vs ball / far-side runner; scan lag in frames (pose matches)."),
    ]

    summary = {
        "n_matches": len(ids),
        "n_phases": int(phases["oop"].isin(ORGANIZED).sum()),
        "n_pose": int(len(pose)),
        "pose_note": pose_note,
        "pose_archives": archives,
        "label_counts": {
            k: {"n": int(len(v)), "fire": int(v["label_fire"].sum())}
            for k, v in label_dfs.items()
            if v is not None and not v.empty
        },
    }
    with open(LABEL_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Writing pages…")
    page_hub(summary, rates)
    page_catalog(rates, figs, definitions)
    page_clips(label_dfs)
    page_pose(pose, pose_figs, pose_note)

    # Link hooks — patch index/cup/graphs if missing
    _ensure_links()
    print("Done → docs/custom-labels.html")


def _ensure_links() -> None:
    """Insert custom-labels hub links into key docs if absent."""
    snippets = {
        "docs/interpretable-graphs.html": (
            "custom-labels.html",
            '  <a class="card-link" href="custom-labels.html">\n'
            '    <span class="title">Custom labels (defensive positioning)</span>\n'
            '    <span class="path">Coverage debt · compact-broken · weak-side · step/hold · pose</span>\n'
            "  </a>\n",
            "</ul>",
        ),
    }
    # Simpler: rewrite small link blocks via string contains check in index/cup/review/graph-tool-notes
    for path, block in [
        (
            DOCS / "cup-ideas.html",
            '    <li><a href="custom-labels.html">Custom labels</a> — high-value defensive positioning tags</li>\n',
        ),
        (
            DOCS / "graph-tool-notes.html",
            "  <p>High-value custom labels (coverage, compact-broken, weak-side, step/hold, pose attention): "
            '<a href="custom-labels.html"><strong>custom-labels.html</strong></a>.</p>\n',
        ),
    ]:
        if not path.is_file():
            continue
        text = path.read_text()
        if "custom-labels.html" in text:
            continue
        # insert before last nav paragraph if possible
        if "</ul>" in text and path.name == "cup-ideas.html":
            text = text.replace("</ul>", block + "</ul>", 1)
        else:
            # before final links
            marker = '<p><a href="cup-ideas.html">'
            if marker in text:
                text = text.replace(marker, block + marker, 1)
            else:
                text = text.replace("</body>", block + "</body>")
        path.write_text(text)

    for path in (DOCS / "index.html", DOCS / "review.html"):
        if not path.is_file():
            continue
        text = path.read_text()
        if "custom-labels.html" in text:
            continue
        card = (
            '      <a href="custom-labels.html">\n'
            "        <span class=\"title\">Custom defense labels</span>\n"
            "        <span class=\"path\">Coverage · compact-broken · weak-side · step/hold · pose</span>\n"
            "      </a>\n"
            if path.name == "index.html"
            else (
                '  <a class="cta" href="custom-labels.html">\n'
                "    Custom defense labels\n"
                "    <span>Coverage · compact-broken · weak-side · pose attention</span>\n"
                "  </a>\n"
            )
        )
        if path.name == "index.html" and 'href="interpretable-graphs.html"' in text:
            # insert after interpretable-graphs card block — find next </a> after it
            anchor = 'href="interpretable-graphs.html"'
            i = text.find(anchor)
            j = text.find("</a>", i)
            if j != -1:
                text = text[: j + 4] + "\n" + card + text[j + 4 :]
                path.write_text(text)
        elif path.name == "review.html" and 'href="interpretable-graphs.html"' in text:
            i = text.find('href="interpretable-graphs.html"')
            j = text.find("</a>", i)
            if j != -1:
                text = text[: j + 4] + "\n" + card + text[j + 4 :]
                path.write_text(text)

    # interpretable hub
    hub = DOCS / "interpretable-graphs.html"
    if hub.is_file():
        text = hub.read_text()
        if "custom-labels.html" not in text:
            card = (
                '  <a class="card-link" href="custom-labels.html">\n'
                '    <span class="title">Custom labels (defensive positioning)</span>\n'
                '    <span class="path">Coverage debt · compact-broken · weak-side · step/hold · pose</span>\n'
                "  </a>\n"
            )
            marker = 'href="graph-case-retrieval.html"'
            i = text.find(marker)
            j = text.find("</a>", i)
            if j != -1:
                text = text[: j + 4] + "\n" + card + text[j + 4 :]
                hub.write_text(text)


if __name__ == "__main__":
    main()
