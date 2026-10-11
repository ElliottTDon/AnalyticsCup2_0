"""Interpretable graph methods for defensive coverage debt (small-n friendly).

Implements approaches that fit the 20-match constraint:
  1. Hand-crafted graph features (no trained GNN)
  2. Event-level coverage-debt graphs (nodes/edges at danger moments)
  6. Regularized sparse residual model (graph features after box + pressure)
  7. Case-based retrieval (graph signature → similar clips)

Writes figures + CSVs under analysis/output/ and phone pages under docs/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_DIR = WORKSPACE / "analysis" / "output"
FIG_DIR = OUT_DIR / "figures" / "graphs"
DOCS = WORKSPACE / "docs"
DOCS_FIG = DOCS / "eda" / "figures" / "graphs"
HUN_PATH = OUT_DIR / "hungarian_free_roles.csv"

ORGANIZED = ("low_block", "medium_block", "high_block")
ATTACK_ROLES = {"CF", "RW", "LW", "AM", "RM", "LM", "RDM", "LDM", "LF", "RF", "SS"}
PRESSURE_ORDER = [
    "no_pressure",
    "low_pressure",
    "medium_pressure",
    "high_pressure",
    "very_high_pressure",
]
PRESSURE_SCORE = {p: i for i, p in enumerate(PRESSURE_ORDER)}
WIDE_LEFT = {"wide_left", "half_space_left", "left_halfspace"}
WIDE_RIGHT = {"wide_right", "half_space_right", "right_halfspace"}

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
  <a href="interpretable-graphs.html">Graph methods hub</a>
  <a href="cup-ideas.html">Cup ideas</a>
  <a href="eda.html">Visual EDA</a>
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


# ---------- loaders ----------


def load_phases(match_ids: list[str]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
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


def load_events(match_ids: list[str]) -> pd.DataFrame:
    want = [
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
        "third_start",
        "frame_start",
        "frame_end",
        "phase_index",
        "team_out_of_possession_phase_type",
        "overall_pressure_start",
        "pressing_chain",
        "pressing_chain_index",
        "pressing_chain_end_type",
        "inside_defensive_shape_start",
        "organised_defense",
    ]
    frames = []
    for mid in match_ids:
        path = DATA_DIR / mid / f"{mid}_dynamic_events.csv"
        cols = [c for c in want if c in pd.read_csv(path, nrows=0).columns]
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        df["match_id"] = int(mid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_hungarian() -> pd.DataFrame:
    if not HUN_PATH.is_file():
        return pd.DataFrame()
    h = pd.read_csv(HUN_PATH)
    h = h.loc[h["free_role"].isin(ATTACK_ROLES) | h["free_role"].isna()].copy()
    return h


# ---------- 2) Event-level coverage debt graphs ----------


def _side(channel: str | float) -> str:
    if pd.isna(channel):
        return "center"
    c = str(channel)
    if c in WIDE_LEFT or "left" in c:
        return "left"
    if c in WIDE_RIGHT or "right" in c:
        return "right"
    return "center"


def build_coverage_debt_events(
    events: pd.DataFrame, phases: pd.DataFrame, hun: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """One row per danger possession = one coverage-debt graph summary + optional draw specs."""
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & (events["lead_to_shot"] | events["lead_to_goal"])
        & events["team_out_of_possession_phase_type"].isin(ORGANIZED)
    ].copy()

    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["in_chain"] = eng["pressing_chain"].fillna(False).astype(bool)
    runs = events.loc[events["event_type"] == "off_ball_run"].copy()

    # free roles aggregated per match-frame
    if not hun.empty:
        free_agg = (
            hun.loc[hun["free_role"].isin(ATTACK_ROLES)]
            .groupby(["match_id", "frame"])
            .agg(
                n_free=("free_role", "nunique"),
                free_roles=("free_role", lambda s: ",".join(sorted(set(s)))),
                lambda2_defense=("lambda2_defense", "mean"),
            )
            .reset_index()
        )
    else:
        free_agg = pd.DataFrame()

    phase_key = phases[
        ["match_id", "phase_index", "box_area", "width", "length", "oop", "frame_start", "frame_end"]
    ].copy()

    rows: list[dict] = []
    examples: list[dict] = []  # for schematic drawings

    for _, p in poss.iterrows():
        mid = int(p["match_id"])
        frame = int(p["frame_end"]) if pd.notna(p["frame_end"]) else None
        if frame is None:
            continue
        # phase join
        ph = phase_key.loc[
            (phase_key["match_id"] == mid) & (phase_key["phase_index"] == p.get("phase_index"))
        ]
        if ph.empty:
            # fallback: overlapping frames
            ph = phase_key.loc[
                (phase_key["match_id"] == mid)
                & (phase_key["frame_start"] <= frame)
                & (phase_key["frame_end"] >= frame - 5)
            ]
        box_area = float(ph["box_area"].iloc[0]) if len(ph) else np.nan
        width = float(ph["width"].iloc[0]) if len(ph) else np.nan
        length = float(ph["length"].iloc[0]) if len(ph) else np.nan
        oop = ph["oop"].iloc[0] if len(ph) else p.get("team_out_of_possession_phase_type")

        # free attackers near this frame
        n_free, free_roles, lam2 = 0, "", np.nan
        if not free_agg.empty:
            sub = free_agg.loc[
                (free_agg["match_id"] == mid)
                & (free_agg["frame"].between(frame - 20, frame + 5))
            ]
            if not sub.empty:
                best = sub.sort_values("n_free", ascending=False).iloc[0]
                n_free = int(best["n_free"])
                free_roles = best["free_roles"]
                lam2 = float(best["lambda2_defense"]) if pd.notna(best["lambda2_defense"]) else np.nan

        # concurrent chain engagements
        win0, win1 = frame - 40, frame + 5
        chain_eng = eng.loc[
            (eng["match_id"] == mid)
            & eng["in_chain"]
            & (eng["frame_start"] <= win1)
            & (eng["frame_end"] >= win0)
        ]
        n_chain = int(len(chain_eng))
        chain_end = (
            chain_eng["pressing_chain_end_type"].dropna().mode().iloc[0]
            if chain_eng["pressing_chain_end_type"].notna().any()
            else "none"
        )
        press_sides = {_side(c) for c in chain_eng["channel_start"].dropna()}

        # far-side runners
        run_win = runs.loc[
            (runs["match_id"] == mid)
            & (runs["frame_start"] <= win1)
            & (runs["frame_end"] >= win0)
        ]
        far_runner = False
        far_role = ""
        for _, rr in run_win.iterrows():
            rs = _side(rr.get("channel_start"))
            if rs == "left" and "right" in press_sides:
                far_runner, far_role = True, rr.get("player_position") or ""
                break
            if rs == "right" and "left" in press_sides:
                far_runner, far_role = True, rr.get("player_position") or ""
                break

        pressure = p.get("overall_pressure_start")
        pressure_score = PRESSURE_SCORE.get(pressure, np.nan)
        inside = bool(p.get("inside_defensive_shape_start") is True)
        carrier_role = p.get("player_position") or p.get("player_in_possession_position") or "?"
        carrier_side = _side(p.get("channel_start"))

        # Hand-crafted debt score (interpretable, not learned)
        free_roles_list = [r for r in free_roles.split(",") if r]
        cf_free = int("CF" in free_roles_list)
        wide_free = int(bool({"RW", "LW", "RM", "LM"} & set(free_roles_list)))
        debt_score = (
            1.0 * n_free
            + 1.5 * cf_free
            + 1.0 * wide_free
            + 1.0 * float(far_runner)
            + 0.5 * float(inside)
            + 0.5 * float(n_chain > 0 and chain_end not in ("regain",))
        )

        # Graph size features
        n_nodes = 1 + n_free + min(n_chain, 5)  # carrier + free + capped pressers
        n_edges = n_free  # mark-debt edges to free attackers
        if n_chain:
            n_edges += n_chain  # press edges to carrier
        if far_runner:
            n_edges += 1  # weak-side leak edge

        row = {
            "match_id": mid,
            "frame": frame,
            "phase_index": p.get("phase_index"),
            "danger": "goal" if p.get("lead_to_goal") else "shot",
            "oop": oop,
            "box_area": box_area,
            "width": width,
            "length": length,
            "pressure": pressure,
            "pressure_score": pressure_score,
            "inside_shape": inside,
            "carrier_role": carrier_role,
            "carrier_side": carrier_side,
            "n_free": n_free,
            "free_roles": free_roles,
            "cf_free": cf_free,
            "wide_free": wide_free,
            "lambda2_defense": lam2,
            "n_chain_presses": n_chain,
            "chain_end": chain_end,
            "far_runner": far_runner,
            "far_role": far_role,
            "debt_score": debt_score,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "edge_density": n_edges / max(n_nodes * (n_nodes - 1) / 2, 1),
            "team_attack": p.get("team_shortname"),
        }
        rows.append(row)

        # keep a few rich examples for schematics
        if len(examples) < 8 and (n_free >= 2 or far_runner):
            examples.append(
                {
                    **row,
                    "press_roles": chain_eng["player_position"].dropna().astype(str).tolist()[:4],
                    "press_channels": chain_eng["channel_start"].dropna().astype(str).tolist()[:4],
                }
            )

    return pd.DataFrame(rows), examples


def draw_coverage_debt_schematic(example: dict, name: str) -> str:
    """Interpretable node-link diagram for one coverage-debt event graph."""
    G = nx.DiGraph()
    G.add_node("BALL", kind="carrier", label=f"Ball\n{example['carrier_role']}")
    free = [r for r in str(example.get("free_roles") or "").split(",") if r]
    for i, role in enumerate(free[:5]):
        nid = f"FREE_{i}"
        G.add_node(nid, kind="free", label=f"Free\n{role}")
        G.add_edge("BALL", nid, kind="debt", label="unmarked")
    for i, role in enumerate((example.get("press_roles") or [])[:3]):
        nid = f"PRESS_{i}"
        G.add_node(nid, kind="press", label=f"Press\n{role}")
        G.add_edge(nid, "BALL", kind="press", label="press")
    if example.get("far_runner"):
        G.add_node("FAR", kind="far", label=f"Far run\n{example.get('far_role') or '?'}")
        G.add_edge("BALL", "FAR", kind="leak", label="weak side")

    pos = nx.spring_layout(G, seed=7, k=1.8)
    # manual layout preference
    pos["BALL"] = np.array([0.0, 0.0])
    free_nodes = [n for n in G if str(n).startswith("FREE")]
    for i, n in enumerate(free_nodes):
        pos[n] = np.array([0.8, 0.6 - 0.3 * i])
    press_nodes = [n for n in G if str(n).startswith("PRESS")]
    for i, n in enumerate(press_nodes):
        pos[n] = np.array([-0.9, 0.4 - 0.35 * i])
    if "FAR" in G:
        pos["FAR"] = np.array([0.9, -0.7])

    colors = {
        "carrier": "#1a5276",
        "free": "#c0392b",
        "press": "#117a65",
        "far": "#e67e22",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    node_colors = [colors[G.nodes[n]["kind"]] for n in G]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, ax=ax, alpha=0.9)
    edge_colors = []
    for u, v, d in G.edges(data=True):
        edge_colors.append(
            {"debt": "#c0392b", "press": "#117a65", "leak": "#e67e22"}.get(d["kind"], "#555")
        )
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True, arrowsize=16, width=2.0, ax=ax, alpha=0.85
    )
    labels = {n: G.nodes[n]["label"] for n in G}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white", ax=ax)
    ax.set_axis_off()
    ax.set_title(
        f"Coverage-debt graph — match {example['match_id']} frame {example['frame']}\n"
        f"debt={example['debt_score']:.1f} · free={example['n_free']} · "
        f"{'goal' if example['danger']=='goal' else 'shot'} · {example['oop']}",
        fontsize=11,
    )
    fig.tight_layout()
    return savefig(fig, name)


# ---------- 1) Hand-crafted feature charts ----------


def fig_debt_distributions(debt: pd.DataFrame) -> list[str]:
    figs = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, danger, color in zip(axes, ["shot", "goal"], ["#e67e22", "#c0392b"]):
        sub = debt.loc[debt["danger"] == danger, "debt_score"]
        ax.hist(sub, bins=20, color=color, edgecolor="white")
        ax.set_title(f"Coverage debt score — {danger}s")
        ax.set_xlabel("Debt score (hand-crafted)")
        ax.set_ylabel("Events")
    fig.suptitle("1. Hand-crafted graph feature: coverage debt", y=1.02)
    fig.tight_layout()
    figs.append(savefig(fig, "01_debt_score_hist.png"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    g = (
        debt.groupby(["oop", "danger"])
        .agg(mean_debt=("debt_score", "mean"), mean_free=("n_free", "mean"), n=("debt_score", "size"))
        .reset_index()
    )
    for danger, color in [("shot", "#e67e22"), ("goal", "#c0392b")]:
        sub = g.loc[g["danger"] == danger]
        order = [b for b in ORGANIZED if b in set(sub["oop"])]
        sub = sub.set_index("oop").reindex(order)
        ax.plot(
            [b.replace("_", "\n") for b in order],
            sub["mean_debt"],
            marker="o",
            label=danger,
            color=color,
        )
    ax.set_ylabel("Mean debt score")
    ax.set_title("Debt by organised block — goals vs shots")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs.append(savefig(fig, "01_debt_by_block.png"))

    # free role composition
    roles = []
    for _, r in debt.iterrows():
        for role in str(r["free_roles"]).split(","):
            if role:
                roles.append({"danger": r["danger"], "role": role})
    rdf = pd.DataFrame(roles)
    if not rdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, danger, color in zip(axes, ["goal", "shot"], ["#c0392b", "#e67e22"]):
            vc = rdf.loc[rdf["danger"] == danger, "role"].value_counts().head(10)
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=color)
            ax.set_title(f"Free roles in debt graph — {danger}s")
            ax.set_xlabel("Count")
        fig.suptitle("Who carries coverage debt when chances happen?", y=1.02)
        fig.tight_layout()
        figs.append(savefig(fig, "01_free_roles_in_debt.png"))
    return figs


# ---------- 6) Regularized sparse residual model ----------


def build_phase_end_features(phases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Event-only coverage-debt features at EVERY organised phase end.

    Deliberately omits Hungarian free-attacker counts: those rows exist only at
    shot/goal frames, so joining them here would leak outcome into the model.
    Free-attacker structure stays in danger-event graphs (approaches 1/2/7).
    """
    org = phases.loc[phases["oop"].isin(ORGANIZED)].copy()
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["in_chain"] = eng["pressing_chain"].fillna(False).astype(bool)
    runs = events.loc[events["event_type"] == "off_ball_run"]
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    poss["pressure_score"] = poss["overall_pressure_start"].map(PRESSURE_SCORE)

    eng_by = {mid: g for mid, g in eng.groupby("match_id")}
    runs_by = {mid: g for mid, g in runs.groupby("match_id")}
    poss_by = {mid: g for mid, g in poss.groupby("match_id")}

    rows = []
    for _, ph in org.iterrows():
        mid = int(ph["match_id"])
        f1 = int(ph["frame_end"])
        win0, win1 = f1 - 40, f1 + 5

        e_mid = eng_by.get(mid)
        eng_win = (
            e_mid.loc[(e_mid["frame_start"] <= win1) & (e_mid["frame_end"] >= win0)]
            if e_mid is not None
            else eng.iloc[0:0]
        )
        chain_eng = eng_win.loc[eng_win["in_chain"]] if len(eng_win) else eng_win
        n_eng = int(len(eng_win))
        n_chain = int(len(chain_eng))
        chain_end = (
            chain_eng["pressing_chain_end_type"].dropna().mode().iloc[0]
            if len(chain_eng) and chain_eng["pressing_chain_end_type"].notna().any()
            else "none"
        )
        chain_failed = int(n_chain > 0 and chain_end not in ("regain", "none"))
        press_sides = {_side(c) for c in chain_eng["channel_start"].dropna()} if len(chain_eng) else set()

        r_mid = runs_by.get(mid)
        n_runs = 0
        far_runner = False
        if r_mid is not None:
            run_win = r_mid.loc[(r_mid["frame_start"] <= win1) & (r_mid["frame_end"] >= win0)]
            n_runs = int(len(run_win))
            if press_sides:
                for _, rr in run_win.iterrows():
                    rs = _side(rr.get("channel_start"))
                    if (rs == "left" and "right" in press_sides) or (
                        rs == "right" and "left" in press_sides
                    ):
                        far_runner = True
                        break

        p_mid = poss_by.get(mid)
        if p_mid is not None:
            p_win = p_mid.loc[(p_mid["frame_start"] <= win1) & (p_mid["frame_end"] >= win0)]
            mean_pressure = (
                float(p_win["pressure_score"].mean())
                if len(p_win) and p_win["pressure_score"].notna().any()
                else np.nan
            )
            inside = (
                bool((p_win.get("inside_defensive_shape_start") == True).any())  # noqa: E712
                if len(p_win)
                else False
            )
            n_poss = int(len(p_win))
        else:
            mean_pressure, inside, n_poss = np.nan, False, 0

        # Event-graph debt (no Hungarian): press cluster + weak-side leak + ball in shape
        event_debt = (
            1.0 * float(far_runner)
            + 0.75 * float(inside)
            + 0.75 * float(chain_failed)
            + 0.25 * min(n_chain, 4)
            + 0.15 * min(n_runs, 4)
        )
        n_nodes = 1 + min(n_chain, 5) + int(far_runner) + int(inside)
        n_edges = n_chain + int(far_runner) + int(inside)

        rows.append(
            {
                "match_id": mid,
                "phase_index": ph["phase_index"],
                "shot": bool(ph["shot"]),
                "goal": bool(ph["goal"]),
                "oop": ph["oop"],
                "box_area": float(ph["box_area"]),
                "width": float(ph["width"]),
                "length": float(ph["length"]),
                "mean_pressure": mean_pressure,
                "event_debt": event_debt,
                "any_far_runner": int(far_runner),
                "any_inside_shape": int(inside),
                "chain_failed": chain_failed,
                "n_chain_presses": float(n_chain),
                "n_engagements": float(n_eng),
                "n_off_ball_runs": float(n_runs),
                "n_possessions": float(n_poss),
                "mean_n_nodes": float(n_nodes),
                "mean_n_edges": float(n_edges),
                "has_debt_event": int(far_runner or inside or chain_failed),
            }
        )
    m = pd.DataFrame(rows)
    m["mean_pressure"] = m["mean_pressure"].fillna(m["mean_pressure"].median())
    for b in ORGANIZED:
        m[f"block_{b}"] = (m["oop"] == b).astype(int)
    return m


def build_phase_modeling_table(
    debt: pd.DataFrame, phases: pd.DataFrame, events: pd.DataFrame, hun: pd.DataFrame
) -> pd.DataFrame:
    """Phase-end event-graph features for ALL organised phases (outcome-blind)."""
    del debt, hun  # danger-only Hungarian tables must not enter the model
    return build_phase_end_features(phases, events)


def _logit(C: float = 0.2) -> LogisticRegression:
    # sklearn ≥1.8 deprecates penalty='l2'; default + C is L2 shrinkage
    return LogisticRegression(C=C, max_iter=500, class_weight="balanced")


def run_sparse_residual(model_df: pd.DataFrame) -> dict:
    """L2-regularized logistic: baselines vs baselines+event-graph; match-grouped CV."""
    baseline_cols = [
        "box_area",
        "width",
        "length",
        "mean_pressure",
        "block_low_block",
        "block_medium_block",
        "block_high_block",
    ]
    graph_cols = [
        "event_debt",
        "any_far_runner",
        "any_inside_shape",
        "chain_failed",
        "n_chain_presses",
        "n_off_ball_runs",
        "mean_n_nodes",
        "mean_n_edges",
        "has_debt_event",
    ]
    use = model_df.dropna(subset=baseline_cols).copy()
    y = use["shot"].astype(int).values
    groups = use["match_id"].values

    def _cv_auc(cols: list[str]) -> tuple[float, list[float], np.ndarray]:
        X = use[cols].astype(float).values
        X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        X = StandardScaler().fit_transform(X)
        gkf = GroupKFold(n_splits=min(5, use["match_id"].nunique()))
        aucs = []
        coefs = []
        for train, test in gkf.split(X, y, groups):
            if y[train].sum() == 0 or y[train].sum() == len(train):
                continue
            clf = _logit(0.2)
            clf.fit(X[train], y[train])
            if len(np.unique(y[test])) < 2:
                continue
            proba = clf.predict_proba(X[test])[:, 1]
            aucs.append(roc_auc_score(y[test], proba))
            coefs.append(clf.coef_.ravel())
        mean_coef = np.mean(coefs, axis=0) if coefs else np.zeros(len(cols))
        return (float(np.mean(aucs)) if aucs else float("nan"), aucs, mean_coef)

    base_auc, base_folds, _ = _cv_auc(baseline_cols)
    full_auc, full_folds, full_coef = _cv_auc(baseline_cols + graph_cols)

    X_full = use[baseline_cols + graph_cols].astype(float).values
    X_full = np.nan_to_num(X_full, nan=np.nanmedian(X_full, axis=0))
    X_full = StandardScaler().fit_transform(X_full)
    clf = _logit(0.2)
    clf.fit(X_full, y)
    coef_table = pd.DataFrame(
        {
            "feature": baseline_cols + graph_cols,
            "coef": clf.coef_.ravel(),
            "abs_coef": np.abs(clf.coef_.ravel()),
            "family": ["baseline"] * len(baseline_cols) + ["graph"] * len(graph_cols),
        }
    ).sort_values("abs_coef", ascending=False)

    # Match-bootstrap residual: debt among shot vs non-shot after regressing debt on baselines
    from sklearn.linear_model import LinearRegression

    Xb = use[baseline_cols].astype(float).values
    Xb = np.nan_to_num(Xb, nan=np.nanmedian(Xb, axis=0))
    debt_y = use["event_debt"].astype(float).values
    lr = LinearRegression().fit(Xb, debt_y)
    resid = debt_y - lr.predict(Xb)
    use = use.copy()
    use["debt_resid"] = resid
    rng = np.random.default_rng(42)
    matches = use["match_id"].unique()
    diffs = []
    for _ in range(400):
        sample = rng.choice(matches, size=len(matches), replace=True)
        parts = [use.loc[use["match_id"] == m] for m in sample]
        boot = pd.concat(parts, ignore_index=True)
        diffs.append(
            boot.loc[boot["shot"], "debt_resid"].mean()
            - boot.loc[~boot["shot"], "debt_resid"].mean()
        )
    diffs = np.array(diffs)
    residual = {
        "mean_diff_shot_minus_nons": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "excludes_zero": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        ["Box + pressure\n(+ block)", "+ event-graph\ndebt features"],
        [base_auc, full_auc],
        color=["#85929e", "#1a5276"],
    )
    ax.set_ylabel("Match-grouped CV AUC (shot)")
    ax.set_ylim(0.45, max(0.75, max(base_auc, full_auc) + 0.05))
    ax.set_title("6. Regularized model — event-graph debt after baselines\n(no shot-conditioned Hungarian)")
    for i, v in enumerate([base_auc, full_auc]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    f1 = savefig(fig, "06_sparse_auc.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    top = coef_table.head(12)
    colors = ["#1a5276" if f == "graph" else "#85929e" for f in top["family"]]
    ax.barh(top["feature"][::-1], top["coef"][::-1], color=colors[::-1])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_title("Shrunk coefficients (L2, C=0.2) — blue = event-graph family")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    f2 = savefig(fig, "06_sparse_coefs.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(diffs, bins=30, color="#5dade2", edgecolor="white")
    ax.axvline(0, color="#333", lw=1)
    ax.axvline(residual["mean_diff_shot_minus_nons"], color="#c0392b", lw=2, label="mean")
    ax.set_title(
        f"Match-bootstrap: residual event_debt (shot − no shot)\n"
        f"mean={residual['mean_diff_shot_minus_nons']:.3f} "
        f"CI [{residual['ci_low']:.3f}, {residual['ci_high']:.3f}]"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    f3 = savefig(fig, "06_debt_residual_bootstrap.png")

    return {
        "base_auc": base_auc,
        "full_auc": full_auc,
        "base_folds": base_folds,
        "full_folds": full_folds,
        "coef_table": coef_table,
        "residual": residual,
        "figs": [f1, f2, f3],
        "n_phases": int(len(use)),
        "n_matches": int(use["match_id"].nunique()),
        "shot_rate": float(y.mean()),
        "leakage_safe": True,
        "graph_feature_note": "event-only (press chain, far runner, inside shape); no Hungarian",
    }


# ---------- 7) Case-based retrieval ----------


def build_case_retrieval(debt: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Graph signature → nearest neighbor clips (cosine on standardized features)."""
    feat_cols = [
        "debt_score",
        "n_free",
        "cf_free",
        "wide_free",
        "far_runner",
        "inside_shape",
        "n_chain_presses",
        "n_nodes",
        "n_edges",
        "pressure_score",
        "box_area",
        "width",
        "length",
    ]
    d = debt.dropna(subset=["debt_score", "box_area"]).copy()
    d["far_runner"] = d["far_runner"].astype(int)
    d["inside_shape"] = d["inside_shape"].astype(int)
    d["pressure_score"] = d["pressure_score"].fillna(d["pressure_score"].median())
    X = d[feat_cols].astype(float).values
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(6, len(d)), metric="cosine")
    nn.fit(Xs)
    dists, idxs = nn.kneighbors(Xs)

    # Query: high-debt goals — retrieve similar non-goal shots (or any)
    queries = d.loc[d["danger"] == "goal"].sort_values("debt_score", ascending=False).head(12)
    rows = []
    for qi, qrow in queries.iterrows():
        # position in d
        pos = d.index.get_loc(qi)
        if isinstance(pos, slice):
            continue
        neighbors = []
        for dist, j in zip(dists[pos][1:], idxs[pos][1:]):  # skip self
            n = d.iloc[j]
            neighbors.append(
                {
                    "query_match": int(qrow["match_id"]),
                    "query_frame": int(qrow["frame"]),
                    "query_debt": float(qrow["debt_score"]),
                    "query_free": qrow["free_roles"],
                    "query_oop": qrow["oop"],
                    "nn_match": int(n["match_id"]),
                    "nn_frame": int(n["frame"]),
                    "nn_debt": float(n["debt_score"]),
                    "nn_danger": n["danger"],
                    "nn_free": n["free_roles"],
                    "nn_oop": n["oop"],
                    "cosine_dist": float(dist),
                }
            )
        rows.extend(neighbors[:3])
    retrieval = pd.DataFrame(rows)

    figs = []
    # embedding-ish 2D via first two standardized PCA-ish (SVD)
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    xy = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mask_g = d["danger"].values == "goal"
    ax.scatter(xy[~mask_g, 0], xy[~mask_g, 1], s=18, alpha=0.35, c="#e67e22", label="shot")
    ax.scatter(xy[mask_g, 0], xy[mask_g, 1], s=36, alpha=0.75, c="#c0392b", label="goal")
    ax.set_xlabel("Graph-signature dim 1")
    ax.set_ylabel("Graph-signature dim 2")
    ax.set_title("7. Case space — coverage-debt graph signatures")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs.append(savefig(fig, "07_case_space.png"))

    if not retrieval.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(retrieval["cosine_dist"], bins=15, color="#1a5276", edgecolor="white")
        ax.set_xlabel("Cosine distance to nearest neighbors")
        ax.set_ylabel("Pairs")
        ax.set_title("Retrieval distances for high-debt goals")
        fig.tight_layout()
        figs.append(savefig(fig, "07_retrieval_distances.png"))

    return retrieval, figs


# ---------- pages ----------


def page_hub(summary: dict) -> Path:
    body = f"""
  <h1>Interpretable graphs (small-n)</h1>
  <p class="meta">Approaches 1, 2, 6, 7 — structure you can read, not a deep GNN on 20 matches.</p>
  <div class="mission">
    <strong>Cup goal.</strong> Use graphs to understand defensive positioning: coverage debt,
    pressing linkage, and free attackers — with honest uncertainty under a small sample.
  </div>
  <ul>
    <li>Danger events in organised blocks: <strong>{summary['n_events']:,}</strong></li>
    <li>Matches: <strong>{summary['n_matches']}</strong></li>
    <li>Sparse CV AUC baseline → +event-graph:
      <strong>{summary['base_auc']:.3f} → {summary['full_auc']:.3f}</strong>
      (leakage-safe event features only)</li>
  </ul>
  <a class="card-link" href="coverage-debt-graph.html">
    <span class="title">2. Coverage-debt event graph</span>
    <span class="path">Nodes = ball, free attackers, pressers, far runner — edges you can explain</span>
  </a>
  <a class="card-link" href="graph-handcrafted.html">
    <span class="title">1. Hand-crafted graph features</span>
    <span class="path">Debt score, free roles, chain/far-side flags — no training</span>
  </a>
  <a class="card-link" href="graph-sparse-model.html">
    <span class="title">6. Regularized sparse residual model</span>
    <span class="path">L2 logistic + match CV — event-graph debt after box &amp; pressure</span>
  </a>
  <a class="card-link" href="graph-case-retrieval.html">
    <span class="title">7. Case-based retrieval</span>
    <span class="path">Find clips with a similar coverage-debt signature</span>
  </a>
"""
    return write_page("interpretable-graphs.html", "Interpretable graphs", body)


def page_coverage_debt(examples_figs: list[str], debt: pd.DataFrame) -> Path:
    figs = "".join(
        f'<figure><img src="eda/figures/graphs/{f}" alt="{f}" loading="lazy">'
        f"<figcaption>Example coverage-debt graph (interpretable structure).</figcaption></figure>"
        for f in examples_figs
    )
    body = f"""
  <h1>Coverage-debt event graph</h1>
  <p class="meta">Approach 2 — event-level graphs at shots and goals in organised blocks.</p>
  <div class="mission">
    <strong>Nodes.</strong> Ball-carrier · free attackers (Hungarian, &gt;8 m or not goal-side) ·
    concurrent pressing-chain defenders · far-side off-ball runner.<br>
    <strong>Edges.</strong> Unmarked (debt) · press-to-ball · weak-side leak.<br>
    No learned weights — the graph <em>is</em> the insight.
  </div>
  <ul>
    <li>Events graphed: <strong>{len(debt):,}</strong> ({int((debt.danger=='goal').sum())} goals)</li>
    <li>Mean free attackers at event: <strong>{debt['n_free'].mean():.1f}</strong></li>
    <li>Share with far-side runner: <strong>{100*debt['far_runner'].mean():.1f}%</strong></li>
    <li>Mean debt score: <strong>{debt['debt_score'].mean():.2f}</strong></li>
  </ul>
  <h2>Example graphs</h2>
  {figs}
  <h2>Why this fits small n</h2>
  <p>Hundreds of danger events give structure to inspect; we do not train a message-passing GNN
  on 20 matches. Hand-crafted scores and case retrieval use these danger graphs directly.
  The sparse residual model uses a parallel <em>event-only</em> debt signature at every phase end
  (press chain / far runner / inside shape) so Hungarian free-attacker rows — which exist only at
  shots and goals — cannot leak outcome into the AUC.</p>
"""
    return write_page("coverage-debt-graph.html", "Coverage-debt event graph", body)


def page_handcrafted(figs: list[str]) -> Path:
    imgs = "".join(
        f'<figure><img src="eda/figures/graphs/{f}" alt="{f}" loading="lazy"></figure>' for f in figs
    )
    body = f"""
  <h1>Hand-crafted graph features</h1>
  <p class="meta">Approach 1 — readable scores from the coverage-debt graph.</p>
  <div class="mission">
    On <em>danger</em> graphs: <code>debt_score</code> = free attackers + bonus for free CF / wide free
    + far-side runner + ball inside shape + pressing chain that did not regain.
    Tunable weights, not fitted weights. Charts below are descriptive of shots/goals.
  </div>
  {imgs}
  <h2>Feature list</h2>
  <ul>
    <li>Danger graphs: <code>n_free</code>, <code>cf_free</code>, <code>wide_free</code>, <code>free_roles</code></li>
    <li><code>n_chain_presses</code>, <code>chain_end</code>, <code>far_runner</code>, <code>inside_shape</code></li>
    <li><code>n_nodes</code>, <code>n_edges</code>, <code>edge_density</code></li>
    <li>Model (all phases): <code>event_debt</code> from press/far/inside only — no Hungarian</li>
  </ul>
"""
    return write_page("graph-handcrafted.html", "Hand-crafted graph features", body)


def page_sparse(stats: dict) -> Path:
    imgs = "".join(
        f'<figure><img src="eda/figures/graphs/{f}" alt="{f}" loading="lazy"></figure>'
        for f in stats["figs"]
    )
    top = stats["coef_table"].head(10)
    rows = "".join(
        f"<tr><td>{r.feature}</td><td>{r.family}</td><td>{r.coef:.3f}</td></tr>"
        for r in top.itertuples()
    )
    r = stats["residual"]
    body = f"""
  <h1>Regularized sparse residual model</h1>
  <p class="meta">Approach 6 — L2 logistic (C=0.2), match-grouped 5-fold CV. Shrinkage over cleverness.</p>
  <div class="mission">
    Question: after OOP box (area/width/length), mean on-ball pressure, and block height,
    do <em>event-graph</em> debt features (press chain, far-side runner, ball inside shape)
    still separate shot phases? Hungarian free attackers are <strong>not</strong> in this model —
    those frames are only labeled at shots/goals.
  </div>
  <ul>
    <li>Phases: <strong>{stats['n_phases']:,}</strong> across <strong>{stats['n_matches']}</strong> matches</li>
    <li>Baseline AUC: <strong>{stats['base_auc']:.3f}</strong></li>
    <li>Baseline + event-graph AUC: <strong>{stats['full_auc']:.3f}</strong>
      (Δ={stats['full_auc']-stats['base_auc']:+.3f})</li>
    <li>Residual <code>event_debt</code> (shot − no shot), match bootstrap mean
      <strong>{r['mean_diff_shot_minus_nons']:.3f}</strong>
      CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]
      — excludes 0: <strong>{r['excludes_zero']}</strong></li>
  </ul>
  {imgs}
  <h2>Largest |coefficients|</h2>
  <table><thead><tr><th>Feature</th><th>Family</th><th>Coef</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <p class="meta">AUC gains can be small under strong shrinkage — that is the point on 20 matches.
  Prefer residual CIs and film lists over claiming a large predictive leap.</p>
"""
    return write_page("graph-sparse-model.html", "Sparse residual model", body)


def page_retrieval(retrieval: pd.DataFrame, figs: list[str]) -> Path:
    imgs = "".join(
        f'<figure><img src="eda/figures/graphs/{f}" alt="{f}" loading="lazy"></figure>' for f in figs
    )
    rows = ""
    for _, r in retrieval.head(30).iterrows():
        rows += (
            f"<tr><td>{int(r['query_match'])}</td><td>{int(r['query_frame'])}</td>"
            f"<td>{r['query_debt']:.1f}</td><td>{r['query_free']}</td>"
            f"<td>{int(r['nn_match'])}</td><td>{int(r['nn_frame'])}</td>"
            f"<td>{r['nn_danger']}</td><td>{r['nn_free']}</td>"
            f"<td>{r['cosine_dist']:.3f}</td></tr>"
        )
    body = f"""
  <h1>Case-based retrieval</h1>
  <p class="meta">Approach 7 — the graph signature is a search key for film, not a leaderboard model.</p>
  <div class="mission">
    For each high-debt <strong>goal</strong>, retrieve the nearest coverage-debt graphs (cosine on
    standardized features). Use the pairs as “watch these together” clips.
  </div>
  {imgs}
  <h2>High-debt goals → nearest cases</h2>
  <table>
    <thead><tr>
      <th>Q match</th><th>Q frame</th><th>Q debt</th><th>Q free</th>
      <th>NN match</th><th>NN frame</th><th>NN type</th><th>NN free</th><th>Dist</th>
    </tr></thead>
    <tbody>{rows or '<tr><td colspan="9">No pairs</td></tr>'}</tbody>
  </table>
"""
    return write_page("graph-case-retrieval.html", "Case-based retrieval", body)


def main() -> None:
    style()
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Missing {DATA_DIR}")
    # networkx may need install
    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "networkx"])

    ids = discover_ids()
    print(f"Loading {len(ids)} matches…")
    events = load_events(ids)
    phases = load_phases(ids)
    hun = load_hungarian()
    print(f"  events={len(events):,} phases={len(phases):,} hun_rows={len(hun):,}")

    print("Building coverage-debt event graphs…")
    debt, examples = build_coverage_debt_events(events, phases, hun)
    debt.to_csv(OUT_DIR / "coverage_debt_events.csv", index=False)
    print(f"  {len(debt)} danger graphs")

    example_figs = []
    for i, ex in enumerate(examples[:4]):
        example_figs.append(draw_coverage_debt_schematic(ex, f"02_debt_graph_example_{i+1}.png"))

    print("Hand-crafted feature charts…")
    hc_figs = fig_debt_distributions(debt)

    print("Sparse residual model (phase-end features, no outcome leak)…")
    model_df = build_phase_modeling_table(debt, phases, events, hun)
    model_df.to_csv(OUT_DIR / "coverage_debt_phase_model.csv", index=False)
    sparse = run_sparse_residual(model_df)
    sparse["coef_table"].to_csv(OUT_DIR / "coverage_debt_sparse_coefs.csv", index=False)

    print("Case retrieval…")
    retrieval, ret_figs = build_case_retrieval(debt)
    retrieval.to_csv(OUT_DIR / "coverage_debt_retrieval.csv", index=False)

    summary = {
        "n_events": int(len(debt)),
        "n_matches": int(debt["match_id"].nunique()) if len(debt) else 0,
        "base_auc": sparse["base_auc"],
        "full_auc": sparse["full_auc"],
        "residual": sparse["residual"],
        "leakage_safe": sparse.get("leakage_safe", True),
        "graph_feature_note": sparse.get("graph_feature_note", ""),
    }
    with open(OUT_DIR / "coverage_debt_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Writing pages…")
    page_hub(summary)
    page_coverage_debt(example_figs, debt)
    page_handcrafted(hc_figs)
    page_sparse(sparse)
    page_retrieval(retrieval, ret_figs)
    print("Done → docs/interpretable-graphs.html")


if __name__ == "__main__":
    main()
