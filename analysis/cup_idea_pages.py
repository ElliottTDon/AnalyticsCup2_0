"""Build separate GitHub Pages for cup-goal ideas from graph-tool-notes.

Pages:
  1. team-zones.html          — already built by eda_visual_defense (linked)
  2. marking-debt.html        — unmarked / chance roles at set pieces & cutbacks
  3. compact-weak-coupling.html — coach clip list: compact box + free attackers
  4. pressing-chain-weak-side.html — pressing chains vs far-side runners
  5. cup-ideas.html           — hub linking all four
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
MATCHES_JSON = WORKSPACE / "opendata" / "data" / "matches.json"
OUT_DIR = WORKSPACE / "analysis" / "output"
FIG_DIR = OUT_DIR / "figures" / "ideas"
DOCS = WORKSPACE / "docs"
DOCS_FIG = DOCS / "eda" / "figures" / "ideas"

ORGANIZED = ("low_block", "medium_block", "high_block")
ATTACK_ROLES = {"CF", "RW", "LW", "AM", "RM", "LM", "RDM", "LDM", "LF", "RF", "SS"}
WIDE = {"wide_left", "wide_right", "half_space_left", "half_space_right", "left_halfspace", "right_halfspace"}
CHANNEL_FLIP = {
    "wide_left": "wide_right",
    "wide_right": "wide_left",
    "half_space_left": "half_space_right",
    "half_space_right": "half_space_left",
    "left_halfspace": "right_halfspace",
    "right_halfspace": "left_halfspace",
}

CSS = """
:root { color-scheme: light dark; }
body {
  margin: 0 auto; max-width: 44rem; padding: 1.1rem 0.9rem 3rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55; font-size: 1.05rem;
}
h1 { font-size: 1.45rem; margin: 0 0 0.4rem; }
h2 { font-size: 1.12rem; margin: 1.6rem 0 0.55rem; padding-top: 0.45rem; border-top: 1px solid #ddd; }
@media (prefers-color-scheme: dark) { h2 { border-top-color: #444; } }
.mission {
  margin: 0.7rem 0 1rem; padding: 0.75rem 0.85rem; border-left: 3px solid #888;
  background: rgba(127,127,127,0.08);
}
.meta { color: #666; font-size: 0.92rem; }
@media (prefers-color-scheme: dark) { .meta { color: #aaa; } }
figure { margin: 0 0 1.25rem; }
img { width: 100%; height: auto; border-radius: 4px; background: #f3f3f3; display: block; }
figcaption { margin-top: 0.4rem; font-size: 0.95rem; color: #444; }
@media (prefers-color-scheme: dark) { figcaption { color: #bbb; } img { background: #222; } }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0.75rem 0 1.25rem; }
th, td { text-align: left; padding: 0.45rem 0.35rem; border-bottom: 1px solid #ddd; }
@media (prefers-color-scheme: dark) { th, td { border-bottom-color: #444; } }
.nav a { display: inline-block; margin: 0.35rem 1rem 0.35rem 0; color: inherit; }
ul { padding-left: 1.2rem; }
li { margin: 0 0 0.5rem; }
.card-link {
  display: block; padding: 0.85rem 0.2rem; border-bottom: 1px solid #eee;
  text-decoration: none; color: inherit;
}
@media (prefers-color-scheme: dark) { .card-link { border-bottom-color: #333; } }
.card-link:hover { background: rgba(127,127,127,0.1); }
.card-link .title { font-weight: 600; display: block; }
.card-link .path { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
"""


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "axes.titlesize": 12})


def discover_ids() -> list[str]:
    return sorted(
        d.name
        for d in DATA_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}_phases_of_play.csv").is_file()
        and (d / f"{d.name}_dynamic_events.csv").is_file()
    )


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
  <a href="cup-ideas.html">Cup ideas hub</a>
  <a href="graph-tool-notes.html">Graph FAQ</a>
  <a href="eda.html">Visual EDA</a>
  <a href="index.html">Index</a>
</p>
</body></html>
"""
    out = DOCS / name
    out.write_text(html)
    return out


# ---------- loaders ----------


def load_all_events(match_ids: list[str]) -> pd.DataFrame:
    usecols = [
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
        "channel_end",
        "third_start",
        "third_end",
        "x_start",
        "y_start",
        "team_out_of_possession_phase_type",
        "pressing_chain",
        "pressing_chain_length",
        "pressing_chain_end_type",
        "pressing_chain_index",
        "frame_start",
        "frame_end",
        "phase_index",
        "associated_off_ball_run_subtype",
        "penalty_area_start",
        "penalty_area_end",
        "n_simultaneous_runs",
        "simultaneous_defensive_engagement_same_target",
        "inside_defensive_shape_start",
        "beaten_by_possession",
        "beaten_by_movement",
        "penalty_area_start",
        "penalty_area_end",
    ]
    frames = []
    for mid in match_ids:
        path = DATA_DIR / mid / f"{mid}_dynamic_events.csv"
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in usecols if c in header]
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        df["match_id"] = int(mid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_all_phases(match_ids: list[str]) -> pd.DataFrame:
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
    return ph


# ---------- 1) Marking debt (set pieces + cutbacks) ----------


def build_marking_debt(events: pd.DataFrame) -> dict:
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    ob = events.loc[events["event_type"] == "off_ball_run"].copy()

    # Set pieces: OOP phase defending_set_play
    set_poss = poss.loc[poss["team_out_of_possession_phase_type"] == "defending_set_play"].copy()
    set_danger = set_poss.loc[set_poss["lead_to_shot"] | set_poss["lead_to_goal"]].copy()
    set_danger["role"] = set_danger["player_position"].fillna("?")

    # Cutbacks / crosses: cross_receiver runs + wide → box possessions that shoot
    cross_runs = ob.loc[ob["event_subtype"] == "cross_receiver"].copy()
    cross_runs["role"] = cross_runs["player_position"].fillna("?")
    wide_box = poss.loc[
        poss["channel_start"].isin(WIDE)
        & (
            (poss.get("penalty_area_end") == True)  # noqa: E712
            | (poss["third_end"] == "attacking_third")
            | (poss["third_start"] == "attacking_third")
        )
    ].copy()
    cutback_proxy = wide_box.loc[wide_box["lead_to_shot"] | wide_box["lead_to_goal"]].copy()
    cutback_proxy["role"] = cutback_proxy["player_position"].fillna("?")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, df, title, color in [
        (axes[0], set_danger, "Set pieces — who has the ball when a shot follows", "#6c3483"),
        (axes[1], cross_runs.loc[cross_runs["lead_to_shot"] | cross_runs["lead_to_goal"]], "Cross receivers — who is free when it becomes a chance", "#c0392b"),
    ]:
        if df.empty:
            ax.text(0.5, 0.5, "No rows", ha="center")
            ax.set_title(title)
            continue
        counts = df["role"].value_counts().head(12)
        ax.barh(counts.index.astype(str)[::-1], counts.values[::-1], color=color)
        ax.set_xlabel("Count")
        ax.set_title(title)
    fig.suptitle("Marking debt by role: set pieces and cutback/cross moments", y=1.02)
    fig.tight_layout()
    f1 = savefig(fig, "marking_debt_roles.png")

    # Shot rates by role in set play vs open play organized
    set_g = (
        set_poss.groupby(set_poss["player_position"].fillna("?"))
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"), goals=("lead_to_goal", "sum"))
        .query("n >= 5")
        .sort_values("shot_rate", ascending=False)
        .head(10)
    )
    org = poss.loc[poss["team_out_of_possession_phase_type"].isin(ORGANIZED)]
    org_g = (
        org.groupby(org["player_position"].fillna("?"))
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .query("n >= 30")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    roles = list(set_g.index)
    x = np.arange(len(roles))
    set_rates = set_g.loc[roles, "shot_rate"] * 100
    org_rates = [org_g.loc[r, "shot_rate"] * 100 if r in org_g.index else np.nan for r in roles]
    ax.bar(x - 0.2, set_rates, 0.4, label="Set piece", color="#6c3483")
    ax.bar(x + 0.2, org_rates, 0.4, label="Organised block", color="#1e8449")
    ax.set_xticks(x)
    ax.set_xticklabels(roles, rotation=30, ha="right")
    ax.set_ylabel("Shot rate after possession (%)")
    ax.set_title("Same role, different context: set piece vs organised block")
    ax.legend(frameon=False)
    fig.tight_layout()
    f2 = savefig(fig, "marking_debt_set_vs_block.png")

    # Cutback channel map
    cut_all = cutback_proxy.copy()
    if not cut_all.empty:
        g = cut_all.groupby(["channel_start", "player_position"]).size().reset_index(name="n")
        top = g.sort_values("n", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"{r.channel_start} · {r.player_position}" for r in top.itertuples()]
        ax.barh(labels[::-1], top["n"].values[::-1], color="#e67e22")
        ax.set_xlabel("Wide → attacking-third shots")
        ax.set_title("Cutback / wide delivery debt — channel × ball-carrier role")
        fig.tight_layout()
        f3 = savefig(fig, "marking_debt_cutback_channels.png")
    else:
        f3 = None

    # Clip table: set-piece goals + cutback goals
    clips = []
    goals_set = set_danger.loc[set_danger["lead_to_goal"], ["match_id", "frame_end", "player_position", "channel_start", "team_shortname"]].copy()
    goals_set["context"] = "set_piece_goal"
    goals_cut = cutback_proxy.loc[cutback_proxy["lead_to_goal"], ["match_id", "frame_end", "player_position", "channel_start", "team_shortname"]].copy()
    goals_cut["context"] = "wide_cutback_goal"
    shots_cross = cross_runs.loc[cross_runs["lead_to_goal"] | cross_runs["lead_to_shot"], ["match_id", "frame_end", "player_position", "channel_start", "team_shortname"]].copy()
    shots_cross["context"] = "cross_receiver_chance"
    for part in (goals_set, goals_cut, shots_cross):
        if not part.empty:
            clips.append(part)
    clip_df = pd.concat(clips, ignore_index=True) if clips else pd.DataFrame()
    if not clip_df.empty:
        clip_df = clip_df.sort_values(["context", "match_id", "frame_end"]).head(40)

    return {
        "figs": [f for f in (f1, f2, f3) if f],
        "set_g": set_g,
        "n_set_poss": int(len(set_poss)),
        "n_set_shots": int(set_poss["lead_to_shot"].sum()),
        "n_cross": int(len(cross_runs)),
        "n_cross_shot": int((cross_runs["lead_to_shot"]).sum()),
        "n_cutback_shots": int(len(cutback_proxy)),
        "clips": clip_df,
    }


def page_marking_debt(stats: dict) -> Path:
    figs = "".join(
        f'<figure><img src="eda/figures/ideas/{f}" alt="{f}" loading="lazy">'
        f"<figcaption>{f.replace('_', ' ').replace('.png', '')}</figcaption></figure>"
        for f in stats["figs"]
    )
    rows = ""
    if not stats["clips"].empty:
        for _, r in stats["clips"].head(25).iterrows():
            rows += (
                f"<tr><td>{r['context']}</td><td>{int(r['match_id'])}</td>"
                f"<td>{int(r['frame_end']) if pd.notna(r['frame_end']) else ''}</td>"
                f"<td>{r['player_position']}</td><td>{r['channel_start']}</td>"
                f"<td>{r['team_shortname']}</td></tr>"
            )
    body = f"""
  <h1>Marking debt by role</h1>
  <p class="meta">Set pieces and cutback / cross moments — who is unmarked when the chance arrives?</p>
  <div class="mission">
    <strong>Cup link.</strong> Defensive positioning is not only block height. At set pieces and wide deliveries,
    debt sits on specific roles (CF vs winger vs far-side full-back).
  </div>
  <ul>
    <li>Set-piece possessions: <strong>{stats['n_set_poss']:,}</strong> ({stats['n_set_shots']:,} → shot)</li>
    <li>Cross-receiver runs: <strong>{stats['n_cross']:,}</strong> ({stats['n_cross_shot']:,} → shot)</li>
    <li>Wide → attacking-third shot proxies (cutback-like): <strong>{stats['n_cutback_shots']:,}</strong></li>
  </ul>
  <h2>Charts</h2>
  {figs}
  <h2>Film list (sample)</h2>
  <p class="meta">Match id + frame end — pull these clips in SkillCorner / your video tool.</p>
  <table>
    <thead><tr><th>Context</th><th>Match</th><th>Frame</th><th>Role</th><th>Channel</th><th>Team on ball</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="6">No goal/shot clips in filters</td></tr>'}</tbody>
  </table>
"""
    return write_page("marking-debt.html", "Marking debt by role", body)


# ---------- 2) Compact box + weak coupling clip list ----------


def build_compact_weak(phases: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Compact OOP box + signs of weak coverage (free/inside shape / beaten)."""
    org = phases.loc[phases["oop"].isin(ORGANIZED)].copy()
    # Compact = bottom tercile of box_area within each block
    org["area_tercile"] = org.groupby("oop")["box_area"].transform(
        lambda s: pd.qcut(s, 3, labels=["tight", "mid", "open"], duplicates="drop")
    )
    tight = org.loc[org["area_tercile"] == "tight"].copy()

    # Weak coverage proxies from engagements in same phase window via phase_index + match
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    # beaten flags if present
    for col in ("beaten_by_possession", "beaten_by_movement"):
        if col not in eng.columns:
            eng[col] = False

    # possessions inside shape
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    if "inside_defensive_shape_start" in poss.columns:
        inside = poss.loc[poss["inside_defensive_shape_start"] == True]  # noqa: E712
    else:
        inside = poss.iloc[0:0]

    # Hungarian free roles (precomputed) — high free count at danger frames
    hun_path = OUT_DIR / "hungarian_free_roles.csv"
    hun = pd.read_csv(hun_path) if hun_path.is_file() else pd.DataFrame()
    if not hun.empty:
        free_by_frame = (
            hun.loc[hun["free_role"].isin(ATTACK_ROLES)]
            .groupby(["match_id", "frame", "danger"])
            .agg(n_free_attack=("free_role", "nunique"), free_roles=("free_role", lambda s: ",".join(sorted(set(s)))))
            .reset_index()
        )
    else:
        free_by_frame = pd.DataFrame()

    # Join tight phases that lead to shot OR have inside-shape possession
    tight["phase_index"] = tight["index"] if "index" in tight.columns else tight.get("phase_index")
    clips = []
    for _, ph in tight.loc[tight["shot"] | tight["goal"]].iterrows():
        mid = int(ph["match_id"])
        f0, f1 = int(ph["frame_start"]), int(ph["frame_end"])
        reason = []
        free_roles = ""
        n_free = 0
        if not free_by_frame.empty:
            sub = free_by_frame.loc[
                (free_by_frame["match_id"] == mid)
                & (free_by_frame["frame"] >= f0)
                & (free_by_frame["frame"] <= f1 + 25)
            ]
            if not sub.empty:
                best = sub.sort_values("n_free_attack", ascending=False).iloc[0]
                n_free = int(best["n_free_attack"])
                free_roles = best["free_roles"]
                if n_free >= 2:
                    reason.append(f"{n_free} free attackers ({free_roles})")
        # inside shape during phase
        if not inside.empty:
            ins = inside.loc[
                (inside["match_id"] == mid)
                & (inside["frame_start"].between(f0, f1) | inside["frame_end"].between(f0, f1))
            ]
            if len(ins):
                reason.append("ball inside defensive shape")
        if not reason:
            reason.append("tight box + shot against (coverage not measured)")
        clips.append(
            {
                "match_id": mid,
                "frame_start": f0,
                "frame_end": f1,
                "oop": ph["oop"],
                "box_area": round(float(ph["box_area"]), 1),
                "width": round(float(ph["width"]), 1),
                "length": round(float(ph["length"]), 1),
                "goal": bool(ph["goal"]),
                "attack_team": ph.get("team_in_possession_shortname"),
                "n_free": n_free,
                "free_roles": free_roles,
                "why": "; ".join(reason),
            }
        )
    clip_df = pd.DataFrame(clips)
    if not clip_df.empty:
        # Prefer high free-attacker disagreements
        clip_df["priority"] = clip_df["n_free"] * 10 + clip_df["goal"].astype(int) * 5
        clip_df = clip_df.sort_values(["priority", "box_area"], ascending=[False, True]).head(50)

    # Chart: shot rate tight vs open by block
    g = (
        org.groupby(["oop", "area_tercile"], observed=False)
        .agg(n=("shot", "size"), shot_rate=("shot", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for block in ORGANIZED:
        sub = g.loc[g["oop"] == block]
        order = ["tight", "mid", "open"]
        sub = sub.set_index("area_tercile").reindex(order)
        ax.plot(order, sub["shot_rate"] * 100, marker="o", label=block.replace("_", " "))
    ax.set_ylabel("Shot rate against (%)")
    ax.set_xlabel("Defensive box size (within block)")
    ax.set_title("Compact boxes still concede — look for coverage failures next")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    f1 = savefig(fig, "compact_vs_open_shot_rate.png")

    # Free attackers among tight+shot clips
    if not clip_df.empty and clip_df["n_free"].gt(0).any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(clip_df["n_free"], bins=range(0, int(clip_df["n_free"].max()) + 2), color="#1a5276", edgecolor="white")
        ax.set_xlabel("Unmarked attacking roles in phase (Hungarian)")
        ax.set_ylabel("Clip count")
        ax.set_title("Tight-box shot phases: how many attackers look free?")
        fig.tight_layout()
        f2 = savefig(fig, "compact_free_attacker_hist.png")
    else:
        f2 = None

    return {"figs": [f for f in (f1, f2) if f], "clips": clip_df, "n_tight_shot": int(len(clips))}


def page_compact_weak(stats: dict) -> Path:
    figs = "".join(
        f'<figure><img src="eda/figures/ideas/{f}" alt="{f}" loading="lazy"></figure>'
        for f in stats["figs"]
    )
    rows = ""
    for _, r in stats["clips"].head(30).iterrows():
        rows += (
            f"<tr><td>{int(r['match_id'])}</td><td>{int(r['frame_start'])}–{int(r['frame_end'])}</td>"
            f"<td>{r['oop']}</td><td>{r['box_area']}</td><td>{r['n_free']}</td>"
            f"<td>{r['free_roles']}</td><td>{'goal' if r['goal'] else 'shot'}</td>"
            f"<td>{r['why']}</td></tr>"
        )
    body = f"""
  <h1>Compact box, weak coverage</h1>
  <p class="meta">Coach clip list — the block looks tight, but marking / coupling still fails.</p>
  <div class="mission">
    <strong>Cup link.</strong> Bounding-box compactness is a baseline. The interesting film is when the
    box is <em>small</em> and attackers are still free (Hungarian free roles ≥ 2) or the ball is inside the shape.
  </div>
  <p>{stats['n_tight_shot']} organised-block phases with a <strong>tight</strong> box that still ended in a shot/goal.</p>
  <h2>Charts</h2>
  {figs}
  <h2>Clip list</h2>
  <p class="meta">Prioritised by free attackers + goals. Open match + frame range in your video tool.</p>
  <table>
    <thead><tr><th>Match</th><th>Frames</th><th>Block</th><th>Box m²</th><th>Free</th><th>Roles</th><th>End</th><th>Why</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="8">No clips</td></tr>'}</tbody>
  </table>
"""
    return write_page("compact-weak-coupling.html", "Compact box, weak coverage", body)


# ---------- 3) Pressing chain vs far-side free runner ----------


def _is_far_side(ch_a: str, ch_b: str) -> bool:
    if pd.isna(ch_a) or pd.isna(ch_b):
        return False
    return CHANNEL_FLIP.get(str(ch_a)) == str(ch_b) or (
        "left" in str(ch_a) and "right" in str(ch_b)
    ) or (
        "right" in str(ch_a) and "left" in str(ch_b)
    )


def build_pressing_chains(events: pd.DataFrame) -> dict:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["in_chain"] = eng["pressing_chain"].fillna(False).astype(bool)
    chains = eng.loc[eng["in_chain"]].copy()
    chains["end"] = chains["pressing_chain_end_type"].fillna("unknown")
    chains["regain"] = chains["end"].eq("regain")

    # Far-side runners: off-ball runs overlapping chain frames on flipped channel
    runs = events.loc[events["event_type"] == "off_ball_run"].copy()

    chain_rows = []
    # Aggregate by match + pressing_chain_index when available
    group_cols = ["match_id"]
    if "pressing_chain_index" in chains.columns and chains["pressing_chain_index"].notna().any():
        group_cols.append("pressing_chain_index")
    else:
        chains["_gid"] = (
            chains["match_id"].astype(str)
            + "_"
            + chains["frame_start"].astype(str)
        )
        group_cols = ["match_id", "_gid"]

    for keys, g in chains.groupby(group_cols, dropna=False):
        if isinstance(keys, tuple):
            mid = int(keys[0])
        else:
            mid = int(keys)
        f0, f1 = int(g["frame_start"].min()), int(g["frame_end"].max())
        press_channels = set(g["channel_start"].dropna().astype(str))
        end = g["end"].mode().iloc[0] if len(g["end"].mode()) else "unknown"
        lead_shot = bool(g["lead_to_shot"].max())
        lead_goal = bool(g["lead_to_goal"].max()) if "lead_to_goal" in g else False
        length = g["pressing_chain_length"].max() if "pressing_chain_length" in g else len(g)

        far_runs = runs.loc[
            (runs["match_id"] == mid)
            & (runs["frame_start"] <= f1 + 15)
            & (runs["frame_end"] >= f0 - 15)
        ]
        far_side = False
        far_role = ""
        far_channel = ""
        for _, rr in far_runs.iterrows():
            for pch in press_channels:
                if _is_far_side(pch, rr.get("channel_start")):
                    far_side = True
                    far_role = rr.get("player_position") or ""
                    far_channel = rr.get("channel_start") or ""
                    break
            if far_side:
                break
        chain_rows.append(
            {
                "match_id": mid,
                "frame_start": f0,
                "frame_end": f1,
                "length": float(length) if pd.notna(length) else len(g),
                "end": end,
                "regain": end == "regain",
                "lead_to_shot": lead_shot,
                "lead_to_goal": lead_goal,
                "far_side_runner": far_side,
                "far_role": far_role,
                "far_channel": far_channel,
                "press_channels": ",".join(sorted(press_channels)),
            }
        )
    cdf = pd.DataFrame(chain_rows)

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if not cdf.empty:
        ends = cdf["end"].value_counts()
        axes[0].barh(ends.index.astype(str), ends.values, color="#117a65")
        axes[0].set_title("How pressing chains end")
        axes[0].set_xlabel("Chains")

        # Shot rate: regain vs not, with/without far-side runner
        pivot = (
            cdf.groupby(["regain", "far_side_runner"])
            .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
            .reset_index()
        )
        labels = [
            f"{'Regain' if r.regain else 'No regain'}\n{'+ far runner' if r.far_side_runner else 'no far runner'}\n(n={int(r.n)})"
            for r in pivot.itertuples()
        ]
        axes[1].bar(range(len(pivot)), pivot["shot_rate"] * 100, color=["#1e8449", "#e67e22", "#85929e", "#c0392b"][: len(pivot)])
        axes[1].set_xticks(range(len(pivot)))
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel("Shot rate after chain (%)")
        axes[1].set_title("Regain vs far-side free runner")
    fig.suptitle("Pressing chains: win the ball or leave the far side open?", y=1.03)
    fig.tight_layout()
    f1 = savefig(fig, "pressing_chain_outcomes.png")

    # Far-side role when chain does not regain
    no_regain_far = cdf.loc[~cdf["regain"] & cdf["far_side_runner"]]
    if not no_regain_far.empty:
        vc = no_regain_far["far_role"].fillna("?").value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color="#c0392b")
        ax.set_xlabel("Chains with far-side runner (no regain)")
        ax.set_title("Which role runs free on the far side?")
        fig.tight_layout()
        f2 = savefig(fig, "pressing_chain_far_roles.png")
    else:
        f2 = None

    # Clip list: no regain + far side + shot
    danger = cdf.loc[(~cdf["regain"]) & cdf["far_side_runner"] & (cdf["lead_to_shot"] | cdf["lead_to_goal"])]
    danger = danger.sort_values(["lead_to_goal", "length"], ascending=[False, False]).head(40)

    summary = {
        "n_chains": int(len(cdf)),
        "regain_rate": float(cdf["regain"].mean()) if len(cdf) else 0,
        "far_rate": float(cdf["far_side_runner"].mean()) if len(cdf) else 0,
        "shot_if_far_no_regain": float(
            cdf.loc[~cdf["regain"] & cdf["far_side_runner"], "lead_to_shot"].mean()
        )
        if len(cdf.loc[~cdf["regain"] & cdf["far_side_runner"]])
        else None,
        "shot_if_regain": float(cdf.loc[cdf["regain"], "lead_to_shot"].mean()) if cdf["regain"].any() else None,
    }
    return {"figs": [f for f in (f1, f2) if f], "clips": danger, "summary": summary, "cdf": cdf}


def page_pressing_chains(stats: dict) -> Path:
    s = stats["summary"]
    figs = "".join(
        f'<figure><img src="eda/figures/ideas/{f}" alt="{f}" loading="lazy"></figure>'
        for f in stats["figs"]
    )
    rows = ""
    for _, r in stats["clips"].head(30).iterrows():
        rows += (
            f"<tr><td>{int(r['match_id'])}</td><td>{int(r['frame_start'])}–{int(r['frame_end'])}</td>"
            f"<td>{r['length']}</td><td>{r['end']}</td><td>{r['far_role']}</td>"
            f"<td>{r['far_channel']}</td><td>{r['press_channels']}</td>"
            f"<td>{'goal' if r['lead_to_goal'] else 'shot'}</td></tr>"
        )
    shot_far = f"{s['shot_if_far_no_regain']*100:.1f}%" if s["shot_if_far_no_regain"] is not None else "n/a"
    shot_reg = f"{s['shot_if_regain']*100:.1f}%" if s["shot_if_regain"] is not None else "n/a"
    body = f"""
  <h1>Pressing chains vs far-side runners</h1>
  <p class="meta">Does the chain win the ball — or open the weak side?</p>
  <div class="mission">
    <strong>Cup link.</strong> On-ball pressure is only half the story. A pressing chain that recovers the ball
    is good defending; a chain that leaves a free runner on the opposite flank is a positioning decision.
  </div>
  <ul>
    <li>Chains analysed: <strong>{s['n_chains']:,}</strong></li>
    <li>End in regain: <strong>{s['regain_rate']*100:.1f}%</strong></li>
    <li>Far-side off-ball runner during chain: <strong>{s['far_rate']*100:.1f}%</strong></li>
    <li>Shot rate after regain: <strong>{shot_reg}</strong></li>
    <li>Shot rate when no regain + far-side runner: <strong>{shot_far}</strong></li>
  </ul>
  <h2>Charts</h2>
  {figs}
  <h2>Film list — no regain, far-side runner, chance</h2>
  <table>
    <thead><tr><th>Match</th><th>Frames</th><th>Length</th><th>End</th><th>Far role</th><th>Far channel</th><th>Press side</th><th>End</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="8">No matching chains</td></tr>'}</tbody>
  </table>
"""
    return write_page("pressing-chain-weak-side.html", "Pressing chains vs far-side runners", body)


# ---------- Hub ----------


def page_hub() -> Path:
    body = """
  <h1>Cup-goal idea pages</h1>
  <p class="meta">Defensive positioning — separate pages for each idea from the block-graph notes.</p>
  <div class="mission">
    How can tracking and contextual data help us understand how players and teams defend?
    Shape, spacing, pressure, compactness, marking, decision-making.
  </div>
  <a class="card-link" href="team-zones.html">
    <span class="title">1. Team-specific shot maps</span>
    <span class="path">Where each club scores and is broken — not the league-average centre blob</span>
  </a>
  <a class="card-link" href="marking-debt.html">
    <span class="title">2. Marking debt by role</span>
    <span class="path">Set pieces and cutbacks / crosses — CF vs wingers vs far-side debt</span>
  </a>
  <a class="card-link" href="compact-weak-coupling.html">
    <span class="title">3. Compact box, weak coverage</span>
    <span class="path">Coach clip list — tight shape that still leaves attackers free</span>
  </a>
  <a class="card-link" href="pressing-chain-weak-side.html">
    <span class="title">4. Pressing chains vs far-side runners</span>
    <span class="path">Regain the ball or leave the opposite flank open</span>
  </a>
  <h2>Related</h2>
  <ul>
    <li><a href="graph-tool-notes.html">Block graph FAQ</a> (timer series, pressure, pre-fracturing)</li>
    <li><a href="eda.html">Visual EDA</a></li>
    <li><a href="defend.html">Graph notebook</a></li>
  </ul>
"""
    return write_page("cup-ideas.html", "Cup-goal idea pages", body)


def main() -> None:
    style()
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Missing {DATA_DIR}")
    ids = discover_ids()
    print(f"Loading {len(ids)} matches…")
    events = load_all_events(ids)
    phases = load_all_phases(ids)
    print(f"  events={len(events):,} phases={len(phases):,}")

    print("Marking debt…")
    debt = build_marking_debt(events)
    page_marking_debt(debt)
    debt["clips"].to_csv(OUT_DIR / "marking_debt_clips.csv", index=False)

    print("Compact + weak coverage…")
    compact = build_compact_weak(phases, events)
    page_compact_weak(compact)
    compact["clips"].to_csv(OUT_DIR / "compact_weak_clips.csv", index=False)

    print("Pressing chains…")
    chains = build_pressing_chains(events)
    page_pressing_chains(chains)
    chains["clips"].to_csv(OUT_DIR / "pressing_chain_clips.csv", index=False)

    print("Hub…")
    page_hub()
    print("Done → docs/cup-ideas.html + three idea pages")


if __name__ == "__main__":
    main()
