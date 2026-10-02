"""Visual EDA: defensive positioning across SkillCorner Open Data.

Focus: team shape, spacing, pressure, compactness, marking, and defensive
decision-making — how contextual tracking-adjacent data helps evaluate defense.

Writes figures + summaries under analysis/output/ and a phone-first HTML report
under docs/eda.html (plus docs/eda/figures/).
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
FIG_DIR = OUT_DIR / "figures"
DOCS_DIR = WORKSPACE / "docs"
DOCS_FIG = DOCS_DIR / "eda" / "figures"

ORGANIZED_BLOCKS = ("low_block", "medium_block", "high_block")
PRESSURE_ORDER = [
    "no_pressure",
    "low_pressure",
    "medium_pressure",
    "high_pressure",
    "very_high_pressure",
]
ENGAGEMENT_SUBTYPES = (
    "pressure",
    "pressing",
    "recovery_press",
    "counter_press",
    "other",
)
BLOCK_COLORS = {
    "low_block": "#1b4f72",
    "medium_block": "#1e8449",
    "high_block": "#b9770e",
    "chaotic": "#7f8c8d",
    "defending_set_play": "#6c3483",
    "defending_direct": "#922b21",
    "defending_transition": "#117a65",
    "defending_quick_break": "#c0392b",
}

# Pitch roughly SkillCorner-style: x in [-52.5, 52.5], y in [-34, 34]
PITCH_X = (-52.5, 52.5)
PITCH_Y = (-34.0, 34.0)

PHASE_COLS = [
    "match_id",
    "duration",
    "team_possession_lead_to_shot",
    "team_possession_lead_to_goal",
    "team_in_possession_shortname",
    "team_out_of_possession_phase_type",
    "team_out_of_possession_width_start",
    "team_out_of_possession_width_end",
    "team_out_of_possession_length_start",
    "team_out_of_possession_length_end",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "third_start",
    "channel_start",
    "third_end",
    "channel_end",
]

EVENT_COLS = [
    "match_id",
    "event_type",
    "event_subtype",
    "end_type",
    "lead_to_shot",
    "lead_to_goal",
    "team_shortname",
    "player_name",
    "player_position",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "third_start",
    "channel_start",
    "overall_pressure_start",
    "overall_pressure_end",
    "possession_epv_delta_against",
    "team_out_of_possession_phase_type",
    "organised_defense",
    "defensive_structure",
    "n_defensive_lines",
    "inside_defensive_shape_start",
    "inside_defensive_shape_end",
    "last_defensive_line_height_start",
    "last_defensive_line_height_end",
    "force_backward",
    "stop_possession_danger",
    "reduce_possession_danger",
    "beaten_by_possession",
    "beaten_by_movement",
    "pressing_chain",
    "pressing_chain_length",
    "pressing_chain_end_type",
    "simultaneous_defensive_engagement_same_target",
    "distance_to_player_in_possession_start",
    "distance_to_player_in_possession_end",
    "interplayer_distance_start",
    "player_in_possession_position",
    "frame_end",
    "phase_index",
    "interplayer_distance_min",
    "angle_of_engagement",
    "goal_side_start",
]


def style_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "grid.color": "#e8e8e8",
            "axes.titleweight": "bold",
        }
    )


def discover_match_ids(data_dir: Path) -> list[str]:
    ids: list[str] = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        mid = d.name
        if (d / f"{mid}_phases_of_play.csv").is_file() and (
            d / f"{mid}_dynamic_events.csv"
        ).is_file():
            ids.append(mid)
    return ids


def load_match_meta(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    return pd.DataFrame(
        [
            {
                "match_id": int(r["id"]),
                "date_time": r["date_time"],
                "home": r["home_team"]["short_name"],
                "away": r["away_team"]["short_name"],
            }
            for r in rows
        ]
    )


def load_all(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    match_ids = discover_match_ids(data_dir)
    phase_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for mid in match_ids:
        d = data_dir / mid
        phases = pd.read_csv(d / f"{mid}_phases_of_play.csv")
        events = pd.read_csv(d / f"{mid}_dynamic_events.csv", low_memory=False)
        # Keep available columns only (schema-safe)
        phase_frames.append(phases[[c for c in PHASE_COLS if c in phases.columns]])
        event_frames.append(events[[c for c in EVENT_COLS if c in events.columns]])
    phases_all = pd.concat(phase_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)
    return phases_all, events_all, match_ids


def enrich_phases(phases: pd.DataFrame) -> pd.DataFrame:
    df = phases.copy()
    df["oop_phase"] = df["team_out_of_possession_phase_type"]
    df["width"] = df[
        ["team_out_of_possession_width_start", "team_out_of_possession_width_end"]
    ].mean(axis=1)
    df["length"] = df[
        ["team_out_of_possession_length_start", "team_out_of_possession_length_end"]
    ].mean(axis=1)
    df["box_area"] = df["width"] * df["length"]
    df["aspect"] = df["width"] / df["length"].replace(0, np.nan)
    df["shot"] = df["team_possession_lead_to_shot"].astype(bool)
    df["goal"] = df["team_possession_lead_to_goal"].astype(bool)
    df["defending_team"] = df["team_in_possession_shortname"]  # placeholder overwritten
    # Defending team = the side NOT in possession
    # We only have team_in_possession; defending team inferred per match later if needed.
    df["is_organized"] = df["oop_phase"].isin(ORGANIZED_BLOCKS)
    return df


def defending_team_from_possession(
    phases: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Map team_in_possession -> defending opponent via home/away."""
    meta = meta.copy()
    meta["match_id"] = meta["match_id"].astype(int)
    df = phases.copy()
    df["match_id"] = df["match_id"].astype(int)
    df = df.merge(meta, on="match_id", how="left")

    def opp(row):
        tip = row["team_in_possession_shortname"]
        if tip == row["home"]:
            return row["away"]
        if tip == row["away"]:
            return row["home"]
        # fuzzy: short names sometimes differ slightly
        return None

    df["defending_team"] = df.apply(opp, axis=1)
    return df


def savefig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(*PITCH_X)
    ax.set_ylim(*PITCH_Y)
    ax.set_aspect("equal")
    # Outer lines
    ax.plot(
        [PITCH_X[0], PITCH_X[1], PITCH_X[1], PITCH_X[0], PITCH_X[0]],
        [PITCH_Y[0], PITCH_Y[0], PITCH_Y[1], PITCH_Y[1], PITCH_Y[0]],
        color="#333",
        lw=1.2,
    )
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.add_patch(
        plt.Circle((0, 0), 9.15, fill=False, color="#888", lw=0.8)
    )
    for x in (PITCH_X[0], PITCH_X[1]):
        # penalty box approx
        sign = 1 if x > 0 else -1
        ax.plot(
            [x, x - sign * 16.5, x - sign * 16.5, x],
            [-20.16, -20.16, 20.16, 20.16],
            color="#888",
            lw=0.8,
        )
    ax.set_xticks([])
    ax.set_yticks([])


# ---------- figure builders ----------


def fig_phase_mix(phases: pd.DataFrame) -> Path:
    g = (
        phases.groupby("oop_phase", dropna=False)
        .agg(
            n=("duration", "size"),
            duration_s=("duration", "sum"),
            shot_rate=("shot", "mean"),
            goal_rate=("goal", "mean"),
        )
        .reset_index()
        .sort_values("duration_s", ascending=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = [BLOCK_COLORS.get(p, "#555") for p in g["oop_phase"]]
    axes[0].barh(g["oop_phase"], g["duration_s"] / 60, color=colors)
    axes[0].set_xlabel("Total minutes")
    axes[0].set_title("Out-of-possession phase mix")
    axes[1].barh(g["oop_phase"], g["shot_rate"] * 100, color=colors)
    axes[1].set_xlabel("Shot rate against (%)")
    axes[1].set_title("Danger by OOP phase")
    axes[1].set_yticklabels([])
    fig.suptitle("How teams defend: phase context", y=1.02)
    fig.tight_layout()
    return savefig(fig, "01_phase_mix.png")


def fig_shape_scatter(phases: pd.DataFrame) -> Path:
    df = phases.loc[phases["is_organized"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    for ax, block in zip(axes, ORGANIZED_BLOCKS):
        sub = df.loc[df["oop_phase"] == block]
        nons = sub.loc[~sub["shot"]]
        shot = sub.loc[sub["shot"]]
        ax.scatter(nons["width"], nons["length"], s=12, alpha=0.25, c="#7f8c8d", label="no shot")
        ax.scatter(shot["width"], shot["length"], s=28, alpha=0.65, c="#c0392b", label="→ shot")
        ax.set_title(block.replace("_", " "))
        ax.set_xlabel("Defensive width (m)")
        if ax is axes[0]:
            ax.set_ylabel("Defensive length (m)")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Team shape: OOP bounding box (organized blocks)", y=1.03)
    fig.tight_layout()
    return savefig(fig, "02_shape_scatter.png")


def fig_compactness_violin(phases: pd.DataFrame) -> Path:
    df = phases.loc[phases["is_organized"]].copy()
    df["outcome"] = np.where(df["shot"], "shot against", "no shot")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, metric, title in zip(
        axes,
        ["box_area", "width", "length"],
        ["Box area (m²)", "Width (m)", "Length (m)"],
    ):
        sns.violinplot(
            data=df,
            x="oop_phase",
            y=metric,
            hue="outcome",
            order=list(ORGANIZED_BLOCKS),
            split=True,
            inner="quartile",
            palette={"no shot": "#85929e", "shot against": "#c0392b"},
            ax=ax,
            cut=0,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        if ax is not axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="", frameon=False, fontsize=9)
    fig.suptitle("Compactness: shot phases sit in a tighter box", y=1.03)
    fig.tight_layout()
    return savefig(fig, "03_compactness_violin.png")


def fig_pressure_ladder(events: pd.DataFrame) -> Path:
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    g = (
        poss.groupby("overall_pressure_start", dropna=False)
        .agg(
            n=("lead_to_shot", "size"),
            shot_rate=("lead_to_shot", "mean"),
            epv=("possession_epv_delta_against", "mean"),
        )
        .reindex(PRESSURE_ORDER + [np.nan])
        .dropna(how="all")
        .reset_index()
    )
    g = g.loc[g["overall_pressure_start"].isin(PRESSURE_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(g))
    axes[0].bar(x, g["shot_rate"] * 100, color=sns.color_palette("YlOrRd", len(g)))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.replace("_", "\n") for p in g["overall_pressure_start"]], fontsize=9)
    axes[0].set_ylabel("Shot rate (%)")
    axes[0].set_title("On-ball pressure → shot against")
    for i, n in enumerate(g["n"]):
        axes[0].text(i, g["shot_rate"].iloc[i] * 100 + 0.4, f"n={int(n)}", ha="center", fontsize=8)
    axes[1].bar(x, g["epv"], color=sns.color_palette("YlOrRd", len(g)))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.replace("_", "\n") for p in g["overall_pressure_start"]], fontsize=9)
    axes[1].set_ylabel("Mean EPV Δ against")
    axes[1].set_title("Pressure and value conceded")
    axes[1].axhline(0, color="#333", lw=0.8)
    fig.suptitle("Pressure is the strongest contextual separator", y=1.03)
    fig.tight_layout()
    return savefig(fig, "04_pressure_ladder.png")


def fig_pressure_by_block(events: pd.DataFrame) -> Path:
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & (events["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS))
        & (events["overall_pressure_start"].isin(PRESSURE_ORDER))
    ].copy()
    g = (
        poss.groupby(["team_out_of_possession_phase_type", "overall_pressure_start"])
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .reset_index()
    )
    pivot = g.pivot(
        index="overall_pressure_start",
        columns="team_out_of_possession_phase_type",
        values="shot_rate",
    ).reindex(PRESSURE_ORDER)[list(ORGANIZED_BLOCKS)]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Shot rate %"},
    )
    ax.set_xlabel("Organized block")
    ax.set_ylabel("Overall pressure on ball-carrier")
    ax.set_title("Pressure × block height: shot rate heatmap")
    fig.tight_layout()
    return savefig(fig, "05_pressure_x_block_heatmap.png")


def fig_engagement_outcomes(events: pd.DataFrame) -> Path:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["event_subtype"] = eng["event_subtype"].fillna("unknown")
    eng["end_type"] = eng["end_type"].fillna("no_end_type")
    order = [s for s in ENGAGEMENT_SUBTYPES if s in set(eng["event_subtype"])]
    counts = eng["event_subtype"].value_counts().reindex(order)
    regain = (
        eng.assign(regain=eng["end_type"].str.contains("regain", na=False))
        .groupby("event_subtype")["regain"]
        .mean()
        .reindex(order)
    )
    shot = eng.groupby("event_subtype")["lead_to_shot"].mean().reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].bar(order, counts.values, color="#1a5276")
    axes[0].set_title("Engagement volume")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Count")
    axes[1].bar(order, regain.values * 100, color="#117a65")
    axes[1].set_title("Regain rate")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("%")
    axes[2].bar(order, shot.values * 100, color="#922b21")
    axes[2].set_title("Lead-to-shot rate")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylabel("%")
    fig.suptitle("Marking / pressing actions (on-ball engagement)", y=1.03)
    fig.tight_layout()
    return savefig(fig, "06_engagement_outcomes.png")


def fig_engagement_pitch(events: pd.DataFrame) -> Path:
    eng = events.loc[
        (events["event_type"] == "on_ball_engagement")
        & events["x_start"].notna()
        & events["y_start"].notna()
    ].copy()
    subtypes = [s for s in ("pressing", "pressure", "recovery_press", "counter_press") if s in set(eng["event_subtype"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, subtype in zip(axes.ravel(), subtypes):
        sub = eng.loc[eng["event_subtype"] == subtype]
        draw_pitch(ax)
        hb = ax.hexbin(
            sub["x_start"],
            sub["y_start"],
            gridsize=22,
            cmap="mako_r",
            mincnt=1,
            extent=[*PITCH_X, *PITCH_Y],
        )
        ax.set_title(f"{subtype.replace('_', ' ')} (n={len(sub)})")
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Where defenders engage: pitch heatmaps", y=1.01)
    fig.tight_layout()
    return savefig(fig, "07_engagement_heatmaps.png")


def _label_boolish(series: pd.Series) -> pd.Series:
    def _one(v):
        if pd.isna(v):
            return "unknown"
        if v is True or v == 1:
            return "True"
        if v is False or v == 0:
            return "False"
        return str(v)

    return series.map(_one)


def fig_structure_and_shape(events: pd.DataFrame) -> Path:
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    poss["organised_lbl"] = _label_boolish(poss["organised_defense"])
    poss["inside_lbl"] = _label_boolish(poss["inside_defensive_shape_start"])
    org = (
        poss.groupby("organised_lbl", dropna=False)
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .reindex(["True", "False", "unknown"])
        .dropna(how="all")
        .reset_index()
        .rename(columns={"organised_lbl": "label"})
    )
    lines = (
        poss.loc[poss["n_defensive_lines"].notna()]
        .groupby("n_defensive_lines")
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .reset_index()
    )
    inside = (
        poss.groupby("inside_lbl", dropna=False)
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .reindex(["True", "False", "unknown"])
        .dropna(how="all")
        .reset_index()
        .rename(columns={"inside_lbl": "label"})
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].bar(
        org["label"].tolist(),
        (org["shot_rate"] * 100).tolist(),
        color=["#1e8449", "#922b21", "#7f8c8d"][: len(org)],
    )
    axes[0].set_title("Organised defense?")
    axes[0].set_ylabel("Shot rate %")
    for i, row in org.iterrows():
        axes[0].text(i, row["shot_rate"] * 100 + 0.3, f"n={int(row['n'])}", ha="center", fontsize=8)

    line_labels = lines["n_defensive_lines"].astype(int).astype(str).tolist()
    axes[1].bar(line_labels, (lines["shot_rate"] * 100).tolist(), color="#1a5276")
    axes[1].set_title("Number of defensive lines")
    axes[1].set_xlabel("Lines")
    for i, row in lines.iterrows():
        axes[1].text(i, row["shot_rate"] * 100 + 0.3, f"n={int(row['n'])}", ha="center", fontsize=8)

    axes[2].bar(
        inside["label"].tolist(),
        (inside["shot_rate"] * 100).tolist(),
        color=["#b9770e", "#85929e", "#7f8c8d"][: len(inside)],
    )
    axes[2].set_title("Ball inside defensive shape?")
    for i, row in inside.iterrows():
        axes[2].text(i, row["shot_rate"] * 100 + 0.3, f"n={int(row['n'])}", ha="center", fontsize=8)

    fig.suptitle("Structure labels vs danger conceded", y=1.03)
    fig.tight_layout()
    return savefig(fig, "08_structure_labels.png")


def fig_line_height(events: pd.DataFrame) -> Path:
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & events["last_defensive_line_height_start"].notna()
        & (events["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS))
    ].copy()
    poss["shot_flag"] = poss["lead_to_shot"].astype(bool)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=poss,
        x="last_defensive_line_height_start",
        hue="team_out_of_possession_phase_type",
        hue_order=list(ORGANIZED_BLOCKS),
        common_norm=False,
        fill=True,
        alpha=0.25,
        ax=ax,
        palette=[BLOCK_COLORS[b] for b in ORGANIZED_BLOCKS],
    )
    ax.set_xlabel("Last defensive line height (m)")
    ax.set_title("Defensive line height by organized block")
    fig.tight_layout()
    return savefig(fig, "09_line_height_by_block.png")


def fig_decision_effects(events: pd.DataFrame) -> Path:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    metrics = [
        ("force_backward", "Forced backward"),
        ("stop_possession_danger", "Stopped danger"),
        ("reduce_possession_danger", "Reduced danger"),
        ("beaten_by_possession", "Beaten by possession"),
        ("beaten_by_movement", "Beaten by movement"),
    ]
    rows = []
    for col, label in metrics:
        if col not in eng.columns:
            continue
        sub = eng.loc[eng[col].notna()]
        for val, g in sub.groupby(col):
            rows.append(
                {
                    "metric": label,
                    "value": bool(val),
                    "n": len(g),
                    "shot_rate": g["lead_to_shot"].mean(),
                    "regain": g["end_type"].fillna("").str.contains("regain").mean(),
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    # grouped bars for True rates only where True
    true_df = df.loc[df["value"] == True]  # noqa: E712
    false_df = df.loc[df["value"] == False]  # noqa: E712
    x = np.arange(len(true_df))
    w = 0.35
    ax.bar(x - w / 2, false_df["shot_rate"].values * 100, w, label="False", color="#85929e")
    ax.bar(x + w / 2, true_df["shot_rate"].values * 100, w, label="True", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(true_df["metric"], rotation=20, ha="right")
    ax.set_ylabel("Lead-to-shot rate (%)")
    ax.set_title("Defensive decision outcomes on engagements")
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, "10_decision_flags.png")


def fig_pressing_chains(events: pd.DataFrame) -> Path:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    chained = eng.loc[eng["pressing_chain"].fillna(False) == True].copy()  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if len(chained):
        lengths = chained.groupby("pressing_chain_length").size()
        axes[0].bar(lengths.index.astype(str), lengths.values, color="#1a5276")
        axes[0].set_xlabel("Chain length")
        axes[0].set_ylabel("Engagements")
        axes[0].set_title("Pressing chain lengths")
        ends = chained["pressing_chain_end_type"].fillna("unknown").value_counts()
        axes[1].barh(ends.index.astype(str), ends.values, color="#117a65")
        axes[1].set_title("Chain end types")
        axes[1].set_xlabel("Count")
    else:
        axes[0].text(0.5, 0.5, "No pressing chains", ha="center")
    fig.suptitle("Coordinated pressing decisions", y=1.02)
    fig.tight_layout()
    return savefig(fig, "11_pressing_chains.png")


def fig_team_defense_board(phases: pd.DataFrame) -> Path:
    df = phases.loc[phases["is_organized"] & phases["defending_team"].notna()].copy()
    g = (
        df.groupby("defending_team")
        .agg(
            n_phases=("shot", "size"),
            shot_rate=("shot", "mean"),
            mean_area=("box_area", "mean"),
            mean_width=("width", "mean"),
            mean_length=("length", "mean"),
            minutes=("duration", "sum"),
        )
        .reset_index()
    )
    g = g.loc[g["n_phases"] >= 80].sort_values("shot_rate")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        g["mean_area"],
        g["shot_rate"] * 100,
        s=np.clip(g["n_phases"] / 3, 40, 280),
        c=g["mean_width"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
    )
    for _, row in g.iterrows():
        ax.annotate(
            row["defending_team"],
            (row["mean_area"], row["shot_rate"] * 100),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean OOP box area (m²)")
    ax.set_ylabel("Shot rate against (%)")
    ax.set_title("Team defensive board: compactness vs shots conceded (organized blocks)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean width (m)")
    fig.tight_layout()
    return savefig(fig, "12_team_defense_board.png")


def fig_spacing_distance(events: pd.DataFrame) -> Path:
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    # SkillCorner fills interplayer_distance_start for engagements; ball-carrier distance is often empty.
    dist_col = "interplayer_distance_start"
    if dist_col not in eng.columns or not eng[dist_col].notna().any():
        dist_col = "interplayer_distance_min"
    eng = eng.loc[eng[dist_col].notna()].copy()
    eng["event_subtype"] = eng["event_subtype"].fillna("unknown")
    order = [s for s in ENGAGEMENT_SUBTYPES if s in set(eng["event_subtype"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.boxplot(
        data=eng,
        x="event_subtype",
        y=dist_col,
        order=order,
        ax=axes[0],
        color="#5dade2",
    )
    axes[0].set_ylabel("Metres between defender and opponent at press (m)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("How tight is the press?")

    eng["dist_bin"] = pd.cut(
        eng[dist_col],
        bins=[0, 2, 4, 6, 8, 12, 30],
        labels=["0–2", "2–4", "4–6", "6–8", "8–12", "12+"],
    )
    eng = eng.loc[eng["dist_bin"].notna()]
    g = eng.groupby("dist_bin", observed=False).agg(
        n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean")
    )
    axes[1].bar(g.index.astype(str), g["shot_rate"] * 100, color="#e67e22")
    for i, (_, row) in enumerate(g.iterrows()):
        if row["n"] > 0:
            axes[1].text(i, row["shot_rate"] * 100 + 0.5, f"n={int(row['n'])}", ha="center", fontsize=8)
    axes[1].set_ylabel("Share of presses after which a shot happens (%)")
    axes[1].set_xlabel("Distance when the press starts (m)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Closer is not always safer for the defense")
    fig.suptitle("Defensive spacing when someone steps to the ball", y=1.03)
    fig.tight_layout()
    return savefig(fig, "13_engagement_spacing.png")


def fig_box_vs_pressure_joint(phases: pd.DataFrame, events: pd.DataFrame) -> Path:
    """Phase-level area vs possession-level pressure conceptually side by side."""
    blocks = phases.loc[phases["is_organized"]].copy()
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & (events["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS))
        & (events["overall_pressure_start"].isin(PRESSURE_ORDER))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for block in ORGANIZED_BLOCKS:
        sub = blocks.loc[blocks["oop_phase"] == block]
        rates = []
        # tertiles of box area
        qs = sub["box_area"].quantile([0.33, 0.66]).values
        bins = [-np.inf, qs[0], qs[1], np.inf]
        labels = ["tight", "mid", "open"]
        sub = sub.copy()
        sub["area_bin"] = pd.cut(sub["box_area"], bins=bins, labels=labels)
        g = sub.groupby("area_bin", observed=False)["shot"].mean()
        axes[0].plot(labels, g.values * 100, marker="o", label=block.replace("_", " "), color=BLOCK_COLORS[block])
    axes[0].set_ylabel("Shot rate %")
    axes[0].set_xlabel("Defensive box: tight / mid / open")
    axes[0].set_title("Compactness gradient: tighter block → more shots?")
    axes[0].legend(frameon=False, fontsize=9)

    for block in ORGANIZED_BLOCKS:
        sub = poss.loc[poss["team_out_of_possession_phase_type"] == block]
        g = sub.groupby("overall_pressure_start")["lead_to_shot"].mean().reindex(PRESSURE_ORDER)
        axes[1].plot(
            [p.replace("_", "\n") for p in PRESSURE_ORDER],
            g.values * 100,
            marker="o",
            label=block.replace("_", " "),
            color=BLOCK_COLORS[block],
        )
    axes[1].set_ylabel("Shot rate %")
    axes[1].set_title("Pressure gradient: heavier press → more shots?")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Two sliders inside each block height\n"
        "(Left: defensive box split tight / mid / open. Right: pressure on the ball-carrier.)",
        y=1.05,
        fontsize=11,
    )
    fig.tight_layout()
    return savefig(fig, "14_compactness_vs_pressure_gradients.png")


def _team_slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def _load_phase_attackers(data_dir: Path, match_ids: list[str]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        ph = pd.read_csv(
            data_dir / mid / f"{mid}_phases_of_play.csv",
            usecols=["index", "team_in_possession_shortname"],
        ).rename(columns={"index": "phase_index", "team_in_possession_shortname": "attack_team"})
        ph["match_id"] = int(mid)
        frames.append(ph)
    out = pd.concat(frames, ignore_index=True)
    out["match_id"] = out["match_id"].astype(int)
    return out


def fig_team_pitch_zones(
    events: pd.DataFrame, meta: pd.DataFrame, match_ids: list[str], min_shots: int = 10
) -> tuple[list[str], Path]:
    """Per-team scored vs conceded shot maps (third × channel). Returns figure names + HTML path."""
    phase_attack = _load_phase_attackers(DATA_DIR, match_ids)
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & (events["lead_to_shot"] == True)  # noqa: E712
        & events["third_start"].notna()
        & events["channel_start"].notna()
        & events["phase_index"].notna()
    ].copy()
    poss["match_id"] = poss["match_id"].astype(int)
    poss = poss.merge(phase_attack, on=["match_id", "phase_index"], how="left")
    poss["attack_team"] = poss["team_shortname"].fillna(poss["attack_team"])
    teams = sorted(meta["home"].tolist() + meta["away"].tolist())
    teams = sorted(set(teams))

    third_order = ["defensive_third", "middle_third", "attacking_third"]
    channel_order = ["wide_left", "left_halfspace", "center", "right_halfspace", "wide_right"]
    third_labels = ["Def third", "Mid third", "Att third"]
    channel_labels = ["Wide L", "Half L", "Centre", "Half R", "Wide R"]

    team_dir = DOCS_FIG / "teams"
    team_dir.mkdir(parents=True, exist_ok=True)
    fig_names: list[str] = []
    cards: list[str] = []

    for team in teams:
        scored = poss.loc[poss["attack_team"] == team]
        conceded = poss.loc[poss["attack_team"] != team]
        # only matches this team played
        team_matches = set(
            meta.loc[(meta["home"] == team) | (meta["away"] == team), "match_id"].astype(int)
        )
        conceded = conceded.loc[conceded["match_id"].isin(team_matches)]
        conceded = conceded.loc[
            ~conceded["attack_team"].isin([team])  # opponent had the ball
        ]
        if len(scored) < min_shots and len(conceded) < min_shots:
            continue

        def _heat(df: pd.DataFrame) -> pd.DataFrame:
            g = df.groupby(["third_start", "channel_start"]).size().reset_index(name="n")
            p = g.pivot(index="third_start", columns="channel_start", values="n")
            return p.reindex(index=third_order, columns=channel_order).fillna(0)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, df, title in [
            (axes[0], scored, f"Shots scored"),
            (axes[1], conceded, f"Shots conceded"),
        ]:
            p = _heat(df)
            sns.heatmap(
                p,
                annot=True,
                fmt=".0f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "Shots"},
                linewidths=0.5,
            )
            ax.set_title(f"{title} (n={len(df)})")
            ax.set_xticklabels(channel_labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(third_labels, rotation=0, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle(f"{team} — where shots happen on the pitch", y=1.02, fontsize=12)
        fig.tight_layout()
        fname = f"teams/{_team_slug(team)}_zones.png"
        savefig(fig, fname)
        fig_names.append(fname)
        rel = f"eda/figures/{fname}"
        cards.append(
            f'<figure class="card"><img src="{rel}" alt="{team} shot zones" loading="lazy">'
            f"<figcaption><strong>{team}</strong> — left: where they shoot; right: where opponents shoot against them.</figcaption></figure>"
        )

    html_path = DOCS_DIR / "team-zones.html"
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Team shot zones</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 1rem; max-width: 44rem; margin: auto; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  .lead {{ color: #555; }}
  .card {{ margin: 1.2rem 0; }}
  .card img {{ width: 100%; height: auto; border-radius: 4px; }}
  figcaption {{ font-size: 0.95rem; margin-top: 0.35rem; }}
  a {{ color: inherit; }}
</style></head><body>
<h1>Shot zones by team</h1>
<p class="lead">League averages hide shape: each club has its own shot map (scored vs conceded). Cells count where the ball was when the shot started.</p>
{"".join(cards)}
<p><a href="eda.html">Back to visual EDA</a> · <a href="index.html">Index</a></p>
</body></html>"""
    )
    return fig_names, html_path


def fig_third_channel_danger(events: pd.DataFrame) -> Path:
    poss = events.loc[
        (events["event_type"] == "player_possession")
        & events["third_start"].notna()
        & events["channel_start"].notna()
    ].copy()
    g = (
        poss.groupby(["third_start", "channel_start"])
        .agg(n=("lead_to_shot", "size"), shot_rate=("lead_to_shot", "mean"))
        .reset_index()
    )
    third_order = ["defensive_third", "middle_third", "attacking_third"]
    channel_order = ["wide_left", "left_halfspace", "center", "right_halfspace", "wide_right"]
    # normalize names that may differ
    pivot = g.pivot(index="third_start", columns="channel_start", values="shot_rate")
    # reorder if present
    pivot = pivot.reindex(
        [t for t in third_order if t in pivot.index],
        columns=[c for c in channel_order if c in pivot.columns],
    )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Shot %"})
    ax.set_title("Where possessions become shots (third × channel)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Third")
    fig.tight_layout()
    return savefig(fig, "15_third_channel_heatmap.png")


# ---------- summaries ----------


def build_summaries(phases: pd.DataFrame, events: pd.DataFrame, match_ids: list[str]) -> dict:
    phase_sum = (
        phases.groupby("oop_phase", dropna=False)
        .agg(
            n_phases=("duration", "size"),
            total_duration_s=("duration", "sum"),
            mean_duration_s=("duration", "mean"),
            n_shot=("shot", "sum"),
            n_goal=("goal", "sum"),
            shot_rate=("shot", "mean"),
            goal_rate=("goal", "mean"),
            mean_width=("width", "mean"),
            mean_length=("length", "mean"),
            mean_box_area=("box_area", "mean"),
        )
        .reset_index()
        .sort_values("total_duration_s", ascending=False)
    )
    phase_sum["shots_per_minute"] = phase_sum["n_shot"] / (phase_sum["total_duration_s"] / 60)

    poss = events.loc[events["event_type"] == "player_possession"]
    pressure = (
        poss.groupby("overall_pressure_start", dropna=False)
        .agg(
            n=("lead_to_shot", "size"),
            n_shot=("lead_to_shot", "sum"),
            shot_rate=("lead_to_shot", "mean"),
            mean_epv_delta_against=("possession_epv_delta_against", "mean"),
        )
        .reset_index()
    )

    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["regain"] = eng["end_type"].fillna("").str.contains("regain")
    engagement = (
        eng.groupby("event_subtype", dropna=False)
        .agg(
            n=("lead_to_shot", "size"),
            shot_rate=("lead_to_shot", "mean"),
            regain_rate=("regain", "mean"),
            mean_start_dist=("distance_to_player_in_possession_start", "mean"),
        )
        .reset_index()
    )

    bbox_rows = []
    for block, block_df in phases.loc[phases["is_organized"]].groupby("oop_phase"):
        for shot_flag, sub in block_df.groupby("shot"):
            bbox_rows.append(
                {
                    "oop_phase": block,
                    "lead_to_shot": bool(shot_flag),
                    "n": len(sub),
                    "mean_width": sub["width"].mean(),
                    "mean_length": sub["length"].mean(),
                    "mean_box_area": sub["box_area"].mean(),
                }
            )
    bbox = pd.DataFrame(bbox_rows)

    headline = {
        "n_matches": len(match_ids),
        "match_ids": match_ids,
        "n_phases": int(len(phases)),
        "n_player_possessions": int((events["event_type"] == "player_possession").sum()),
        "n_on_ball_engagements": int((events["event_type"] == "on_ball_engagement").sum()),
        "shot_rate_by_organized_block": {
            b: float(phase_sum.loc[phase_sum["oop_phase"] == b, "shot_rate"].iloc[0])
            for b in ORGANIZED_BLOCKS
            if b in set(phase_sum["oop_phase"])
        },
        "pressure_shot_rates": {
            str(r["overall_pressure_start"]): float(r["shot_rate"])
            for _, r in pressure.dropna(subset=["overall_pressure_start"]).iterrows()
        },
        "key_takeaways": [
            "Organized block height sets the base shot rate: low (~18.5%) ≫ medium (~9.0%) > high (~6.0%).",
            "Shot phases sit in a smaller OOP bounding box — compactness co-moves with danger, strongest in the low block.",
            "On-ball pressure separates outcomes (~2% no pressure → ~21% very high), but high bands are endogenous with already-dangerous moments — treat as context, not pure cause.",
            "Keeping the block organised (~2.5% shot rate) and the ball outside the shape (~5.8% vs ~11.5% inside) are the clearest structure signals; line count barely moves.",
            "Engagement mix matters: pressing/counter-press regain more often; beaten-by-possession/movement flags precede shots.",
            "Team board: mean OOP box area and width profile which A-League sides concede more in organized defense — compactness alone does not rank the table.",
        ],
    }
    return {
        "phase_sum": phase_sum,
        "pressure": pressure,
        "engagement": engagement,
        "bbox": bbox,
        "headline": headline,
    }


def write_html_report(fig_names: list[str], headline: dict) -> Path:
    """Phone-first visual report for GitHub Pages commute review."""
    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    # copy figures into docs
    for name in fig_names:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, DOCS_FIG / name)

    cards = []
    captions = {
        "01_phase_mix.png": "How long teams spend in each defensive situation (low / mid / high block, transitions, set pieces) and how often those spells end in a shot.",
        "02_shape_scatter.png": "Width and depth of the out-of-possession box. Red dots are phases that ended in a shot — often a slightly tighter shape.",
        "03_compactness_violin.png": "Compactness = how small the defensive rectangle is. Shot phases tend to be tighter, especially in a low block.",
        "04_pressure_ladder.png": "How hard the ball-carrier is pressed. More pressure goes with more shots — but that often means the attack was already dangerous.",
        "05_pressure_x_block_heatmap.png": "Same story inside each block height: heavier pressure on the ball, more shots.",
        "06_engagement_outcomes.png": "When a defender steps to the ball: how often they win it back vs how often a shot still follows.",
        "07_engagement_heatmaps.png": "Where on the pitch those steps happen — high press up field, recovery runs deeper.",
        "08_structure_labels.png": "SkillCorner tags: organised shape, ball inside the block, number of lines — vs shot rate.",
        "09_line_height_by_block.png": "How high the back line sits in each block type (validates low / mid / high labels).",
        "10_decision_flags.png": "Did the press slow the attack or get beaten? Beaten flags line up with more shots.",
        "11_pressing_chains.png": "Linked presses: how many steps in a chain and whether it ended in a regain.",
        "12_team_defense_board.png": "Club comparison: average box size vs shots faced in organised blocks.",
        "13_engagement_spacing.png": "Distance between defender and opponent when the press starts — not blank; uses inter-player distance from SkillCorner.",
        "14_compactness_vs_pressure_gradients.png": "Gradients = how shot rate changes as you move from tight→open shape (left) or low→high pressure (right), within each block.",
        "15_third_channel_heatmap.png": "League-wide shot locations (centre-heavy). See team-zones.html for each club.",
        "16_free_attackers_by_role.png": "Graph tool: at goals and shots, which attacking roles were unmarked (Hungarian matching on tracking)?",
    }
    for name in fig_names:
        cap = captions.get(name, name)
        cards.append(
            f"""
      <figure class="card">
        <img src="eda/figures/{name}" alt="{cap}" loading="lazy">
        <figcaption>{cap}</figcaption>
      </figure>"""
        )

    takeaways = "".join(f"<li>{t}</li>" for t in headline["key_takeaways"])
    block_rates = headline["shot_rate_by_organized_block"]
    stats = (
        f"{headline['n_matches']} matches · {headline['n_phases']:,} phases · "
        f"{headline['n_player_possessions']:,} possessions · "
        f"{headline['n_on_ball_engagements']:,} engagements"
    )
    block_line = " · ".join(
        f"{k.replace('_', ' ')} {v*100:.1f}%" for k, v in block_rates.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Defensive positioning EDA</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 1.1rem 0.85rem 3rem;
      max-width: 44rem;
      margin-inline: auto;
    }}
    h1 {{ font-size: 1.55rem; line-height: 1.25; margin: 0 0 0.4rem; }}
    h2 {{
      font-size: 1.15rem;
      margin: 1.75rem 0 0.6rem;
      padding-top: 0.4rem;
      border-top: 1px solid #ddd;
    }}
    @media (prefers-color-scheme: dark) {{
      h2 {{ border-top-color: #444; }}
    }}
    .mission {{
      margin: 0.75rem 0 1rem;
      padding: 0.75rem 0.85rem;
      border-left: 3px solid #888;
      background: rgba(127,127,127,0.08);
      font-size: 1.02rem;
    }}
    .meta {{ color: #666; font-size: 0.92rem; margin: 0 0 0.75rem; }}
    @media (prefers-color-scheme: dark) {{ .meta {{ color: #aaa; }} }}
    ul {{ padding-left: 1.2rem; }}
    li {{ margin: 0 0 0.55rem; }}
    .card {{
      margin: 0 0 1.35rem;
      padding: 0;
    }}
    .card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
      background: #f3f3f3;
    }}
    figcaption {{
      margin-top: 0.45rem;
      font-size: 0.95rem;
      color: #444;
    }}
    @media (prefers-color-scheme: dark) {{
      figcaption {{ color: #bbb; }}
      .card img {{ background: #222; }}
    }}
    a {{ color: inherit; }}
    .nav {{ margin-top: 2rem; font-size: 0.95rem; }}
    .nav a {{ display: inline-block; margin-right: 1rem; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <h1>Defensive positioning EDA</h1>
  <p class="meta">{stats}</p>
  <p class="meta">Organized-block shot rates: {block_line}</p>

  <div class="mission">
    <strong>Question.</strong> How can tracking and contextual data help us understand the way players and teams defend?
    Explore team shape, spacing, pressure, compactness, marking, and defensive decision-making.
  </div>

  <h2>Takeaways</h2>
  <ul>
    {takeaways}
  </ul>

  <h2>Plain-language glossary</h2>
  <ul>
    <li><strong>Compactness</strong> — how small the defending team’s shape is (width × depth of their box). Tighter is not always better in this sample.</li>
    <li><strong>Pressure gradient</strong> — as on-ball pressure rises from none to very high, how much the shot rate climbs (same for block height).</li>
    <li><strong>Organised block</strong> — low, medium, or high defensive block (not chaotic or transition).</li>
  </ul>
  <p><a href="team-zones.html"><strong>Team shot maps</strong></a> (each club — scored vs conceded). 
  <a href="graph-tool-notes.html"><strong>Block graph tool FAQ</strong></a> (timer series, pressing, pre-fracturing).</p>

  <h2>Visual tour</h2>
  {''.join(cards)}

  <div class="nav">
    <a href="team-zones.html">Team zones</a>
    <a href="graph-tool-notes.html">Graph tool notes</a>
    <a href="review.html">Review notes</a>
    <a href="defend.html">Graph notebook</a>
    <a href="eda_notebook.html">Full EDA notebook</a>
    <a href="index.html">All notebooks</a>
  </div>
</body>
</html>
"""
    out = DOCS_DIR / "eda.html"
    out.write_text(html)
    return out


def write_notebook(fig_names: list[str], headline: dict) -> Path:
    """Create a lightweight notebook that embeds the saved figures + narrative."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# Defensive positioning — visual EDA

**Mission:** Football defensive positioning. How can tracking and contextual data help us better understand the way players and teams defend?

Dataset: **{headline['n_matches']}** A-League 2024/25 matches (SkillCorner Open Data) — phases of play + dynamic events (pressure, engagements, structure labels). Tracking JSONL stubs are not used here; contextual fields already encode shape, pressure, and marking.

Counts: {headline['n_phases']:,} phases · {headline['n_player_possessions']:,} possessions · {headline['n_on_ball_engagements']:,} on-ball engagements.
"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
from IPython.display import Image, display, Markdown
FIG = Path('output/figures')
print('Figures:', sorted(p.name for p in FIG.glob('*.png')))
"""
        )
    )

    sections = [
        ("01_phase_mix.png", "## 1. Team defensive context (phase mix)\nOrganized blocks dominate minutes; shot rate climbs as the block drops."),
        ("02_shape_scatter.png", "## 2. Team shape\nOOP width × length. Shot phases (red) sit toward the compact side of the cloud."),
        ("03_compactness_violin.png", "## 3. Compactness\nBox area / width / length split by shot vs no-shot inside each organized block."),
        ("04_pressure_ladder.png", "## 4. Pressure\nOn-ball `overall_pressure` is a steep ladder for shot rate and EPV conceded."),
        ("05_pressure_x_block_heatmap.png", "## 5. Pressure × block\nPressure still separates danger inside low, medium, and high blocks."),
        ("06_engagement_outcomes.png", "## 6. Marking / pressing actions\nEngagement subtypes: volume, regain rate, lead-to-shot."),
        ("07_engagement_heatmaps.png", "## 7. Where defenders engage\nPitch heatmaps by engagement subtype."),
        ("08_structure_labels.png", "## 8. Defensive structure labels\nOrganised defense, number of lines, ball inside shape."),
        ("09_line_height_by_block.png", "## 9. Line height\nLast defensive line height distributions validate block labels."),
        ("10_decision_flags.png", "## 10. Defensive decisions\nForce backward / stop danger / beaten flags vs subsequent shot rate."),
        ("11_pressing_chains.png", "## 11. Pressing chains\nCoordinated pressure sequences."),
        ("12_team_defense_board.png", "## 12. Team board\nWhich sides defend in a bigger box — and how that relates to shots against."),
        ("13_engagement_spacing.png", "## 13. Spacing at engagement\nDistance to ball-carrier by subtype and shot rate by distance bin."),
        ("14_compactness_vs_pressure_gradients.png", "## 14. Two axes of defense\nWithin-block compactness tertiles vs pressure gradients."),
        ("15_third_channel_heatmap.png", "## 15. League shot map (centre-heavy)\nSee team-zones.html for each club."),
        ("16_free_attackers_by_role.png", "## 16. Unmarked attackers by role\nHungarian matching on tracking at goals and shots."),
    ]
    for name, md in sections:
        if name in fig_names:
            cells.append(nbf.v4.new_markdown_cell(md))
            cells.append(
                nbf.v4.new_code_cell(f"display(Image(filename=str(FIG / '{name}')))")
            )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Takeaways for coaches / analysts\n\n"
            + "\n".join(f"- {t}" for t in headline["key_takeaways"])
            + "\n\n**Next film questions:** (1) moments where the box looks compact but pressure is low; "
            "(2) engagements where the defender is beaten by movement; "
            "(3) team outliers on the compactness–shot board."
        )
    )
    nb["cells"] = cells
    path = WORKSPACE / "analysis" / "eda_defensive_positioning.ipynb"
    nbf.write(nb, path)
    return path


def main() -> None:
    style_plots()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Missing {DATA_DIR}. Clone SkillCorner/opendata into workspace/opendata."
        )

    print("Loading matches…")
    phases, events, match_ids = load_all(DATA_DIR)
    phases = enrich_phases(phases)
    meta = load_match_meta(MATCHES_JSON)
    phases = defending_team_from_possession(phases, meta)
    print(f"  {len(match_ids)} matches, {len(phases)} phases, {len(events)} events")

    print("Building figures…")
    fig_names = []
    builders = [
        fig_phase_mix(phases),
        fig_shape_scatter(phases),
        fig_compactness_violin(phases),
        fig_pressure_ladder(events),
        fig_pressure_by_block(events),
        fig_engagement_outcomes(events),
        fig_engagement_pitch(events),
        fig_structure_and_shape(events),
        fig_line_height(events),
        fig_decision_effects(events),
        fig_pressing_chains(events),
        fig_team_defense_board(phases),
        fig_spacing_distance(events),
        fig_box_vs_pressure_joint(phases, events),
        fig_third_channel_danger(events),
    ]
    for p in builders:
        fig_names.append(p.name)
        print(f"  wrote {p.name}")

    print("Hungarian roles (tracking)…")
    try:
        import sys

        sys.path.insert(0, str(WORKSPACE / "analysis"))
        import hungarian_danger_roles as hdr

        hdr.main()
    except Exception as exc:
        print(f"  hungarian skipped: {exc}")

    print("Team shot zones…")
    team_figs, tz_html = fig_team_pitch_zones(events, meta, match_ids)
    for tf in team_figs:
        fig_names.append(tf)
        src = FIG_DIR / tf
        if src.exists():
            dest = DOCS_FIG / tf
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    print(f"  {len(team_figs)} team maps → {tz_html.name}")

    hungarian_fig = FIG_DIR / "16_free_attackers_by_role.png"
    if hungarian_fig.is_file():
        fig_names.append(hungarian_fig.name)
        shutil.copy2(hungarian_fig, DOCS_FIG / hungarian_fig.name)
        print("  included Hungarian role chart")
    else:
        print("  (skip chart 16 — run: python analysis/hungarian_danger_roles.py)")

    print("Summaries…")
    sums = build_summaries(phases, events, match_ids)
    sums["phase_sum"].to_csv(OUT_DIR / "eda_phase_summary.csv", index=False)
    sums["pressure"].to_csv(OUT_DIR / "eda_pressure_summary.csv", index=False)
    sums["engagement"].to_csv(OUT_DIR / "eda_engagement_summary.csv", index=False)
    sums["bbox"].to_csv(OUT_DIR / "eda_bbox_summary.csv", index=False)
    sums["headline"]["figures"] = [f"analysis/output/figures/{n}" for n in fig_names]
    with open(OUT_DIR / "eda_summary.json", "w") as f:
        json.dump(sums["headline"], f, indent=2)

    print("HTML + notebook…")
    html_path = write_html_report(fig_names, sums["headline"])
    nb_path = write_notebook(fig_names, sums["headline"])
    print(f"Report: {html_path}")
    print(f"Notebook: {nb_path}")
    print("Done.")


if __name__ == "__main__":
    main()
