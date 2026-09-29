"""EDA of defensive context across SkillCorner Open Data matches.

Loads all matches under opendata/data/matches/<id>/ (phases + dynamic events;
no tracking). Writes machine-readable summaries under analysis/output/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "opendata" / "data" / "matches"
OUT_DIR = WORKSPACE / "analysis" / "output"

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

PHASE_USECOLS = [
    "match_id",
    "duration",
    "team_possession_lead_to_shot",
    "team_possession_lead_to_goal",
    "team_out_of_possession_phase_type",
    "team_out_of_possession_width_start",
    "team_out_of_possession_width_end",
    "team_out_of_possession_length_start",
    "team_out_of_possession_length_end",
]

EVENT_USECOLS = [
    "match_id",
    "event_type",
    "event_subtype",
    "end_type",
    "lead_to_shot",
    "overall_pressure_start",
    "possession_epv_delta_against",
    "team_out_of_possession_phase_type",
]


def discover_match_ids(data_dir: Path) -> list[str]:
    ids: list[str] = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        mid = d.name
        phases = d / f"{mid}_phases_of_play.csv"
        events = d / f"{mid}_dynamic_events.csv"
        if phases.is_file() and events.is_file():
            ids.append(mid)
    return ids


def load_all(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    match_ids = discover_match_ids(data_dir)
    phase_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for mid in match_ids:
        d = data_dir / mid
        phases = pd.read_csv(d / f"{mid}_phases_of_play.csv", usecols=PHASE_USECOLS)
        events = pd.read_csv(
            d / f"{mid}_dynamic_events.csv",
            usecols=EVENT_USECOLS,
            low_memory=False,
        )
        phase_frames.append(phases)
        event_frames.append(events)
    phases_all = pd.concat(phase_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)
    return phases_all, events_all, match_ids


def phase_mix(phases: pd.DataFrame) -> pd.DataFrame:
    """Out-of-possession phase mix: counts and total duration."""
    g = (
        phases.groupby("team_out_of_possession_phase_type", dropna=False)
        .agg(
            n_phases=("duration", "size"),
            total_duration_s=("duration", "sum"),
            mean_duration_s=("duration", "mean"),
            n_shot=("team_possession_lead_to_shot", "sum"),
            n_goal=("team_possession_lead_to_goal", "sum"),
        )
        .reset_index()
        .rename(columns={"team_out_of_possession_phase_type": "oop_phase_type"})
    )
    g["shot_rate"] = g["n_shot"] / g["n_phases"]
    g["goal_rate"] = g["n_goal"] / g["n_phases"]
    g["shots_per_minute"] = g["n_shot"] / (g["total_duration_s"] / 60.0)
    g["goals_per_minute"] = g["n_goal"] / (g["total_duration_s"] / 60.0)
    g["is_organized_block"] = g["oop_phase_type"].isin(ORGANIZED_BLOCKS)
    return g.sort_values("total_duration_s", ascending=False).reset_index(drop=True)


def pressure_summary(events: pd.DataFrame) -> pd.DataFrame:
    """overall_pressure_start vs lead_to_shot and possession_epv_delta_against."""
    poss = events.loc[events["event_type"] == "player_possession"].copy()
    g = (
        poss.groupby("overall_pressure_start", dropna=False)
        .agg(
            n=("lead_to_shot", "size"),
            n_shot=("lead_to_shot", "sum"),
            mean_epv_delta_against=("possession_epv_delta_against", "mean"),
            median_epv_delta_against=("possession_epv_delta_against", "median"),
            std_epv_delta_against=("possession_epv_delta_against", "std"),
        )
        .reset_index()
    )
    g["shot_rate"] = g["n_shot"] / g["n"]
    order = {p: i for i, p in enumerate(PRESSURE_ORDER)}
    g["_ord"] = g["overall_pressure_start"].map(order)
    g = g.sort_values("_ord", na_position="last").drop(columns="_ord").reset_index(drop=True)
    return g


def engagement_summary(events: pd.DataFrame) -> pd.DataFrame:
    """on_ball_engagement subtype vs end_type / lead_to_shot, crossed with OOP phase."""
    eng = events.loc[events["event_type"] == "on_ball_engagement"].copy()
    eng["event_subtype"] = eng["event_subtype"].fillna("unknown")
    eng["end_type"] = eng["end_type"].fillna("no_end_type")
    eng["team_out_of_possession_phase_type"] = eng[
        "team_out_of_possession_phase_type"
    ].fillna("unknown")

    rows: list[dict] = []

    # Overall by subtype
    for subtype, sub in eng.groupby("event_subtype", dropna=False):
        rows.append(
            {
                "slice": "subtype_overall",
                "event_subtype": subtype,
                "end_type": None,
                "oop_phase_type": None,
                "n": len(sub),
                "n_shot": int(sub["lead_to_shot"].sum()),
                "shot_rate": float(sub["lead_to_shot"].mean()),
            }
        )

    # Subtype x end_type
    for (subtype, end_type), sub in eng.groupby(
        ["event_subtype", "end_type"], dropna=False
    ):
        rows.append(
            {
                "slice": "subtype_x_end_type",
                "event_subtype": subtype,
                "end_type": end_type,
                "oop_phase_type": None,
                "n": len(sub),
                "n_shot": int(sub["lead_to_shot"].sum()),
                "shot_rate": float(sub["lead_to_shot"].mean()),
            }
        )

    # Subtype x OOP phase (organized blocks + overall other)
    for (subtype, oop), sub in eng.groupby(
        ["event_subtype", "team_out_of_possession_phase_type"], dropna=False
    ):
        rows.append(
            {
                "slice": "subtype_x_oop_phase",
                "event_subtype": subtype,
                "end_type": None,
                "oop_phase_type": oop,
                "n": len(sub),
                "n_shot": int(sub["lead_to_shot"].sum()),
                "shot_rate": float(sub["lead_to_shot"].mean()),
            }
        )

    # Subtype x end_type x organized block
    blocks = eng.loc[
        eng["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS)
    ]
    for (subtype, end_type, oop), sub in blocks.groupby(
        ["event_subtype", "end_type", "team_out_of_possession_phase_type"],
        dropna=False,
    ):
        rows.append(
            {
                "slice": "subtype_x_end_type_x_block",
                "event_subtype": subtype,
                "end_type": end_type,
                "oop_phase_type": oop,
                "n": len(sub),
                "n_shot": int(sub["lead_to_shot"].sum()),
                "shot_rate": float(sub["lead_to_shot"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    subtype_rank = {s: i for i, s in enumerate(ENGAGEMENT_SUBTYPES)}
    out["_ord"] = out["event_subtype"].map(subtype_rank)
    return out.sort_values(
        ["slice", "_ord", "oop_phase_type", "end_type"], na_position="last"
    ).drop(columns="_ord").reset_index(drop=True)


def bbox_baseline(phases: pd.DataFrame) -> pd.DataFrame:
    """Width, length, and product for shot vs non-shot in organized blocks."""
    df = phases.loc[
        phases["team_out_of_possession_phase_type"].isin(ORGANIZED_BLOCKS)
    ].copy()
    df["width"] = df[
        ["team_out_of_possession_width_start", "team_out_of_possession_width_end"]
    ].mean(axis=1)
    df["length"] = df[
        ["team_out_of_possession_length_start", "team_out_of_possession_length_end"]
    ].mean(axis=1)
    df["box_area"] = df["width"] * df["length"]
    df["shot"] = df["team_possession_lead_to_shot"].astype(bool)

    rows: list[dict] = []
    for block, block_df in df.groupby("team_out_of_possession_phase_type"):
        for shot_flag, sub in block_df.groupby("shot"):
            rows.append(
                {
                    "oop_phase_type": block,
                    "lead_to_shot": bool(shot_flag),
                    "n_phases": len(sub),
                    "mean_width": float(sub["width"].mean()),
                    "median_width": float(sub["width"].median()),
                    "mean_length": float(sub["length"].mean()),
                    "median_length": float(sub["length"].median()),
                    "mean_box_area": float(sub["box_area"].mean()),
                    "median_box_area": float(sub["box_area"].median()),
                    "std_box_area": float(sub["box_area"].std()),
                }
            )
        # Separation: difference of means (shot - non-shot)
        shot = block_df.loc[block_df["shot"]]
        nons = block_df.loc[~block_df["shot"]]
        if len(shot) and len(nons):
            rows.append(
                {
                    "oop_phase_type": block,
                    "lead_to_shot": "diff_shot_minus_nons",
                    "n_phases": len(block_df),
                    "mean_width": float(shot["width"].mean() - nons["width"].mean()),
                    "median_width": float(
                        shot["width"].median() - nons["width"].median()
                    ),
                    "mean_length": float(
                        shot["length"].mean() - nons["length"].mean()
                    ),
                    "median_length": float(
                        shot["length"].median() - nons["length"].median()
                    ),
                    "mean_box_area": float(
                        shot["box_area"].mean() - nons["box_area"].mean()
                    ),
                    "median_box_area": float(
                        shot["box_area"].median() - nons["box_area"].median()
                    ),
                    "std_box_area": None,
                }
            )
    out = pd.DataFrame(rows)
    block_rank = {b: i for i, b in enumerate(ORGANIZED_BLOCKS)}
    out["_ord"] = out["oop_phase_type"].map(block_rank)
    return out.sort_values(["_ord", "lead_to_shot"]).drop(columns="_ord").reset_index(
        drop=True
    )


def headline_json(
    match_ids: list[str],
    phases: pd.DataFrame,
    phase_sum: pd.DataFrame,
    pressure: pd.DataFrame,
    bbox: pd.DataFrame,
    events: pd.DataFrame,
) -> dict:
    blocks = phase_sum.loc[phase_sum["oop_phase_type"].isin(ORGANIZED_BLOCKS)].set_index(
        "oop_phase_type"
    )
    shot_rate_by_block = {
        b: {
            "n_phases": int(blocks.loc[b, "n_phases"]),
            "n_shot": int(blocks.loc[b, "n_shot"]),
            "shot_rate": float(blocks.loc[b, "shot_rate"]),
            "goal_rate": float(blocks.loc[b, "goal_rate"]),
            "total_duration_s": float(blocks.loc[b, "total_duration_s"]),
        }
        for b in ORGANIZED_BLOCKS
        if b in blocks.index
    }

    # Pressure band separation: shot rate spread across non-null bands
    press_nonnull = pressure.loc[pressure["overall_pressure_start"].notna()].copy()
    press_rates = press_nonnull.set_index("overall_pressure_start")["shot_rate"]
    pressure_separates = bool(
        len(press_rates) >= 2 and (press_rates.max() - press_rates.min()) > 0.01
    )
    pressure_shot_rates = {
        str(k): float(v) for k, v in press_rates.items() if pd.notna(k)
    }

    # Box separation: any organized block with |mean box_area diff| > 5 m^2
    diffs = bbox.loc[bbox["lead_to_shot"] == "diff_shot_minus_nons"]
    box_diffs = {
        row["oop_phase_type"]: float(row["mean_box_area"])
        for _, row in diffs.iterrows()
    }
    box_separates = any(abs(v) > 5.0 for v in box_diffs.values())

    n_poss = int((events["event_type"] == "player_possession").sum())
    n_eng = int((events["event_type"] == "on_ball_engagement").sum())

    return {
        "n_matches": len(match_ids),
        "match_ids": match_ids,
        "n_phases": int(len(phases)),
        "n_player_possessions": n_poss,
        "n_on_ball_engagements": n_eng,
        "oop_phase_mix": phase_sum[
            ["oop_phase_type", "n_phases", "total_duration_s", "shot_rate", "goal_rate"]
        ].to_dict(orient="records"),
        "shot_rate_by_organized_block": shot_rate_by_block,
        "pressure_shot_rates": pressure_shot_rates,
        "pressure_bands_separate_shots": pressure_separates,
        "pressure_shot_rate_spread": float(press_rates.max() - press_rates.min())
        if len(press_rates)
        else None,
        "bbox_mean_area_diff_shot_minus_nons": box_diffs,
        "bbox_width_x_length_separates_shots": box_separates,
        "outputs": {
            "phase_summary": "analysis/output/eda_phase_summary.csv",
            "pressure_summary": "analysis/output/eda_pressure_summary.csv",
            "engagement_summary": "analysis/output/eda_engagement_summary.csv",
            "bbox_summary": "analysis/output/eda_bbox_summary.csv",
            "headline": "analysis/output/eda_summary.json",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phases, events, match_ids = load_all(DATA_DIR)
    print(f"Loaded {len(match_ids)} matches, {len(phases)} phases, {len(events)} events")

    phase_sum = phase_mix(phases)
    pressure = pressure_summary(events)
    engagements = engagement_summary(events)
    bbox = bbox_baseline(phases)
    summary = headline_json(match_ids, phases, phase_sum, pressure, bbox, events)

    phase_sum.to_csv(OUT_DIR / "eda_phase_summary.csv", index=False)
    pressure.to_csv(OUT_DIR / "eda_pressure_summary.csv", index=False)
    engagements.to_csv(OUT_DIR / "eda_engagement_summary.csv", index=False)
    bbox.to_csv(OUT_DIR / "eda_bbox_summary.csv", index=False)
    with open(OUT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OOP phase mix ===")
    print(phase_sum.to_string(index=False))
    print("\n=== Shot rate by organized block ===")
    for b, info in summary["shot_rate_by_organized_block"].items():
        print(
            f"  {b}: shot_rate={info['shot_rate']:.4f} "
            f"({info['n_shot']}/{info['n_phases']}), "
            f"goal_rate={info['goal_rate']:.4f}"
        )
    print("\n=== Pressure vs shot ===")
    print(pressure.to_string(index=False))
    print(
        f"\nPressure bands separate shots: {summary['pressure_bands_separate_shots']} "
        f"(spread={summary['pressure_shot_rate_spread']:.4f})"
    )
    print("\n=== Bounding box (organized blocks) ===")
    print(bbox.to_string(index=False))
    print(
        f"\nWidth×length separates shot phases: "
        f"{summary['bbox_width_x_length_separates_shots']}"
    )
    print(f"\nWrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
