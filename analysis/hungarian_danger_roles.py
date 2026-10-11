"""Free attackers by position at goals and shot-ending possessions (Hungarian marking).

Requires tracking JSONL (Git LFS). Run from repo root after:
  cd opendata && git lfs pull   # or pull per-match in script

Writes analysis/output/figures/16_free_attackers_by_role.png and CSV.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from coupled_formation import DATA_DIR, OUT_DIR, load_frame_geometry, load_match_meta

ORGANIZED = {"low_block", "medium_block", "high_block"}
ATTACK_ROLES = {
    "CF",
    "RW",
    "LW",
    "AM",
    "RM",
    "LM",
    "RDM",
    "LDM",
    "LF",
    "RF",
    "SS",
}
POSITION_ORDER = ["CF", "RW", "LW", "AM", "RM", "LM", "RDM", "LDM", "LF", "RF"]


def ensure_tracking(match_id: str) -> bool:
    import shutil

    path = DATA_DIR / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    if path.is_file() and path.stat().st_size > 10_000:
        return True
    sc_repo = Path("/tmp/sc-opendata")
    rel = f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    if sc_repo.is_dir():
        subprocess.run(
            ["git", "lfs", "pull", "--include", rel],
            cwd=sc_repo,
            check=False,
            capture_output=True,
        )
        src = sc_repo / rel
        if src.is_file() and src.stat().st_size > 10_000:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, path)
    return path.is_file() and path.stat().st_size > 10_000


def danger_events(match_id: str) -> pd.DataFrame:
    path = DATA_DIR / match_id / f"{match_id}_dynamic_events.csv"
    use = [
        "event_type",
        "event_subtype",
        "frame_end",
        "lead_to_shot",
        "lead_to_goal",
        "team_out_of_possession_phase_type",
        "team_shortname",
        "player_in_possession_position",
    ]
    e = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    poss = e.loc[e["event_type"] == "player_possession"].copy()
    poss = poss.loc[poss["team_out_of_possession_phase_type"].isin(ORGANIZED)]
    poss["danger"] = "other"
    poss.loc[poss["lead_to_goal"], "danger"] = "goal"
    poss.loc[poss["lead_to_shot"] & ~poss["lead_to_goal"], "danger"] = "shot"
    poss = poss.loc[poss["danger"].isin(["goal", "shot"])]
    poss["match_id"] = match_id
    return poss


def analyze_match(match_id: str) -> list[dict]:
    if not ensure_tracking(match_id):
        return []
    meta = load_match_meta(DATA_DIR, match_id)
    rows: list[dict] = []
    for _, ev in danger_events(match_id).iterrows():
        frame = int(ev["frame_end"]) if pd.notna(ev["frame_end"]) else None
        if frame is None:
            continue
        # defending team = not in possession
        att_team_name = ev["team_shortname"]
        if att_team_name == meta.home_name:
            def_team_id = meta.away_id
        else:
            def_team_id = meta.home_id
        geom = load_frame_geometry(DATA_DIR, meta, frame, defending_team_id=def_team_id)
        if geom is None:
            continue
        att_free = [r for r in geom["free_roles"] if r in ATTACK_ROLES]
        for role in att_free:
            rows.append(
                {
                    "match_id": match_id,
                    "danger": ev["danger"],
                    "frame": frame,
                    "free_role": role,
                    "n_free": len(att_free),
                    "lambda2_defense": geom["lambda2_defense"],
                    "ball_carrier_role": ev.get("player_in_possession_position"),
                }
            )
        if not att_free:
            rows.append(
                {
                    "match_id": match_id,
                    "danger": ev["danger"],
                    "frame": frame,
                    "free_role": None,
                    "n_free": len(att_free),
                    "lambda2_defense": geom["lambda2_defense"],
                    "ball_carrier_role": ev.get("player_in_possession_position"),
                }
            )
    return rows


def plot_role_counts(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.loc[df["free_role"].notna()].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No tracking frames — run git lfs pull on opendata", ha="center")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, danger, title in zip(
        axes,
        ["goal", "shot"],
        ["Goals — free attackers by position", "Shots (big chances) — free attackers by position"],
    ):
        c = Counter(sub.loc[sub["danger"] == danger, "free_role"])
        order = [p for p in POSITION_ORDER if p in c] + sorted(k for k in c if k not in POSITION_ORDER)
        vals = [c[p] for p in order]
        ax.barh(order, vals, color="#c0392b" if danger == "goal" else "#e67e22")
        ax.set_title(title)
        ax.set_xlabel("Count (free attacker frames)")
        ax.invert_yaxis()
    fig.suptitle(
        "Hungarian marking: who is unmarked when chances happen?\n"
        "(Free = >8 m from nearest assigned marker or attacker goal-side of marker)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    match_ids = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    all_rows: list[dict] = []
    for mid in match_ids:
        print(f"  {mid}…", end=" ", flush=True)
        part = analyze_match(mid)
        print(len(part), "rows")
        all_rows.extend(part)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "hungarian_free_roles.csv", index=False)
    plot_role_counts(df, fig_dir / "16_free_attackers_by_role.png")
    print(f"Wrote {len(df)} rows → {OUT_DIR / 'hungarian_free_roles.csv'}")


if __name__ == "__main__":
    main()
