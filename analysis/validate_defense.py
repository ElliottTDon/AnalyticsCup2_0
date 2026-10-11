"""Validate coupled graph features against box and pressure baselines.

Aggregates phase-level contrasts within team and block type. Tests whether
graph features separate shot vs non-shot phases beyond OOP width/length,
nearest-defender distance, and overall_pressure.

Twenty matches is a small sample; phases are nested in matches (cluster note).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coupled_formation import OUT_DIR, ORGANIZED_BLOCKS, process_all_matches

WORKSPACE = Path(__file__).resolve().parents[1]

GRAPH_FEATURES = [
    "mean_lambda2_defense",
    "mean_n_free_attackers",
    "mean_uncovered_danger",
    "mean_structural_pressure",
    "mean_structural_pressure_frac",
    "mean_lambda2_supra",
    "lambda2_drop_pre2s",
]

BASELINE_FEATURES = [
    "mean_oop_width",
    "mean_oop_length",
    "mean_oop_area",
    "mean_nearest_defender_to_ball",
    "mean_overall_pressure_score",
]


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference of means a - b with percentile bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {
            "diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": float(np.nanmean(a)) if len(a) else float("nan"),
            "mean_b": float(np.nanmean(b)) if len(b) else float("nan"),
        }
    rng = np.random.default_rng(seed)
    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[i] = aa.mean() - bb.mean()
    return {
        "diff": obs,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on columns of X (with intercept)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < X.shape[1] + 2:
        return resid
    Xm = np.column_stack([np.ones(mask.sum()), X[mask]])
    ym = y[mask]
    try:
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        resid[mask] = ym - Xm @ beta
    except np.linalg.LinAlgError:
        pass
    return resid


def logistic_auc_simple(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC for a continuous score vs binary y."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y, score = y[mask], score[mask]
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # AUC = P(score_pos > score_neg) + 0.5 P(equal)
    # Efficient via ranks
    order = np.argsort(score)
    ranks = np.empty_like(score, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    # simple version without tie correction is fine enough
    sum_ranks_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def stratified_contrasts(phase_df: pd.DataFrame) -> list[dict]:
    """Within defending_team_id x block_type, shot vs non-shot mean diffs."""
    results = []
    # Use defending team as the unit (OOP team)
    df = phase_df.copy()
    df["def_team_key"] = df["match_id"].astype(str) + ":" + df["defending_team_id"].astype(str)

    features = GRAPH_FEATURES + BASELINE_FEATURES
    for block in ORGANIZED_BLOCKS:
        sub = df[df["block_type"] == block]
        if sub.empty:
            continue
        # Pool within block but difference of means still reported globally
        # and also residualized within team
        shot = sub[sub["lead_to_shot"] == True]  # noqa: E712
        nons = sub[sub["lead_to_shot"] == False]  # noqa: E712
        for feat in features:
            if feat not in sub.columns:
                continue
            raw = bootstrap_mean_diff(shot[feat].values, nons[feat].values, seed=hash(feat) % 10_000)
            # Within-team residualized: demean by team then compare
            demeaned = []
            for _, g in sub.groupby("defending_team_id"):
                mu = g[feat].mean()
                demeaned.append(g.assign(_dm=g[feat] - mu))
            dem = pd.concat(demeaned, ignore_index=True) if demeaned else sub.assign(_dm=np.nan)
            dem_shot = dem.loc[dem["lead_to_shot"] == True, "_dm"].values  # noqa: E712
            dem_nons = dem.loc[dem["lead_to_shot"] == False, "_dm"].values  # noqa: E712
            within = bootstrap_mean_diff(dem_shot, dem_nons, seed=17 + hash(feat) % 1000)
            results.append(
                {
                    "slice": "within_block",
                    "block_type": block,
                    "feature": feat,
                    "family": "graph" if feat in GRAPH_FEATURES else "baseline",
                    "shot_mean": raw["mean_a"],
                    "nons_mean": raw["mean_b"],
                    "diff_shot_minus_nons": raw["diff"],
                    "ci_low": raw["ci_low"],
                    "ci_high": raw["ci_high"],
                    "n_shot": raw["n_a"],
                    "n_nons": raw["n_b"],
                    "within_team_diff": within["diff"],
                    "within_team_ci_low": within["ci_low"],
                    "within_team_ci_high": within["ci_high"],
                    "ci_excludes_zero": bool(
                        np.isfinite(raw["ci_low"])
                        and np.isfinite(raw["ci_high"])
                        and (raw["ci_low"] > 0 or raw["ci_high"] < 0)
                    ),
                }
            )
    return results


def residual_beyond_baseline(phase_df: pd.DataFrame) -> list[dict]:
    """Does each graph feature separate shots after residualizing on baselines?"""
    rows = []
    base_cols = [c for c in BASELINE_FEATURES if c in phase_df.columns]
    y = phase_df["lead_to_shot"].astype(int).values
    X = phase_df[base_cols].values
    # Also add block dummies
    for block in ORGANIZED_BLOCKS:
        phase_df = phase_df.copy()
    block_dummies = pd.get_dummies(phase_df["block_type"], prefix="block")
    X_full = np.column_stack([X, block_dummies.values.astype(float)])

    baseline_auc = {}
    for b in base_cols:
        # Direction: smaller box / closer defender / higher pressure?
        # Use absolute association via max(auc, 1-auc) reported as oriented auc
        auc = logistic_auc_simple(y, phase_df[b].values)
        baseline_auc[b] = auc

    y_resid = residualize(y.astype(float), X_full)

    for feat in GRAPH_FEATURES:
        if feat not in phase_df.columns:
            continue
        raw_auc = logistic_auc_simple(y, phase_df[feat].values)
        # Residualize feature on baselines, then AUC of residuals vs shot
        feat_resid = residualize(phase_df[feat].values, X_full)
        # Correlate residuals with shot label via AUC of feat_resid
        resid_auc = logistic_auc_simple(y, feat_resid)
        # Mean residual difference
        shot_mask = y == 1
        diff = float(np.nanmean(feat_resid[shot_mask]) - np.nanmean(feat_resid[~shot_mask]))
        # Bootstrap CI on residual mean diff
        boot = bootstrap_mean_diff(feat_resid[shot_mask], feat_resid[~shot_mask], seed=99)
        # Compare |auc-0.5| to best baseline
        best_base = max(baseline_auc, key=lambda k: abs(baseline_auc[k] - 0.5))
        best_base_auc = baseline_auc[best_base]
        beats = abs(resid_auc - 0.5) > 0.02 and boot["ci_excludes_zero"] if False else (
            bool(np.isfinite(boot["ci_low"]) and (boot["ci_low"] > 0 or boot["ci_high"] < 0))
        )
        # More honest: beats if residual CI excludes 0
        beats_baseline = bool(
            np.isfinite(boot["ci_low"]) and np.isfinite(boot["ci_high"]) and (boot["ci_low"] > 0 or boot["ci_high"] < 0)
        )
        rows.append(
            {
                "feature": feat,
                "raw_auc": raw_auc,
                "residual_auc": resid_auc,
                "residual_diff_shot_minus_nons": diff,
                "residual_ci_low": boot["ci_low"],
                "residual_ci_high": boot["ci_high"],
                "beats_baseline": beats_baseline,
                "best_baseline": best_base,
                "best_baseline_auc": best_base_auc,
                "baseline_aucs": {k: float(v) for k, v in baseline_auc.items()},
            }
        )
    return rows


def pre_shot_lambda2_check(phase_df: pd.DataFrame) -> dict:
    """In the 2s before phase end, does lambda2 fall more before shots?"""
    if "lambda2_drop_pre2s" not in phase_df.columns:
        return {"available": False}
    shot = phase_df.loc[phase_df["lead_to_shot"] == True, "lambda2_drop_pre2s"].values  # noqa: E712
    nons = phase_df.loc[phase_df["lead_to_shot"] == False, "lambda2_drop_pre2s"].values  # noqa: E712
    boot = bootstrap_mean_diff(shot, nons, seed=123)
    by_block = {}
    for block in ORGANIZED_BLOCKS:
        sub = phase_df[phase_df["block_type"] == block]
        s = sub.loc[sub["lead_to_shot"] == True, "lambda2_drop_pre2s"].values  # noqa: E712
        n = sub.loc[sub["lead_to_shot"] == False, "lambda2_drop_pre2s"].values  # noqa: E712
        by_block[block] = bootstrap_mean_diff(s, n, seed=200 + hash(block) % 50)
    return {
        "available": True,
        "metric": "lambda2_drop_pre2s (early_mean - last_2s_mean; positive = connectivity fell)",
        "overall": boot,
        "by_block": by_block,
        "interpretation": (
            "Positive diff means shot phases show a larger lambda2 drop into the last 2s "
            "than non-shot phases."
        ),
    }


def shot_rates_by_block(phase_df: pd.DataFrame) -> dict:
    out = {}
    for block in ORGANIZED_BLOCKS:
        sub = phase_df[phase_df["block_type"] == block]
        n = len(sub)
        n_shot = int(sub["lead_to_shot"].sum()) if n else 0
        out[block] = {
            "n_phases": n,
            "n_shot": n_shot,
            "shot_rate": (n_shot / n) if n else float("nan"),
        }
    return out


def team_summary(phase_df: pd.DataFrame) -> pd.DataFrame:
    """Per defending team across matches: mean graph metrics and shot rate."""
    rows = []
    for (mid, def_id), g in phase_df.groupby(["match_id", "defending_team_id"]):
        # defending team name: opposite of team_in_possession on a row... use home/away
        home = g["home_name"].iloc[0]
        away = g["away_name"].iloc[0]
        # Infer name: if defending_team_id matches... we don't have id->name map in phase table.
        # Use: when team_in_possession is home, defense is away, etc.
        # Majority vote from phase
        tip_names = g["team_in_possession"].value_counts()
        # defending is the other of home/away relative to possession
        # Simple: name = home if most possessions against home? 
        # defending_team_id — compare via whether tip is home or away by name match
        def_name_guess = None
        for _, r in g.iterrows():
            tip = r["team_in_possession"]
            if tip == home:
                def_name_guess = away
                break
            if tip == away:
                def_name_guess = home
                break
        rows.append(
            {
                "match_id": mid,
                "defending_team_id": def_id,
                "defending_team": def_name_guess,
                "n_phases": len(g),
                "shot_rate_against": float(g["lead_to_shot"].mean()),
                "mean_lambda2_defense": float(g["mean_lambda2_defense"].mean()),
                "mean_uncovered_danger": float(g["mean_uncovered_danger"].mean()),
                "mean_n_free_attackers": float(g["mean_n_free_attackers"].mean()),
                "mean_structural_pressure": float(g["mean_structural_pressure"].mean()),
                "mean_oop_area": float(g["mean_oop_area"].mean()),
                "mean_nearest_defender_to_ball": float(g["mean_nearest_defender_to_ball"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_validation(
    phase_df: pd.DataFrame | None = None,
    out_dir: Path = OUT_DIR,
    rebuild: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_path = out_dir / "phase_features.csv"
    if rebuild or phase_df is None:
        if rebuild or not phase_path.exists():
            # Fresh run: truncate frame file first
            frame_path = out_dir / "frame_features.csv"
            if frame_path.exists():
                frame_path.unlink()
            _, phase_df = process_all_matches(out_dir=out_dir)
        else:
            phase_df = pd.read_csv(phase_path)

    contrasts = stratified_contrasts(phase_df)
    residual = residual_beyond_baseline(phase_df)
    pre_shot = pre_shot_lambda2_check(phase_df)
    rates = shot_rates_by_block(phase_df)
    teams = team_summary(phase_df)
    teams.to_csv(out_dir / "team_summary.csv", index=False)

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out_dir / "contrast_by_block.csv", index=False)

    # Headline: which graph features beat baseline
    beating = [r for r in residual if r["beats_baseline"]]
    not_beating = [r for r in residual if not r["beats_baseline"]]

    summary = {
        "n_matches": int(phase_df["match_id"].nunique()),
        "n_phases": int(len(phase_df)),
        "n_shot_phases": int(phase_df["lead_to_shot"].sum()),
        "sample_note": (
            "Twenty matches is a small sample. Phases are nested within matches; "
            "bootstrap CIs treat phases as independent and therefore overstate precision."
        ),
        "cluster_note": "Phases nested in matches; within-team demeaning used for one contrast layer.",
        "laplacian": "combinatorial L = D - W; lambda2 = algebraic connectivity",
        "shot_rate_by_block": rates,
        "graph_vs_baseline": residual,
        "features_beating_baseline": [r["feature"] for r in beating],
        "features_not_beating_baseline": [r["feature"] for r in not_beating],
        "pre_shot_lambda2": {
            "overall_diff_shot_minus_nons": pre_shot.get("overall", {}).get("diff"),
            "ci_low": pre_shot.get("overall", {}).get("ci_low"),
            "ci_high": pre_shot.get("overall", {}).get("ci_high"),
            "by_block": {
                b: {
                    "diff": v["diff"],
                    "ci_low": v["ci_low"],
                    "ci_high": v["ci_high"],
                    "mean_shot": v["mean_a"],
                    "mean_nons": v["mean_b"],
                }
                for b, v in (pre_shot.get("by_block") or {}).items()
            },
            "metric": pre_shot.get("metric"),
            "interpretation": pre_shot.get("interpretation"),
        },
        "contrast_highlights": [
            {
                "block_type": r["block_type"],
                "feature": r["feature"],
                "diff": r["diff_shot_minus_nons"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "family": r["family"],
            }
            for r in contrasts
            if r["ci_excludes_zero"]
        ],
        "outputs": {
            "phase_features": str(out_dir / "phase_features.csv"),
            "frame_features": str(out_dir / "frame_features.csv"),
            "contrast_by_block": str(out_dir / "contrast_by_block.csv"),
            "team_summary": str(out_dir / "team_summary.csv"),
            "contrast_summary": str(out_dir / "contrast_summary.json"),
        },
    }

    # JSON-safe
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            x = float(o)
            if not np.isfinite(x):
                return None
            return x
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    summary = _clean(summary)
    with open(out_dir / "contrast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("n_matches", "n_phases", "features_beating_baseline", "features_not_beating_baseline")}, indent=2))
    return summary


if __name__ == "__main__":
    run_validation(rebuild=True)
