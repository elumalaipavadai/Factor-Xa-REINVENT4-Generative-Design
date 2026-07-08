"""
STEP 07: Prioritize TL-generated Factor Xa candidates.

Purpose
-------
Use the scored TL-generated molecules and create practical candidate shortlists.

Main filters
------------
1. Valid RDKit molecule
2. Novel vs Project 1 FXA reference set
3. Basic druglike flag = True
4. High predicted FXA pKi

Candidate tiers
---------------
Tier 1:
    Novel + druglike + predicted pKi >= 8.0

Tier 2:
    Novel + druglike + 7.5 <= predicted pKi < 8.0

Tier 3:
    Novel + druglike + 7.0 <= predicted pKi < 7.5

Outputs
-------
results/tables/tl_fxa_tier1_pki8_druglike_candidates.csv
results/tables/tl_fxa_tier2_pki7p5_druglike_candidates.csv
results/tables/tl_fxa_top100_ranked_candidates.csv
results/tables/tl_fxa_docking_queue_top30.csv
results/metrics/tl_fxa_candidate_prioritization_summary.json

How to run
----------
conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\07_prioritize_tl_fxa_candidates.py
python .\\scripts\\07_prioritize_tl_fxa_candidates.py *> .\\scripts\\07.log
Get-Content .\\scripts\\07.log -Tail 160
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_scored_fxa.csv"
)

OUT_TIER1 = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tl_fxa_tier1_pki8_druglike_candidates.csv"
)

OUT_TIER2 = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tl_fxa_tier2_pki7p5_druglike_candidates.csv"
)

OUT_TOP100 = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tl_fxa_top100_ranked_candidates.csv"
)

OUT_DOCKING_QUEUE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tl_fxa_docking_queue_top30.csv"
)

OUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "tl_fxa_candidate_prioritization_summary.json"
)


def to_bool(series):
    """
    Safely convert CSV-loaded boolean columns to True/False.
    """
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def require_columns(df, cols):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main():
    print("=" * 90)
    print("STEP 07: PRIORITIZE TL-GENERATED FACTOR XA CANDIDATES")
    print("=" * 90)

    print(f"Input CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded dataframe: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    required = [
        "SMILES",
        "Predicted_FXA_pKi",
        "RF_Tree_Uncertainty",
        "Valid_RDKit",
        "Novel_vs_FXA_reference_flat",
        "Basic_Druglike_Flag",
        "MW",
        "cLogP",
        "QED",
    ]
    require_columns(df, required)

    df["Valid_RDKit_bool"] = to_bool(df["Valid_RDKit"])
    df["Novel_bool"] = to_bool(df["Novel_vs_FXA_reference_flat"])
    df["Druglike_bool"] = to_bool(df["Basic_Druglike_Flag"])

    # Core candidate set: valid, novel, druglike.
    cand = df[
        (df["Valid_RDKit_bool"])
        & (df["Novel_bool"])
        & (df["Druglike_bool"])
    ].copy()

    print(f"\nValid + novel + druglike candidates: {len(cand)}")

    if len(cand) == 0:
        raise ValueError("No valid, novel, druglike TL candidates found.")

    # Selection score:
    # Predicted pKi is primary.
    # Penalize high RF uncertainty.
    # Slightly reward QED.
    cand["Selection_Score"] = (
        cand["Predicted_FXA_pKi"]
        - 0.25 * cand["RF_Tree_Uncertainty"]
        + 0.50 * cand["QED"]
    )

    cand["Candidate_Tier"] = "Below_Tier"

    cand.loc[
        cand["Predicted_FXA_pKi"] >= 8.0,
        "Candidate_Tier",
    ] = "Tier_1_pKi_ge_8_druglike"

    cand.loc[
        (cand["Predicted_FXA_pKi"] >= 7.5)
        & (cand["Predicted_FXA_pKi"] < 8.0),
        "Candidate_Tier",
    ] = "Tier_2_pKi_7p5_to_8_druglike"

    cand.loc[
        (cand["Predicted_FXA_pKi"] >= 7.0)
        & (cand["Predicted_FXA_pKi"] < 7.5),
        "Candidate_Tier",
    ] = "Tier_3_pKi_7_to_7p5_druglike"

    # Sort by tier-relevant practical priority.
    cand = cand.sort_values(
        by=[
            "Predicted_FXA_pKi",
            "Selection_Score",
            "RF_Tree_Uncertainty",
            "QED",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    cand["Candidate_Rank"] = np.arange(1, len(cand) + 1)

    tier1 = cand[cand["Predicted_FXA_pKi"] >= 8.0].copy()

    tier2 = cand[
        (cand["Predicted_FXA_pKi"] >= 7.5)
        & (cand["Predicted_FXA_pKi"] < 8.0)
    ].copy()

    top100 = cand.head(100).copy()

    # Docking queue:
    # Prioritize high pKi, druglike, novelty, and reasonable uncertainty.
    # Do not over-filter uncertainty yet because this is exploration.
    docking_queue = cand[
        cand["Predicted_FXA_pKi"] >= 7.0
    ].copy().head(30)

    keep_cols = [
        "Candidate_Rank",
        "Candidate_Tier",
        "Selection_Score",
        "Predicted_FXA_pKi",
        "RF_Tree_Uncertainty",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "HeavyAtoms",
        "QED",
        "SMILES",
        "Canonical_Flat_SMILES",
    ]

    keep_cols = [col for col in keep_cols if col in cand.columns]

    OUT_TIER1.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    tier1[keep_cols].to_csv(OUT_TIER1, index=False)
    tier2[keep_cols].to_csv(OUT_TIER2, index=False)
    top100[keep_cols].to_csv(OUT_TOP100, index=False)
    docking_queue[keep_cols].to_csv(OUT_DOCKING_QUEUE, index=False)

    summary = {
        "input_csv": str(INPUT_CSV),
        "n_total_scored_rows": int(len(df)),
        "n_valid_novel_druglike": int(len(cand)),
        "n_tier1_pki_ge_8_druglike": int(len(tier1)),
        "n_tier2_pki_7p5_to_8_druglike": int(len(tier2)),
        "n_tier3_pki_7_to_7p5_druglike": int(
            ((cand["Predicted_FXA_pKi"] >= 7.0)
             & (cand["Predicted_FXA_pKi"] < 7.5)).sum()
        ),
        "top_candidate_predicted_pKi": float(cand["Predicted_FXA_pKi"].max()),
        "top_candidate_selection_score": float(cand["Selection_Score"].max()),
        "median_candidate_uncertainty": float(cand["RF_Tree_Uncertainty"].median()),
        "outputs": {
            "tier1_csv": str(OUT_TIER1),
            "tier2_csv": str(OUT_TIER2),
            "top100_csv": str(OUT_TOP100),
            "docking_queue_top30_csv": str(OUT_DOCKING_QUEUE),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nTop 10 prioritized candidates:")
    print(cand[keep_cols].head(10).to_string(index=False))

    print("\nSaved outputs:")
    print(f"Tier 1 CSV: {OUT_TIER1}")
    print(f"Tier 2 CSV: {OUT_TIER2}")
    print(f"Top 100 CSV: {OUT_TOP100}")
    print(f"Docking queue top 30 CSV: {OUT_DOCKING_QUEUE}")
    print(f"Summary JSON: {OUT_JSON}")

    print("\nSTEP 07 COMPLETE")


if __name__ == "__main__":
    main()