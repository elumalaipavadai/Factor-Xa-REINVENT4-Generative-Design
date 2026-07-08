"""
STEP 06C: Compare original REINVENT prior vs FXA transfer-learned model.

Inputs
------
results/metrics/reinvent_prior_1000_score_summary.json
results/metrics/reinvent_tl_fxa_pki8_all_flat_sanity_1000_score_summary.json

Outputs
-------
results/tables/prior_vs_tl_1000_score_comparison.csv
results/metrics/prior_vs_tl_1000_score_comparison.json

How to run
----------
python -m py_compile .\\scripts\\06_compare_prior_vs_tl_scores.py
python .\\scripts\\06_compare_prior_vs_tl_scores.py *> .\\scripts\\06_compare.log
Get-Content .\\scripts\\06_compare.log -Tail 120
"""

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRIOR_JSON = PROJECT_ROOT / "results" / "metrics" / "reinvent_prior_1000_score_summary.json"

TL_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_score_summary.json"
)

OUT_CSV = PROJECT_ROOT / "results" / "tables" / "prior_vs_tl_1000_score_comparison.csv"
OUT_JSON = PROJECT_ROOT / "results" / "metrics" / "prior_vs_tl_1000_score_comparison.json"


METRICS = [
    "n_input_rows",
    "n_valid_rdkit",
    "n_invalid_rdkit",
    "n_unique_valid_canonical",
    "n_novel_vs_reference",
    "n_basic_druglike",
    "predicted_pKi_mean",
    "predicted_pKi_median",
    "predicted_pKi_max",
    "predicted_pKi_top10_mean",
    "n_pred_pKi_ge_7",
    "n_pred_pKi_ge_8",
    "rf_uncertainty_median",
    "rf_uncertainty_top10_mean",
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    print("=" * 90)
    print("STEP 06C: COMPARE PRIOR 1000 VS TL 1000")
    print("=" * 90)

    prior = load_json(PRIOR_JSON)
    tl = load_json(TL_JSON)

    rows = []

    for metric in METRICS:
        prior_value = prior.get(metric, None)
        tl_value = tl.get(metric, None)

        delta = None
        fold_change = None

        if isinstance(prior_value, (int, float)) and isinstance(tl_value, (int, float)):
            delta = tl_value - prior_value
            if prior_value != 0:
                fold_change = tl_value / prior_value

        rows.append(
            {
                "Metric": metric,
                "Prior_1000": prior_value,
                "TL_1000": tl_value,
                "Delta_TL_minus_Prior": delta,
                "Fold_TL_over_Prior": fold_change,
            }
        )

    df = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_CSV, index=False)

    comparison = {
        "prior_summary_json": str(PRIOR_JSON),
        "tl_summary_json": str(TL_JSON),
        "comparison_csv": str(OUT_CSV),
        "metrics": rows,
    }

    OUT_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\nComparison table:")
    print(df.to_string(index=False))

    print("\nKey interpretation:")
    print(
        f"Mean pKi shift: {tl['predicted_pKi_mean'] - prior['predicted_pKi_mean']:.3f}"
    )
    print(
        f"Top10 mean pKi shift: {tl['predicted_pKi_top10_mean'] - prior['predicted_pKi_top10_mean']:.3f}"
    )
    print(
        f"pKi >= 7 count shift: {tl['n_pred_pKi_ge_7'] - prior['n_pred_pKi_ge_7']}"
    )
    print(
        f"pKi >= 8 count shift: {tl['n_pred_pKi_ge_8'] - prior['n_pred_pKi_ge_8']}"
    )

    print("\nSaved outputs:")
    print(f"Comparison CSV: {OUT_CSV}")
    print(f"Comparison JSON: {OUT_JSON}")

    print("\nSTEP 06C COMPLETE")


if __name__ == "__main__":
    main()