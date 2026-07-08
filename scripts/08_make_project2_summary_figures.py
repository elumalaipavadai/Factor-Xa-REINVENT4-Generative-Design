"""
STEP 08: Create Project 2 summary figures and final tables.

Purpose
-------
Create final portfolio-ready result tables and figures for:

1. Prior vs TL score comparison
2. Predicted pKi distributions
3. Candidate tier counts
4. Final concise Project 2 summary

Inputs
------
results/tables/prior_vs_tl_1000_score_comparison.csv
data/generated/reinvent_prior_1000_scored_fxa.csv
data/generated/reinvent_tl_fxa_pki8_all_flat_sanity_1000_scored_fxa.csv
results/metrics/tl_fxa_candidate_prioritization_summary.json

Outputs
-------
results/figures/prior_vs_tl_key_metrics.png
results/figures/prior_vs_tl_predicted_pki_distribution.png
results/figures/tl_candidate_tier_counts.png

results/tables/project2_final_key_metrics.csv
results/metrics/project2_final_summary.json
results/metrics/project2_final_summary.txt

How to run
----------
conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\08_make_project2_summary_figures.py
python .\\scripts\\08_make_project2_summary_figures.py *> .\\scripts\\08.log
Get-Content .\\scripts\\08.log -Tail 160
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "prior_vs_tl_1000_score_comparison.csv"
)

PRIOR_SCORED_CSV = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reinvent_prior_1000_scored_fxa.csv"
)

TL_SCORED_CSV = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_scored_fxa.csv"
)

CANDIDATE_SUMMARY_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "tl_fxa_candidate_prioritization_summary.json"
)

OUT_FIG_DIR = PROJECT_ROOT / "results" / "figures"
OUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUT_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

OUT_KEY_METRICS_FIG = OUT_FIG_DIR / "prior_vs_tl_key_metrics.png"
OUT_PKI_DIST_FIG = OUT_FIG_DIR / "prior_vs_tl_predicted_pki_distribution.png"
OUT_TIER_COUNTS_FIG = OUT_FIG_DIR / "tl_candidate_tier_counts.png"

OUT_FINAL_METRICS_CSV = OUT_TABLE_DIR / "project2_final_key_metrics.csv"
OUT_FINAL_SUMMARY_JSON = OUT_METRICS_DIR / "project2_final_summary.json"
OUT_FINAL_SUMMARY_TXT = OUT_METRICS_DIR / "project2_final_summary.txt"


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def get_metric(df, metric_name, col):
    row = df[df["Metric"] == metric_name]
    if row.empty:
        raise ValueError(f"Metric not found: {metric_name}")
    return float(row.iloc[0][col])


def main():
    print("=" * 90)
    print("STEP 08: CREATE PROJECT 2 SUMMARY FIGURES AND TABLES")
    print("=" * 90)

    for path in [
        COMPARISON_CSV,
        PRIOR_SCORED_CSV,
        TL_SCORED_CSV,
        CANDIDATE_SUMMARY_JSON,
    ]:
        require_file(path)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(COMPARISON_CSV)
    prior = pd.read_csv(PRIOR_SCORED_CSV)
    tl = pd.read_csv(TL_SCORED_CSV)
    candidate_summary = json.loads(CANDIDATE_SUMMARY_JSON.read_text(encoding="utf-8"))

    print(f"Comparison table: {comparison.shape}")
    print(f"Prior scored molecules: {prior.shape}")
    print(f"TL scored molecules: {tl.shape}")

    key_metrics = [
        "predicted_pKi_mean",
        "predicted_pKi_median",
        "predicted_pKi_max",
        "predicted_pKi_top10_mean",
        "n_pred_pKi_ge_7",
        "n_pred_pKi_ge_8",
        "n_basic_druglike",
    ]

    final_rows = []

    for metric in key_metrics:
        prior_value = get_metric(comparison, metric, "Prior_1000")
        tl_value = get_metric(comparison, metric, "TL_1000")
        delta = tl_value - prior_value

        fold = None
        if prior_value != 0:
            fold = tl_value / prior_value

        final_rows.append(
            {
                "Metric": metric,
                "Prior_1000": prior_value,
                "TL_1000": tl_value,
                "Delta_TL_minus_Prior": delta,
                "Fold_TL_over_Prior": fold,
            }
        )

    final_metrics = pd.DataFrame(final_rows)
    final_metrics.to_csv(OUT_FINAL_METRICS_CSV, index=False)

    # -------------------------------------------------------------------------
    # Figure 1: key prior vs TL metrics
    # -------------------------------------------------------------------------
    plot_metrics = [
        "predicted_pKi_mean",
        "predicted_pKi_top10_mean",
        "predicted_pKi_max",
    ]

    plot_df = final_metrics[final_metrics["Metric"].isin(plot_metrics)].copy()

    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], plot_df["Prior_1000"], width, label="Prior")
    plt.bar([i + width / 2 for i in x], plot_df["TL_1000"], width, label="TL")
    plt.xticks(list(x), plot_df["Metric"], rotation=25, ha="right")
    plt.ylabel("Predicted pKi")
    plt.title("REINVENT Prior vs FXA Transfer-Learned Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_KEY_METRICS_FIG, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: predicted pKi distribution
    # -------------------------------------------------------------------------
    prior_valid = prior[prior["Valid_RDKit"] == True].copy()
    tl_valid = tl[tl["Valid_RDKit"] == True].copy()

    plt.figure(figsize=(8, 5))
    plt.hist(
        prior_valid["Predicted_FXA_pKi"].dropna(),
        bins=30,
        alpha=0.6,
        label="Prior",
    )
    plt.hist(
        tl_valid["Predicted_FXA_pKi"].dropna(),
        bins=30,
        alpha=0.6,
        label="TL",
    )
    plt.xlabel("Predicted FXA pKi")
    plt.ylabel("Molecule count")
    plt.title("Predicted FXA pKi Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PKI_DIST_FIG, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: candidate tier counts
    # -------------------------------------------------------------------------
    tier_counts = {
        "Tier 1\npKi â‰¥ 8": candidate_summary["n_tier1_pki_ge_8_druglike"],
        "Tier 2\n7.5â€“8": candidate_summary["n_tier2_pki_7p5_to_8_druglike"],
        "Tier 3\n7â€“7.5": candidate_summary["n_tier3_pki_7_to_7p5_druglike"],
    }

    plt.figure(figsize=(7, 5))
    plt.bar(list(tier_counts.keys()), list(tier_counts.values()))
    plt.ylabel("Candidate count")
    plt.title("Novel Druglike TL Candidate Tiers")
    plt.tight_layout()
    plt.savefig(OUT_TIER_COUNTS_FIG, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    summary = {
        "project_title": "REINVENT-Based Generative Design of Factor Xa Inhibitors Using Scaffold-Aware ML Scoring",
        "prior_vs_tl": {
            "prior_n_valid": get_metric(comparison, "n_valid_rdkit", "Prior_1000"),
            "tl_n_valid": get_metric(comparison, "n_valid_rdkit", "TL_1000"),
            "prior_mean_predicted_pKi": get_metric(comparison, "predicted_pKi_mean", "Prior_1000"),
            "tl_mean_predicted_pKi": get_metric(comparison, "predicted_pKi_mean", "TL_1000"),
            "mean_predicted_pKi_shift": get_metric(comparison, "predicted_pKi_mean", "Delta_TL_minus_Prior"),
            "prior_top10_mean_predicted_pKi": get_metric(comparison, "predicted_pKi_top10_mean", "Prior_1000"),
            "tl_top10_mean_predicted_pKi": get_metric(comparison, "predicted_pKi_top10_mean", "TL_1000"),
            "top10_mean_predicted_pKi_shift": get_metric(comparison, "predicted_pKi_top10_mean", "Delta_TL_minus_Prior"),
            "prior_n_pki_ge_7": get_metric(comparison, "n_pred_pKi_ge_7", "Prior_1000"),
            "tl_n_pki_ge_7": get_metric(comparison, "n_pred_pKi_ge_7", "TL_1000"),
            "prior_n_pki_ge_8": get_metric(comparison, "n_pred_pKi_ge_8", "Prior_1000"),
            "tl_n_pki_ge_8": get_metric(comparison, "n_pred_pKi_ge_8", "TL_1000"),
        },
        "candidate_prioritization": candidate_summary,
        "outputs": {
            "key_metrics_csv": str(OUT_FINAL_METRICS_CSV),
            "key_metrics_figure": str(OUT_KEY_METRICS_FIG),
            "pki_distribution_figure": str(OUT_PKI_DIST_FIG),
            "candidate_tier_counts_figure": str(OUT_TIER_COUNTS_FIG),
            "summary_json": str(OUT_FINAL_SUMMARY_JSON),
            "summary_txt": str(OUT_FINAL_SUMMARY_TXT),
        },
    }

    OUT_FINAL_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_text = f"""
Project 2 Summary
=================

Title:
REINVENT-Based Generative Design of Factor Xa Inhibitors Using Scaffold-Aware ML Scoring

Main result:
A short REINVENT transfer-learning run shifted generated molecules toward higher predicted Factor Xa activity.

Prior vs TL:
- Mean predicted pKi: {summary['prior_vs_tl']['prior_mean_predicted_pKi']:.3f} -> {summary['prior_vs_tl']['tl_mean_predicted_pKi']:.3f}
- Top10 mean predicted pKi: {summary['prior_vs_tl']['prior_top10_mean_predicted_pKi']:.3f} -> {summary['prior_vs_tl']['tl_top10_mean_predicted_pKi']:.3f}
- Predicted pKi >= 7 molecules: {int(summary['prior_vs_tl']['prior_n_pki_ge_7'])} -> {int(summary['prior_vs_tl']['tl_n_pki_ge_7'])}
- Predicted pKi >= 8 molecules: {int(summary['prior_vs_tl']['prior_n_pki_ge_8'])} -> {int(summary['prior_vs_tl']['tl_n_pki_ge_8'])}

Candidate prioritization:
- Valid + novel + druglike TL candidates: {candidate_summary['n_valid_novel_druglike']}
- Tier 1 candidates, predicted pKi >= 8: {candidate_summary['n_tier1_pki_ge_8_druglike']}
- Tier 2 candidates, predicted pKi 7.5-8: {candidate_summary['n_tier2_pki_7p5_to_8_druglike']}
- Tier 3 candidates, predicted pKi 7-7.5: {candidate_summary['n_tier3_pki_7_to_7p5_druglike']}

Interpretation:
Transfer learning produced a measurable enrichment of high-scoring Factor Xa-like molecules compared with the original REINVENT prior. The final prioritized output is a novel, druglike candidate set and a top-30 docking queue for structure-based follow-up.

Caution:
Predicted activity comes from a scaffold-aware ML model and should be treated as a prioritization signal, not experimental activity. The next validation step is docking/structure-based filtering.
""".strip()

    OUT_FINAL_SUMMARY_TXT.write_text(summary_text, encoding="utf-8")

    print("\nFinal key metrics:")
    print(final_metrics.to_string(index=False))

    print("\nProject 2 summary text:")
    print(summary_text)

    print("\nSaved outputs:")
    print(f"Key metrics CSV: {OUT_FINAL_METRICS_CSV}")
    print(f"Key metrics figure: {OUT_KEY_METRICS_FIG}")
    print(f"pKi distribution figure: {OUT_PKI_DIST_FIG}")
    print(f"Candidate tier figure: {OUT_TIER_COUNTS_FIG}")
    print(f"Summary JSON: {OUT_FINAL_SUMMARY_JSON}")
    print(f"Summary TXT: {OUT_FINAL_SUMMARY_TXT}")

    print("\nSTEP 08 COMPLETE")


if __name__ == "__main__":
    main()