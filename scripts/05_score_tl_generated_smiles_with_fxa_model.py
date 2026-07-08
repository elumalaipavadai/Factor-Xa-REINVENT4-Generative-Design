"""
STEP 05D: Score TL-generated REINVENT molecules with the Factor Xa RF model.

Purpose
-------
This script scores molecules generated from the REINVENT transfer-learned model
using the Project 1 scaffold-aware Factor Xa RandomForest model.

Input
-----
data/generated/reinvent_tl_fxa_pki8_all_flat_sanity_1000.csv

Model
-----
models/fxa_05b_best_scaffold_feature_model.joblib

Reference set
-------------
data/reference/fxa_reference_all_valid_unique_smiles.csv

Outputs
-------
data/generated/reinvent_tl_fxa_pki8_all_flat_sanity_1000_scored_fxa.csv
results/tables/reinvent_tl_fxa_pki8_all_flat_sanity_1000_top50_fxa.csv
results/metrics/reinvent_tl_fxa_pki8_all_flat_sanity_1000_score_summary.json

How to run
----------
conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\05_score_tl_generated_smiles_with_fxa_model.py
python .\\scripts\\05_score_tl_generated_smiles_with_fxa_model.py *> .\\scripts\\05_score.log
Get-Content .\\scripts\\05_score.log -Tail 160
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import rdFingerprintGenerator


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "fxa_05b_best_scaffold_feature_model.joblib"
)

REFERENCE_CSV = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "fxa_reference_all_valid_unique_smiles.csv"
)

OUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_scored_fxa.csv"
)

OUT_TOP50 = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_top50_fxa.csv"
)

OUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "reinvent_tl_fxa_pki8_all_flat_sanity_1000_score_summary.json"
)


# =============================================================================
# Fingerprint settings
# =============================================================================

# These must match the Project 1 Morgan_2048 model.
FP_RADIUS = 2
FP_BITS = 2048

# RDKit generator API replacement for deprecated GetMorganFingerprintAsBitVect.
# This is configured to match standard binary Morgan bit fingerprints:
# radius=2, fpSize=2048, no count simulation, no chirality.
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
)


# =============================================================================
# Helper functions
# =============================================================================

def detect_smiles_column(df):
    """
    Detect the SMILES column in a REINVENT output CSV.
    """
    candidates = [
        "SMILES",
        "smiles",
        "Smiles",
        "canonical_smiles",
        "Generated_SMILES",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"No SMILES column found. Columns are: {list(df.columns)}")


def mol_from_smiles(smi):
    """
    Convert a SMILES string to an RDKit molecule.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None

    return Chem.MolFromSmiles(smi.strip())


def canonical_flat_smiles(mol):
    """
    Generate flat canonical SMILES.

    isomericSmiles=False is used so novelty comparison is consistent with the
    stereo-free REINVENT TL workflow.
    """
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def morgan_fp_array(mol):
    """
    Generate a binary Morgan fingerprint using RDKit's generator API.

    Parameters are set to match the Project 1 Morgan_2048 setup:
        radius = 2
        fpSize = 2048
        countSimulation = False
        includeChirality = False
        useBondTypes = True
    """
    fp = MORGAN_GENERATOR.GetFingerprint(mol)

    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def basic_properties(mol):
    """
    Compute simple molecular properties for prioritization and filtering.
    """
    return {
        "MW": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "QED": QED.qed(mol),
    }


def basic_druglike_flag(props):
    """
    Simple drug-like filter used for ranking and summary only.

    This does not remove molecules. It only labels them.
    """
    return (
        200 <= props["MW"] <= 600
        and -2 <= props["cLogP"] <= 5
        and props["HBD"] <= 5
        and props["HBA"] <= 10
        and props["RotBonds"] <= 12
        and props["HeavyAtoms"] <= 50
    )


def load_reference_flat_smiles(path):
    """
    Load Project 1 reference molecules and convert them to flat canonical SMILES.

    Used for novelty checking:
        Novel_vs_FXA_reference_flat = True/False
    """
    if not path.exists():
        print(f"WARNING: Reference CSV not found: {path}")
        return set()

    ref = pd.read_csv(path)

    smiles_col = None
    for col in [
        "canonical_smiles",
        "canonical_smiles_rdkit",
        "model_smiles",
        "smiles",
        "SMILES",
    ]:
        if col in ref.columns:
            smiles_col = col
            break

    if smiles_col is None:
        print(f"WARNING: No reference SMILES column found in {path}")
        print(f"Reference columns: {list(ref.columns)}")
        return set()

    ref_set = set()

    for smi in ref[smiles_col].dropna().astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_set.add(canonical_flat_smiles(mol))

    return ref_set


def transform_pipeline_features(model, X):
    """
    Apply fitted preprocessing steps before accessing the final RF estimator.

    Project 1 Step 05b model is expected to be a sklearn Pipeline, likely:

        SimpleImputer -> VarianceThreshold -> RandomForestRegressor

    model.predict(X) works directly on raw Morgan fingerprints because the
    pipeline handles preprocessing internally.

    But for RF tree uncertainty, individual trees belong to the final estimator.
    Therefore the raw Morgan X must first be transformed through all fitted
    preprocessing steps.
    """
    if not hasattr(model, "steps"):
        return X, model

    X_model = X

    print("\nPipeline preprocessing steps:")
    for step_name, step in model.steps[:-1]:
        print(f"  applying: {step_name} -> {type(step).__name__}")
        X_model = step.transform(X_model)
        print(f"    shape after {step_name}: {X_model.shape}")

    final_estimator = model.steps[-1][1]
    print(f"Final estimator: {type(final_estimator).__name__}")

    return X_model, final_estimator


def rf_uncertainty(model, X):
    """
    Estimate RandomForest uncertainty as standard deviation across tree predictions.

    Returns NaN if the final estimator is not tree-ensemble-like.
    """
    X_model, base = transform_pipeline_features(model, X)

    if hasattr(base, "estimators_"):
        preds = np.vstack([tree.predict(X_model) for tree in base.estimators_])
        return preds.std(axis=0)

    return np.full(X.shape[0], np.nan)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 05D: SCORE TL-GENERATED MOLECULES WITH FACTOR XA RF MODEL")
    print("=" * 90)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Model file: {MODEL_FILE}")
    print(f"Reference CSV: {REFERENCE_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    df = pd.read_csv(INPUT_CSV)

    print(f"\nLoaded generated dataframe: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    smi_col = detect_smiles_column(df)
    print(f"Detected SMILES column: {smi_col}")

    model = joblib.load(MODEL_FILE)
    print(f"Loaded model type: {type(model)}")

    if hasattr(model, "steps"):
        print("Loaded sklearn Pipeline steps:")
        for step_name, step in model.steps:
            print(f"  {step_name}: {type(step).__name__}")

    reference_flat = load_reference_flat_smiles(REFERENCE_CSV)
    print(f"Reference flat SMILES count: {len(reference_flat)}")

    records = []
    fps = []

    n_invalid = 0

    for idx, row in df.iterrows():
        smi = row[smi_col]
        mol = mol_from_smiles(smi)

        if mol is None:
            n_invalid += 1

            rec = {
                "Input_Row": idx,
                "SMILES": smi,
                "Canonical_Flat_SMILES": None,
                "Valid_RDKit": False,
                "Novel_vs_FXA_reference_flat": None,
                "Basic_Druglike_Flag": None,
            }

            for col in df.columns:
                if col not in rec:
                    rec[col] = row[col]

            records.append(rec)
            continue

        can_flat = canonical_flat_smiles(mol)
        props = basic_properties(mol)

        rec = {
            "Input_Row": idx,
            "SMILES": smi,
            "Canonical_Flat_SMILES": can_flat,
            "Valid_RDKit": True,
            "Novel_vs_FXA_reference_flat": can_flat not in reference_flat,
            **props,
        }

        rec["Basic_Druglike_Flag"] = basic_druglike_flag(props)

        for col in df.columns:
            if col not in rec:
                rec[col] = row[col]

        records.append(rec)
        fps.append(morgan_fp_array(mol))

    out = pd.DataFrame(records)

    valid_mask = out["Valid_RDKit"] == True
    n_valid = int(valid_mask.sum())

    print(f"\nInput rows: {len(df)}")
    print(f"Valid RDKit molecules: {n_valid}")
    print(f"Invalid RDKit molecules: {n_invalid}")

    if n_valid == 0:
        raise ValueError("No valid RDKit molecules to score.")

    X = np.vstack(fps)

    print(f"\nRaw Morgan X shape: {X.shape}")
    print(f"Expected Morgan bits: {FP_BITS}")

    if X.shape[0] != n_valid:
        raise RuntimeError(
            f"Feature/prediction alignment error: X rows={X.shape[0]}, "
            f"valid molecules={n_valid}"
        )

    if X.shape[1] != FP_BITS:
        raise RuntimeError(
            f"Unexpected fingerprint size: X columns={X.shape[1]}, "
            f"expected={FP_BITS}"
        )

    # Core prediction. Since model is a pipeline, preprocessing is handled here.
    pred = model.predict(X)

    # Tree uncertainty. This manually applies pipeline preprocessing first.
    uncert = rf_uncertainty(model, X)

    if len(pred) != n_valid:
        raise RuntimeError(
            f"Prediction length mismatch: len(pred)={len(pred)}, n_valid={n_valid}"
        )

    if len(uncert) != n_valid:
        raise RuntimeError(
            f"Uncertainty length mismatch: len(uncert)={len(uncert)}, n_valid={n_valid}"
        )

    # Safe positional assignment:
    # fps and valid rows were created in the same single pass, so pred maps to
    # valid_mask rows in exact order.
    out.loc[valid_mask, "Predicted_FXA_pKi"] = pred
    out.loc[valid_mask, "RF_Tree_Uncertainty"] = uncert

    # Rank valid rows first by predicted pKi, then drug-like flag, then QED.
    # Invalid rows have NaN predictions and will sort to the bottom.
    out = out.sort_values(
        by=["Predicted_FXA_pKi", "Basic_Druglike_Flag", "QED"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out["FXA_Rank"] = np.arange(1, len(out) + 1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TOP50.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_CSV, index=False)
    out.head(50).to_csv(OUT_TOP50, index=False)

    valid_out = out[out["Valid_RDKit"] == True].copy()

    summary = {
        "input_csv": str(INPUT_CSV),
        "model_file": str(MODEL_FILE),
        "reference_csv": str(REFERENCE_CSV),
        "output_csv": str(OUT_CSV),
        "top50_csv": str(OUT_TOP50),
        "fingerprint": {
            "type": "MorganGenerator binary fingerprint",
            "radius": FP_RADIUS,
            "fpSize": FP_BITS,
            "countSimulation": False,
            "includeChirality": False,
            "useBondTypes": True,
        },
        "n_input_rows": int(len(df)),
        "n_valid_rdkit": int(n_valid),
        "n_invalid_rdkit": int(n_invalid),
        "n_unique_valid_canonical": int(valid_out["Canonical_Flat_SMILES"].nunique()),
        "n_novel_vs_reference": int(valid_out["Novel_vs_FXA_reference_flat"].sum()),
        "n_basic_druglike": int(valid_out["Basic_Druglike_Flag"].sum()),
        "predicted_pKi_mean": float(valid_out["Predicted_FXA_pKi"].mean()),
        "predicted_pKi_median": float(valid_out["Predicted_FXA_pKi"].median()),
        "predicted_pKi_max": float(valid_out["Predicted_FXA_pKi"].max()),
        "predicted_pKi_top10_mean": float(
            valid_out.head(10)["Predicted_FXA_pKi"].mean()
        ),
        "n_pred_pKi_ge_7": int((valid_out["Predicted_FXA_pKi"] >= 7.0).sum()),
        "n_pred_pKi_ge_8": int((valid_out["Predicted_FXA_pKi"] >= 8.0).sum()),
        "rf_uncertainty_median": float(valid_out["RF_Tree_Uncertainty"].median()),
        "rf_uncertainty_top10_mean": float(
            valid_out.head(10)["RF_Tree_Uncertainty"].mean()
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nTop 10 predicted FXA molecules:")
    cols = [
        "FXA_Rank",
        "Predicted_FXA_pKi",
        "RF_Tree_Uncertainty",
        "Basic_Druglike_Flag",
        "Novel_vs_FXA_reference_flat",
        "MW",
        "cLogP",
        "QED",
        "SMILES",
    ]

    available_cols = [col for col in cols if col in out.columns]
    print(out[available_cols].head(10).to_string(index=False))

    print("\nSaved outputs:")
    print(f"Scored CSV: {OUT_CSV}")
    print(f"Top 50 CSV: {OUT_TOP50}")
    print(f"Summary JSON: {OUT_JSON}")

    print("\nSTEP 05D COMPLETE")


if __name__ == "__main__":
    main()