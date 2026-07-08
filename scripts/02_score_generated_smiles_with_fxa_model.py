#!/usr/bin/env python
"""
02_score_reinvent_with_fxa_model.py

Score REINVENT-generated SMILES with the Project 1 Factor Xa model.

Loads the canonical Step 05b model (a scikit-learn Pipeline:
SimpleImputer -> VarianceThreshold -> RandomForest), featurizes generated
SMILES with the SAME Morgan settings used in Project 1, predicts pKi, and
computes RandomForest ensemble-disagreement uncertainty.

Notes
-----
- The Morgan settings (radius, fpSize, binary on-bits) MUST match Project 1
  Step 04 exactly, or predictions will be silently meaningless.
- Uncertainty (RF tree std) is computed on the pipeline-transformed features,
  not the raw fingerprint, so it works with the VarianceThreshold step.
- Novelty is compared on stereo-flattened canonical SMILES on both sides so it
  stays consistent with the flat reinvent.prior (which emits no stereo tokens).


How to run:
----------

python -m py_compile .\scripts\02_score_generated_smiles_with_fxa_model.py
python .\scripts\02_score_generated_smiles_with_fxa_model.py *> .\scripts\02.log
Get-Content .\scripts\02.log -Tail 160
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.pipeline import Pipeline


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "generated" / "reinvent_sanity_100.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "fxa_05b_best_scaffold_feature_model.joblib"
REFERENCE_CSV = PROJECT_ROOT / "data" / "reference" / "fxa_reference_all_valid_unique_smiles.csv"

OUT_CSV = PROJECT_ROOT / "data" / "generated" / "reinvent_sanity_100_scored_fxa.csv"
OUT_TOP_CSV = PROJECT_ROOT / "results" / "tables" / "reinvent_sanity_100_top20_fxa.csv"
OUT_JSON = PROJECT_ROOT / "results" / "metrics" / "reinvent_sanity_100_fxa_score_summary.json"

# These MUST match Project 1 Step 04 feature generation.
FP_SIZE = 2048
RADIUS = 2


# =============================================================================
# Helpers
# =============================================================================

def canonicalize_smiles(smi):
    """Return (mol, isomeric_canonical_smiles) or (None, None)."""
    if not isinstance(smi, str) or not smi.strip():
        return None, None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return mol, can


def flat_canonical(smi):
    """Stereo-flattened canonical SMILES (no @, /, \\) for novelty comparison."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def mol_props(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "QED": QED.qed(mol),
    }


def load_estimator(path):
    """Load a predict-capable estimator, whether saved bare or inside a dict."""
    obj = joblib.load(path)

    if hasattr(obj, "predict"):
        return obj, {"loaded_object_type": type(obj).__name__}

    if isinstance(obj, dict):
        info = {"loaded_object_type": "dict", "keys": list(obj.keys())}
        for key in ["model", "best_model", "estimator", "pipeline", "trained_model", "regressor"]:
            if key in obj and hasattr(obj[key], "predict"):
                info["selected_key"] = key
                return obj[key], info
        for key, value in obj.items():
            if hasattr(value, "predict"):
                info["selected_key"] = key
                return value, info

    raise TypeError(f"Could not find predict-capable estimator inside {path}")


def get_expected_features(model):
    """Number of input features the (pipeline) model expects."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    return None


def morgan_features(mols):
    """Binary Morgan fingerprints, matching Project 1 Step 04."""
    gen = GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)
    arr = np.zeros((len(mols), FP_SIZE), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = gen.GetFingerprint(mol)
        arr[i, list(fp.GetOnBits())] = 1.0
    return arr


def rf_uncertainty(model, X):
    """
    RandomForest ensemble-disagreement (std across trees).

    If the model is a Pipeline, X is transformed through every step EXCEPT the
    final estimator first, so the per-tree predictions see the same feature
    space the forest was trained on (e.g. after VarianceThreshold). Feeding the
    raw fingerprint here is a feature-count mismatch and will crash.
    """
    base = model
    X_model = X

    if hasattr(model, "steps"):
        base = model.steps[-1][1]
        preprocessor = Pipeline(model.steps[:-1])
        X_model = preprocessor.transform(X)

    if hasattr(base, "estimators_"):
        preds = np.vstack([tree.predict(X_model) for tree in base.estimators_])
        return preds.std(axis=0)

    return np.full(X.shape[0], np.nan)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("STEP 04: SCORE REINVENT-GENERATED SMILES WITH FACTOR XA MODEL")
    print("=" * 90)
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Model file: {MODEL_FILE}")

    for path in [INPUT_CSV, MODEL_FILE, REFERENCE_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(INPUT_CSV)

    if "SMILES" not in df.columns:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    model, model_info = load_estimator(MODEL_FILE)
    expected = get_expected_features(model)

    print("Loaded model info:")
    print(model_info)
    print(f"Expected feature count: {expected}")

    if expected is not None and expected != FP_SIZE:
        raise ValueError(
            f"Model expects {expected} features, but this script creates {FP_SIZE} Morgan features."
        )

    records = []
    valid_mols = []
    valid_indices = []

    for idx, row in df.iterrows():
        smi = row["SMILES"]
        mol, can = canonicalize_smiles(smi)

        rec = row.to_dict()
        rec["Canonical_SMILES"] = can
        rec["Flat_Canonical_SMILES"] = flat_canonical(smi)
        rec["Valid_RDKit"] = mol is not None

        if mol is not None:
            rec.update(mol_props(mol))
            valid_mols.append(mol)
            valid_indices.append(idx)
        else:
            rec.update({
                "MW": np.nan, "LogP": np.nan, "TPSA": np.nan, "HBD": np.nan,
                "HBA": np.nan, "RotBonds": np.nan, "HeavyAtoms": np.nan, "QED": np.nan,
            })

        records.append(rec)

    out = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Novelty vs Project 1 reference (compared on stereo-flattened canonical)
    # -------------------------------------------------------------------------
    ref = pd.read_csv(REFERENCE_CSV)
    ref_smiles_col = None
    for col in ["canonical_smiles", "Canonical_SMILES", "canonical_smiles_rdkit",
                "model_smiles", "SMILES", "smiles"]:
        if col in ref.columns:
            ref_smiles_col = col
            break

    if ref_smiles_col is None:
        raise ValueError(f"Could not find reference SMILES column. Columns: {list(ref.columns)}")

    ref_set = set()
    for smi in ref[ref_smiles_col].dropna().astype(str):
        flat = flat_canonical(smi)
        if flat:
            ref_set.add(flat)

    out["Novel_vs_FXA_reference"] = ~out["Flat_Canonical_SMILES"].isin(ref_set)

    # -------------------------------------------------------------------------
    # Score with the Factor Xa model
    # -------------------------------------------------------------------------
    out["Predicted_pKi_FXA"] = np.nan
    out["RF_tree_std_uncertainty"] = np.nan

    if valid_mols:
        X = morgan_features(valid_mols)
        preds = model.predict(X)
        unc = rf_uncertainty(model, X)
        out.loc[valid_indices, "Predicted_pKi_FXA"] = preds
        out.loc[valid_indices, "RF_tree_std_uncertainty"] = unc

    out["Basic_Druglike"] = (
        (out["MW"].between(200, 650)) &
        (out["LogP"].between(-1, 6)) &
        (out["HBD"] <= 5) &
        (out["HBA"] <= 12) &
        (out["TPSA"] <= 160) &
        (out["RotBonds"] <= 12)
    )

    out = out.sort_values(
        ["Predicted_pKi_FXA", "RF_tree_std_uncertainty"],
        ascending=[False, True],
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    for p in [OUT_CSV, OUT_TOP_CSV, OUT_JSON]:
        p.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_CSV, index=False)
    out.head(20).to_csv(OUT_TOP_CSV, index=False)

    pred_series = out["Predicted_pKi_FXA"].dropna()

    summary = {
        "input_csv": str(INPUT_CSV),
        "model_file": str(MODEL_FILE),
        "output_csv": str(OUT_CSV),
        "n_input_rows": int(len(df)),
        "n_valid_rdkit": int(out["Valid_RDKit"].sum()),
        "n_unique_valid_canonical": int(out.loc[out["Valid_RDKit"], "Canonical_SMILES"].nunique()),
        "n_novel_vs_reference": int(out["Novel_vs_FXA_reference"].sum()),
        "n_basic_druglike": int(out["Basic_Druglike"].sum()),
        "predicted_pKi_mean": float(pred_series.mean()) if len(pred_series) else None,
        "predicted_pKi_median": float(pred_series.median()) if len(pred_series) else None,
        "predicted_pKi_max": float(pred_series.max()) if len(pred_series) else None,
        "n_pred_pKi_ge_7": int((out["Predicted_pKi_FXA"] >= 7.0).sum()),
        "n_pred_pKi_ge_8": int((out["Predicted_pKi_FXA"] >= 8.0).sum()),
        "model_info": model_info,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"Scored CSV: {OUT_CSV}")
    print(f"Top 20 CSV: {OUT_TOP_CSV}")
    print(f"Summary JSON: {OUT_JSON}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nTop 10 predicted FXA molecules:")
    cols = [
        "SMILES", "Predicted_pKi_FXA", "RF_tree_std_uncertainty",
        "Novel_vs_FXA_reference", "Basic_Druglike", "MW", "LogP", "QED",
    ]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()