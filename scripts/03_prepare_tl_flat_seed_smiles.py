"""
STEP 03: Prepare flat, transfer-learning-ready FXA seed SMILES for REINVENT4.

Purpose
-------
This script prepares the potent Factor Xa seed molecules for REINVENT4 transfer learning.

Why this step is needed
-----------------------
REINVENT's standard prior vocabulary may not safely handle all stereochemical SMILES
tokens from curated ChEMBL data. Therefore, before transfer learning, we convert the
potent FXA seed molecules to flat canonical SMILES:

    RDKit MolToSmiles(canonical=True, isomericSmiles=False)

This removes stereochemical tokens such as @, /, and \\ from the TL input file.

Input
-----
data/reference/fxa_reinvent_seed_pki_ge_8p0.csv

Main outputs
------------
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat.csv
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat_smiles_only.smi

data/reference/fxa_reinvent_tl_pki_ge_8p0_druglike_flat.csv
data/reference/fxa_reinvent_tl_pki_ge_8p0_druglike_flat_smiles_only.smi

results/metrics/fxa_reinvent_tl_flat_seed_summary.json

How to run
----------
From the project root:

conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\03_prepare_tl_flat_seed_smiles.py
python .\\scripts\\03_prepare_tl_flat_seed_smiles.py *> .\\scripts\\03.log
Get-Content .\\scripts\\03.log -Tail 120
"""

from pathlib import Path
import json
import pandas as pd
from rdkit import Chem


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "reference" / "fxa_reinvent_seed_pki_ge_8p0.csv"

OUT_ALL_CSV = PROJECT_ROOT / "data" / "reference" / "fxa_reinvent_tl_pki_ge_8p0_flat.csv"
OUT_ALL_SMI = PROJECT_ROOT / "data" / "reference" / "fxa_reinvent_tl_pki_ge_8p0_flat_smiles_only.smi"

OUT_DRUGLIKE_CSV = PROJECT_ROOT / "data" / "reference" / "fxa_reinvent_tl_pki_ge_8p0_druglike_flat.csv"
OUT_DRUGLIKE_SMI = PROJECT_ROOT / "data" / "reference" / "fxa_reinvent_tl_pki_ge_8p0_druglike_flat_smiles_only.smi"

OUT_JSON = PROJECT_ROOT / "results" / "metrics" / "fxa_reinvent_tl_flat_seed_summary.json"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def find_col(df, candidates, required=True):
    """
    Find the first matching column from a list of possible column names.

    This makes the script robust to small naming differences between
    datasets, for example:
        model_smiles vs Canonical_SMILES
        target_pKi vs pKi
        basic_druglike_flag vs Basic_Druglike
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def flat_canonical_smiles(smi):
    """
    Convert input SMILES to RDKit flat canonical SMILES.

    isomericSmiles=False removes stereochemical information from output SMILES.
    This is safer for REINVENT transfer learning with the standard prior.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def parse_bool_value(value):
    """
    Safely convert common boolean-like values to True/False.

    This avoids a common pandas mistake:
        bool("False") == True

    Handles:
        True, False
        "True", "False"
        "1", "0"
        "yes", "no"
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value_str = str(value).strip().lower()

    if value_str in {"true", "1", "yes", "y"}:
        return True

    if value_str in {"false", "0", "no", "n", ""}:
        return False

    return False


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    print("=" * 90)
    print("STEP 03: PREPARE FLAT SMILES FOR REINVENT TRANSFER LEARNING")
    print("=" * 90)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    # Load potent FXA seed molecules from Step 01.
    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded input dataframe: {df.shape}")
    print(f"Columns:\n{list(df.columns)}")

    # Detect the SMILES, pKi, and drug-likeness columns automatically.
    smiles_col = find_col(
        df,
        [
            "Canonical_SMILES",
            "canonical_smiles",
            "canonical_smiles_rdkit",
            "model_smiles",
            "SMILES",
            "smiles",
            "Ionized_SMILES",
            "Sanitized_SMILES",
        ],
        required=True,
    )

    pki_col = find_col(
        df,
        ["target_pKi", "pKi", "pchembl_value", "pChEMBL", "activity_pKi"],
        required=False,
    )

    druglike_col = find_col(
        df,
        ["basic_druglike_flag", "Basic_Druglike", "druglike", "Druglike"],
        required=False,
    )

    print(f"\nDetected SMILES column: {smiles_col}")
    print(f"Detected pKi column: {pki_col}")
    print(f"Detected druglike column: {druglike_col}")

    # Convert every seed molecule to flat canonical SMILES.
    rows = []

    for _, row in df.iterrows():
        original_smi = row[smiles_col]
        flat_smi = flat_canonical_smiles(original_smi)

        rec = row.to_dict()
        rec["Original_SMILES_for_TL_input"] = original_smi
        rec["TL_Flat_SMILES"] = flat_smi
        rec["Valid_TL_SMILES"] = flat_smi is not None

        rows.append(rec)

    out = pd.DataFrame(rows)

    n_invalid = int((~out["Valid_TL_SMILES"]).sum())
    print(f"\nInvalid SMILES after RDKit parsing: {n_invalid}")

    # Keep only valid flat SMILES.
    out = out[out["Valid_TL_SMILES"]].copy()

    # If pKi is available, sort so that duplicate flat SMILES keep the most potent record.
    if pki_col is not None:
        out = out.sort_values(pki_col, ascending=False)

    before_dedup = len(out)

    # Deduplicate flat SMILES. This is important because removing stereo can collapse
    # multiple stereoisomeric records into one flat parent molecule.
    out = out.drop_duplicates("TL_Flat_SMILES", keep="first").copy()

    after_dedup = len(out)
    n_removed_duplicates = before_dedup - after_dedup

    print(f"Valid flat SMILES before deduplication: {before_dedup}")
    print(f"Valid unique flat SMILES after deduplication: {after_dedup}")
    print(f"Duplicates removed after flattening: {n_removed_duplicates}")

    # Check for remaining stereo-like tokens after flattening.
    # Ideally this should be zero.
    out["Contains_stereo_token"] = (
        out["TL_Flat_SMILES"]
        .astype(str)
        .str.contains(r"@|/|\\", regex=True)
    )

    n_stereo_tokens_all = int(out["Contains_stereo_token"].sum())

    # Create a druglike subset if the druglike flag is available from Step 01.
    # This is usually the better TL input set because it avoids pushing REINVENT
    # toward extreme large/lipophilic seed molecules.
    if druglike_col is not None:
        out["_druglike_bool_for_filter"] = out[druglike_col].apply(parse_bool_value)
        druglike = out[out["_druglike_bool_for_filter"]].copy()
        druglike = druglike.drop(columns=["_druglike_bool_for_filter"])
        out = out.drop(columns=["_druglike_bool_for_filter"])
    else:
        druglike = out.copy()

    n_stereo_tokens_druglike = int(druglike["Contains_stereo_token"].sum())

    # Create output folders if needed.
    OUT_ALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save full flat seed set.
    out.to_csv(OUT_ALL_CSV, index=False)
    OUT_ALL_SMI.write_text(
        "\n".join(out["TL_Flat_SMILES"].tolist()) + "\n",
        encoding="utf-8",
    )

    # Save druglike flat seed set.
    druglike.to_csv(OUT_DRUGLIKE_CSV, index=False)
    OUT_DRUGLIKE_SMI.write_text(
        "\n".join(druglike["TL_Flat_SMILES"].tolist()) + "\n",
        encoding="utf-8",
    )

    # Summary for reproducibility and portfolio documentation.
    summary = {
        "input_csv": str(INPUT_CSV),
        "smiles_column": smiles_col,
        "pki_column": pki_col,
        "druglike_column": druglike_col,
        "n_input_rows": int(len(df)),
        "n_invalid_smiles_after_rdkit": n_invalid,
        "n_valid_flat_smiles_before_deduplication": int(before_dedup),
        "n_valid_flat_smiles_unique": int(len(out)),
        "n_duplicates_removed_after_flattening": int(n_removed_duplicates),
        "n_druglike_flat_smiles_unique": int(len(druglike)),
        "n_with_stereo_tokens_after_flattening_all": n_stereo_tokens_all,
        "n_with_stereo_tokens_after_flattening_druglike": n_stereo_tokens_druglike,
        "output_all_csv": str(OUT_ALL_CSV),
        "output_all_smi": str(OUT_ALL_SMI),
        "output_druglike_csv": str(OUT_DRUGLIKE_CSV),
        "output_druglike_smi": str(OUT_DRUGLIKE_SMI),
        "canonicalization": "RDKit MolToSmiles(canonical=True, isomericSmiles=False)",
        "recommended_tl_smiles_file": str(OUT_DRUGLIKE_SMI),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"All TL CSV: {OUT_ALL_CSV}")
    print(f"All TL SMI: {OUT_ALL_SMI}")
    print(f"Druglike TL CSV: {OUT_DRUGLIKE_CSV}")
    print(f"Druglike TL SMI: {OUT_DRUGLIKE_SMI}")
    print(f"Summary JSON: {OUT_JSON}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nFirst 10 druglike TL SMILES:")
    for smi in druglike["TL_Flat_SMILES"].head(10):
        print(smi)

    print("\nSTEP 03 COMPLETE")


if __name__ == "__main__":
    main()