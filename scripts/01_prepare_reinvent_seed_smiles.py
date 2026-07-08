#!/usr/bin/env python
"""
01_prepare_reinvent_seed_smiles.py

Project 2: FXA_REINVENT_Portfolio

Purpose
-------
Prepare potent Factor Xa seed/reference SMILES from the curated Project 1 dataset.

This script:
1. Loads the curated Project 1 Factor Xa modeling dataset.
2. Detects the SMILES, target pKi, and molecule ID columns.
3. Validates and canonicalizes SMILES using RDKit.
4. Filters potent molecules using a pKi threshold, default pKi >= 8.0.
5. Removes duplicate canonical SMILES, keeping the most potent record.
6. Computes simple RDKit molecular properties.
7. Saves both REINVENT-compatible seed formats:
   - SMILES<TAB>ID
   - SMILES only
8. Saves all-valid unique reference SMILES for novelty checks.
9. Saves a JSON summary.

Inputs
------
data/reference/fxa_04_modeling_dataset.csv

Outputs
-------
data/reference/fxa_reinvent_seed_pki_ge_8p0.smi
data/reference/fxa_reinvent_seed_pki_ge_8p0_smiles_only.smi
data/reference/fxa_reinvent_seed_pki_ge_8p0.csv
data/reference/fxa_reference_all_valid_unique_smiles.smi
data/reference/fxa_reference_all_valid_unique_smiles_smiles_only.smi
data/reference/fxa_reference_all_valid_unique_smiles.csv
results/tables/fxa_reinvent_invalid_smiles_rows.csv
results/metrics/fxa_reinvent_seed_summary.json

How to run
----------
python -m py_compile .\\scripts\\01_prepare_reinvent_seed_smiles.py
python .\\scripts\\01_prepare_reinvent_seed_smiles.py *> .\\scripts\\01.log
Get-Content .\\scripts\\01.log -Tail 120

Optional lower threshold
------------------------
python .\\scripts\\01_prepare_reinvent_seed_smiles.py --pki_threshold 7.5 *> .\\scripts\\01_pki75.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = ROOT / "data" / "reference" / "fxa_04_modeling_dataset.csv"

REFERENCE_DIR = ROOT / "data" / "reference"
METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"

for directory in [REFERENCE_DIR, METRICS_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def safe_threshold_label(value: float) -> str:
    """
    Convert 8.0 -> 8p0 for filename safety.
    """
    return str(value).replace(".", "p")


def dedupe_column_list(cols: List[str]) -> List[str]:
    """
    Preserve order while removing duplicate column names.
    """
    seen = set()
    out = []

    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)

    return out


def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    """
    Detect the first matching column from a candidate list.
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            "Could not detect required column. Tried candidates:\n"
            + "\n".join(candidates)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns.astype(str).tolist())
        )

    return None


def canonicalize_smiles(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return canonical SMILES and an error message.

    If valid:
        return canonical_smiles, None

    If invalid:
        return None, error_message
    """
    if pd.isna(smiles):
        return None, "missing_smiles"

    smiles = str(smiles).strip()

    if smiles == "":
        return None, "empty_smiles"

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, "rdkit_parse_failed"

    try:
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        return None, f"canonicalization_failed: {exc}"

    if canonical is None or canonical.strip() == "":
        return None, "empty_canonical_smiles"

    return canonical, None


def compute_rdkit_properties(smiles: str) -> Dict[str, float | int | None]:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "mw": None,
            "clogp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "heavy_atoms": None,
            "qed": None,
        }

    return {
        "mw": float(Descriptors.MolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "qed": float(QED.qed(mol)),
    }


def basic_druglike_flag(row: pd.Series) -> bool:
    """
    Simple soft drug-like flag for seed review.

    This flag is not used to remove seeds by default.
    It is only reported as metadata.
    """
    try:
        return bool(
            200.0 <= float(row["mw"]) <= 600.0
            and -2.0 <= float(row["clogp"]) <= 6.0
            and int(row["hbd"]) <= 5
            and int(row["hba"]) <= 10
            and int(row["rotatable_bonds"]) <= 12
            and int(row["heavy_atoms"]) <= 60
        )
    except Exception:
        return False


def write_smi(
    df: pd.DataFrame,
    smiles_col: str,
    out_path: Path,
    include_id: bool = True,
):
    """
    Write a .smi file.

    If include_id=True:
        SMILES<TAB>ID

    If include_id=False:
        SMILES only
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            smiles = str(row[smiles_col]).strip()

            if include_id:
                mol_id = str(row["seed_id"]).strip()
                f.write(f"{smiles}\t{mol_id}\n")
            else:
                f.write(f"{smiles}\n")


def summarize_numeric(series: pd.Series) -> Dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }

    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


# =============================================================================
# Main workflow
# =============================================================================

def prepare_seed_smiles(
    input_csv: Path,
    pki_threshold: float,
    smiles_col_arg: Optional[str] = None,
    target_col_arg: Optional[str] = None,
    id_col_arg: Optional[str] = None,
    max_seeds: Optional[int] = None,
    default_smi_smiles_only: bool = False,
):
    print("=" * 90)
    print("STEP 01: PREPARE REINVENT SEED SMILES")
    print("=" * 90)

    print(f"Project root: {ROOT}")
    print(f"Input CSV: {input_csv}")
    print(f"pKi threshold: >= {pki_threshold}")

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n\n"
            "Copy the Project 1 curated dataset first, for example:\n"
            "Copy-Item ..\\FXA_GNN_Portfolio\\data\\processed\\fxa_04_modeling_dataset.csv .\\data\\reference\\"
        )

    df = pd.read_csv(input_csv)

    print(f"Loaded input dataframe: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    smiles_col = smiles_col_arg or detect_column(
        df,
        candidates=[
            "model_smiles",
            "standardized_smiles",
            "canonical_smiles",
            "canonical_smiles_rdkit",
            "smiles",
            "SMILES",
        ],
        required=True,
    )

    target_col = target_col_arg or detect_column(
        df,
        candidates=[
            "target_pKi",
            "pKi",
            "standard_value_pKi",
            "activity_pKi",
        ],
        required=True,
    )

    id_col = id_col_arg or detect_column(
        df,
        candidates=[
            "molecule_chembl_id",
            "Molecule ChEMBL ID",
            "chembl_id",
            "compound_id",
            "molecule_id",
            "id",
        ],
        required=False,
    )

    print(f"Detected SMILES column: {smiles_col}")
    print(f"Detected target column: {target_col}")
    print(f"Detected ID column: {id_col}")

    work = df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    before_target = len(work)
    work = work.dropna(subset=[target_col]).copy()
    after_target = len(work)

    print(f"Rows before target cleanup: {before_target}")
    print(f"Rows after target cleanup: {after_target}")

    canonical_smiles = []
    smiles_errors = []

    for smi in work[smiles_col].tolist():
        can, err = canonicalize_smiles(smi)
        canonical_smiles.append(can)
        smiles_errors.append(err)

    work["canonical_smiles"] = canonical_smiles
    work["smiles_error"] = smiles_errors
    work["is_valid_smiles"] = work["canonical_smiles"].notna()

    invalid_df = work[~work["is_valid_smiles"]].copy()
    valid_df = work[work["is_valid_smiles"]].copy()

    print(f"Valid SMILES rows: {len(valid_df)}")
    print(f"Invalid SMILES rows: {len(invalid_df)}")

    if len(valid_df) == 0:
        raise ValueError("No valid SMILES found after RDKit parsing.")

    if id_col is None:
        valid_df["source_molecule_id"] = [
            f"FXA_REF_{i:06d}" for i in range(len(valid_df))
        ]
    else:
        valid_df["source_molecule_id"] = valid_df[id_col].astype(str)

    # Keep the most potent record per canonical SMILES.
    valid_df = valid_df.sort_values(
        by=[target_col, "canonical_smiles"],
        ascending=[False, True],
    ).copy()

    all_valid_unique_df = (
        valid_df.drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
        .copy()
    )

    print(f"All valid unique canonical SMILES: {len(all_valid_unique_df)}")

    potent_df = all_valid_unique_df[
        all_valid_unique_df[target_col] >= pki_threshold
    ].copy()

    potent_df = potent_df.sort_values(
        by=[target_col, "canonical_smiles"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if max_seeds is not None:
        potent_df = potent_df.head(max_seeds).copy()

    print(f"Potent unique seeds passing pKi >= {pki_threshold}: {len(potent_df)}")

    if len(potent_df) == 0:
        raise ValueError(
            f"No potent seeds found at pKi >= {pki_threshold}. "
            "Try a lower threshold such as 7.5."
        )

    # Add seed/reference IDs.
    threshold_label = safe_threshold_label(pki_threshold)

    potent_df["seed_rank"] = np.arange(1, len(potent_df) + 1)
    potent_df["seed_id"] = potent_df.apply(
        lambda row: (
            f"FXA_PKI_GE_{threshold_label}_{int(row['seed_rank']):05d}"
        ),
        axis=1,
    )

    all_valid_unique_df["reference_rank"] = np.arange(
        1,
        len(all_valid_unique_df) + 1,
    )

    all_valid_unique_df["seed_id"] = all_valid_unique_df.apply(
        lambda row: f"FXA_REF_{int(row['reference_rank']):06d}",
        axis=1,
    )

    # Compute simple RDKit properties for potent seeds.
    print("Computing RDKit properties for potent seeds...")

    prop_rows = [
        compute_rdkit_properties(smi)
        for smi in potent_df["canonical_smiles"].tolist()
    ]

    prop_df = pd.DataFrame(prop_rows)
    potent_df = pd.concat([potent_df.reset_index(drop=True), prop_df], axis=1)

    potent_df["basic_druglike_flag"] = potent_df.apply(
        basic_druglike_flag,
        axis=1,
    )

    # Reorder columns.
    keep_cols = [
        "seed_id",
        "seed_rank",
        "source_molecule_id",
        "canonical_smiles",
        smiles_col,
        target_col,
        "mw",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "heavy_atoms",
        "qed",
        "basic_druglike_flag",
    ]

    optional_cols = [
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "document_chembl_id",
    ]

    for col in optional_cols:
        if col in potent_df.columns:
            keep_cols.append(col)

    keep_cols = dedupe_column_list(
        [col for col in keep_cols if col in potent_df.columns]
    )

    potent_out_df = potent_df[keep_cols].copy()

    ref_keep_cols = [
        "seed_id",
        "reference_rank",
        "source_molecule_id",
        "canonical_smiles",
        smiles_col,
        target_col,
    ]

    ref_keep_cols = dedupe_column_list(
        [col for col in ref_keep_cols if col in all_valid_unique_df.columns]
    )

    ref_out_df = all_valid_unique_df[ref_keep_cols].copy()

    # Output filenames.
    seed_smi = REFERENCE_DIR / f"fxa_reinvent_seed_pki_ge_{threshold_label}.smi"
    seed_smi_smiles_only = (
        REFERENCE_DIR
        / f"fxa_reinvent_seed_pki_ge_{threshold_label}_smiles_only.smi"
    )
    seed_csv = REFERENCE_DIR / f"fxa_reinvent_seed_pki_ge_{threshold_label}.csv"

    ref_smi = REFERENCE_DIR / "fxa_reference_all_valid_unique_smiles.smi"
    ref_smi_smiles_only = (
        REFERENCE_DIR / "fxa_reference_all_valid_unique_smiles_smiles_only.smi"
    )
    ref_csv = REFERENCE_DIR / "fxa_reference_all_valid_unique_smiles.csv"

    invalid_csv = TABLES_DIR / "fxa_reinvent_invalid_smiles_rows.csv"
    summary_json = METRICS_DIR / "fxa_reinvent_seed_summary.json"

    # Main seed file can be SMILES<TAB>ID or SMILES only, controlled by CLI.
    # A dedicated SMILES-only file is always written for REINVENT modes that require it.
    write_smi(
        potent_out_df,
        smiles_col="canonical_smiles",
        out_path=seed_smi,
        include_id=not default_smi_smiles_only,
    )

    write_smi(
        potent_out_df,
        smiles_col="canonical_smiles",
        out_path=seed_smi_smiles_only,
        include_id=False,
    )

    # Reference files for novelty checks.
    write_smi(
        ref_out_df,
        smiles_col="canonical_smiles",
        out_path=ref_smi,
        include_id=True,
    )

    write_smi(
        ref_out_df,
        smiles_col="canonical_smiles",
        out_path=ref_smi_smiles_only,
        include_id=False,
    )

    potent_out_df.to_csv(seed_csv, index=False)
    ref_out_df.to_csv(ref_csv, index=False)

    if len(invalid_df) > 0:
        invalid_df.to_csv(invalid_csv, index=False)
        invalid_csv_value = str(invalid_csv)
    else:
        invalid_csv_value = None

    summary = {
        "script": "01_prepare_reinvent_seed_smiles.py",
        "input_csv": str(input_csv),
        "settings": {
            "pki_threshold": pki_threshold,
            "max_seeds": max_seeds,
            "default_smi_smiles_only": default_smi_smiles_only,
            "canonicalization": {
                "canonical": True,
                "isomericSmiles": True,
            },
        },
        "detected_columns": {
            "smiles_col": smiles_col,
            "target_col": target_col,
            "id_col": id_col,
        },
        "counts": {
            "input_rows": int(len(df)),
            "rows_with_numeric_target": int(len(work)),
            "valid_smiles_rows": int(len(valid_df)),
            "invalid_smiles_rows": int(len(invalid_df)),
            "all_valid_unique_canonical_smiles": int(len(all_valid_unique_df)),
            "potent_unique_seeds": int(len(potent_out_df)),
            "basic_druglike_seed_count": int(
                potent_out_df["basic_druglike_flag"].sum()
            ),
        },
        "target_summary_all_unique": summarize_numeric(
            all_valid_unique_df[target_col]
        ),
        "target_summary_potent_seeds": summarize_numeric(
            potent_out_df[target_col]
        ),
        "property_summary_potent_seeds": {
            "mw": summarize_numeric(potent_out_df["mw"]),
            "clogp": summarize_numeric(potent_out_df["clogp"]),
            "tpsa": summarize_numeric(potent_out_df["tpsa"]),
            "qed": summarize_numeric(potent_out_df["qed"]),
        },
        "outputs": {
            "seed_smi_default": str(seed_smi),
            "seed_smi_smiles_only": str(seed_smi_smiles_only),
            "seed_csv": str(seed_csv),
            "reference_smi_with_ids": str(ref_smi),
            "reference_smi_smiles_only": str(ref_smi_smiles_only),
            "reference_csv": str(ref_csv),
            "invalid_csv": invalid_csv_value,
            "summary_json": str(summary_json),
        },
        "reinvent_compatibility_note": (
            "Different REINVENT run modes may expect different input formats. "
            "Use the *_smiles_only.smi file for modes requiring one SMILES per line. "
            "Use the default .smi file when SMILES<TAB>ID is accepted or useful."
        ),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\nSaved outputs:")
    print(f"Seed SMILES default: {seed_smi}")
    print(f"Seed SMILES only: {seed_smi_smiles_only}")
    print(f"Seed CSV: {seed_csv}")
    print(f"Reference SMILES with IDs: {ref_smi}")
    print(f"Reference SMILES only: {ref_smi_smiles_only}")
    print(f"Reference CSV: {ref_csv}")

    if invalid_csv_value is not None:
        print(f"Invalid SMILES rows: {invalid_csv}")

    print(f"Summary JSON: {summary_json}")

    print("\nPotent seed summary:")
    print(f"Number of potent seeds: {len(potent_out_df)}")
    print(f"pKi min: {potent_out_df[target_col].min():.3f}")
    print(f"pKi median: {potent_out_df[target_col].median():.3f}")
    print(f"pKi max: {potent_out_df[target_col].max():.3f}")
    print(
        "Basic drug-like seeds: "
        f"{int(potent_out_df['basic_druglike_flag'].sum())}"
    )

    print("\nTop 10 potent seeds:")
    preview_cols = [
        "seed_id",
        "source_molecule_id",
        "canonical_smiles",
        target_col,
        "mw",
        "clogp",
        "qed",
        "basic_druglike_flag",
    ]

    preview_cols = [
        col for col in preview_cols if col in potent_out_df.columns
    ]

    print(potent_out_df[preview_cols].head(10).to_string(index=False))

    print("\n" + "=" * 90)
    print("STEP 01 COMPLETE")
    print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare potent Factor Xa seed SMILES for REINVENT."
    )

    parser.add_argument(
        "--input_csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input curated Project 1 dataset CSV.",
    )

    parser.add_argument(
        "--pki_threshold",
        type=float,
        default=8.0,
        help="Minimum pKi threshold for potent seed molecules.",
    )

    parser.add_argument(
        "--smiles_col",
        type=str,
        default=None,
        help="Optional explicit SMILES column name.",
    )

    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Optional explicit pKi target column name.",
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default=None,
        help="Optional explicit molecule ID column name.",
    )

    parser.add_argument(
        "--max_seeds",
        type=int,
        default=None,
        help="Optional maximum number of potent seeds to save.",
    )

    parser.add_argument(
        "--default_smi_smiles_only",
        action="store_true",
        help=(
            "Write the default seed .smi as SMILES only instead of "
            "SMILES<TAB>ID. The dedicated *_smiles_only.smi file is "
            "always written regardless of this flag."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    prepare_seed_smiles(
        input_csv=args.input_csv,
        pki_threshold=args.pki_threshold,
        smiles_col_arg=args.smiles_col,
        target_col_arg=args.target_col,
        id_col_arg=args.id_col,
        max_seeds=args.max_seeds,
        default_smi_smiles_only=args.default_smi_smiles_only,
    )


if __name__ == "__main__":
    main()