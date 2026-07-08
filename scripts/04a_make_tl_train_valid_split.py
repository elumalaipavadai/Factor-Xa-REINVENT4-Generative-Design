"""
STEP 04A: Create train/validation split for REINVENT4 transfer learning.

Purpose
-------
This script splits curated FXA transfer-learning SMILES into:

1. Training SMILES
2. Validation SMILES

Why this step is needed
-----------------------
REINVENT transfer learning should be monitored with a validation set so that
we can check whether the model is learning FXA-like chemistry without simply
memorizing the training molecules.

Important design choice
-----------------------
By default, this script uses the ALL flat potent FXA SMILES:

    fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_smiles_only.smi

This keeps the broader FXA-active chemistry distribution for transfer learning.

If you want a cleaner but narrower seed set, change:

    INPUT_MODE = "all_flat"

to:

    INPUT_MODE = "all_flat"

Inputs
------
All-flat input:
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_smiles_only.smi

Druglike-flat input:
data/reference/fxa_reinvent_tl_pki_ge_8p0_druglike_flat_smiles_only.smi

Outputs
-------
data/reference/tl_split/fxa_tl_train_pki8_all_flat_elements.smi
data/reference/tl_split/fxa_tl_valid_pki8_all_flat_elements.smi

or, if INPUT_MODE = "all_flat":

data/reference/tl_split/fxa_tl_train_pki8_druglike_flat.smi
data/reference/tl_split/fxa_tl_valid_pki8_druglike_flat.smi

results/metrics/fxa_tl_train_valid_split_summary.json

How to run
----------
conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\04a_make_tl_train_valid_split.py
python .\\scripts\\04a_make_tl_train_valid_split.py *> .\\scripts\\04a.log
Get-Content .\\scripts\\04a.log -Tail 120
"""

from pathlib import Path
import random
import json


# ---------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------

# Recommended default:
#   "all_flat"       = broader FXA-active distribution for TL
#   "druglike_flat"  = narrower, cleaner druglike subset
#INPUT_MODE = "all_flat"
INPUT_MODE = "all_flat"

TRAIN_FRAC = 0.80
SEED = 42

# Guardrail: fail early if upstream files are unexpectedly tiny.
MIN_UNIQUE_SMILES = 10


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "all_flat": (
        PROJECT_ROOT
        / "data"
        / "reference"
        / "fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_smiles_only.smi"
    ),
    "druglike_flat": (
        PROJECT_ROOT
        / "data"
        / "reference"
        / "fxa_reinvent_tl_pki_ge_8p0_druglike_flat_smiles_only.smi"
    ),
}

OUTPUT_FILES = {
    "all_flat": {
        "train": PROJECT_ROOT / "data" / "reference" / "tl_split" / "fxa_tl_train_pki8_all_flat_elements.smi",
        "valid": PROJECT_ROOT / "data" / "reference" / "tl_split" / "fxa_tl_valid_pki8_all_flat_elements.smi",
    },
    "druglike_flat": {
        "train": PROJECT_ROOT / "data" / "reference" / "tl_split" / "fxa_tl_train_pki8_druglike_flat.smi",
        "valid": PROJECT_ROOT / "data" / "reference" / "tl_split" / "fxa_tl_valid_pki8_druglike_flat.smi",
    },
}

OUT_JSON = PROJECT_ROOT / "results" / "metrics" / "fxa_tl_train_valid_split_summary.json"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def read_smiles_file(path):
    """
    Read a one-SMILES-per-line .smi file and remove empty lines.
    """
    smiles = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return smiles


def write_smiles_file(path, smiles):
    """
    Write one SMILES per line.

    The explicit empty-list check prevents accidentally writing files that only
    contain a trailing newline.
    """
    if len(smiles) == 0:
        raise ValueError(f"Refusing to write empty SMILES file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(smiles) + "\n", encoding="utf-8")


def main():
    print("=" * 90)
    print("STEP 04A: CREATE TRAIN/VALIDATION SPLIT FOR REINVENT TL")
    print("=" * 90)

    if INPUT_MODE not in INPUT_FILES:
        raise ValueError(
            f"Invalid INPUT_MODE='{INPUT_MODE}'. "
            f"Allowed values: {list(INPUT_FILES.keys())}"
        )

    if not (0.0 < TRAIN_FRAC < 1.0):
        raise ValueError(f"TRAIN_FRAC must be between 0 and 1. Got {TRAIN_FRAC}")

    input_smi = INPUT_FILES[INPUT_MODE]
    out_train = OUTPUT_FILES[INPUT_MODE]["train"]
    out_valid = OUTPUT_FILES[INPUT_MODE]["valid"]

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input mode: {INPUT_MODE}")
    print(f"Input SMI: {input_smi}")
    print(f"Train fraction: {TRAIN_FRAC}")
    print(f"Random seed: {SEED}")
    print(f"Minimum unique SMILES guard: {MIN_UNIQUE_SMILES}")

    if not input_smi.exists():
        raise FileNotFoundError(f"Input SMILES file not found: {input_smi}")

    smiles_raw = read_smiles_file(input_smi)
    n_input_lines = len(smiles_raw)

    # Deduplicate before splitting to avoid identical molecules appearing in both
    # train and validation sets.
    smiles_unique = sorted(set(smiles_raw))
    n_unique = len(smiles_unique)
    n_duplicates_removed = n_input_lines - n_unique

    print(f"\nInput lines: {n_input_lines}")
    print(f"Unique SMILES after deduplication: {n_unique}")
    print(f"Duplicates removed: {n_duplicates_removed}")

    # Critical guardrail: do not create degenerate train/valid splits.
    if n_unique < MIN_UNIQUE_SMILES:
        raise ValueError(
            f"Only {n_unique} unique SMILES found. "
            "Too few for a reliable train/validation split. "
            "Check the Step 03 TL seed file."
        )

    rng = random.Random(SEED)
    rng.shuffle(smiles_unique)

    n_train = int(TRAIN_FRAC * n_unique)

    train = smiles_unique[:n_train]
    valid = smiles_unique[n_train:]

    if len(train) == 0 or len(valid) == 0:
        raise ValueError(
            f"Degenerate split created: n_train={len(train)}, n_valid={len(valid)}. "
            "Adjust TRAIN_FRAC or check the input file."
        )

    # Confirm no identical SMILES leakage between train and validation.
    overlap = set(train).intersection(set(valid))

    if overlap:
        raise RuntimeError(
            f"Train/validation leakage detected: {len(overlap)} overlapping SMILES."
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    write_smiles_file(out_train, train)
    write_smiles_file(out_valid, valid)

    # Re-read outputs to confirm they are non-empty and line counts match.
    train_check = read_smiles_file(out_train)
    valid_check = read_smiles_file(out_valid)

    if len(train_check) != len(train) or len(valid_check) != len(valid):
        raise RuntimeError(
            "Output file line-count check failed. "
            f"Expected train/valid = {len(train)}/{len(valid)}, "
            f"read back = {len(train_check)}/{len(valid_check)}"
        )

    summary = {
        "input_mode": INPUT_MODE,
        "input_smi": str(input_smi),
        "n_input_lines": int(n_input_lines),
        "n_unique_smiles": int(n_unique),
        "n_duplicates_removed": int(n_duplicates_removed),
        "train_fraction": TRAIN_FRAC,
        "random_seed": SEED,
        "minimum_unique_smiles_guard": MIN_UNIQUE_SMILES,
        "n_train": int(len(train)),
        "n_validation": int(len(valid)),
        "n_train_valid_overlap": int(len(overlap)),
        "train_smi": str(out_train),
        "validation_smi": str(out_valid),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nFirst 5 train SMILES:")
    for smi in train[:5]:
        print(smi)

    print("\nFirst 5 validation SMILES:")
    for smi in valid[:5]:
        print(smi)

    print("\nSTEP 04A COMPLETE")


if __name__ == "__main__":
    main()
