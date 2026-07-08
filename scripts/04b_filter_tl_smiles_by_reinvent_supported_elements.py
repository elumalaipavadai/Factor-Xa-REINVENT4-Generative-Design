"""
STEP 04B: Filter TL SMILES by elements supported by the REINVENT prior.

Purpose
-------
The first REINVENT TL launch failed because one TL molecule contained boron:

    ValueError: Tokens {'B'} ... are not supported by the model

The official reinvent.prior vocabulary supports common organic atoms seen in
its token set, but not standalone boron or unusual elements. Instead of using
a fragile hard-coded SMILES token allow-list, this script filters by RDKit atom
symbols.

Why element filtering is better here
------------------------------------
A token allow-list can falsely remove valid molecules because of SMILES notation,
for example ring labels, bracket notation, or protonation states.

An element filter removes the true problem class:
    unsupported elements such as B, Si, metals, etc.

Input
-----
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat_smiles_only.smi

Outputs
-------
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_smiles_only.smi
data/reference/fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_removed.tsv
results/metrics/fxa_reinvent_tl_element_filter_summary.json

How to run
----------
conda activate fxa_reinvent4_py311
cd .

python -m py_compile .\\scripts\\04b_filter_tl_smiles_by_reinvent_supported_elements.py
python .\\scripts\\04b_filter_tl_smiles_by_reinvent_supported_elements.py *> .\\scripts\\04b.log
Get-Content .\\scripts\\04b.log -Tail 120
"""

from pathlib import Path
import json
from collections import Counter
from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_SMI = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "fxa_reinvent_tl_pki_ge_8p0_flat_smiles_only.smi"
)

OUT_KEEP = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_smiles_only.smi"
)

OUT_REMOVE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "fxa_reinvent_tl_pki_ge_8p0_flat_reinvent_elements_removed.tsv"
)

OUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "fxa_reinvent_tl_element_filter_summary.json"
)

# Strict element set based on the official reinvent.prior token error you observed.
# Do not include B, Si, P, I, metals, etc. unless a later model/prior confirms support.
SUPPORTED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br"}

# Guardrail: if too many molecules are removed, stop and inspect.
MAX_DROP_FRACTION = 0.05


def read_smiles(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unsupported_elements_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return ["INVALID_RDKIT_SMILES"]

    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    unsupported = sorted(set(elements) - SUPPORTED_ELEMENTS)

    return unsupported


def main():
    print("=" * 90)
    print("STEP 04B: FILTER TL SMILES BY REINVENT-SUPPORTED ELEMENTS")
    print("=" * 90)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input SMI: {INPUT_SMI}")
    print(f"Supported elements: {sorted(SUPPORTED_ELEMENTS)}")
    print(f"Max allowed drop fraction: {MAX_DROP_FRACTION}")

    if not INPUT_SMI.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_SMI}")

    smiles = read_smiles(INPUT_SMI)
    n_input = len(smiles)

    if n_input == 0:
        raise ValueError(f"Input file is empty: {INPUT_SMI}")

    keep = []
    removed = []
    element_counter = Counter()

    for smi in smiles:
        unsupported = unsupported_elements_from_smiles(smi)

        if unsupported:
            removed.append(
                {
                    "SMILES": smi,
                    "Unsupported_Elements": ",".join(unsupported),
                }
            )
            element_counter.update(unsupported)
        else:
            keep.append(smi)

    n_keep = len(keep)
    n_removed = len(removed)
    drop_fraction = n_removed / n_input

    if n_keep == 0:
        raise ValueError("All SMILES were removed. Check supported element list.")

    OUT_KEEP.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    OUT_KEEP.write_text("\n".join(keep) + "\n", encoding="utf-8")

    with OUT_REMOVE.open("w", encoding="utf-8") as f:
        f.write("SMILES\tUnsupported_Elements\n")
        for rec in removed:
            f.write(f"{rec['SMILES']}\t{rec['Unsupported_Elements']}\n")

    summary = {
        "input_smi": str(INPUT_SMI),
        "output_supported_smi": str(OUT_KEEP),
        "output_removed_tsv": str(OUT_REMOVE),
        "supported_elements": sorted(SUPPORTED_ELEMENTS),
        "n_input_smiles": int(n_input),
        "n_supported_smiles": int(n_keep),
        "n_removed_smiles": int(n_removed),
        "drop_fraction": float(drop_fraction),
        "unsupported_element_counts": dict(element_counter.most_common()),
        "max_drop_fraction_guard": MAX_DROP_FRACTION,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nFirst 20 removed molecules:")
    for rec in removed[:20]:
        print(f"{rec['Unsupported_Elements']}\t{rec['SMILES']}")

    if drop_fraction > MAX_DROP_FRACTION:
        raise ValueError(
            f"High drop fraction: {drop_fraction:.3f}. "
            "More than 5% of TL seeds were removed. Inspect removed TSV before TL."
        )

    print("\nSTEP 04B COMPLETE")


if __name__ == "__main__":
    main()