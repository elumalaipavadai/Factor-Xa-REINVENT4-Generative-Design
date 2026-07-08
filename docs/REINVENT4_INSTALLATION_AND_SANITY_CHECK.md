# REINVENT4 Installation, Troubleshooting, and Sanity Check

Project: `FXA_REINVENT_Portfolio`  
Environment: `fxa_reinvent4_py311`  
Platform: Windows PowerShell  
REINVENT version: `4.8.24`  
PyTorch: `2.12.0+cpu`  
Run mode tested: CPU sampling

---

## 1. Project location

```powershell
cd .
```

Final project structure used:

```text
FXA_REINVENT_Portfolio/
â”œâ”€â”€ configs/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ generated/
â”‚   â””â”€â”€ reference/
â”œâ”€â”€ docs/
â”œâ”€â”€ external/
â”‚   â””â”€â”€ REINVENT4/
â”œâ”€â”€ models/
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ metrics/
â”‚   â”œâ”€â”€ tables/
â”‚   â””â”€â”€ figures/
â””â”€â”€ scripts/
```

---

## 2. Conda environment

The first attempt used Python 3.10, but the installed REINVENT4 package required Python 3.11 or newer. Therefore, a new environment was created.

```powershell
conda create -n fxa_reinvent4_py311 python=3.11 -y
conda activate fxa_reinvent4_py311
python --version
where python
```

Expected environment:

```text
fxa_reinvent4_py311
Python 3.11.x
```

---

## 3. Git installation issue

Initial problem:

```text
git : The term 'git' is not recognized
```

Fix:

```powershell
conda activate fxa_reinvent4_py311
conda install -c conda-forge git -y
git --version
```

Reason: REINVENT installation later needed Git to install the `iSIM` dependency from GitHub.

---

## 4. Clone REINVENT4

```powershell
cd .

New-Item -ItemType Directory -Force external
cd external

git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
```

A Windows warning appeared about case-sensitive file collisions in tutorial map files. This did not block the core REINVENT installation.

---

## 5. Check installer help

```powershell
python install.py --help
```

The installer showed:

```text
usage: install.py [-h] [-d PKGS [PKGS ...]] [-e] [--dry-run] NAME

NAME: cpu, cu124, rocm6.2.4, mac, etc.
```

For this project, CPU installation was selected first for stability.

---

## 6. Dry run

```powershell
python install.py cpu --dry-run *> ..\..\scripts\02_reinvent_install_dryrun_py311.log
Get-Content ..\..\scripts\02_reinvent_install_dryrun_py311.log -Tail 80
```

Dry run command shown:

```text
pip install .[all,chemprop2] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.anaconda.org/OpenEye/simple
```

---

## 7. Install REINVENT4

```powershell
python install.py cpu *> ..\..\scripts\02_reinvent_install_py311_retry2.log
Get-Content ..\..\scripts\02_reinvent_install_py311_retry2.log -Tail 120
```

Successful install evidence:

```text
Successfully built reinvent molvs iSIM
Successfully installed ... reinvent-4.8.24 ... torch-2.12.0+cpu ... torchvision-0.27.0+cpu ...
```

The install log confirmed that `reinvent-4.8.24`, `torch-2.12.0+cpu`, `torchvision-0.27.0+cpu`, `chemprop`, `OpenEye-toolkits`, and other dependencies were installed successfully.

---

## 8. Troubleshooting during install

### Problem A: Python 3.10 was incompatible

Error:

```text
ERROR: Package 'reinvent' requires a different Python: 3.10.20 not in '>=3.11'
```

Fix:

```powershell
conda create -n fxa_reinvent4_py311 python=3.11 -y
conda activate fxa_reinvent4_py311
```

---

### Problem B: Missing torch and RDKit

Errors:

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'rdkit'
```

Fix:

```powershell
conda install -c conda-forge rdkit pandas numpy scipy scikit-learn tqdm pydantic toml -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Problem C: Git missing during iSIM installation

Error:

```text
ERROR: Cannot find command 'git'
```

Fix:

```powershell
conda activate fxa_reinvent4_py311
conda install -c conda-forge git -y
```

Then rerun:

```powershell
cd .\external\REINVENT4
python install.py cpu *> ..\..\scripts\02_reinvent_install_py311_retry2.log
```

---

### Problem D: Windows `resource` module error

Error:

```text
ModuleNotFoundError: No module named 'resource'
```

Reason: Pythonâ€™s `resource` module is Unix/Linux-specific. REINVENT imported it in `hw_report.py`.

Fix: patch `hw_report.py` to make `resource` optional.

```powershell
@'
from pathlib import Path
import site

matches = []

for sp in site.getsitepackages():
    p = Path(sp) / "reinvent" / "utils" / "hw_report.py"
    if p.exists():
        matches.append(p)

if not matches:
    raise SystemExit("Could not find reinvent/utils/hw_report.py")

path = matches[0]
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.bak_windows_resource_patch")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

old = "import resource"

new = """try:
    import resource
except ModuleNotFoundError:
    class _DummyResource:
        RUSAGE_SELF = 0
        RUSAGE_CHILDREN = 0

        @staticmethod
        def getrusage(*args, **kwargs):
            class _Usage:
                ru_maxrss = 0
            return _Usage()

    resource = _DummyResource()
"""

if "class _DummyResource" in text:
    print("Already patched:", path)
else:
    if old not in text:
        raise SystemExit("Could not find 'import resource' line to patch.")
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("Patched:", path)
    print("Backup:", backup)
'@ | python
```

---

### Problem E: Pillow / PIL DLL error

Error:

```text
ImportError: DLL load failed while importing _imaging
```

Reason: `torchvision` imported `PIL.Image`, but the Pillow DLL was broken.

Fix:

```powershell
python -m pip uninstall -y pillow PIL
conda remove pillow -y
python -m pip install --no-cache-dir --force-reinstall pillow
```

Then test:

```powershell
python -c "from PIL import Image; print('PIL Image OK')"
```

---

### Problem F: Missing RDKit and matplotlib after Pillow cleanup

Warning:

```text
reinvent 4.8.24 requires matplotlib<4,>=3.7, which is not installed.
reinvent 4.8.24 requires rdkit>=2025.09.1, which is not installed.
```

Fix:

```powershell
python -m pip install --no-cache-dir --only-binary=:all: "rdkit>=2025.09.1" "matplotlib>=3.7,<4"
```

---

### Problem G: Missing charset-normalizer

Error from `pip check`:

```text
requests 2.34.2 requires charset-normalizer, which is not installed.
```

Fix:

```powershell
python -m pip install charset-normalizer
python -m pip check
```

Final result:

```text
No broken requirements found.
```

The final dependency check and import tests passed: `pip check` returned no broken requirements, and PIL, RDKit, torchvision, and REINVENT all imported correctly.

---

## 9. Final environment verification

Run from project root:

```powershell
cd .
conda activate fxa_reinvent4_py311

python -m pip check

reinvent --help *> .\scripts\03_reinvent_help_after_dependency_fix.log
Get-Content .\scripts\03_reinvent_help_after_dependency_fix.log -Head 120

python -c "from PIL import Image; print('PIL Image OK')"
python -c "from rdkit import Chem; print('RDKit OK', Chem.MolToSmiles(Chem.MolFromSmiles('CCO')))"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import reinvent; print('REINVENT import OK')"
```

Expected output:

```text
No broken requirements found.
REINVENT 4.8.24 using PyTorch 2.12.0+cpu
PIL Image OK
RDKit OK CCO
torchvision 0.27.0+cpu
REINVENT import OK
```

---

## 10. Download REINVENT prior model

The cloned REINVENT4 repository did not include a `priors` folder or `reinvent.prior`. This caused the first sampling run to fail with:

```text
RuntimeError: model file ...\external\REINVENT4\priors\reinvent.prior is not accessible
```

Download the official prior:

```powershell
cd .
conda activate fxa_reinvent4_py311

New-Item -ItemType Directory -Force .\external\REINVENT4\priors

curl.exe -L "https://zenodo.org/records/15641297/files/reinvent.prior?download=1" -o ".\external\REINVENT4\priors\reinvent.prior"
```

Verify:

```powershell
Test-Path .\external\REINVENT4\priors\reinvent.prior

Get-Item .\external\REINVENT4\priors\reinvent.prior | Select-Object FullName,Length

Get-FileHash .\external\REINVENT4\priors\reinvent.prior -Algorithm MD5
```

Expected:

```text
Test-Path = True
Length = 23226277
MD5 = F268EB072F4FCA69CA9434768D3CD461
```

---

## 11. Minimal REINVENT sampling config

Create:

```powershell
@'
run_type = "sampling"
device = "cpu"
json_out_config = "results/metrics/03_sanity_sampling_config.json"

[parameters]

model_file = "external/REINVENT4/priors/reinvent.prior"
sample_strategy = "multinomial"
temperature = 1.0

output_file = "data/generated/reinvent_sanity_100.csv"

num_smiles = 100
unique_molecules = true
randomize_smiles = true
'@ | Set-Content .\configs\03_sanity_sampling.toml
```

Check:

```powershell
Get-Content .\configs\03_sanity_sampling.toml
```

Important: do not type this line directly into PowerShell:

```text
model_file = "external/REINVENT4/priors/reinvent.prior"
```

That line belongs inside the TOML config file, not as a PowerShell command. Typing it directly causes:

```text
model_file : The term 'model_file' is not recognized
```

This happened during troubleshooting and was only a PowerShell usage mistake, not a REINVENT issue.

---

## 12. Run sanity sampling

```powershell
reinvent -l .\results\metrics\03_sanity_sampling.log .\configs\03_sanity_sampling.toml
```

Inspect log:

```powershell
Get-Content .\results\metrics\03_sanity_sampling.log -Tail 100
```

Inspect generated molecules:

```powershell
Get-ChildItem .\data\generated

Import-Csv .\data\generated\reinvent_sanity_100.csv | Select-Object -First 10
```

---

## 13. Sanity sampling result

Final successful result:

```text
Started REINVENT 4.8.24
Python version 3.11.15
PyTorch version 2.12.0+cpu
RDKit version 2026.03.3
Platform Windows
Using CPU
Starting Sampling
Writing sampled SMILES to CSV file data/generated/reinvent_sanity_100.csv
Sampling 100 SMILES from model external/REINVENT4/priors/reinvent.prior
Removed 1 invalid SMILES
Finished REINVENT
```

Generated file:

```text
data/generated/reinvent_sanity_100.csv
```

The first generated molecules were successfully read with:

```powershell
Import-Csv .\data\generated\reinvent_sanity_100.csv | Select-Object -First 10
```

---

## 14. Final status

REINVENT4 installation status:

```text
Conda environment: fxa_reinvent4_py311
Python: 3.11.15
REINVENT: 4.8.24
PyTorch: 2.12.0+cpu
torchvision: 0.27.0+cpu
RDKit: 2026.03.3
PIL/Pillow: working
pip check: No broken requirements found
REINVENT CLI: working
Prior model: downloaded and checksum verified
Sanity sampling: completed
Generated output: data/generated/reinvent_sanity_100.csv
```

Conclusion:

REINVENT4 is installed and functional on Windows CPU mode after small Windows-specific fixes. The environment can now generate molecules from the official `reinvent.prior`. The next project step is to score generated molecules using the Project 1 Factor Xa scaffold-aware Random Forest model.
