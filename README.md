# ReSTIR-GS

Minimal Windows-native prototype setup for ReSTIR-GS. Phase 1 uses deterministic synthetic Gaussians to verify `gsplat` RGB/depth/alpha rendering before adding `.ply` loading, normal estimation, lighting, or ReSTIR.

## Environment

Create the conda environment and install Python dependencies:

```powershell
conda env create -f environment.yml
conda activate restirgs
pip install -r requirements.txt
```

The current verified stack is Python 3.10, CUDA toolkit 12.4, PyTorch `2.5.1+cu124`, and `gsplat==1.5.3`.

## Windows gsplat Patch

`gsplat==1.5.3` passes the GCC-only flag `-Wno-attributes` to MSVC during JIT extension builds. Apply the local compatibility patch after installing dependencies:

```powershell
python scripts/patch_gsplat_windows.py
python scripts/patch_gsplat_windows.py --check
```

The patch is idempotent and only changes the installed package inside the active Python environment.

## Smoke Demo

Run the repo-native synthetic RGB+D render from Windows:

```powershell
scripts\run_smoke_windows.bat
```

It writes:

```text
outputs/synthetic_rgb.png
outputs/synthetic_depth.png
outputs/synthetic_alpha.png
```

This is the gate before adding pseudo G-buffer normals, deferred lighting, or real Gaussian `.ply` assets.

