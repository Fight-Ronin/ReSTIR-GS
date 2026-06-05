# Phase 20: DXGL Apple Splat Fidelity Smoke

Phase 20 connects the aligned DXGL Apple cameras from Phase 19 to the pretrained 3DGS splat advertised by DXGL. The goal is not ReSTIR tuning. The goal is to prove that:

```text
DXGL pretrained splat -> existing PLY loader -> gsplat renderer -> imported transforms.json camera
```

produces a non-empty render that roughly aligns with the DXGL reference RGB under the provided mask.

## Download

Download and validate the Apple pretrained splat:

```powershell
python scripts/download_dxgl_apple_splat.py --dry-run
python scripts/download_dxgl_apple_splat.py
```

The default path is:

```text
outputs/aligned_assets/dxgl/apple_splat/apple.ply
```

The downloader validates the file through the existing GraphDECO/Nerfstudio-compatible 3DGS PLY loader. This requires fields such as `opacity`, `scale_0..2`, `rot_0..3`, and color fields.

## Fidelity Smoke

Run the aligned render smoke:

```powershell
scripts\run_dxgl_splat_fidelity_windows.bat
```

or directly:

```powershell
python scripts/demo_18_dxgl_splat_fidelity.py --width 256 --height 256
```

The default selected frames are:

```text
0, 49, 98, 147
```

The demo writes:

```text
outputs/aligned_fidelity/dxgl_apple_splat_contact.png
outputs/aligned_fidelity/dxgl_apple_splat_fidelity_summary.json
outputs/aligned_fidelity/dxgl_apple_splat_frame_<index>_render.png
outputs/aligned_fidelity/dxgl_apple_splat_frame_<index>_reference.png
outputs/aligned_fidelity/dxgl_apple_splat_frame_<index>_alpha.png
outputs/aligned_fidelity/dxgl_apple_splat_frame_<index>_abs_error.png
```

## Camera Scaling

The DXGL source frames are 1024x1024. The smoke defaults to 256x256 for speed. Intrinsics are scaled as:

```text
fx' = fx * width' / width
cx' = cx * width' / width
fy' = fy * height' / height
cy' = cy * height' / height
```

Extrinsics are unchanged.

## Metrics

The demo reports masked RGB:

```text
MAE
RMSE
PSNR
```

The mask comes from `masks/` when available, otherwise from the RGBA alpha channel. These are smoke metrics, not final benchmark numbers. The first thing to inspect is the contact sheet alignment.

## Interpretation

If the splat downloads, loads, and renders aligned views, then the project has a real aligned 3DGS path and can move back toward G-buffer/deferred/proposal experiments.

If the splat fails schema validation, the next step is a PLY schema adapter. If the splat renders but is not aligned, the next step is camera convention debugging, not ReSTIR tuning.
