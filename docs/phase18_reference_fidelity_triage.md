# Phase 18: Real-Asset Reference / Fidelity Triage

## Goal

Phase 18 diagnoses why Phase 17 benchmark previews should not be treated as real-image quality comparisons yet.

This phase does not change ReSTIR, proposal sampling, deferred lighting, spatial MIS, or benchmark row schemas. It only checks whether Voxel51 provides reference images and camera metadata that make Phase 17 auto-camera previews comparable to real images.

## Reference Intake

Download scene-root Voxel51 reference images:

```powershell
conda activate restirgs
python scripts/download_voxel51_references.py --dry-run
python scripts/download_voxel51_references.py
```

The script reads the Hugging Face tree for:

```text
drjohnson
playroom
train
truck
```

It downloads only scene-root `.jpg`, `.jpeg`, and `.png` files into:

```text
outputs/references/voxel51_<scene>/
```

It records:

```text
outputs/references/voxel51_reference_inventory.json
```

The inventory also searches for camera-like metadata paths:

```text
transforms.json
cameras.json
cameras.txt
images.txt
sparse/
colmap/
```

If none are found, `camera_alignment` is marked `unavailable`.

## Fidelity Triage

After Phase 17 benchmark outputs and Phase 18 references exist, run:

```powershell
python scripts/demo_16_render_fidelity_triage.py
```

Outputs:

```text
outputs/fidelity/voxel51_reference_vs_probe_contact.png
outputs/fidelity/fidelity_triage_summary.json
```

The contact sheet places Voxel51 reference images beside Phase 17 selected auto-camera previews. If camera metadata is unavailable, the preview is labeled `not camera-aligned`.

## Interpretation

Do not compute PSNR, SSIM, or photometric error unless real camera metadata is found.

If Voxel51 has reference images but no usable camera poses, the Phase 17 benchmark remains:

```text
multi-scene algorithm smoke benchmark
```

It is not:

```text
camera-aligned photometric benchmark
```

In that case, poor visual agreement with reference images is not enough evidence that ReSTIR, spatial MIS, or deferred lighting is broken. The next dataset step should be a real 3DGS package with COLMAP or equivalent camera metadata.

## Known Limits

Phase 18 keeps the 7000-iteration Voxel51 assets from Phase 17. It does not download 30000-iteration PLYs or render full-density visual comparisons.

The Phase 17 previews still use:

```text
auto-probed cameras
DC-only Gaussian color
max_gaussians=200000
synthetic lighting for ReSTIR experiments
```

Those choices are useful for algorithm smoke tests, not for faithful reproduction of dataset reference photographs.
