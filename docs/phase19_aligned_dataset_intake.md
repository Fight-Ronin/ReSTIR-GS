# Phase 19: Aligned Dataset Intake Spike

Phase 19 stops treating the Voxel51 assets as the active dataset path. Those assets were useful for historical algorithm smoke tests, but they do not provide the aligned camera/reference package we need for real fidelity checks.

The active smoke target is the DXGL Polyhaven 10 `apple` object. It is small, CC0, single-object, and includes RGB images, masks, depth, normals, `points3D.ply`, and `transforms.json`.

## Workflow

Download and validate the aligned asset:

```powershell
python scripts/download_dxgl_apple.py --dry-run
python scripts/download_dxgl_apple.py
```

Run the intake demo:

```powershell
python scripts/demo_17_dxgl_aligned_intake.py
```

The downloader extracts into:

```text
outputs/aligned_assets/dxgl/apple/
```

The intake demo writes:

```text
outputs/aligned_fidelity/dxgl_apple_contact.png
outputs/aligned_fidelity/dxgl_apple_intake_summary.json
outputs/aligned_fidelity/dxgl_apple_frame_000_camera.json
outputs/aligned_fidelity/dxgl_apple_frame_049_camera.json
outputs/aligned_fidelity/dxgl_apple_frame_098_camera.json
outputs/aligned_fidelity/dxgl_apple_frame_147_camera.json
```

## Required Dataset Entries

The downloader validates:

```text
transforms.json
images/
depth/
depth_16bit/
normals/
masks/
points3D.ply
```

Missing entries fail loudly because this phase is about confirming aligned dataset intake, not silently falling back to incomplete data.

## Camera Convention

DXGL/Nerfstudio-style `transforms.json` stores camera-to-world transforms in an OpenGL-like convention where the camera looks down `-Z`.

The project renderer expects world-to-camera matrices with camera `+Z` forward and image `+Y` downward. The importer converts each frame by:

```text
w2c_opengl = inverse(camera_to_world)
project_w2c = diag(1, -1, -1, 1) * w2c_opengl
```

Intrinsics are loaded from `fl_x`, `fl_y`, `cx`, `cy`, `w`, and `h`. If `fl_y` is absent, it falls back to `fl_x`. If focal length is described by camera angle, the importer computes focal length from image size and field of view.

Imported cameras are returned as the existing `PinholeCamera` dataclass, so the `gsplat` render path does not need an API change.

## points3D.ply Probe

The demo probes `points3D.ply` with the existing 3DGS PLY loader. This point cloud is expected to be incompatible if it lacks GraphDECO 3DGS fields such as `opacity`, `scale_0..2`, and `rot_0..3`.

That is not a Phase 19 failure. It simply means the next phase should discover or obtain a compatible pretrained Gaussian splat for the same DXGL camera set before computing PSNR/SSIM or running ReSTIR experiments.

## Interpretation

Phase 19 answers only:

```text
Can we download and validate an aligned dataset?
Can we import real camera metadata into PinholeCamera?
Can we inspect RGB/depth/normal/mask frames side by side?
Does the included point cloud already match our 3DGS loader schema?
```

It does not train, download GraphDECO pretrained models, compute photometric metrics, or tune ReSTIR.
