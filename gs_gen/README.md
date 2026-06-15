# gs_gen

Standalone helper for preparing local Gaussian Splatting assets before any manual integration with ReSTIR-GS.

`gs_gen` is intentionally separate from the active renderer path. It does not edit `configs/aligned_assets.json`, does not make staged assets available to the aligned asset registry, and does not install or run external reconstruction/training tools.

## Scope

Current commands:

```text
probe-source      validate an image directory or video path
extract-frames    extract a frame sequence from a video
plan              print a Nerfstudio/Splatfacto command plan
validate          validate processed transforms and exported splat fields
stage             copy validated outputs into a stable local folder
```

Supported sources:

```text
image directory
video file
JSON config with asset_id, source, and optional workspace
```

## Local Files

Capture videos, image sequences, generated datasets, trained models, and exported splats should stay in ignored local directories such as:

```text
data/gs_gen/
outputs/gsgen/
```

## Probe Sources

```powershell
python -m gs_gen probe-source --images data\gs_gen\room_capture\my_room\images
python -m gs_gen probe-source --video data\gs_gen\room_capture\my_room\walkthrough.mp4
```

Add `--json` for machine-readable output.

## Extract Video Frames

```powershell
python -m gs_gen extract-frames --video data\gs_gen\room_capture\my_room\walkthrough.mp4 --output-dir outputs\gsgen\my_room\source_images --target-fps 5
```

Useful options:

```text
--max-frames N
--dry-run
--json
```

## Plan External Commands

From an image directory:

```powershell
python -m gs_gen plan --asset-id my_room --images data\gs_gen\room_capture\my_room\images
```

From a video:

```powershell
python -m gs_gen plan --asset-id my_room --video data\gs_gen\room_capture\my_room\walkthrough.mp4
```

From a config file:

```powershell
python -m gs_gen plan --config gs_gen\configs\my_room.example.json
```

The printed plan follows this shape:

```text
ns-process-data images|video
ns-train splatfacto
ns-export gaussian-splat
python -m gs_gen validate
python -m gs_gen stage
```

Those external commands are informational; `gs_gen plan` only prints them.

## Validate

```powershell
python -m gs_gen validate --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply
```

Validation checks:

```text
transforms.json exists and contains frames
global image size and focal metadata exist
frame image paths resolve
exported PLY has compatible 3DGS fields
```

## Stage

```powershell
python -m gs_gen stage --asset-id my_room --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply --copy-images
```

Staging writes:

```text
outputs/gsgen/<asset_id>/staged/
  transforms.json
  splat.ply
  asset_info.json
  images/              optional, with --copy-images
```

Use `--dry-run` to validate and preview the staged path without writing files.

## Output Layout

The default workspace is `outputs/gsgen/`.

```text
outputs/gsgen/<asset_id>/
  processed/
  train/
  export/
  staged/
```
