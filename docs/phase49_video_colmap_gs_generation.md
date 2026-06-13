# Phase 49: Video To COLMAP GS Generation

This note records the first successful public-video Gaussian Splatting generation path for `gs_gen`.

The goal is to validate this route:

```text
source video
-> extracted image sequence
-> COLMAP via Nerfstudio ns-process-data
-> splatfacto training
-> ns-export gaussian-splat
-> ReSTIR-GS interactive viewer
```

This path is intentionally separate from the active aligned asset registry. It does not modify `configs/aligned_assets.json` or the active renderer manifest.

## Current Successful Asset

Tanks and Temples `Family.mp4` was used as the first stable video-to-COLMAP sample.

```text
video:
outputs/gsgen/tnt/videos/Family.mp4

processed dataset:
outputs/gsgen/tnt_family/ns_processed_nocuda_1fps_1600

trained run:
outputs/gsgen/tnt_family/train_smoke/ns_processed_nocuda_1fps_1600/splatfacto/2026-06-08_105851

exported splat:
outputs/gsgen/tnt_family/exports_smoke/family_splat.ply

viewer preview:
outputs/interactive_viewer/tnt_family/current_rgb.png
```

The validated dataset has:

```text
registered frames: 147 / 147
COLMAP sparse points: 80178
exported Gaussians: 120672
```

The 1000-step run is a smoke-quality training run, not a final-quality asset. It proves that the pipeline is connected end to end.

## Important COLMAP Model Selection Note

For `Family.mp4`, COLMAP produced multiple sparse models:

```text
colmap/sparse/0  bad small model: 4 registered images, 2 points
colmap/sparse/1  good model: 147 registered images, 80178 points
```

Nerfstudio initially reported a poor match count because it used the small model. The correct recovery was to regenerate `transforms.json` from the good COLMAP model:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
conda run --no-capture-output -n gs_gen ns-process-data images `
  --data outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600\images `
  --output-dir outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600 `
  --skip-colmap `
  --skip-image-processing `
  --colmap-model-path colmap\sparse\1 `
  --matching-method sequential `
  --sfm-tool colmap `
  --colmap-cmd C:\Users\yinha\Documents\ReSTIR-GS\gs_gen\tools\colmap_ns_compat.cmd `
  --no-gpu `
  --num-downscales 2
```

Do not judge the video-to-COLMAP result only from Nerfstudio's final report if multiple COLMAP sparse models exist. Inspect each model:

```powershell
gs_gen\tools\colmap-4.0.4-nocuda\bin\colmap.exe model_analyzer `
  --path outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600\colmap\sparse\1
```

## Reproducing The Smoke Run

Extract frames from the video at 1 fps and resize to a 1600px long edge:

```powershell
C:\Users\yinha\miniconda3\envs\gs_gen\Library\bin\ffmpeg.exe `
  -y `
  -i outputs\gsgen\tnt\videos\Family.mp4 `
  -vf fps=1,scale=1600:-2 `
  outputs\gsgen\tnt_family\source_images_1600\frame_%06d.png
```

Process with COLMAP:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
conda run --no-capture-output -n gs_gen ns-process-data images `
  --data outputs\gsgen\tnt_family\source_images_1600 `
  --output-dir outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600 `
  --matching-method sequential `
  --sfm-tool colmap `
  --colmap-cmd C:\Users\yinha\Documents\ReSTIR-GS\gs_gen\tools\colmap_ns_compat.cmd `
  --no-gpu `
  --num-downscales 2
```

If Nerfstudio selects the wrong sparse model, rerun the `--skip-colmap --skip-image-processing` command in the previous section and point it at the good `colmap\sparse\<id>` model.

Train a smoke-quality splat:

```powershell
conda run --no-capture-output -n gs_gen gs_gen\tools\run_gs_gen_vs.cmd ns-train splatfacto `
  --data outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600 `
  --output-dir outputs\gsgen\tnt_family\train_smoke `
  --max-num-iterations 1000 `
  --steps-per-save 1000 `
  --vis tensorboard `
  --viewer.quit-on-train-completion True
```

Export the trained Gaussian splat:

```powershell
conda run --no-capture-output -n gs_gen gs_gen\tools\run_gs_gen_vs.cmd ns-export gaussian-splat `
  --load-config outputs\gsgen\tnt_family\train_smoke\ns_processed_nocuda_1fps_1600\splatfacto\2026-06-08_105851\config.yml `
  --output-dir outputs\gsgen\tnt_family\exports_smoke `
  --output-filename family_splat.ply
```

Validate the generated asset:

```powershell
conda run --no-capture-output -n gs_gen python -m gs_gen validate `
  --dataset-root outputs\gsgen\tnt_family\ns_processed_nocuda_1fps_1600 `
  --splat outputs\gsgen\tnt_family\exports_smoke\family_splat.ply
```

## Inspecting In The Interactive Viewer

The interactive viewer can load a generic GraphDECO/Nerfstudio-style 3DGS PLY through `interactive.launcher --ply`.

Use the `gs_gen` runner rather than `scripts\run_interactive_viewer_windows.bat` for this generated asset, because the project runner defaults to the `restirgs` environment while this asset was trained in `gs_gen`.

Open the live viewer:

```powershell
conda run --no-capture-output -n gs_gen gs_gen\tools\run_gs_gen_vs.cmd python -m interactive.launcher `
  --ply outputs\gsgen\tnt_family\exports_smoke\family_splat.ply `
  --width 768 `
  --height 768 `
  --device cuda
```

For this smoke asset, outlier Gaussians can make the default generic auto-camera too far away. A tighter initial view is often easier to inspect:

```powershell
conda run --no-capture-output -n gs_gen gs_gen\tools\run_gs_gen_vs.cmd python -m interactive.launcher `
  --ply outputs\gsgen\tnt_family\exports_smoke\family_splat.ply `
  --width 768 `
  --height 768 `
  --device cuda `
  --auto-camera-bbox-percentile 0.85 `
  --auto-camera-radius-scale 0.8
```

Save a non-interactive preview:

```powershell
conda run --no-capture-output -n gs_gen gs_gen\tools\run_gs_gen_vs.cmd python -m interactive.launcher `
  --ply outputs\gsgen\tnt_family\exports_smoke\family_splat.ply `
  --width 768 `
  --height 768 `
  --output-dir outputs\interactive_viewer\tnt_family `
  --save-and-exit `
  --device cuda
```

The save-and-exit command writes:

```text
outputs/interactive_viewer/tnt_family/current_camera.json
outputs/interactive_viewer/tnt_family/current_rgb.png
outputs/interactive_viewer/tnt_family/current_alpha.png
outputs/interactive_viewer/tnt_family/current_normal.png
outputs/interactive_viewer/tnt_family/current_blinn_phong.png
```

Useful controls in the live viewer:

```text
W / S        move forward / backward
A / D        move left / right
Shift / Ctrl move up / down
Left drag    free-look yaw/pitch
Middle drag  pan camera target
Mouse wheel  dolly focus distance
1            RGB
2            Alpha
3            Depth
4            Normal
5            Lambertian
6            Blinn-Phong
Ctrl+S       save current camera and previews
q            quit
```

## Current Limitations

- The exported `family_splat.ply` is suitable for viewer inspection, but it is only a 1000-step smoke result.
- `sparse_pc.ply` is only the COLMAP sparse point cloud. It is not a Gaussian splat and should not be loaded into the GS renderer.
- The generated asset is not registered in `configs/aligned_assets.json`; it is inspected through generic `--ply` mode.
- CPU COLMAP on 4K frames is heavy. The successful smoke used 1 fps and 1600px resized frames.
