# Phase 31: Aligned ReSTIR Renderer Path

Phase 31 turns the existing aligned components into one coherent renderer path instead of another ablation sweep.

```text
aligned asset registry
-> generic 3DGS loader
-> aligned camera + pseudo G-buffer
-> scene-stable world-space lights
-> geometric proposal
-> initial RIS
-> previous-frame temporal reservoir reuse
-> CSV/JSON/debug images
```

The default run targets the `testing` asset set from `configs/aligned_assets.json`.

## Renderer Contract

The reusable renderer core lives under `restir_gs.restir.renderer`. It has two entry levels:

- `evaluate_restir_frame_from_gbuffer(...)` runs the ReSTIR path from an existing G-buffer and camera. This is the CPU-testable layer.
- `render_restir_frame(...)` renders a 3DGS scene with `gsplat`, builds the pseudo G-buffer, converts world lights into the current camera, then calls the G-buffer layer.

The algorithm is intentionally fixed for this phase:

- target mode: diffuse
- proposal: geometric per-pixel light proposal
- default candidates: `K=8`
- lights: scene-stable world-space point lights
- reuse: one previous carried temporal reservoir

For the first frame, temporal output is exactly the initial RIS output. For later frames, the previous reservoir is reprojected into the current frame, the selected previous light is re-evaluated at the current pixel, and the combined reservoir is carried forward.

## Outputs

Run:

```powershell
scripts\run_aligned_restir_renderer_windows.bat
```

Outputs are written under:

```text
outputs/aligned_restir/
```

The global artifacts are:

```text
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
```

Each asset also gets:

```text
outputs/aligned_restir/<asset_id>/contact.png
outputs/aligned_restir/<asset_id>/final_reference.png
outputs/aligned_restir/<asset_id>/final_initial_ris.png
outputs/aligned_restir/<asset_id>/final_temporal_ris.png
outputs/aligned_restir/<asset_id>/final_reuse_mask.png
```

The CSV records `initial_ris` and `temporal_ris` side by side for contribution and composite RGB. Temporal improvement is not required for correctness in this phase; the important checks are finite metrics, correct first-frame fallback, and nonzero aligned reuse on later frames where reprojection succeeds.

## Viewer Inspection

The interactive viewer now accepts registered aligned assets:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Press `4` for the single-frame ReSTIR inspection panel:

```text
reference / geometric MC / initial RIS / initial error / proposal max / alpha
```

The viewer remains a debug tool. Temporal carry is inspected through the renderer contact sheets, not through live UI state.

## Limits

This phase does not add new proposals, Blinn-Phong RIS targets, spatial reuse, visibility, shadows, denoising, or broad ablations. It is a renderer-path consolidation step so later algorithm changes have one stable aligned baseline to plug into.
