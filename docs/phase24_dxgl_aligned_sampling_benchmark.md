# Phase 24: DXGL Aligned Multi-Frame Sampling Benchmark

Phase 24 combines the active aligned path into one sampling readout:

```text
DXGL Apple cameras + compatible splat
-> RGB/expected-depth/alpha
-> pseudo G-buffer
-> asset-scaled lights
-> Lambertian and Blinn-Phong all-lights references
-> uniform/geometric MC/RIS estimates
```

The goal is to compare diffuse and Blinn-Phong RIS targets across multiple aligned frames without changing the proposal distribution or adding reuse.

## What Is Compared

Each frame records rows for:

```text
target_mode: diffuse, blinn_phong
proposal: uniform, geometric
estimator: mc, ris
K: 1, 2, 4, 8, 16, 32
reference_quantity: contribution_rgb, composite_rgb
```

`target_mode="diffuse"` uses the Lambertian all-lights reference:

```text
contribution_rgb = lambertian.diffuse_rgb
composite_rgb = lambertian.composite_rgb
```

`target_mode="blinn_phong"` uses the Blinn-Phong all-lights reference:

```text
contribution_rgb = blinn.diffuse_rgb + blinn.specular_rgb
composite_rgb = blinn.composite_rgb
```

The geometric proposal is intentionally unchanged:

```text
q(l|x) proportional to intensity_l * luminance(color_l) * abs(dot(N, wi)) / dist2
```

This isolates whether changing the RIS target alone is useful before designing a specular-aware proposal.

## Run

```powershell
scripts\run_dxgl_sampling_benchmark_windows.bat
```

The default run uses 8 evenly spaced frames, 256x256 resolution, 128 lights, K values `[1,2,4,8,16,32]`, and 4 seeds.

Outputs:

```text
outputs/aligned_sampling/dxgl_sampling_rows.csv
outputs/aligned_sampling/dxgl_sampling_summary.json
outputs/aligned_sampling/dxgl_sampling_contact.png
```

The contact sheet shows the aligned reference RGB, rendered RGB, alpha, Lambertian composite, and Blinn-Phong composite for each selected frame.

## Interpretation

If the Blinn-Phong target improves Blinn-Phong reference error at fixed K, the next step can investigate specular-aware proposals or a broader target benchmark.

If the Blinn-Phong target does not help, keep the default RIS target diffuse and focus on proposal quality, pseudo-normal quality, visibility, or dataset expansion.

If both target modes are unstable across frames, fix G-buffer/reference/lighting consistency before adding spatial or temporal reuse.

## Scope

This phase does not add temporal reuse, spatial reuse, shadows, visibility, denoising, new scene assets, or a new proposal distribution. It is a multi-frame aligned smoke benchmark, not the final multi-scene research readout.
