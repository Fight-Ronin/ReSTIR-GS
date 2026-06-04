# Phase 5: RIS Ablation Harness

This phase evaluates the Phase 4 initial RIS estimator across multiple candidate counts and random seeds before adding temporal or spatial reuse.

## Sweep

The default sweep is:

```text
K = [1, 2, 4, 8, 16, 32]
seed_count = 8
candidate_seed = 3100 + seed_index
selection_seed = 4100 + seed_index
num_lights = 128
```

Each row compares either `uniform` or `ris` against the all-lights reference.

## Quantities

Metrics are recorded for two quantities:

```text
diffuse_rgb
composite_rgb
```

`diffuse_rgb` is the primary estimator quantity because it excludes ambient and base RGB. `composite_rgb` is included because it reflects the final visible output.

## Metrics

Metrics are computed over valid G-buffer pixels:

```text
mae           = mean(abs(estimate - reference))
rmse          = sqrt(mean((estimate - reference)^2))
bias_r/g/b    = mean signed RGB error by channel
mean_abs_bias = mean(abs([bias_r, bias_g, bias_b]))
```

Invalid pixels are excluded from metrics.

## Outputs

The demo writes:

```text
outputs/ris_ablation.csv
outputs/ris_ablation_summary.json
```

The CSV contains one row per estimator, K value, seed, and reference quantity. The JSON stores run metadata plus grouped mean/std summaries by `reference_quantity`, `estimator`, and `K`.

## Interpretation

Uniform K-sample MAE should generally decrease as K increases. Initial RIS is reported beside uniform baselines, but it is not required to beat uniform for every K or seed with the current uniform proposal and luminance target.

This phase is only an evaluation harness. It does not add temporal reuse, spatial reuse, visibility, shadowing, proposal improvements, `.ply` loading, or real scene assets.
