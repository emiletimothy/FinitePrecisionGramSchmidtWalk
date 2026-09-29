# Gram–Schmidt walk under reduced-precision arithmetic

Code for studying how low-precision (reduced-mantissa) arithmetic degrades the
Gram–Schmidt walk (Bansal–Dadush–Garg–Lovett). Given `B ∈ R^{m×n}` with
unit-norm columns, the walk produces a signing `z ∈ {−1,+1}^n` such that the
discrepancy `Bz` is small and every linear functional `d·Bz` is approximately
1-subgaussian. The experiments measure how the discrepancy and the subgaussian
constant degrade as the walk's inner least-squares solve is rounded to few
mantissa bits, and compare deterministic rounding against matched additive
noise.

Main phenomena studied:

- A **precision plateau**: at moderate mantissa widths the walk absorbs
  rounding error and matches the fp64 baseline; below a threshold the
  subgaussian scale inflates.
- **Scaling laws**: on real data (Higgs) the inflation is well described by
  `n·2^-b`; on an adversarial matrix it reaches the lower-bound rate
  `n²·2^-b`.
- **Rounding vs. noise**: matched i.i.d. additive noise accumulates
  incoherently (`√n·ε`), while rounding accumulates coherently.
- **Mechanism**: the per-step solve error is concentrated where the free-set
  size `k ≈ m`, i.e. where the active submatrix is nearly singular.

## Setup

```sh
conda env create -f environment.yml   # env name: gsw, python 3.13
conda activate gsw
```

or a plain venv with `numpy`, `scipy`, `matplotlib`, `tqdm`, `pandas`,
`scikit-learn`, `pytest`. Run the tests with `python -m unittest discover -s tests`.
The solver self-check is `python core/lpla.py` (compares the hand-rolled solver
against `np.linalg.lstsq` at full precision).

## Layout

```
core/            reusable library (imported by everything else)
  gsw.py           the walk; lpla.py the precision model; rollouts.py the
                   parallel MC driver + stats caches; matrices.py the matrix
                   registry; paths.py the figures/<experiment>/ output helper
scripts/         runnable experiments — `python scripts/<name>.py ...`
tests/           test_rollouts.py
figures/         generated output, one subdirectory per experiment kind
dtype_stability/  self-contained IEEE-dtype solver-comparison suite
```

## Core modules (`core/`)

| file | contents |
|---|---|
| `gsw.py` | the walk. `gram_schmidt_walk(B, chop=None, noise=None, record_trajectory=False, quantize_input=True, snap_c=10.0, max_steps=None) → WalkResult`. `chop` routes the direction solve through `lpla.lstsq`; `noise` perturbs unfrozen `z` coordinates each step. Tolerances scale with the format ulp; binding coordinates are force-snapped to ±1; `WalkResult.failed` records early termination. `Bz` is always measured in fp64 on the unrounded input; `Bz_hat` scores the same signs against `chop(B)`, so `Bz − Bz_hat` isolates input quantization. |
| `lpla.py` | the precision model. `make_round(sig_bits)` is a round-to-nearest-even mantissa rounder with unbounded exponent (rounding, not IEEE range); the returned callable carries `.sig_bits` for tolerance scaling. `lstsq` is a shape-adaptive Householder QR / LQ min-norm solver with every elementary op rounded; reductions accumulate in fp64. `round_inputs=False` keeps A and c exact so arithmetic error can be isolated from input quantization. |
| `rollouts.py` | parallel Monte-Carlo walks over `ProcessPoolExecutor` with per-rollout `SeedSequence` streams; returns a 4-tuple `(bz_means, projections, bz_samples, bz_hat_samples)`. Also owns the provenance-checked stats caches (`save_cache`/`load_cache`), `output_path` noise-suffixing, and `add_noise_arguments`. |
| `matrices.py` | the matrix registry: `clustered`, `gaussian`, `higgs`, `identity`, `lower_bound`. `matrix_family(name, m, n_max, seed)` returns `get_B(n)` giving a nested column set (B(n) = n-column prefix of one draw), so n-sweeps aren't confounded by redrawn matrices. `add_matrix_arguments(parser)` gives every experiment the same `--matrix/--m` flags. |
| `paths.py` | `figure_path(experiment, filename)` → `figures/<experiment>/<filename>`; every script's default output goes there (stats caches land beside the figure they describe). |

## Gaussian noise controls and new-run provenance

Main-walk sweeps now default to **no added Gaussian noise**, including their
float64 references. Use `--noise-std VALUE` for a finite, nonnegative standard
deviation, or `--no-noise` to disable it explicitly. These options are available
in `n_subgauss.py`, `bits_subgauss.py`, `shape_persist.py`, `phase_heatmap.py`,
`noise_vs_chop.py`, `norm_dists.py`, and `qq_diagnostics.py`. Their callable
functions also accept `noise_std=0.0`. The lower-bound experiments retain their
separate deterministic error model; the dtype-based suite in `dtype_stability/`
is unchanged.

```sh
python scripts/n_subgauss.py --no-noise --save ablation.png
python scripts/n_subgauss.py --noise-std 2.3283064365386963e-10 --save ablation.png
python scripts/n_subgauss.py --matrix identity --n 4 8 --sig-bits 3 52 --num-samples 10 --workers 1 --no-noise
```

The first two commands use standard deviations 0 and `2^-32`, respectively.
Outputs receive a `_noise<standard-deviation>` suffix even with `--save`:
`ablation_noise0.png` and `ablation_noise2.3283064365386963e-10.png`, with
correspondingly named caches. Existing unlabelled figures and caches are not
renamed or rewritten. Each rollout batch prints its noise parameters, and new
figures display the selected setting. New NPZ caches store a JSON `metadata`
field; PNGs embed the same JSON in their `Description` field. Metadata records
noise parameters, seed, experiment settings, RNG scheme, timestamp, Git revision,
and whether the checkout has local changes. A dirty checkout is recorded as
such, not represented as an exact committed revision.

Cache readers reject missing metadata and mismatched requested settings. In
particular, an old cache cannot be silently reused as a zero-noise result.
For `--plot-only`, pass the same experiment settings and base `--save` path as
in the generating command. Replotting preserves the cache's generating metadata.
Heatmap reuse is explicit and checked:

```sh
python scripts/phase_heatmap.py --no-noise --rounding-cache figures/n_subgauss/n_subgauss_higgs_noise0_cache.npz
```

The dedicated `noise_vs_chop.py` comparison has two arms. `--noise-std` /
`--no-noise` select the noise setting of the cached rounding arm. Its intentional
noise-only arm uses `--noise-scale S` times `2^-b` (default `S=1`); set
`--noise-scale 0` to disable that arm's added noise. Both settings are recorded
and labelled, and the output filename also includes `_scale<S>`. Generate a
compatible `n_subgauss.py` cache first or provide `--rounding-cache PATH`.

Gaussian draws now use a separate child RNG stream per rollout, so sampling
noise does not itself consume walk random draws. Results are reproducible across
worker counts for fixed inputs and seeds, but adaptive trajectories can still
diverge. This RNG change means noisy runs are not bitwise reproductions of the
old shared-RNG implementation.


## Experiment scripts (`scripts/`)

Scripts are run directly, e.g. `python scripts/n_subgauss.py --matrix higgs`.
Each script's output lands in `figures/<script-name>/` (the stats NPZ cache sits
next to the figure it describes, and `--plot-only` replots from it).

Every experiment accepts `--matrix {clustered,gaussian,higgs,identity,lower_bound}`
(and `--m` for the families that need a row count) plus `--num-samples`,
`--workers`, `--seed`, `--noise-std`/`--no-noise`, and `--plot-only` where a
stats cache exists. Figure filenames embed the matrix by default.

| script | figure (under `figures/`) | question |
|---|---|---|
| `n_subgauss.py` | `n_subgauss/n_subgauss_<matrix>.png` | σ̂ vs n: worst-direction moment/ψ₂ estimates (selection/evaluation split, centered σ + bias), fixed-threshold exceedance, direction-spread bands. |
| `bits_subgauss.py` | `bits_subgauss/bits_subgauss_<matrix>.png` | σ̂ vs mantissa bits at fixed n, with plateau/growth model fits. |
| `shape_persist.py` | `shape_persist/shape_persist_<matrix>.png` | worst-direction tail shape vs n — does `d·Bz` stay Gaussian-shaped as σ inflates? |
| `phase_heatmap.py` | `phase_heatmap/phase_heatmap_<matrix>.png` | σ̂ over the (n, bits) plane with the empirical knee `n ≈ 10·2^b`. |
| `noise_vs_chop.py` | `noise_vs_chop/noise_vs_chop_<matrix>.png` | rounding vs matched additive noise, and the `n·ε` / `√n·ε` collapse comparison. |
| `step_landing.py` | `step_landing/step_landing_*.png` | non-accumulating diagnostic: runs the exact fp64 trajectory while also solving each step in low precision, reporting `u_lp − u_id` keyed by free-set size `k`. Shows the `k ≈ m` spike. |
| `lb_subgauss.py` | `lb_subgauss/lb_n_subgauss.png`, `lb_bits_subgauss.png` | adversarial matrix `v_i = (e_0 + e_i)/√2` with injected per-step error (Theorem-4.1 model); realizes `Ω(a·n²)` growth. Open markers flag cells violating `a < 1/(8n)`. |
| `norm_dists.py` | `norm_dists/norm_dists_<matrix>.png` | Mahalanobis-whitened norm distributions of `Bz` vs max-of-iid / χ_m references. |
| `qq_diagnostics.py` | `qq_diagnostics/qq_diagnostics_<matrix>.png` | Q–Q scatter, standardised tail, and density of `Bz` coords vs N(0,1). |
| `walk_step.py` | — | interactive single-walk step-through. |
| `scaling_collapse.py` | `scaling_collapse/scaling_collapse_<matrix>.png` | composite of the `n_subgauss` + `lb_subgauss` caches (no new rollouts): uncentered Ŝ = √(σ̂²+b̂²_max) collapse on `n·2^-b` vs `n²·2^-b`, with fitted `√(σ₀²+(cx)²)` and `k·x/√2` curves, the σ=1 guarantee line, and open markers outside the `a < 1/(8n)` precondition. |
| `ensemble_panels.py` | `ensemble_panels/ensemble_panels.png` | composite 2×2 (no new rollouts): clustered and identity rows, worst-direction Ŝ vs n (left) and vs mantissa bits (right), from the corresponding `n_subgauss`/`bits_subgauss` caches. `--centered` plots plain σ̂. |

## Conventions

- `sig_bits` = mantissa fraction bits retained; `2^-sig_bits` is the rounding
  resolution. fp8-E5M2 ≙ 2, fp8-E4M3 ≙ 3, bf16 ≙ 7, fp16 ≙ 10, fp32 ≙ 23,
  fp64 ≙ 52.
- The clustered ensemble is `B = u·1ᵀ + m^{-1/2}·G` (Gaussian `G`, random unit
  `u`), columns normalized; `gaussian` is the plain i.i.d. ensemble. Higgs uses
  the two low-level angular-momentum features of the Higgs two-sample benchmark.
  All families live in `matrices.py`.
- σ̂ is estimated per direction three ways: a moment bound over `k ≤ 3`, a tail
  bound from empirical order statistics, and the ψ₂ Orlicz norm by bisection.
  The worst direction is selected on the first half of the walks and evaluated
  on the held-out half; σ is computed on centered samples and the bias
  (`|mean|`) is reported separately.
- Plots use matplotlib's `Agg` backend and save under `figures/<experiment>/`.
- The low-precision path is ~20× slower than fp64. Use the `workers` argument
  to parallelize; near-singular solves around `k ≈ m` emit warnings at very
  low bits, which is expected.

## `dtype_stability/`

A self-contained second study (`gsw_stability` package) using **true IEEE
float64/32/16 dtypes** rather than the mantissa-only model, comparing three
step-direction implementations (from-scratch QR least squares, the
Cholesky/Woodbury scheme of GSWDesign.jl, and the explicit-inverse
Sherman–Morrison scheme) under a shared walk skeleton with common random
numbers. Includes a pytest suite, resumable result grids, and a cited error
analysis in `stability_references.md`. See `dtype_stability/README.md`.

