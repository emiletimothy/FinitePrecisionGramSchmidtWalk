"""Parallel Monte-Carlo rollouts of the Gram-Schmidt Walk.

Each rollout is an independent walk on a fixed B. They parallelize across *processes*:
the low-precision walk is Python/CPU-bound, so threads are GIL-limited — only separate
processes give a real speedup (which saturates near the physical-core count).

Every rollout is seeded from an independent numpy SeedSequence, so the rollouts are
statistically independent (and reproducible when a base `seed` is supplied). Without
this, forked workers would inherit one RNG state and produce identical walks.

Keep BLAS single-threaded (see the NUM_THREADS env vars set by the sweep scripts) so
N worker processes don't each spawn a thread pool and oversubscribe the cores.
"""

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gsw, lpla

# Per-worker globals, populated once by _init so B isn't re-sent with every task.
_B = _DIRS = _CHOP = _NOISE = _QIN = None


def _init(B, directions, sig_bits, noise_mean, noise_std, quantize_input):
    global _B, _DIRS, _CHOP, _NOISE, _QIN
    _B, _DIRS, _QIN = B, directions, quantize_input
    _CHOP = lpla.make_round(sig_bits) if sig_bits is not None else None
    has_noise = noise_mean != 0.0 or noise_std != 0.0
    _NOISE = (noise_mean, noise_std) if has_noise else None


def _rollout(seed_seq):
    # Independent global RNG state per task; SeedSequence keeps the streams decorrelated.
    np.random.seed(seed_seq.generate_state(8))
    noise = None
    if _NOISE is not None:
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        noise = lambda size: rng.normal(*_NOISE, size=size)
    r = gsw.gram_schmidt_walk(_B, chop=_CHOP, noise=noise, quantize_input=_QIN)
    # final Bz is full float64; discrepancy is mean of |Bz|, subgaussianity uses signed
    # projections; Bz_hat scores the same signs against the matrix the walk saw
    return np.abs(r.Bz).mean(), _DIRS @ r.Bz, r.Bz, r.Bz_hat


def default_workers():
    # Half the logical cores ≈ physical cores, where the CPU-bound speedup plateaus.
    return max(1, (os.cpu_count() or 2) // 2)


def run_samples(B, directions, num_samples, *, sig_bits=None,
                noise_mean=0.0, noise_std=0.0, workers=None, seed=None,
                quantize_input=True):
    """Run `num_samples` walks on B; return (bz_means, projections, bz_samples, bz_hat_samples).

    bz_means:       shape (num_samples,)                 mean of |Bz| per rollout
    projections:    shape (len(directions), num_samples) directions @ Bz per rollout
    bz_samples:     shape (m, num_samples)               Bz = B0 @ z per rollout
    bz_hat_samples: shape (m, num_samples)               B_hat @ z per rollout, where
                    B_hat is the (possibly rounded) matrix the walk solved against.
                    bz_samples - bz_hat_samples isolates input quantization.

    chop mode (sig_bits) and noise (noise_mean/noise_std) may be combined — mutual
    exclusion, if wanted, is the caller's policy. workers=1 runs inline (no pool).
    """
    validate_noise(noise_mean, noise_std)
    print(f"Gaussian noise: mean={noise_mean:g}, std={noise_std:g}; seed={seed}")
    workers = default_workers() if workers is None else workers
    seed_seqs = np.random.SeedSequence(seed).spawn(num_samples)

    if workers == 1:
        _init(B, directions, sig_bits, noise_mean, noise_std, quantize_input)
        state = np.random.get_state()
        try:
            results = [_rollout(ss) for ss in seed_seqs]
        finally:
            np.random.set_state(state)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init,
            initargs=(B, directions, sig_bits, noise_mean, noise_std, quantize_input),
        ) as ex:
            chunk = max(1, num_samples // (workers * 4))
            results = list(ex.map(_rollout, seed_seqs, chunksize=chunk))

    bz_means = np.array([m for m, _, _, _ in results])
    projections = np.stack([p for _, p, _, _ in results], axis=1)
    bz_samples = np.stack([b for _, _, b, _ in results], axis=1)
    bz_hat_samples = np.stack([b for _, _, _, b in results], axis=1)
    return bz_means, projections, bz_samples, bz_hat_samples


RNG_SCHEME = "per-rollout-seedsequence/separate-gaussian-v1"


def validate_noise(mean, std):
    if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
        raise ValueError("Gaussian mean must be finite and standard deviation finite and nonnegative")


def noise_std_value(value):
    value = float(value)
    validate_noise(0.0, value)
    return value


def add_noise_arguments(parser, *, scope="all runs, including the reference"):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--noise-std", type=noise_std_value, default=0.0,
                       help=f"added Gaussian standard deviation for {scope} (default: 0)")
    group.add_argument("--no-noise", dest="noise_std", action="store_const", const=0.0,
                       help=f"disable added Gaussian noise for {scope}")


def output_path(path, noise_std):
    validate_noise(0.0, noise_std)
    path = Path(path)
    if path.suffix != ".png":
        raise ValueError(f"figure path must end in .png, got {path}")
    return str(path.with_name(f"{path.stem}_noise{noise_std:.17g}{path.suffix}"))


def experiment_metadata(noise_std, *, seed=0, noise_mean=0.0, **settings):
    validate_noise(noise_mean, noise_std)
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return dict(settings, schema_version=1, noise_std=float(noise_std), noise_mean=float(noise_mean),
                seed=seed, rng_scheme=RNG_SCHEME, git_revision=revision, git_dirty=dirty,
                created_utc=datetime.now(timezone.utc).isoformat())


def save_cache(path, metadata, **arrays):
    np.savez(path, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)


def load_cache(path, **expected):
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{path}: missing provenance metadata; regenerate rather than relabel this cache")
        metadata = json.loads(str(archive["metadata"]))
        required = {"noise_mean": 0.0, **expected, "schema_version": 1, "rng_scheme": RNG_SCHEME}
        mismatches = [key for key, value in required.items() if key not in metadata or metadata[key] != value]
        if mismatches:
            raise ValueError(f"{path}: cache settings differ for {', '.join(mismatches)}; use matching settings or regenerate")
        return {key: archive[key] for key in archive.files}


def metadata_from_cache(cache):
    return json.loads(str(cache["metadata"]))


def save_figure(fig, path, metadata):
    fig.savefig(path, dpi=150, metadata={"Description": json.dumps(metadata, sort_keys=True)})
