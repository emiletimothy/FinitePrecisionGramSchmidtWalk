"""Composite 2x2 ensemble panel: two matrix families x two sweeps.

Rows: clustered (top), identity (bottom). Columns: worst-direction
uncentered scale S-hat = sqrt(sigma^2 + bias^2) vs n (left) and vs
mantissa bits (right). Reads existing n_subgauss and bits_subgauss
stats caches -- no new rollouts.
"""

import argparse
import numpy as np

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import rollouts
from core.paths import figure_path


def _uncentered(stats, i=None, centered=False):
    sig = stats["sig_max"][i] if i is not None else stats["sig_max"]
    sig = np.asarray(sig, float)
    bias = stats.get("max_bias")
    if bias is None or centered:
        return sig
    bias = bias[i] if i is not None else bias
    return np.hypot(sig, np.asarray(bias, float))


def plot_panels(n_cl_cache, n_id_cache, b_cl_cache, b_id_cache,
                noise_std=0.0, save=None, centered=False):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    n_cl = rollouts.load_cache(n_cl_cache, matrix="clustered",
                               noise_std=noise_std)
    n_id = rollouts.load_cache(n_id_cache, matrix="identity",
                               noise_std=noise_std)
    b_cl = rollouts.load_cache(b_cl_cache, matrix="clustered",
                               noise_std=noise_std)
    b_id = rollouts.load_cache(b_id_cache, matrix="identity",
                               noise_std=noise_std)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cmap = plt.get_cmap("viridis")
    m_meta = rollouts.metadata_from_cache(n_cl).get("m")

    for row, (n_st, b_st, label) in enumerate([
            (n_cl, b_cl, f"clustered (m={m_meta})" if m_meta else "clustered"),
            (n_id, b_id, "identity (control)")]):

        scale = r"$\hat{\sigma}$" if centered else r"$\hat{S}$"

        # left: S vs n, one line per precision
        ax = axes[row][0]
        nval = np.asarray(n_st["n_values"], float)
        sb = np.asarray(n_st["sig_bits"], float)
        for j, b in enumerate(sb):
            color = "k" if b == 52 else cmap(0.15 + 0.8 * j / max(len(sb) - 1, 1))
            ax.plot(nval, _uncentered(n_st, j, centered), "-o", ms=4,
                    color=color, label="fp64" if b == 52 else f"{int(b)}")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(nval))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(int(v)) for v in nval])
        ax.set_xlabel("n"); ax.set_ylabel(scale + " (worst dir)")
        ax.set_title(f"{label}: {scale} vs n")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, title="bits", title_fontsize=7,
                      loc="upper left")

        # right: S vs bits at fixed n -- markers with a shaded 95% band
        ax = axes[row][1]
        bits = np.asarray(b_st["sig_bits"], float)
        med = _uncentered(b_st, centered=centered)
        err = 1.96 * np.asarray(b_st.get("sig_max_err",
                                np.zeros_like(b_st["sig_max"])), float)
        ax.plot(bits, med, "o-", ms=5)
        ax.fill_between(bits, med - err, med + err, alpha=0.2)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5,
                   label="guarantee $\\sigma=1$")
        ax.set_xlabel("mantissa bits")
        ax.set_ylabel(scale + " (worst dir)")
        n_meta = rollouts.metadata_from_cache(b_st).get("matrix_n")
        ax.set_title(f"{label}: {scale} vs bits (n={n_meta})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Worst-direction centered scale $\\hat{\\sigma}$ across "
                 "ensembles" if centered else
                 "Worst-direction uncentered scale "
                 "$\\hat{S}=\\sqrt{\\hat{\\sigma}^2+\\hat{b}^2}$ "
                 "across ensembles", y=0.995)
    fig.tight_layout()
    if save:
        from pathlib import Path
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        rollouts.save_figure(fig, save, dict(script="ensemble_panels",
                                             noise_std=noise_std))
        print("saved", save)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cluster-cache",
                   default=figure_path(
                       "n_subgauss", "n_subgauss_clustered_noise0_cache.npz"))
    p.add_argument("--n-identity-cache",
                   default=figure_path(
                       "n_subgauss", "n_subgauss_identity_noise0_cache.npz"))
    p.add_argument("--bits-cluster-cache",
                   default=figure_path(
                       "bits_subgauss",
                       "bits_subgauss_clustered_noise0_cache.npz"))
    p.add_argument("--bits-identity-cache",
                   default=figure_path(
                       "bits_subgauss",
                       "bits_subgauss_identity_noise0_cache.npz"))
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--centered", action="store_true",
                   help="plot centered worst-direction sigma-hat instead of "
                        "the uncentered S-hat")
    p.add_argument("--save", default=None)
    a = p.parse_args()
    plot_panels(a.n_cluster_cache, a.n_identity_cache,
                a.bits_cluster_cache, a.bits_identity_cache,
                a.noise_std, a.save, a.centered)


if __name__ == "__main__":
    main()
