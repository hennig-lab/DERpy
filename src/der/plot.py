# SPDX-License-Identifier: MPL-2.0
"""Optional diagnostic plot; detection does not depend on matplotlib."""


def plot_correlations(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = sorted(
        {(r["channel_a"], r["cluster_a"]) for r in result.correlations}
        | {(r["channel_b"], r["cluster_b"]) for r in result.correlations}
    )
    index = {key: i for i, key in enumerate(keys)}
    matrix = np.full((len(keys), len(keys)), np.nan)
    for r in result.correlations:
        i = index[r["channel_a"], r["cluster_a"]]
        j = index[r["channel_b"], r["cluster_b"]]
        matrix[i, j] = matrix[j, i] = r["z"]
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    if len(keys):
        im = ax.imshow(matrix, vmin=-5, vmax=30, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Central-bin z-score")
        if len(keys) <= 50:
            ticks = [f"{ch}:{cl}" for ch, cl in keys]
            ax.set_xticks(range(len(keys)), ticks, rotation=90)
            ax.set_yticks(range(len(keys)), ticks)
    ax.set(xlabel="Channel:cluster", ylabel="Channel:cluster", title="DER zero-lag cross-correlations")
    fig.savefig(path, dpi=200)
    plt.close(fig)
