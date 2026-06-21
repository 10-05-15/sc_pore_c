



def compute_interaction_distance_cardinality(data_dict, max_diag=None):
    """
    For each genomic distance (diagonal offset), count the number of 
    non-zero contacts in each matrix, normalized by the number of 
    possible contacts at that distance.

    Parameters:
        data_dict:  dict with two matrix entries {label: matrix}
        max_diag:   int, maximum diagonal offset to consider.
                    Defaults to full matrix size - 1.

    Returns:
        distances:      1D array of diagonal offsets
        counts:         dict {label: 1D array of raw non-zero counts per diagonal}
        norm_counts:    dict {label: 1D array of normalized counts per diagonal}
        possible:       1D array of possible contacts at each diagonal offset
    """
    assert len(data_dict) == 2, "Dictionary must contain exactly two matrices."

    labels  = list(data_dict.keys())
    matrices = list(data_dict.values())

    # Convert sparse to dense if needed
    dense = []
    for mat in matrices:
        dense.append(mat.toarray() if sp.issparse(mat) else np.array(mat))

    assert dense[0].shape == dense[1].shape, (
        f"Shape mismatch: {dense[0].shape} vs {dense[1].shape}"
    )

    n        = dense[0].shape[0]
    max_diag = max_diag or (n - 1)

    distances = np.arange(0, max_diag + 1)

    # Number of possible contacts shrinks by 1 at each diagonal offset
    possible = np.array([n - d for d in distances], dtype=float)

    counts      = {label: lnp.zeros(len(distances), dtype=int)  for label in labels}
    norm_counts = {label: np.zeros(len(distances), dtype=float) for label in labels}

    for d in distances:
        for label, mat in zip(labels, dense):
            diag_vals         = mat.ravel()[d::n + 1][:n - d]   # fast strided indexing
            counts[label][d]  = np.count_nonzero(diag_vals)

        # Normalize by possible contacts at this distance
        for label in labels:
            norm_counts[label][d] = counts[label][d] / possible[d]

    return distances, counts, norm_counts, possible

def plot_interaction_distance_cardinality(distances, norm_counts, chrom="chr1", log_scale=True):
    """
    Two-panel plot:
        Top:    Normalized non-zero contact fraction vs genomic distance
        Bottom: Difference panel (label1 - label2) per diagonal offset

    Parameters:
        distances:    1D array of diagonal offsets
        norm_counts:  dict {label: 1D array of normalized counts}
        chrom:        chromosome label for plot title
        log_scale:    bool, log scale on y-axis of top panel
    """
    labels  = list(norm_counts.keys())
    colors  = ["steelblue", "crimson"]

    label1, label2  = labels[0], labels[1]
    counts1, counts2 = norm_counts[label1], norm_counts[label2]
    difference       = counts1 - counts2

    # --- Layout: top panel tall, bottom panel shorter ---
    fig, (ax_main, ax_diff) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # ── Top Panel: Normalized Cardinality ──────────────────────────────────
    for (label, cnt), color in zip(norm_counts.items(), colors):
        ax_main.plot(distances, cnt,
                     label=label,
                     color=color,
                     linewidth=1.8,
                     alpha=0.85)

    if log_scale:
        ax_main.set_yscale("log")
        ax_main.set_ylabel("Normalized contact fraction (log scale)", fontsize=9, fontweight='bold')
    else:
        ax_main.set_ylabel("Normalized contact fraction", fontsize=9, fontweight='bold')

    ax_main.set_title(
        f"Interaction Distance Cardinality",
        fontsize=13, fontweight="bold"
    )
    ax_main.legend(fontsize=11)
    ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

    # ── Bottom Panel: Difference ────────────────────────────────────────────
    ax_diff.axhline(0, color="black", linewidth=1.0, linestyle="--")

    # Fill positive (label1 > label2) and negative (label2 > label1) regions
    ax_diff.fill_between(distances, difference,
                         where=(difference >= 0),
                         color="steelblue", alpha=0.5,
                         label=f"scPoreC > scHiC")

    ax_diff.fill_between(distances, difference,
                         where=(difference < 0),
                         color="crimson", alpha=0.5,
                         label=f"scHiC > scPoreC")

    ax_diff.plot(distances, difference,
                 color="black", linewidth=0.8, alpha=0.6)

    ax_diff.set_ylabel(f"Difference", fontsize=9, fontweight='bold')
    ax_diff.set_xlabel("Genomic distance", fontsize=9, fontweight='bold')
    ax_diff.legend(fontsize=9, loc="upper right")
    ax_diff.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"interaction_distance_cardinality_{chrom}.png", dpi=150)
    plt.show()
    print("Plot saved.")


def compute_difference_matrix(data_dict, normalize=True, pseudocount=1):
    """
    Compute element-wise difference matrix (label1 - label2),
    optionally normalizing each matrix to [0, 1] before subtraction.

    Parameters:
        data_dict:   dict with two matrix entries {label: matrix}
        normalize:   bool, normalize each matrix before differencing
        pseudocount: int, added before normalization to avoid log(0)

    Returns:
        diff_matrix: 2D numpy array (label1 - label2)
        label1:      string, first dict key
        label2:      string, second dict key
    """
    assert len(data_dict) == 2, "Dictionary must contain exactly two matrices."

    labels   = list(data_dict.keys())
    matrices = list(data_dict.values())

    label1, label2 = labels[0], labels[1]

    # Convert sparse to dense
    dense = []
    for mat in matrices:
        dense.append(mat.toarray() if sp.issparse(mat) else np.array(mat, dtype=float))

    assert dense[0].shape == dense[1].shape, (
        f"Shape mismatch: {dense[0].shape} vs {dense[1].shape}"
    )

    if normalize:
        # Log1p normalize then scale to [0, 1]
        normed = []
        for mat in dense:
            m = np.log1p(mat + pseudocount - 1)
            m = (m - m.min()) / (m.max() - m.min() + 1e-10)
            normed.append(m)
        diff_matrix = normed[0] - normed[1]
    else:
        diff_matrix = dense[0] - dense[1]

    return diff_matrix, label1, label2


def plot_difference_heatmap(diff_matrix, label1, label2, chrom="chr1"):
    """
    Full 2D heatmap of element-wise difference (label1 - label2).
    Uses a diverging colormap centered at zero.
    """
    vmax = np.abs(diff_matrix).max()

    diff_matrix_alter = diff_matrix[3:, 3:]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        diff_matrix_alter,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="none",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label(f"scPoreC - scHiC (normalized)", fontsize=9)

    ax.set_title(
        f"Element-wise Difference Matrix",
        fontsize=9, fontweight="bold"
    )

     # ── Remove ticks, labels, and borders ──────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ───────────────────────────────────────────────────────────────────────


    plt.tight_layout()
    plt.savefig(f"difference_heatmap_2D.png", dpi=150)
    plt.show()

def plot_difference_triangle(diff_matrix, label1, label2, chrom="chr1"):
    """
    Plot only the upper triangle of the difference matrix,
    masking the lower triangle for a cleaner Hi-C style view.
    """
    # Mask lower triangle
    mask        = np.tril(np.ones_like(diff_matrix, dtype=bool), k=-1)
    diff_masked = np.where(mask, np.nan, diff_matrix)

    vmax = np.nanmax(np.abs(diff_masked))
    diff_masked_alter = diff_masked[3:, 3:]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        diff_masked_alter,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="none"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label(f"scPoreC - scHiC (normalized)", fontsize=9, fontweight = 'bold')

    ax.set_title(
        f"Difference Matrix",
        fontsize=9, fontweight="bold"
    )

     # ── Remove ticks, labels, and borders ──────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ──────────────────────────────────────────────────────────────────────

    plt.tight_layout()
    plt.savefig(f"difference_heatmap_triangle.png", dpi=150)
    plt.show()


