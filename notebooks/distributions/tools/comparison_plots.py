import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import glob
import time
import scipy
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
import anndata as an
import scanpy as sc
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import matplotlib.patches as mpatches
import networkx as nx
import random
from importlib import reload
import warnings
import ot
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from itertools import chain
import re
import pickle as pkl
from sklearn.decomposition import TruncatedSVD
import pyarrow
from sklearn.preprocessing import MinMaxScaler
import cooler
import pairtools
from matplotlib_venn import venn2

"""WARNING: no warnings"""
warnings.filterwarnings("ignore")

source_path = os.path.abspath("../utilities/")
sys.path.append(source_path)
import matrix as mtrx
import utils as ut
#import plotting as plt2
source_path = os.path.abspath("../utilities/calculations/")
sys.path.append(source_path)
import centrality as central



def clique_expand(H):
    """H is (n_nodes, n_edges) incidence matrix -> returns (n_nodes, n_nodes) contact matrix."""
    if hasattr(H, 'toarray'):
        H = H.tocsr()
    return (H @ H.T)


def add_corner_colorbar(fig, ax, im, label, fontsize=7):
    """
    Add a small colorbar anchored to the bottom-right corner of an axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The parent axes the colorbar should sit inside.
    im : mappable
        The image/mappable to attach the colorbar to.
    label : str
        Colorbar label.
    fontsize : int
        Font size for colorbar tick labels and title.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
    """
    # Place a small inset axes in the bottom-right corner of the parent axes
    # [x0, y0, width, height] in axes-fraction coordinates
    cax = ax.inset_axes([0.72, 0.02, 0.25, 0.03])          # wide & short → horizontal bar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(label, fontsize=fontsize, labelpad=2)
    cbar.ax.tick_params(labelsize=fontsize - 1, length=2, pad=1)

    # Make the colorbar frame stand out slightly against the heatmap
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.8)

    return cbar

def pad_to_shape_centered(mat, target_shape):
    """Pad a 2-D numpy array with zeros, centering the original content."""
    result = np.zeros(target_shape, dtype=float)
    row_offset = (target_shape[0] - mat.shape[0]) 
    col_offset = (target_shape[1] - mat.shape[1])
    result[
        row_offset : row_offset + mat.shape[0],
        col_offset : col_offset + mat.shape[1]
    ] = mat
    return result


def plot_panel_contact(
    sparse_matrix,
    numpy_matrix,
    spath,
    titles,
    log_transform=True,
    cmap='Reds',
    dpi=200,
    figsize=(7, 7),
    xticklabel_interval=5,
    yticklabel_interval=5,
):
    """
    Plot two contact matrices in a single heatmap split along the diagonal.
    The upper triangle shows the sparse matrix and the lower triangle shows
    the numpy matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.csr_matrix
        Sparse incidence matrix to clique-expand (H @ H^T).
    numpy_matrix : np.ndarray
        Pre-computed contact matrix (plotted as-is).
    spath : str
        Output file path for the saved figure.
    titles : tuple of str
        Labels for (upper-triangle, lower-triangle), used in a legend.
    log_transform : bool
        If True, apply np.log1p to both matrices before plotting.
    cmap : str
        Colormap for both heatmaps.
    dpi : int
        Figure resolution.
    figsize : tuple
        Overall figure size (width, height) in inches.
    xticklabel_interval : int
        Show every Nth x-tick label.
    yticklabel_interval : int
        Show every Nth y-tick label.

    Returns
    -------
    ax : matplotlib.axes.Axes
    matrices : dict
        Raw (untransformed) matrices keyed by panel title.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not issparse(sparse_matrix):
        raise TypeError(
            f"sparse_matrix must be a scipy sparse matrix, got {type(sparse_matrix)}. "
            "Convert with scipy.sparse.csr_matrix(your_array)."
        )
    if not isinstance(numpy_matrix, np.ndarray):
        raise TypeError(
            f"numpy_matrix must be a numpy ndarray, got {type(numpy_matrix)}."
        )

    # ------------------------------------------------------------------
    # Build raw matrices
    # Force materialize any AnnData sparse view into a real scipy CSR
    # matrix BEFORE passing to clique_expand
    # ------------------------------------------------------------------
    if hasattr(sparse_matrix, 'toarray'):
        _H = sp.csr_matrix(sparse_matrix)
    else:
        _H = sparse_matrix

    _expanded = clique_expand(_H)

    # Guarantee we have a dense numpy array regardless of what
    # clique_expand returns
    if hasattr(_expanded, 'toarray'):
        contact_sparse = _expanded.toarray().astype(float)
    else:
        contact_sparse = np.asarray(_expanded, dtype=float)

    contact_np1 = numpy_matrix.astype(float)

    # ------------------------------------------------------------------
    # Pad to matching shape if matrices differ slightly in size
    # ------------------------------------------------------------------
    if contact_sparse.shape != contact_np1.shape:
        target = (
            max(contact_sparse.shape[0], contact_np1.shape[0]),
            max(contact_sparse.shape[1], contact_np1.shape[1]),
        )
        if contact_sparse.shape != target:
            contact_sparse = pad_to_shape_centered(contact_sparse, target)
        if contact_np1.shape != target:
            contact_np1 = pad_to_shape_centered(contact_np1, target)

    raw_matrices = {
        titles[0]: contact_sparse,
        titles[1]: contact_np1,
    }

    # ------------------------------------------------------------------
    # Optional log transform
    # ------------------------------------------------------------------
    if log_transform:
        m_upper = np.log1p(contact_sparse)
        m_lower = np.log1p(contact_np1)
        cbar_label = 'log(1 + contacts)'
    else:
        m_upper = contact_sparse.copy()
        m_lower = contact_np1.copy()
        cbar_label = 'Contacts'

    # ------------------------------------------------------------------
    # Build the composite matrix
    #   - upper triangle (k >= 1) taken from m_upper
    #   - lower triangle (k <= -1) taken from m_lower
    #   - diagonal (k = 0) averaged for a smooth transition
    # ------------------------------------------------------------------
    n = m_upper.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    diag_mask  = np.eye(n, dtype=bool)

    composite = np.zeros((n, n), dtype=float)
    composite[upper_mask] = m_upper[upper_mask]
    composite[lower_mask] = m_lower[lower_mask]
    composite[diag_mask]  = (np.diag(m_upper) + np.diag(m_lower)) / 2.0
    composite2 = composite[3:, 3:]

    # ------------------------------------------------------------------
    # Global style
    # ------------------------------------------------------------------
    plt.rcParams.update({
        'figure.dpi':     dpi,
        'axes.edgecolor': 'black',
        'axes.linewidth': 2.0,
        'font.family':    'sans-serif',
    })

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ------------------------------------------------------------------
    # Draw composite heatmap
    # ------------------------------------------------------------------
    sns.heatmap(
        composite2,
        ax=ax,
        cmap=cmap,
        square=True,
        cbar=False,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
    )

    # ------------------------------------------------------------------
    # Draw diagonal dividing line
    # ------------------------------------------------------------------
    ax.plot([0, n], [0, n], color='black', linewidth=1.2, linestyle='--')

    # ------------------------------------------------------------------
    # Triangle labels as a text legend in the corners
    # ------------------------------------------------------------------
    # ax.text(
    #     0.97, 0.03, titles[1],
    #     transform=ax.transAxes,
    #     ha='left', va='bottom',
    #     fontsize=9, fontweight='bold',
    #     color='black',
    #     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
    # )
    # ax.text(
    #     0.03, 0.97, titles[0],
    #     transform=ax.transAxes,
    #     ha='right', va='top',
    #     fontsize=9, fontweight='bold',
    #     color='black',
    #     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
    # )

    ax.set_xlabel('', fontsize=9, fontweight='bold')
    ax.set_ylabel('', fontsize=9, fontweight='bold')

    # ------------------------------------------------------------------
    # Single small colorbar anchored below the bottom-right corner
    # ------------------------------------------------------------------
    plt.tight_layout()
    fig.canvas.draw()

    pos = ax.get_position()

    cbar_width  = pos.width  * 0.15
    cbar_height = 0.010
    cbar_left   = pos.x1 - cbar_width
    cbar_bottom = pos.y0 - 0.07

    cax  = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    im   = ax.collections[0]
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.8)

    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    plt.show()
    

    return ax, raw_matrices




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

def compute_contact_set_cardinality(data_dict):
    """
    Compute the set cardinality of non-zero contacts between two matrices.
    Identifies contacts unique to each matrix and shared by both.

    Parameters:
        data_dict:  dict with two matrix entries {label: matrix}

    Returns:
        sets:       dict with keys 'unique_label1', 'unique_label2', 'shared'
                    each containing a set of (i, j) tuples
        counts:     dict with integer counts for each category
        label1:     string, first dict key
        label2:     string, second dict key
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

    # Get non-zero indices as sets of (i, j) tuples
    nonzero1 = set(zip(*np.nonzero(dense[0])))
    nonzero2 = set(zip(*np.nonzero(dense[1])))

    # Compute set operations
    unique1 = nonzero1 - nonzero2
    unique2 = nonzero2 - nonzero1
    shared  = nonzero1 & nonzero2

    sets = {
        f"Unique to {label1}": unique1,
        f"Unique to {label2}": unique2,
        "Shared":              shared
    }

    counts = {key: len(val) for key, val in sets.items()}

    # Print summary
    total = len(nonzero1 | nonzero2)
    print(f"Total non-zero contacts (union): {total:,}")
    for key, val in counts.items():
        print(f"  {key}: {val:,}  ({100 * val / total:.1f}%)")

    return sets, counts, label1, label2


def plot_contact_barchart(counts, label1, label2, chrom="chr1",
                          xlabel_names=None, bar_spacing=0.6):
    """
    Grouped bar chart showing unique and shared non-zero contact counts.

    Parameters
    ----------
    counts       : dict of category -> count
    label1       : first dataset label
    label2       : second dataset label
    chrom        : chromosome name (used in output filename)
    xlabel_names : list of custom x-axis tick labels
    bar_spacing  : distance between bar centers (default 0.6, lower = closer)
    """
    categories = list(counts.keys())
    values     = list(counts.values())
    colors     = ["dodgerblue", "crimson", "mediumorchid"]
    total      = sum(values)

    # Custom x positions based on spacing
    x_pos = np.array([i * bar_spacing for i in range(len(categories))])

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(x_pos, values, color=colors, edgecolor="white", width=0.5)

    # Annotate bars with counts and percentages
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}\n({100 * val / total:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight='bold'
        )

    # Fit plot window to new positions
    ax.set_xlim(x_pos[0] - bar_spacing * 0.75, x_pos[-1] + bar_spacing * 0.75)
    ax.set_ylim(0, max(values) * 1.2)

    if xlabel_names:
        if len(xlabel_names) != len(categories):
            raise ValueError(
                f"xlabel_names length ({len(xlabel_names)}) must match "
                f"number of categories ({len(categories)})"
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabel_names, fontsize=9, fontweight='bold')

    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)  # optional: clean up bottom spine too

    plt.tight_layout()
    plt.savefig(f"contact_barchart_{chrom}.png", dpi=150)
    plt.show()

def plot_contact_venn(sets, counts, label1, label2, chrom="chr1"):
    """
    Venn diagram of non-zero contact overlap between two matrices.
    Requires: pip install matplotlib-venn
    """
    unique1 = counts[f"Unique to {label1}"]
    unique2 = counts[f"Unique to {label2}"]
    shared  = counts["Shared"]

    fig, ax = plt.subplots(figsize=(7, 6))

    venn = venn2(
        subsets=(unique1, unique2, shared),
        set_labels=(label1, label2),
        set_colors=("dodgerblue", "crimson"),
        alpha=0.99,
        ax=ax
    )

    # Style the labels
    for text in venn.set_labels:
        if text:
            text.set_visible(False)

    for text in venn.subset_labels:
        if text:
            if text.get_text().strip() == '0':
                text.set_visible(False)
            else:
                text.set_fontsize(9)
                text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(f"contact_venn_{chrom}.png", dpi=150)
    plt.show()



