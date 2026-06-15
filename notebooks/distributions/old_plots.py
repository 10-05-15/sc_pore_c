import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import glob
import time
#import gget
import scipy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
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
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from itertools import chain
import re
import pickle as pkl
import HAT
from HAT import draw

def merge_preserve_order(index_a, index_b):
    """
    Merge two genomic indices, preserving and enforcing genomic sort order.
    
    Chromosomes are ordered chr1-chr19, chrX.
    Within each chromosome, positions are sorted numerically.
    
    Any label present in either index appears exactly once in the output,
    inserted at its correct genomic position rather than appended at the end.
    """

    def chrom_sort_key(label):
        """
        Returns a (chrom_rank, position) tuple for genomic sorting.
        Expects labels in the format 'chr1:12345' or similar.
        """
        chrom_order = {f'chr{i}': i for i in range(1, 20)}
        chrom_order['chrX'] = 20  # X comes last

        parts = label.split(':')
        chrom = parts[0]

        # Numeric position — handle missing or non-numeric gracefully
        try:
            pos = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            pos = 0

        # Unknown chromosomes sort to the very end
        rank = chrom_order.get(chrom, 999)
        return (rank, pos)

    # Union of both indices, deduplicated
    seen   = set()
    merged = []
    for label in list(index_a) + list(index_b):
        if label not in seen:
            seen.add(label)
            merged.append(label)

    # Re-sort the unified list according to genomic coordinates
    merged.sort(key=chrom_sort_key)
    return merged



def genomic_incidence_plot_main(
    incidence_df,
    spath,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    row_colors=None,
    node_labels=True,
    col_labels=False,
    figsize=(16, 8),
    row_spacing=1.0,
    col_spacing=1.0,        
):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    matrix = incidence_df.values
    n, m = matrix.shape

    fig, ax = plt.subplots(figsize=figsize)

    ###### Extract chromosome group from each row index ###### 
    row_index = incidence_df.index.tolist()
    chrom_labels = [label.split(':')[0] for label in row_index]

    seen = {}
    unique_chroms = []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)

    n_chroms = len(unique_chroms)

    ###### Assign a color to each unique chromosome ###### 
    if row_colors is None:
        cmap = plt.cm.tab10 if n_chroms <= 10 else plt.cm.tab20
        color_list = cmap(np.linspace(0, 1, n_chroms))
        chrom_to_color = {chrom: color_list[i] for i, chrom in enumerate(unique_chroms)}
        row_colors = np.array([chrom_to_color[c] for c in chrom_labels])
    else:
        row_colors = np.asarray(row_colors)
        chrom_to_color = None

    ######  Plot scatter + connecting lines ###### 
    for i in range(m):
        y_pts = np.where(matrix[:, i] != 0)[0]

        if len(y_pts) == 0:
            continue

        x_scaled = i * col_spacing              
        y_scaled = y_pts * row_spacing

        ax.scatter(
            x_scaled * np.ones(len(y_pts)),  
            y_scaled,
            color='k',
            edgecolor='k',
            linewidths=0.3,
            s=10,
            zorder=2
        )

        if connect_nodes and len(y_scaled) > 1:
            ax.plot(
                [x_scaled, x_scaled],        
                [y_scaled.min(), y_scaled.max()],
                c='k',
                lw=0.8,
                zorder=1
            )

    ###### Chromosome-colored background bands ###### 
    if shade_rows and chrom_to_color is not None:
        for chrom in unique_chroms:
            indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
            row_min = (min(indices) - 0.5) * row_spacing
            row_max = (max(indices) + 0.5) * row_spacing
            band_height = row_max - row_min

            chrom_matrix_rows = matrix[indices, :]
            active_cols = np.where(chrom_matrix_rows.any(axis=0))[0]

            if len(active_cols) == 0:
                continue

            x_end = active_cols.max() * col_spacing + 0.5  
            band_width = x_end - (-0.5)

            ax.barh(
                (row_min + row_max) / 2,
                band_width,
                height=band_height,
                left=-0.5,
                color=chrom_to_color[chrom],
                alpha=0.15,
                zorder=0
            )

    ax.set_xlim([-0.5, (m - 0.5) * col_spacing])
    ax.set_ylim([-0.5, (n - 0.5) * row_spacing])

    ###### Y-axis: grouped chromosome labels with "chr" prefix stripped ###### 
    if node_labels:
        group_tick_positions = []
        group_tick_labels = []
    
        for chrom in unique_chroms:
            indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
            midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
            group_tick_positions.append(midpoint)
            group_tick_labels.append(chrom.replace('chr', '').replace('Chr', ''))
    
        # Only apply truncation if there are more than 4 labels
        if len(group_tick_positions) > 4:
            ellipsis_pos = (group_tick_positions[2] + group_tick_positions[-1]) / 5
    
            display_positions = group_tick_positions[:3] + [ellipsis_pos] + [group_tick_positions[-1]]
            display_labels    = group_tick_labels[:3]    + ['⋮']          + [group_tick_labels[-1]]
        else:
            display_positions = group_tick_positions
            display_labels    = group_tick_labels
    
        ax.set_yticks(display_positions)
        ax.set_yticklabels(display_labels, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', length=0)
    else:
        ax.set_yticks([])

    ######  X-axis ###### 
    if col_labels:
        ax.set_xticks([i * col_spacing for i in range(m)]) 
        ax.set_xticklabels(
            incidence_df.columns.tolist(),
            rotation=90,
            fontsize=6
        )
    else:
        ax.set_xticks([])

    ###### Dropping xLabel, yLabel, and the Title ###### 

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')

def genomic_degree_histogram(
    incidence_df,
    spath,
    dpi=200,
    figsize=(16, 5),
):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50


    ###### Compute normalized degree for each locus ######
    # Degree = number of hyperedges a locus participates in
    # Normalized = degree / total number of hyperedges
    matrix = incidence_df.values
    n, m = matrix.shape

    degrees = matrix.sum(axis=1)                      # raw degree per locus
    norm_degrees = degrees / m                        # normalized

    row_index = incidence_df.index.tolist()
    chrom_labels = [label.split(':')[0] for label in row_index]

    ###### Build ordered unique chromosome list ######
    seen = {}
    unique_chroms = []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)

    n_chroms = len(unique_chroms)

    ###### Assign colors ######
    cmap = plt.cm.tab10 if n_chroms <= 10 else plt.cm.tab20
    color_list = cmap(np.linspace(0, 1, n_chroms))
    chrom_to_color = {chrom: color_list[i] for i, chrom in enumerate(unique_chroms)}

    ###### Plot ######
    fig, ax = plt.subplots(figsize=figsize)

    # Draw one bar per locus, colored background by chromosome group
    bar_width = 1.0

    for i, (locus, nd) in enumerate(zip(row_index, norm_degrees)):
        chrom = chrom_labels[i]

        # Black bar for the degree value
        ax.bar(
            i,
            nd,
            width=bar_width,
            color='black',
            linewidth=0,       # no edge so white gap comes purely from spacing
            zorder=2
        )

    ###### Chromosome-colored background bands ######
    for chrom in unique_chroms:
        indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
        x_min = min(indices) - 0.5
        x_max = max(indices) + 0.5
        band_width = x_max - x_min

        ax.axvspan(
            x_min,
            x_max,
            color=chrom_to_color[chrom],
            alpha=0.15,
            zorder=0
        )

    ###### White separators between individual loci ######
    # Achieved by setting bar width slightly less than 1 so gaps appear
    # Re-draw bars with a small gap (width=0.85 leaves visible white space)
    ax.cla()    # clear and redraw cleanly with gap width

    for i, (locus, nd) in enumerate(zip(row_index, norm_degrees)):
        ax.bar(
            i,
            nd,
            width=0.85,        # gap between bars acts as white separator
            color='black',
            linewidth=0,
            zorder=2
        )

    # Redraw chromosome bands after cla()
    for chrom in unique_chroms:
        indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
        x_min = min(indices) - 0.5
        x_max = max(indices) + 0.5

        ax.axvspan(
            x_min,
            x_max,
            color=chrom_to_color[chrom],
            alpha=0.15,
            zorder=0
        )

    ###### Mean degree line ######
    mean_nd = norm_degrees.mean()
    ax.axhline(
        mean_nd,
        color='red',
        linestyle='--',
        linewidth=1.2,
        zorder=3,
        #label=f'Mean = {mean_nd:.3f}'
    )

    ###### Axes formatting ######
    ax.set_xlim([-0.5, n - 0.5])
    ax.set_ylim([0, norm_degrees.max() * 1.15])   # headroom above tallest bar

    # X-ticks: one per chromosome group, centered, label stripped of "chr"
    group_tick_positions = []
    group_tick_labels = []
    for chrom in unique_chroms:
        indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
        midpoint = (min(indices) + max(indices)) / 2.0
        group_tick_positions.append(midpoint)
        group_tick_labels.append(chrom.replace('chr', '').replace('Chr', ''))

    ax.tick_params(axis='both', length=0)
        
    ax.tick_params(axis='both', which='both', labelsize=0)

    plt.tight_layout()
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')

    return ax



def genomic_incidence_plot_dual_v1(
    incidence_df_a,
    incidence_df_b,
    spath,
    shared_dict=None,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    node_labels=True,
    col_labels=False,
    figsize=(16, 8),
    row_spacing=1.0,
    col_spacing=1.0,
    alpha_scatter=0.85,
):
    """
    Plot two incidence matrices on the same axes.

    Columns are ordered by chromosome group, interleaving one read from
    matrix A then one from matrix B per chromosome. If one matrix has more
    reads for a chromosome than the other, the remainder are plotted
    consecutively before moving to the next chromosome.

    Parameters
    ----------
    incidence_df_a : pd.DataFrame
        First incidence matrix (UUID column names).
    incidence_df_b : pd.DataFrame
        Second incidence matrix (UUID column names).
    shared_dict : dict, optional
        Dictionary where each value is a collection of row-index label sets
        defining shared reads. A column is highlighted if its non-zero row
        indices are an EXACT match to any entry.
        Example: {0: ['chr1:0', 'chr3:50'], 1: ['chr2:25']}
    """

    SHARED_COLOR = '#39FF14'  # Neon green

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.00

    # ── Align both matrices to a shared row universe ──────────────────────────
    all_rows     = merge_preserve_order(incidence_df_a.index, incidence_df_b.index)
    all_rows_arr = np.array(all_rows)

    df_a = incidence_df_a.reindex(index=all_rows, fill_value=0)
    df_b = incidence_df_b.reindex(index=all_rows, fill_value=0)

    n_rows        = len(all_rows)
    matrix_a_full = df_a.values   # (n_rows, n_cols_a)
    matrix_b_full = df_b.values   # (n_rows, n_cols_b)

    # ── Extract chromosome groups from row index ──────────────────────────────
    chrom_labels = [label.split(':')[0] for label in all_rows]

    seen, unique_chroms = {}, []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)
    n_chroms = len(unique_chroms)

    # chrom -> sorted list of row indices belonging to it
    chrom_to_row_indices = {
        chrom: [i for i, c in enumerate(chrom_labels) if c == chrom]
        for chrom in unique_chroms
    }

    # row index -> chrom
    row_to_chrom = {i: c for i, c in enumerate(chrom_labels)}

    # chrom -> its earliest row index (used for tie-breaking)
    chrom_first_row = {
        chrom: min(idxs)
        for chrom, idxs in chrom_to_row_indices.items()
    }

    # ── Assign each column to exactly one chromosome ──────────────────────────
    # Rule: use the chromosome of the FIRST non-zero row in the column.
    # For multi-chromosome reads this anchors them to the earliest chromosome
    # they touch, which matches the natural row ordering of the data.
    def assign_chrom(col_vec):
        nonzero_rows = np.where(col_vec != 0)[0]
        if len(nonzero_rows) == 0:
            return None
        # Sort nonzero rows (np.where already returns sorted) and
        # take the chromosome of the very first hit
        return row_to_chrom[nonzero_rows[-1]]

    # ── Build per-chromosome column descriptor lists for A and B ─────────────
    # Preserves original DataFrame column order within each chromosome bucket
    chrom_cols_a = {ch: [] for ch in unique_chroms}
    chrom_cols_b = {ch: [] for ch in unique_chroms}
    
    def assign_chrom_info(col_vec):
        """
        Returns:
            primary_chrom : str   — chrom of the LAST nonzero row (for x-position)
            all_chroms    : list  — all unique chroms touched, in order of first hit
        """
        nonzero_rows = np.where(col_vec != 0)[0]
        if len(nonzero_rows) == 0:
            return None, []
        
        seen_c, all_chroms = set(), []
        for r in nonzero_rows:
            c = row_to_chrom[r]
            if c not in seen_c:
                seen_c.add(c)
                all_chroms.append(c)
        
        primary_chrom = row_to_chrom[nonzero_rows[0]]   # first hit for x-ordering
        return primary_chrom, all_chroms


    # ── Build per-chromosome column descriptor lists for A and B ─────────────────
    chrom_cols_a = {ch: [] for ch in unique_chroms}
    chrom_cols_b = {ch: [] for ch in unique_chroms}
    
    # Track which col_names have already been assigned to avoid duplicates
    for j, col_name in enumerate(df_a.columns):
        col_vec = matrix_a_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_a[primary_chrom].append({
            'source':     'a',
            'col_name':   col_name,
            'col_vec':    col_vec,
            'chrom':      primary_chrom,
            'all_chroms': all_chroms,     # ← carries full chrom membership
        })
    
    for j, col_name in enumerate(df_b.columns):
        col_vec = matrix_b_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_b[primary_chrom].append({
            'source':     'b',
            'col_name':   col_name,
            'col_vec':    col_vec,
            'chrom':      primary_chrom,
            'all_chroms': all_chroms,
        })

    # ── Interleave A and B columns per chromosome ─────────────────────────────
    # Pattern per chrom: A[0], B[0], A[1], B[1], ...
    # Whichever matrix has more reads for this chrom finishes alone at the end.
    plot_columns = []

    for chrom in unique_chroms:
        cols_a  = chrom_cols_a[chrom]
        cols_b  = chrom_cols_b[chrom]
        max_len = max(len(cols_a), len(cols_b), 0)

        for k in range(max_len):
            if k < len(cols_a):
                plot_columns.append(cols_a[k])
            if k < len(cols_b):
                plot_columns.append(cols_b[k])

    m_plot = len(plot_columns)

    # ── Build shared column mask by content matching ──────────────────────────
    shared_col_mask = np.zeros(m_plot, dtype=bool)

    if shared_dict:
        target_sets = {frozenset(reads) for reads in shared_dict.values()}

        for i, pcol in enumerate(plot_columns):
            nonzero_rows = np.where(pcol['col_vec'] != 0)[0]
            col_set      = frozenset(all_rows_arr[nonzero_rows])
            if col_set in target_sets:
                shared_col_mask[i] = True

    # ── Build per-row colors ──────────────────────────────────────────────────
    def make_row_colors(color_shift=0.0):
        if n_chroms <= 20:
            # Use discrete tab20 indices directly to avoid pale colors
            cmap = plt.cm.get_cmap('tab20', 20)
            color_list = [cmap(i % 20) for i in range(n_chroms)]
        else:
            cmap = plt.cm.hsv
            color_list = [
                cmap((i / max(n_chroms, 1) + color_shift) % 1.0)
                for i in range(n_chroms)
            ]
        chrom_to_color = {chrom: color_list[i]
                          for i, chrom in enumerate(unique_chroms)}
        row_colors = np.array([chrom_to_color[c] for c in chrom_labels])
        return row_colors, chrom_to_color

    row_colors_a, chrom_to_color_a = make_row_colors(color_shift=0.0)
    row_colors_b, _                = make_row_colors(color_shift=0.0)

    fig, ax = plt.subplots(figsize=figsize)

    # ── Draw all plot columns ─────────────────────────────────────────────────
    for i, pcol in enumerate(plot_columns):
        col_vec    = pcol['col_vec']
        source     = pcol['source']
        row_colors = row_colors_a if source == 'a' else row_colors_b
        marker     = 'o'          if source == 'a' else 's'

        y_pts = np.where(col_vec != 0)[0]
        if len(y_pts) == 0:
            continue

        x_scaled = i * col_spacing
        y_scaled = y_pts * row_spacing

        if shared_col_mask[i]:
            colors     = [SHARED_COLOR] * len(y_pts)
            line_color = SHARED_COLOR
        else:
            colors     = row_colors[y_pts]
            line_color = 'k'

        ax.scatter(
            x_scaled * np.ones(len(y_pts)),
            y_scaled,
            color=colors,
            edgecolor='k',
            linewidths=0.3,
            s=14,
            marker=marker,
            alpha=alpha_scatter,
            zorder=2,
        )

        if connect_nodes and len(y_scaled) > 1:
            ax.plot(
                [x_scaled, x_scaled],
                [y_scaled.min(), y_scaled.max()],
                c=line_color,
                lw=0.8,
                zorder=1,
            )

    # ── Background chromosome bands ───────────────────────────────────────────
    if shade_rows:
        for chrom in unique_chroms:
            row_idxs    = chrom_to_row_indices[chrom]
            row_min     = (min(row_idxs) - 0.5) * row_spacing
            row_max     = (max(row_idxs) + 0.5) * row_spacing
            band_height = row_max - row_min
    
            # ← check all_chroms instead of just pc['chrom']
            chrom_plot_col_idxs = [
                i for i, pc in enumerate(plot_columns)
                if chrom in pc['all_chroms']
            ]
            if not chrom_plot_col_idxs:
                continue
    
            x_left     = -0.5
            x_right    = max(chrom_plot_col_idxs) * col_spacing + 0.5
            band_width = x_right - x_left
    
            ax.barh(
                (row_min + row_max) / 2,
                band_width,
                height=band_height,
                left=x_left,
                color=chrom_to_color_a[chrom],
                alpha=0.12,
                zorder=0,
            )

    # ── Axes limits ───────────────────────────────────────────────────────────
    ax.set_xlim([-0.5, (m_plot - 0.6) * col_spacing])
    ax.set_ylim([-0.5, (n_rows - 0.6) * row_spacing])

    # ── Y-axis chromosome labels ──────────────────────────────────────────────
    if node_labels:
        group_tick_positions = []
        group_tick_labels = []
    
        for chrom in unique_chroms:
            indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
            midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
            group_tick_positions.append(midpoint)
            group_tick_labels.append(chrom.replace('chr', '').replace('Chr', ''))
    
        # Only apply truncation if there are more than 4 labels
        if len(group_tick_positions) > 4:
            ellipsis_pos = (group_tick_positions[2] + group_tick_positions[-1]) / 5
    
            display_positions = group_tick_positions[:3] + [ellipsis_pos] + [group_tick_positions[-1]]
            display_labels    = group_tick_labels[:3]    + ['⋮']          + [group_tick_labels[-1]]
        else:
            display_positions = group_tick_positions
            display_labels    = group_tick_labels
    
        ax.set_yticks(display_positions)
        ax.set_yticklabels(display_labels, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', length=0)
    else:
        ax.set_yticks([])
    # ── X-axis ────────────────────────────────────────────────────────────────
    if col_labels:
        ax.set_xticks([i * col_spacing for i in range(m_plot)])
        ax.set_xticklabels(
            [pc['col_name'] for pc in plot_columns],
            rotation=90, fontsize=6
        )
    else:
        ax.set_xticks([])

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')

    return ax


def genomic_incidence_plot_dual_v2(
    incidence_df_a,
    incidence_df_b,
    spath,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    node_labels=True,
    col_labels=False,
    figsize=(16, 8),
    row_spacing=1.0,
    col_spacing=1.0,
    alpha_scatter=0.85,
    label_a="Single Cell",
    label_b="Core",
    divider_color="black",
    divider_lw=1.5,
    divider_gap=3,
):
    plt.rcParams['figure.dpi'] = dpi

    # ── Align both matrices to a shared row universe ──────────────────────────
    all_rows = merge_preserve_order(incidence_df_a.index, incidence_df_b.index)

    cols_a = incidence_df_a.columns.tolist()
    cols_b = incidence_df_b.columns.tolist()
    m_a = len(cols_a)
    m_b = len(cols_b)

    # Reindex BOTH matrices against the same all_rows so row i in matrix_a
    # and row i in matrix_b always refer to the same genomic bin
    df_a = incidence_df_a.reindex(index=all_rows, columns=cols_a, fill_value=0)
    df_b = incidence_df_b.reindex(index=all_rows, columns=cols_b, fill_value=0)

    matrix_a = df_a.values
    matrix_b = df_b.values
    n = len(all_rows)

    # ── X positions ───────────────────────────────────────────────────────────
    x_positions_a = np.arange(m_a) * col_spacing
    x_start_b     = (m_a + divider_gap) * col_spacing
    x_positions_b = x_start_b + np.arange(m_b) * col_spacing

    divider_x_left  = (m_a - 0.5) * col_spacing
    divider_x_right = x_start_b - 0.5 * col_spacing

    # ── Extract chromosome groups from all_rows ───────────────────────────────
    # all_rows is the single source of truth for row ordering
    all_rows_list = list(all_rows)
    chrom_labels  = [label.split(':')[0] for label in all_rows_list]

    seen, unique_chroms = {}, []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)
    n_chroms = len(unique_chroms)

    # ── Per-chromosome colors ─────────────────────────────────────────────────
    cmap = plt.cm.tab20 if n_chroms <= 20 else plt.cm.hsv
    color_list     = [cmap(i / max(n_chroms, 1)) for i in range(n_chroms)]
    chrom_to_color = {chrom: color_list[i] for i, chrom in enumerate(unique_chroms)}

    # Build a color lookup by ROW LABEL (not position) so there is no
    # positional ambiguity between the two matrices
    row_label_to_color = {
        row_label: chrom_to_color[row_label.split(':')[0]]
        for row_label in all_rows_list
    }

    fig, ax = plt.subplots(figsize=figsize)

    # ── Helper: draw one matrix ───────────────────────────────────────────────
    def draw_matrix(matrix, x_positions):
        for i in range(matrix.shape[1]):
            y_pts = np.where(matrix[:, i] != 0)[0]
            if len(y_pts) == 0:
                continue

            x_val  = x_positions[i]
            y_vals = y_pts * row_spacing

            # Look up color by the actual row LABEL at each non-zero position
            # This is safe regardless of how the two matrices were aligned
            colors = [row_label_to_color[all_rows_list[row_idx]] for row_idx in y_pts]

            ax.scatter(
                x_val * np.ones(len(y_pts)),
                y_vals,
                color=colors,
                edgecolor='k',
                linewidths=0.3,
                s=14,
                marker='o',
                alpha=alpha_scatter,
                zorder=2,
            )

            if connect_nodes and len(y_vals) > 1:
                ax.plot(
                    [x_val, x_val],
                    [y_vals.min(), y_vals.max()],
                    c='k',
                    lw=0.8,
                    zorder=1,
                )

    draw_matrix(matrix_a, x_positions_a)
    draw_matrix(matrix_b, x_positions_b)

    # ── Divider bar ───────────────────────────────────────────────────────────
    ax.axvspan(
        divider_x_left,
        divider_x_right,
        color=divider_color,
        alpha=0.15,
        zorder=0,
        label='_nolegend_',
    )
    ax.axvline(
        x=(divider_x_left + divider_x_right) / 2.0,
        color=divider_color,
        linewidth=divider_lw,
        linestyle='-',
        zorder=3,
        label='_nolegend_',
    )

    # ── Background: alternating white/gray chromosome bands ───────────────────
    if shade_rows:
        for idx, chrom in enumerate(unique_chroms):
            indices     = [i for i, c in enumerate(chrom_labels) if c == chrom]
            row_min     = (min(indices) - 0.5) * row_spacing
            row_max     = (max(indices) + 0.5) * row_spacing
            band_height = row_max - row_min
            band_color  = 'lightgray' if idx % 2 == 0 else 'white'

            x_min_band = -0.5 * col_spacing
            x_max_band = (
                x_positions_b[-1] + 0.5 * col_spacing
                if m_b > 0
                else x_positions_a[-1] + 0.5 * col_spacing
            )
            band_width = x_max_band - x_min_band

            ax.barh(
                (row_min + row_max) / 2,
                band_width,
                height=band_height,
                left=x_min_band,
                color=band_color,
                alpha=1.0,
                zorder=0,
            )

    # ── Axes limits ───────────────────────────────────────────────────────────
    x_min = -0.5 * col_spacing
    x_max = (
        x_positions_b[-1] + 0.5 * col_spacing
        if m_b > 0
        else x_positions_a[-1] + 0.5 * col_spacing
    )
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-0.5 * row_spacing, (n - 0.5) * row_spacing])

    # ── Y-axis chromosome labels ──────────────────────────────────────────────
    if node_labels:
        ticks, tick_labels = [], []
        for chrom in unique_chroms:
            indices  = [i for i, c in enumerate(chrom_labels) if c == chrom]
            midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
            ticks.append(midpoint)
            tick_labels.append(chrom.replace('chr', '').replace('Chr', ''))
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=9)
    else:
        ax.set_yticks([])

    # ── X-axis: section labels via secondary axis ─────────────────────────────
    if col_labels:
        all_x    = np.concatenate([x_positions_a, x_positions_b])
        all_lbls = cols_a + cols_b
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_lbls, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    mid_a = (x_positions_a[0] + x_positions_a[-1]) / 2.0
    mid_b = (x_positions_b[0] + x_positions_b[-1]) / 2.0

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_position(('outward', 30))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.set_xticks([mid_a, mid_b])
    ax2.set_xticklabels([label_a, label_b], fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', length=0)


    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')

    return ax



def genomic_combined_dual_plot(
    incidence_df_a,
    incidence_df_b,
    spath,
    shared_color = '#39FF14',
    shared_dict=None,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    node_labels=True,
    col_labels=False,
    figsize=(22, 8),
    row_spacing=1.9,
    col_spacing=1.0,
    alpha_scatter=0.85,
    marker_size=10,          # base marker size
    point_padding=0.25,      # fraction of col_spacing to leave as whitespace (0–1)
    hist_width_ratio=0.07,   # fraction of total width given to EACH histogram
    panel_wspace=0.04,       # white space between panels
    x_offset=-1.0,
):
    """
    Plot layout (left to right):
        [hist of degree IM-A] [hist of degree IM-B] [dual incidence matrix]

    Y-axis chromosome labels are on the LEFT of hist-A.
    The two histograms share the same x-scale so they are visually comparable.
    A small wspace gap separates every panel.
    """

    SHARED_COLOR = shared_color

    plt.rcParams['figure.dpi']        = dpi
    plt.rcParams["axes.edgecolor"]    = "black"
    plt.rcParams["axes.linewidth"]    = 2.00

    ##### Align both matrices to a shared row universe ##### 
    all_rows     = merge_preserve_order(incidence_df_a.index, incidence_df_b.index)
    all_rows_arr = np.array(all_rows)

    df_a = incidence_df_a.reindex(index=all_rows, fill_value=0)
    df_b = incidence_df_b.reindex(index=all_rows, fill_value=0)

    n_rows        = len(all_rows)
    matrix_a_full = df_a.values
    matrix_b_full = df_b.values

    #####  Chromosome metadata ##### 
    chrom_labels = [label.split(':')[0] for label in all_rows]

    seen, unique_chroms = {}, []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)
    n_chroms = len(unique_chroms)

    chrom_to_row_indices = {
        chrom: [i for i, c in enumerate(chrom_labels) if c == chrom]
        for chrom in unique_chroms
    }
    row_to_chrom = {i: c for i, c in enumerate(chrom_labels)}

    # Shared colour map for rows
    def make_chrom_colors(color_shift=0.0):
        if n_chroms <= 20:
            cmap_fn    = plt.cm.get_cmap('tab20', 20)
            color_list = [cmap_fn(i % 20) for i in range(n_chroms)]
        else:
            cmap_fn    = plt.cm.hsv
            color_list = [
                cmap_fn((i / max(n_chroms, 1) + color_shift) % 1.0)
                for i in range(n_chroms)
            ]
        chrom_to_color = {ch: color_list[i] for i, ch in enumerate(unique_chroms)}
        row_colors     = np.array([chrom_to_color[c] for c in chrom_labels])
        return row_colors, chrom_to_color

    row_colors_a, chrom_to_color = make_chrom_colors()

    ##### Degree histograms ##### 
    m_a = matrix_a_full.shape[1]
    m_b = matrix_b_full.shape[1]

    norm_degrees_a = matrix_a_full.sum(axis=1) / m_a
    norm_degrees_b = matrix_b_full.sum(axis=1) / m_b

    # Shared x-limit so both histograms are on the same scale
    global_max_nd = max(norm_degrees_a.max(), norm_degrees_b.max()) * 1.15

    ##### Build plot_columns for the dual incidence matrix ##### 
    def assign_chrom_info(col_vec):
        nonzero_rows = np.where(col_vec != 0)[0]
        if len(nonzero_rows) == 0:
            return None, []
        seen_c, all_chroms = set(), []
        for r in nonzero_rows:
            c = row_to_chrom[r]
            if c not in seen_c:
                seen_c.add(c)
                all_chroms.append(c)
        primary_chrom = row_to_chrom[nonzero_rows[0]]
        return primary_chrom, all_chroms
    
    chrom_cols_a = {ch: [] for ch in unique_chroms}
    chrom_cols_b = {ch: [] for ch in unique_chroms}
    
    for j, col_name in enumerate(df_a.columns):
        col_vec = matrix_a_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_a[primary_chrom].append({
            'source':     'a',
            'col_name':   col_name,
            'col_vec':    col_vec,
            'chrom':      primary_chrom,
            'all_chroms': all_chroms,
        })

    for j, col_name in enumerate(df_b.columns):
        col_vec = matrix_b_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_b[primary_chrom].append({
            'source':     'b',
            'col_name':   col_name,
            'col_vec':    col_vec,
            'chrom':      primary_chrom,
            'all_chroms': all_chroms,
        })

    plot_columns = []
    for chrom in unique_chroms:
        cols_a  = chrom_cols_a[chrom]
        cols_b  = chrom_cols_b[chrom]
        max_len = max(len(cols_a), len(cols_b), 0)
        for k in range(max_len):
            if k < len(cols_a):
                plot_columns.append(cols_a[k])
            if k < len(cols_b):
                plot_columns.append(cols_b[k])

    m_plot = len(plot_columns)

    ##### Shared-column mask ##### 
    shared_col_mask = np.zeros(m_plot, dtype=bool)
    if shared_dict:
        target_sets = {frozenset(reads) for reads in shared_dict.values()}
        for i, pcol in enumerate(plot_columns):
            nonzero_rows = np.where(pcol['col_vec'] != 0)[0]
            col_set      = frozenset(all_rows_arr[nonzero_rows])
            if col_set in target_sets:
                shared_col_mask[i] = True

    #####  Figure creation using GridSpec (3 columns) ##### 
    matrix_ratio = 1.0 - 2 * hist_width_ratio

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[hist_width_ratio, hist_width_ratio, matrix_ratio],
        wspace=panel_wspace,
        figure=fig,
    )

    ax_hist_a = fig.add_subplot(gs[0])   # leftmost  – IM-A histogram
    ax_hist_b = fig.add_subplot(gs[1])   # middle    – IM-B histogram
    ax_matrix = fig.add_subplot(gs[2])   # rightmost – dual incidence matrix

    ##### Helper: draw a vertical (horizontal-bar) histogram ##### 
    data_pad_y  = 1.0 * row_spacing
    def draw_hist(ax, norm_deg, is_left_panel):
        bar_height = 0.85 * row_spacing
        mean_nd    = norm_deg.mean()

        for i, nd in enumerate(norm_deg):
            chrom    = chrom_labels[i]
            y_center = i * row_spacing
            ax.barh(
                y_center, nd,
                height=bar_height,
                color='black',
                linewidth=0,
                zorder=2,
            )

        y_axis_min = -data_pad_y - 1   # the extreme end of ylim (visual top after invert)
        y_axis_max = (n_rows - 1) * row_spacing + data_pad_y  # visual bottom after invert
        
        for idx, chrom in enumerate(unique_chroms):
            indices = chrom_to_row_indices[chrom]
            y_min   = min(indices) * row_spacing - 0.5 * row_spacing
            y_max   = max(indices) * row_spacing + 0.5 * row_spacing
        
            # Extend first chrom band all the way to the top of the axis
            if idx == 0:
                y_min = y_axis_min
        
            # Extend last chrom band all the way to the bottom of the axis
            if idx == len(unique_chroms) - 1:
                y_max = y_axis_max

            ax.axhspan(y_min, y_max,
                       color=chrom_to_color[chrom], alpha=0.15, zorder=0)

        # Mean line
        ax.axvline(mean_nd, color='red', linestyle='--',
                   linewidth=1.2, zorder=3)

        # Shared y / x limits
        ax.set_ylim([
            -data_pad_y - 1,
            (n_rows - 1) * row_spacing + data_pad_y
        ])
        ax.invert_yaxis()
        ax.set_xlim([global_max_nd, 0])   # bars grow right-to-left toward matrix

        # Suppress x ticks
        ax.set_xticks([])
        ax.tick_params(axis='x', length=0)

        # Y-axis chromosome labels only on the leftmost panel
        if is_left_panel and node_labels:
            group_tick_positions = []
            group_tick_labels    = []

            for chrom in unique_chroms:
                indices  = chrom_to_row_indices[chrom]
                midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
                group_tick_positions.append(midpoint)
                group_tick_labels.append(
                    chrom.replace('chr', '').replace('Chr', '')
                )

            if len(group_tick_positions) > 4:
                ellipsis_pos      = (group_tick_positions[2] + group_tick_positions[-1]) / 5
                display_positions = group_tick_positions[:3] + [ellipsis_pos] + [group_tick_positions[-1]]
                display_labels    = group_tick_labels[:3]    + ['⋮']          + [group_tick_labels[-1]]
            else:
                display_positions = group_tick_positions
                display_labels    = group_tick_labels

            ax.set_yticks(display_positions)
            ax.set_yticklabels(display_labels, fontsize=12, fontweight='bold')
            ax.yaxis.set_tick_params(length=0)
            ax.yaxis.set_label_position('left')
            ax.yaxis.tick_left()
        else:
            ax.set_yticks([])

        # Keep the right spine so it sits flush against the next panel
        ax.spines['right'].set_visible(True)

    draw_hist(ax_hist_a, norm_degrees_a, is_left_panel=True)
    draw_hist(ax_hist_b, norm_degrees_b, is_left_panel=False)

    ##### Dual incidence matrix ##### 
    spine_lw    = 2.0
    data_pad_x  = 0.5 * col_spacing   # padding left/right of first/last column
    data_pad_y  = 1.0 * row_spacing   # padding above/below first/last row

    row_colors_b, _ = make_chrom_colors()  # same colour shift = same palette

    fig_width_inches  = figsize[0]
    matrix_ratio      = 1.0 - 2 * hist_width_ratio
    ax_width_inches   = fig_width_inches * matrix_ratio
    
    x_range           = (x_offset + 2) + (m_plot - 1) * col_spacing + 0.5 * col_spacing - (x_offset - 0.5 * col_spacing)
    points_per_data   = (ax_width_inches * 72) / x_range          # 72 pts per inch
    col_spacing_pts   = col_spacing * points_per_data
    effective_s       = ((1.0 - point_padding) * col_spacing_pts) ** 2  # s is area in pts²

    for i, pcol in enumerate(plot_columns):
        col_vec    = pcol['col_vec']
        source     = pcol['source']
        row_colors = row_colors_a if source == 'a' else row_colors_b
        marker     = 'o'          if source == 'a' else 's'

        y_pts = np.where(col_vec != 0)[0]
        if len(y_pts) == 0:
            continue

        x_scaled = i * col_spacing
        y_scaled = y_pts * row_spacing

        if shared_col_mask[i]:
            colors     = [SHARED_COLOR] * len(y_pts)
            line_color = SHARED_COLOR
        else:
            colors     = 'k'
            line_color = 'k'

        ax_matrix.scatter(
            x_scaled * np.ones(len(y_pts)), y_scaled,
            color=colors, edgecolor='k', linewidths=0.3,
            s=effective_s, marker=marker, alpha=alpha_scatter, zorder=2,
        )

        if connect_nodes and len(y_scaled) > 1:
            ax_matrix.plot(
                [x_scaled, x_scaled],
                [y_scaled.min(), y_scaled.max()],
                c=line_color, lw=0.8, zorder=1,
            )

    # Background chromosome bands on the matrix
    if shade_rows:
        for chrom in unique_chroms:
            row_idxs    = chrom_to_row_indices[chrom]
            row_min     = (min(row_idxs) - 0.5) * row_spacing
            row_max     = (max(row_idxs) + 0.5) * row_spacing
            band_height = row_max - row_min + 0.5
    
            # ← all_chroms instead of pc['chrom']
            chrom_plot_col_idxs = [
                i for i, pc in enumerate(plot_columns)
                if chrom in pc['all_chroms']
            ]
            if not chrom_plot_col_idxs:
                continue
    
            x_left     = -1.0
            x_right    = max(chrom_plot_col_idxs) * col_spacing + 0.6
            band_width = x_right - x_left
    
            ax_matrix.barh(
                (row_min + row_max) / 2,
                band_width,
                height=band_height,
                left=x_left,
                color=chrom_to_color[chrom],
                alpha=0.12,
                zorder=0,
            )

    ax_matrix.set_xlim([
        x_offset - data_pad_x,                              # left boundary
        (x_offset + 2) + (m_plot - 1) * col_spacing + data_pad_x  # right boundary
    ])
    
    ax_matrix.set_ylim([
        -data_pad_y,                                         # top (pre-invert)
        (n_rows - 1) * row_spacing + data_pad_y              # bottom (pre-invert)
    ])
    ax_matrix.set_yticks([])
    ax_matrix.tick_params(axis='y', length=0)
    ax_matrix.spines['left'].set_visible(True)
    ax_matrix.invert_yaxis()

    if col_labels:
        ax_matrix.set_xticks([i * col_spacing for i in range(m_plot)])
        ax_matrix.set_xticklabels(
            [pc['col_name'] for pc in plot_columns],
            rotation=90, fontsize=6,
        )
    else:
        ax_matrix.set_xticks([])

    ##### Spine styling ##### 
    def style_spines(ax, linewidth=2.0):
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
            spine.set_zorder(0)          # render behind data (zorder=1+)
            spine.set_capstyle('round')   # prevents rounded ends overshooting corners
    
    for ax in [ax_hist_a, ax_hist_b, ax_matrix]:
        style_spines(ax, linewidth=2.0)

    ##### Save & show ##### 
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    plt.show()

    return fig, ax_hist_a, ax_hist_b, ax_matrix


def genomic_combined_dual_plot_v3(
    incidence_df_a,
    incidence_df_b,
    spath,
    shared_color='#39FF14',
    shared_dict=None,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    node_labels=True,
    col_labels=False,
    figsize=(22, 8),
    row_spacing=1.9,
    col_spacing=1.0,
    alpha_scatter=0.85,
    marker_size=10,
    point_padding=0.25,
    hist_width_ratio=0.10,
    panel_wspace=0.04,
    x_offset=-1.0,
    label_a='Cell',          # Label for matrix A
    label_b='Core',          # Label for matrix B
    color_a='#808080',       # Bar colour for IM-A
    color_b='#000000',       # Bar colour for IM-B
    bar_alpha=0.75,       
):
    """
    Plot layout (left to right):
        [combined degree histogram (IM-A + IM-B)] [dual incidence matrix]

    Both normalised-degree distributions are drawn as horizontal bar charts
    inside the single histogram panel.  IM-A bars grow left-to-right and
    IM-B bars grow right-to-left (mirror / back-to-back style) so they are
    easy to compare without obscuring each other.

    Y-axis chromosome labels are on the LEFT of the histogram panel.
    """

    SHARED_COLOR = shared_color

    plt.rcParams['figure.dpi']     = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.00

    ##### Align both matrices to a shared row universe ##### 
    all_rows     = merge_preserve_order(incidence_df_a.index, incidence_df_b.index)
    all_rows_arr = np.array(all_rows)

    df_a = incidence_df_a.reindex(index=all_rows, fill_value=0)
    df_b = incidence_df_b.reindex(index=all_rows, fill_value=0)

    n_rows        = len(all_rows)
    matrix_a_full = df_a.values
    matrix_b_full = df_b.values

    ##### Chromosome metadata ##### 
    chrom_labels = [label.split(':')[0] for label in all_rows]

    seen, unique_chroms = {}, []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)
    n_chroms = len(unique_chroms)

    chrom_to_row_indices = {
        chrom: [i for i, c in enumerate(chrom_labels) if c == chrom]
        for chrom in unique_chroms
    }
    row_to_chrom = {i: c for i, c in enumerate(chrom_labels)}

    def make_chrom_colors(color_shift=0.0):
        if n_chroms <= 20:
            cmap_fn    = plt.cm.get_cmap('tab20', 20)
            color_list = [cmap_fn(i % 20) for i in range(n_chroms)]
        else:
            cmap_fn    = plt.cm.hsv
            color_list = [
                cmap_fn((i / max(n_chroms, 1) + color_shift) % 1.0)
                for i in range(n_chroms)
            ]
        chrom_to_color = {ch: color_list[i] for i, ch in enumerate(unique_chroms)}
        row_colors     = np.array([chrom_to_color[c] for c in chrom_labels])
        return row_colors, chrom_to_color

    row_colors_a, chrom_to_color = make_chrom_colors()

    ##### Degree histograms ##### 
    m_a = matrix_a_full.shape[1]
    m_b = matrix_b_full.shape[1]

    norm_degrees_a = matrix_a_full.sum(axis=1) / m_a
    norm_degrees_b = matrix_b_full.sum(axis=1) / m_b

    # Shared x-limit so both distributions are on the same scale
    global_max_nd = max(norm_degrees_a.max(), norm_degrees_b.max()) * 1.15

    ##### Build plot_columns for the dual incidence matrix ##### 
    def assign_chrom_info(col_vec):
        nonzero_rows = np.where(col_vec != 0)[0]
        if len(nonzero_rows) == 0:
            return None, []
        seen_c, all_chroms = set(), []
        for r in nonzero_rows:
            c = row_to_chrom[r]
            if c not in seen_c:
                seen_c.add(c)
                all_chroms.append(c)
        primary_chrom = row_to_chrom[nonzero_rows[0]]
        return primary_chrom, all_chroms

    chrom_cols_a = {ch: [] for ch in unique_chroms}
    chrom_cols_b = {ch: [] for ch in unique_chroms}

    for j, col_name in enumerate(df_a.columns):
        col_vec = matrix_a_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_a[primary_chrom].append({
            'source': 'a', 'col_name': col_name,
            'col_vec': col_vec, 'chrom': primary_chrom,
            'all_chroms': all_chroms,
        })

    for j, col_name in enumerate(df_b.columns):
        col_vec = matrix_b_full[:, j]
        primary_chrom, all_chroms = assign_chrom_info(col_vec)
        if primary_chrom is None:
            continue
        chrom_cols_b[primary_chrom].append({
            'source': 'b', 'col_name': col_name,
            'col_vec': col_vec, 'chrom': primary_chrom,
            'all_chroms': all_chroms,
        })

    plot_columns = []
    for chrom in unique_chroms:
        cols_a  = chrom_cols_a[chrom]
        cols_b  = chrom_cols_b[chrom]
        max_len = max(len(cols_a), len(cols_b), 0)
        for k in range(max_len):
            if k < len(cols_a):
                plot_columns.append(cols_a[k])
            if k < len(cols_b):
                plot_columns.append(cols_b[k])

    m_plot = len(plot_columns)

    ##### Shared-column mask ##### 
    shared_col_mask = np.zeros(m_plot, dtype=bool)
    if shared_dict:
        target_sets = {frozenset(reads) for reads in shared_dict.values()}
        for i, pcol in enumerate(plot_columns):
            nonzero_rows = np.where(pcol['col_vec'] != 0)[0]
            col_set      = frozenset(all_rows_arr[nonzero_rows])
            if col_set in target_sets:
                shared_col_mask[i] = True

    #####  Figure creation using GridSpec (2 columns) ##### 
    matrix_ratio = 1.0 - hist_width_ratio      # histogram takes hist_width_ratio

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[hist_width_ratio, matrix_ratio],
        wspace=panel_wspace,
        figure=fig,
    )

    ax_hist   = fig.add_subplot(gs[0])   # single combined histogram
    ax_matrix = fig.add_subplot(gs[1])   # dual incidence matrix

       ##### Combined histogram (both pointing right, side-by-side bars) ##### 
    data_pad_y = 1.0 * row_spacing
    bar_height = 0.85 * row_spacing          # total height split between two bars
    half_bar   = bar_height / 2.0 * 0.92    # slight gap between the two halves

    mean_nd_a = norm_degrees_a.mean()
    mean_nd_b = norm_degrees_b.mean()

    for i in range(n_rows):
        y_center = i * row_spacing

        # IM-A: upper half of the row
        ax_hist.barh(
            y_center - half_bar / 2,
            norm_degrees_a[i],
            height=half_bar,
            color=color_a,
            alpha=bar_alpha,
            linewidth=0,
            zorder=2,
        )

        # IM-B: lower half of the row
        ax_hist.barh(
            y_center + half_bar / 2,
            norm_degrees_b[i],
            height=half_bar,
            color=color_b,
            alpha=bar_alpha,
            linewidth=0,
            zorder=2,
        )

    # Chromosome background bands
    y_axis_min = -data_pad_y - 1
    y_axis_max = (n_rows - 1) * row_spacing + data_pad_y

    for idx, chrom in enumerate(unique_chroms):
        indices = chrom_to_row_indices[chrom]
        y_min   = min(indices) * row_spacing - 0.5 * row_spacing
        y_max   = max(indices) * row_spacing + 0.5 * row_spacing
        if idx == 0:
            y_min = y_axis_min
        if idx == len(unique_chroms) - 1:
            y_max = y_axis_max
        ax_hist.axhspan(y_min, y_max,
                        color=chrom_to_color[chrom], alpha=0.15, zorder=0)

    # Mean lines
    ax_hist.axvline(mean_nd_a, color='#CE2029', linestyle='--', linewidth=1.4,
                    zorder=3, label=f'{label_a} mean')
    ax_hist.axvline(mean_nd_b, color='#FF69B4', linestyle='--', linewidth=1.4,
                    zorder=3, label=f'{label_b} mean')

    # Axis limits
    ax_hist.set_xlim([global_max_nd, 0])     # bars grow right-to-left toward matrix
    ax_hist.set_ylim([y_axis_min, y_axis_max])
    ax_hist.invert_yaxis()

    # X ticks suppressed (same style as original single histograms)
    ax_hist.set_xticks([])
    ax_hist.tick_params(axis='x', length=0)

    # Y-axis chromosome tick labels
    if node_labels:
        group_tick_positions, group_tick_labels = [], []
        for chrom in unique_chroms:
            indices  = chrom_to_row_indices[chrom]
            midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
            group_tick_positions.append(midpoint)
            group_tick_labels.append(
                chrom.replace('chr', '').replace('Chr', '')
            )

        if len(group_tick_positions) > 4:
            ellipsis_pos      = (group_tick_positions[2] + group_tick_positions[-1]) / 5
            display_positions = group_tick_positions[:3] + [ellipsis_pos] + [group_tick_positions[-1]]
            display_labels    = group_tick_labels[:3]    + ['⋮']          + [group_tick_labels[-1]]
        else:
            display_positions = group_tick_positions
            display_labels    = group_tick_labels

        ax_hist.set_yticks(display_positions)
        ax_hist.set_yticklabels(display_labels, fontsize=12, fontweight='bold')
        ax_hist.yaxis.set_tick_params(length=0)
        ax_hist.yaxis.set_label_position('left')
        ax_hist.yaxis.tick_left()
    else:
        ax_hist.set_yticks([])

    ax_hist.spines['right'].set_visible(True)

    ##### Dual incidence matrix (unchanged from original) ##### 
    data_pad_x = 0.5 * col_spacing

    row_colors_b, _ = make_chrom_colors()

    fig_width_inches = figsize[0]
    matrix_ratio     = 1.0 - hist_width_ratio
    ax_width_inches  = fig_width_inches * matrix_ratio

    x_range         = ((x_offset + 2) + (m_plot - 1) * col_spacing
                       + 0.5 * col_spacing - (x_offset - 0.5 * col_spacing))
    points_per_data = (ax_width_inches * 72) / x_range
    col_spacing_pts = col_spacing * points_per_data
    effective_s     = ((1.0 - point_padding) * col_spacing_pts) ** 2

    for i, pcol in enumerate(plot_columns):
        col_vec    = pcol['col_vec']
        source     = pcol['source']
        row_colors = row_colors_a if source == 'a' else row_colors_b
        marker     = 'o'          if source == 'a' else 's'

        y_pts = np.where(col_vec != 0)[0]
        if len(y_pts) == 0:
            continue

        x_scaled = i * col_spacing
        y_scaled = y_pts * row_spacing

        if shared_col_mask[i]:
            colors     = [SHARED_COLOR] * len(y_pts)
            line_color = SHARED_COLOR
        else:
            colors     = 'k'
            line_color = 'k'

        ax_matrix.scatter(
            x_scaled * np.ones(len(y_pts)), y_scaled,
            color=colors, edgecolor='k', linewidths=0.3,
            s=effective_s, marker=marker, alpha=alpha_scatter, zorder=2,
        )

        if connect_nodes and len(y_scaled) > 1:
            ax_matrix.plot(
                [x_scaled, x_scaled],
                [y_scaled.min(), y_scaled.max()],
                c=line_color, lw=0.8, zorder=1,
            )

    if shade_rows:
        for chrom in unique_chroms:
            row_idxs    = chrom_to_row_indices[chrom]
            row_min     = (min(row_idxs) - 0.5) * row_spacing
            row_max     = (max(row_idxs) + 0.5) * row_spacing
            band_height = row_max - row_min + 0.5

            chrom_plot_col_idxs = [
                i for i, pc in enumerate(plot_columns)
                if chrom in pc['all_chroms']
            ]
            if not chrom_plot_col_idxs:
                continue

            x_left     = -1.0
            x_right    = max(chrom_plot_col_idxs) * col_spacing + 0.6
            band_width = x_right - x_left

            ax_matrix.barh(
                (row_min + row_max) / 2,
                band_width,
                height=band_height,
                left=x_left,
                color=chrom_to_color[chrom],
                alpha=0.12,
                zorder=0,
            )

    ax_matrix.set_xlim([
        x_offset - data_pad_x,
        (x_offset + 2) + (m_plot - 1) * col_spacing + data_pad_x,
    ])
    ax_matrix.set_ylim([
        -data_pad_y,
        (n_rows - 1) * row_spacing + data_pad_y,
    ])
    ax_matrix.set_yticks([])
    ax_matrix.tick_params(axis='y', length=0)
    ax_matrix.spines['left'].set_visible(True)
    ax_matrix.invert_yaxis()

    if col_labels:
        ax_matrix.set_xticks([i * col_spacing for i in range(m_plot)])
        ax_matrix.set_xticklabels(
            [pc['col_name'] for pc in plot_columns],
            rotation=90, fontsize=6,
        )
    else:
        ax_matrix.set_xticks([])

    ##### Spine styling ##### 
    def style_spines(ax, linewidth=2.0):
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
            spine.set_zorder(0)
            spine.set_capstyle('round')

    for ax in [ax_hist, ax_matrix]:
        style_spines(ax, linewidth=2.0)

    ##### Save & show ##### 
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    plt.show()

    return fig, ax_hist, ax_matrix
    