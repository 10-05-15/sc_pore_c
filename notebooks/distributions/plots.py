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



def genomic_combined_plot(
    incidence_df,
    spath,
    shade_rows=True,
    connect_nodes=True,
    dpi=200,
    row_colors=None,
    node_labels=True,
    col_labels=False,
    figsize=(18, 8),
    row_spacing=1.0,
    col_spacing=1.0,
    hist_width_ratio=0.08,   # fraction of total width given to histogram
):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    matrix = incidence_df.values
    n, m = matrix.shape

    #####  Shared chromosome metadata ##### 
    row_index    = incidence_df.index.tolist()
    chrom_labels = [label.split(':')[0] for label in row_index]

    seen, unique_chroms = {}, []
    for c in chrom_labels:
        if c not in seen:
            seen[c] = len(unique_chroms)
            unique_chroms.append(c)
    n_chroms = len(unique_chroms)

    cmap       = plt.cm.tab10 if n_chroms <= 10 else plt.cm.tab20
    color_list = cmap(np.linspace(0, 1, n_chroms))
    chrom_to_color = {ch: color_list[i] for i, ch in enumerate(unique_chroms)}

    if row_colors is None:
        row_colors = np.array([chrom_to_color[c] for c in chrom_labels])
    else:
        row_colors = np.asarray(row_colors)

    ##### Degree computation ##### 
    degrees     = matrix.sum(axis=1)
    norm_degrees = degrees / m

    ##### 
    #  Figure layout: [hist | incidence matrix]                           
    #  hist is narrow; incidence matrix is wide 
    ##### 
    hist_ratio   = hist_width_ratio
    matrix_ratio = 1.0 - hist_width_ratio

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[hist_ratio, matrix_ratio],
        wspace=0.04,          # <- Adjust Panel Gap here
        figure=fig
    )

    ax_hist   = fig.add_subplot(gs[0])  # LEFT  – vertical histogram
    ax_matrix = fig.add_subplot(gs[1])   # RIGHT – incidence matrix

    ##### INCIDENCE MATRIX  (ax_matrix) ##### 

    ##### scatter + connecting lines ##### 
    for i in range(m):
        y_pts = np.where(matrix[:, i] != 0)[0]
        if len(y_pts) == 0:
            continue
        x_scaled = i * col_spacing
        y_scaled = y_pts * row_spacing
        ax_matrix.scatter(
            x_scaled * np.ones(len(y_pts)), y_scaled,
            color='k', edgecolor='k', linewidths=0.3, s=10, zorder=2
        )
        if connect_nodes and len(y_scaled) > 1:
            ax_matrix.plot(
                [x_scaled, x_scaled], [y_scaled.min(), y_scaled.max()],
                c='k', lw=0.8, zorder=1
            )

    ##### chromosome background bands ##### 
    if shade_rows:
        for chrom in unique_chroms:
            indices  = [i for i, c in enumerate(chrom_labels) if c == chrom]
            row_min  = (min(indices) - 0.5) * row_spacing
            row_max  = (max(indices) + 0.5) * row_spacing
            band_h   = row_max - row_min

            chrom_matrix_rows = matrix[indices, :]
            active_cols = np.where(chrom_matrix_rows.any(axis=0))[0]
            if len(active_cols) == 0:
                continue

            x_end      = active_cols.max() * col_spacing + 0.5
            band_width = x_end - (-0.5)

            ax_matrix.barh(
                (row_min + row_max) / 2, band_width,
                height=band_h, left=-0.5,
                color=chrom_to_color[chrom], alpha=0.15, zorder=0
            )

    ax_matrix.set_xlim([-0.5, (m - 0.5) * col_spacing])
    ax_matrix.set_ylim([-0.5, (n - 0.5) * row_spacing])

    ##### x-axis ticks ##### 
    if col_labels:
        ax_matrix.set_xticks([i * col_spacing for i in range(m)])
        ax_matrix.set_xticklabels(
            incidence_df.columns.tolist(), rotation=90, fontsize=6
        )
    else:
        ax_matrix.set_xticks([])

    ##### suppress y-axis ticks on the matrix (labels go on hist) ##### 
    ax_matrix.set_yticks([])
    ax_matrix.tick_params(axis='y', length=0)

    ax_matrix.invert_yaxis()

    ##### VERTICAL HISTOGRAM  (ax_hist) ##### 
    #
    #  Strategy: plot bars with barh() so bars grow LEFT (negative width)
    #  from x=0.  The y-axis of ax_hist must share the same range as
    #  ax_matrix so chromosomes align.
    #
    #  ax_hist y-range  : [-0.5, (n-0.5)*row_spacing]  ← same as matrix
    #  ax_hist x-range  : [max_norm_deg * 1.15, 0]      ← bars grow left

    bar_height = 0.85 * row_spacing   # analogous to width=0.85 in original

    for i, nd in enumerate(norm_degrees):
        chrom = chrom_labels[i]
        y_center = i * row_spacing
        ax_hist.barh(
            y_center,
            nd,                    # bar length (will be mirrored by xlim)
            height=bar_height,
            color='black',
            linewidth=0,
            zorder=2
        )

    # Chromosome background bands (horizontal spans in hist)
    for chrom in unique_chroms:
        indices = [i for i, c in enumerate(chrom_labels) if c == chrom]
        y_min   = min(indices) * row_spacing - 0.5 * row_spacing
        y_max   = max(indices) * row_spacing + 0.5 * row_spacing
        ax_hist.axhspan(
            y_min, y_max,
            color=chrom_to_color[chrom], alpha=0.15, zorder=0
        )

    # Mean degree vertical line (now a vertical line in ax_hist x-space)
    mean_nd = norm_degrees.mean()
    ax_hist.axvline(
        mean_nd, color='red', linestyle='--', linewidth=1.2, zorder=3
    )

    # Match y-limits exactly to the matrix so rows align
    ax_hist.set_ylim([-0.5 * row_spacing, (n - 0.5) * row_spacing])
    ax_hist.invert_yaxis()   # top of hist = row 0, matching matrix

    # Flip x so bars grow left-to-right toward the matrix
    max_nd = norm_degrees.max() * 1.15
    ax_hist.set_xlim([max_nd, 0])    # reversed: 0 is on the RIGHT edge

    ##### chromosome group y-tick labels on the LEFT of the histogram ##### 
    if node_labels:
        group_tick_positions = []
        group_tick_labels    = []

        for chrom in unique_chroms:
            indices  = [i for i, c in enumerate(chrom_labels) if c == chrom]
            midpoint = ((min(indices) + max(indices)) / 2.0) * row_spacing
            group_tick_positions.append(midpoint)
            group_tick_labels.append(
                chrom.replace('chr', '').replace('Chr', '')
            )

        # Truncation when many chromosomes
        if len(group_tick_positions) > 4:
            ellipsis_pos = (
                group_tick_positions[2] + group_tick_positions[-1]
            ) / 5
            display_positions = (
                group_tick_positions[:3]
                + [ellipsis_pos]
                + [group_tick_positions[-1]]
            )
            display_labels = (
                group_tick_labels[:3] + ['⋮'] + [group_tick_labels[-1]]
            )
        else:
            display_positions = group_tick_positions
            display_labels    = group_tick_labels

        ax_hist.set_yticks(display_positions)
        ax_hist.set_yticklabels(
            display_labels, fontsize=12, fontweight='bold'
        )
        ax_hist.yaxis.set_tick_params(length=0)
        ax_hist.yaxis.set_label_position('left')
        ax_hist.yaxis.tick_left()
    else:
        ax_hist.set_yticks([])

    # Clean up x-ticks on histogram (optional: keep or remove)
    ax_hist.set_xticks([])
    ax_hist.tick_params(axis='x', length=0)

    # Turn on the inner spines between the panels.
    ax_hist.spines['right'].set_visible(True)
    ax_matrix.spines['left'].set_visible(True)
    def style_spines(ax, linewidth=2.0):
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
            spine.set_zorder(0)          # render behind data (zorder=1+)
            spine.set_capstyle('butt')   # prevents rounded ends overshooting corners
    
    style_spines(ax_hist, linewidth=2.0)

    style_spines(ax_matrix, linewidth=2.0)
    

    ##### Save and show ##### 
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    plt.show()
    return fig, ax_hist, ax_matrix

def plot_contact_matrix(
    incidence_df,
    spath,
    log_transform=True,
    cmap='Reds',
    dpi=200,
    figsize=(10, 8),
):

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    
    H = incidence_df.values.astype(float)
    contact_matrix = H @ H.T  # shape: (n_bins, n_bins)

    if log_transform:
        plot_data = np.log1p(contact_matrix)
        cbar_label = 'log(1 + Shared Reads)'
    else:
        plot_data = contact_matrix
        cbar_label = 'Shared Reads'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=cmap,
        square=True,                        # keep cells square like a Hi-C map,
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        linewidths=0.3,
        linecolor='white',
        xticklabels=5,
        yticklabels=5,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha='right',
        fontsize=7
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=7
    )
    ax.set_xlabel('Genomic Bin (25 Mb)', fontsize=10)
    ax.set_ylabel('Genomic Bin (25 Mb)', fontsize=10)

    plt.tight_layout()
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    return ax, contact_matrix   # return raw matrix for further analysis



def plot_contact_matrices_panel(
    incidence_dfs,          # list of 8 incidence DataFrames
    spath,
    log_transform=True,
    cmap='Reds',
    dpi=200,
    figsize=(35,5),       # adjusted for 2x4 grid
):
    assert len(incidence_dfs) == 8, "Exactly 8 incidence matrices are required."

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    ###### 
    #  Pre-compute all contact matrices & find global min/max for a 
    #  shared color scale 
    ##### 
    contact_matrices = []
    plot_data_list   = []

    for df in incidence_dfs:
        H = df.values.astype(float)
        C = H @ H.T
        contact_matrices.append(C)
        plot_data_list.append(np.log1p(C) if log_transform else C)

    global_min = min(p.min() for p in plot_data_list)
    global_max = max(p.max() for p in plot_data_list)

    cbar_label = 'log(1 + Shared Reads)' if log_transform else 'Shared Reads'
    axis_label = 'Chr 1 - 19, X'

    ##### Build figure ##### 
    fig, axes = plt.subplots(
        1, 8,
        figsize=figsize,
        constrained_layout=True,
    )

    fig.subplots_adjust(wspace=0, hspace=0)
    axes_flat = axes.flatten()   # flatten to iterate easily

    for i, (ax, plot_data) in enumerate(zip(axes_flat, plot_data_list)):

        show_cbar = (i == 7)     # only draw colorbar on the last panel (bottom-right)

        sns.heatmap(
            plot_data,
            ax=ax,
            cmap=cmap,
            square=True,
            vmin=global_min,
            vmax=global_max,
            cbar=show_cbar,
            cbar_kws={
                'label'      : cbar_label,
                'orientation': 'horizontal',
                'shrink'     : 0.6,
                'fraction'   : 0.035,
                'pad'        : 0.08,
            } if show_cbar else {},
            linewidths=0.3,
            linecolor='white',
            xticklabels=False,
            yticklabels=False,
        )

        ##### title ##### 
        ax.set_title(f'Cell {i + 1}', fontsize=10, pad=6)

        #####  axis labels (first panel only) ##### 
        if i == 0:
            ax.set_xlabel(axis_label, fontsize=9)
            ax.set_ylabel(axis_label, fontsize=9)
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

        #####  border ##### 
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2.0)
            
    plt.savefig(spath, dpi=dpi, bbox_inches='tight')
    plt.show()

    return axes, contact_matrices


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

def genomic_combined_dual_plot_v2(
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
    color_a='#808080',       # bar colour for IM-A
    color_b='#808080',       # bar colour for IM-B
    bar_alpha=1.0,
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

    #####  Build plot_columns for the dual incidence matrix ##### 
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

    ##### Combined (back-to-back) histogram ##### 
    data_pad_y = 1.0 * row_spacing
    bar_height = 0.85 * row_spacing

    mean_nd_a = norm_degrees_a.mean()
    mean_nd_b = norm_degrees_b.mean()

    for i in range(n_rows):
        y_center = i * row_spacing

        # Cell (IM-A): bars grow to the RIGHT  (positive direction)
        ax_hist.barh(
            y_center, norm_degrees_a[i],
            height=bar_height,
            color=color_a,
            alpha=bar_alpha,
            linewidth=0,
            edgecolor='black',
            zorder=2,
        )

        # Core (IM-B): bars grow to the LEFT  (negative direction → mirror)
        ax_hist.barh(
            y_center, -norm_degrees_b[i],
            height=bar_height,
            color=color_b,
            alpha=bar_alpha,
            linewidth=0,
            edgecolor = 'black', 
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

    # Mean lines for each distribution
    ax_hist.axvline( mean_nd_a, color='#CE2029', linestyle='--', linewidth=1.4,
                     zorder=3, label=f'{label_a} mean')
    ax_hist.axvline(-mean_nd_b, color='#CE2029', linestyle='--', linewidth=1.4,
                     zorder=3, label=f'{label_b} mean')

    # Centre line at x = 0
    ax_hist.axvline(0, color='black', linewidth=1.0, zorder=3)

    # Axis limits: symmetric around 0 so both sides have equal visual weight
    ax_hist.set_xlim([-global_max_nd, global_max_nd])
    ax_hist.set_ylim([y_axis_min, y_axis_max])
    ax_hist.invert_yaxis()

    # X-axis: show absolute tick labels so negative side reads naturally
    n_ticks   = 3   # number of ticks on each side (excluding 0)
    tick_vals = np.linspace(0, global_max_nd, n_ticks + 1)
    all_ticks = np.concatenate([-tick_vals[1:][::-1], tick_vals])
    
    ax_hist.tick_params(axis='x', length=0)
    ax_hist.set_xticks([])

    # Small axis labels so the reader knows which side is which
    ax_hist.text( global_max_nd * 0.55, y_axis_min * -0.92,
                  label_a, color='#000000', fontsize=9, fontweight='bold',
                  ha='center', va='bottom')
    ax_hist.text(-global_max_nd * 0.55, y_axis_min * -0.92,
                  label_b, color='#000000', fontsize=9, fontweight='bold',
                  ha='center', va='bottom')

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

    ##### Dual incidence matrix ##### 
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