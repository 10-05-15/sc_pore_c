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





def load_pairs(filepath):
    """Load a .pairs file with extended column set"""
    
    cols = [
        "readID", "chrom1", "pos1", "chrom2", "pos2",
        "strand1", "strand2", "pair_type",
        "walk_pair_index", "walk_pair_type",
        "mapq1", "mapq2",
        "read_len1", "read_len2",
        "algn_read_span1", "algn_read_span2",
        "algn_ref_span1", "algn_ref_span2",
        "matched_bp1", "matched_bp2",
        "rfrag1", "rfrag_start1", "rfrag_end1",
        "rfrag2", "rfrag_start2", "rfrag_end2"
    ]
    
    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        header=None,
        names=cols
    )
    
    # Cast numeric columns
    int_cols = [
        "pos1", "pos2",
        "mapq1", "mapq2",
        "read_len1", "read_len2",
        "algn_read_span1", "algn_read_span2",
        "algn_ref_span1", "algn_ref_span2",
        "matched_bp1", "matched_bp2",
        "rfrag1", "rfrag_start1", "rfrag_end1",
        "rfrag2", "rfrag_start2", "rfrag_end2"
    ]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce")
    
    return df

def filter_pairs(df, min_mapq=30, valid_pair_types=None):
    """
    Filter contacts by quality metrics.
    
    Parameters:
        min_mapq         : minimum mapping quality for both reads
        valid_pair_types : list of pair_type values to keep
                           e.g. ['UU', 'RU', 'UR'] for unique alignments
                           None = no pair_type filter
    """
    original_len = len(df)
    
    # Filter by mapping quality
    df = df[(df["mapq1"] >= min_mapq) & (df["mapq2"] >= min_mapq)]
    
    # Filter by pair type (e.g. keep only uniquely mapped pairs)
    if valid_pair_types is not None:
        df = df[df["pair_type"].isin(valid_pair_types)]
    
    # Remove unmapped or low-confidence pair types
    exclude_types = ["NN", "NM", "MN", "XX"]  # adjust as needed
    df = df[~df["pair_type"].isin(exclude_types)]
    
    # print(f"Retained {len(df)}/{original_len} contacts after filtering "
    #       f"({len(df)/original_len*100:.1f}%)")
    
    return df

def build_contact_matrix(df, chrom, bin_size=50_000):
    """
    Build an intra-chromosomal contact matrix for one chromosome.

    Returns (matrix, max_bin) so callers can align sizes across cells.
    """
    intra = df[(df["chrom1"] == chrom) & (df["chrom2"] == chrom)].copy()
    print(f"    {len(intra)} intra-chromosomal contacts on {chrom}")
    intra = intra.dropna(subset=["pos1", "pos2"])
    intra["bin1"] = (intra["pos1"] // bin_size).astype(int)
    intra["bin2"] = (intra["pos2"] // bin_size).astype(int)

    if intra.empty:
        return np.zeros((0, 0), dtype=float), 0

    max_bin = int(max(intra["bin1"].max(), intra["bin2"].max())) + 1
    matrix = np.zeros((max_bin, max_bin), dtype=float)
    np.add.at(matrix, (intra["bin1"].values, intra["bin2"].values), 1)
    np.add.at(matrix, (intra["bin2"].values, intra["bin1"].values), 1)
    return matrix, max_bin

def build_master_matrix(
    directory,
    bin_size=50_000,
    chrom=None,
    chrom_sizes=None,
    min_mapq=30,
    valid_pair_types=None,
    suffix=".GRCm39.filtered.pairs"
):
    """
    Read all *{suffix} files in `directory`, filter them, and accumulate
    contacts into a single master numpy matrix.

    Parameters
    ----------
    directory        : path to folder containing .pairs files
    bin_size         : genomic bin width in bp
    chrom            : str or None
                       • str  → single-chromosome intra mode (e.g. "chr1")
                       • None → whole-genome mode (requires chrom_sizes)
    chrom_sizes      : dict {chrom: length} required when chrom=None
    min_mapq         : passed to filter_pairs
    valid_pair_types : passed to filter_pairs
    suffix           : filename suffix used to glob for files

    Returns
    -------
    master_matrix : np.ndarray  — aggregated contact matrix
    file_list     : list of file paths that were processed
    """
    pattern = os.path.join(directory, f"*{suffix}")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        raise FileNotFoundError(
            f"No files matching '*{suffix}' found in: {directory}"
        )
    print(f"Found {len(file_list)} pairs file(s) in '{directory}'")

    # ── Whole-genome mode validation ──
    if chrom is None and chrom_sizes is None:
        raise ValueError(
            "Provide chrom_sizes dict when running in whole-genome mode "
            "(chrom=None)."
        )

    master_matrix = None

    for i, fpath in enumerate(file_list, 1):
        fname = os.path.basename(fpath)
        #print(f"\n[{i}/{len(file_list)}] Processing: {fname}")

        df = load_pairs(fpath)
        df = filter_pairs(df, min_mapq=min_mapq,
                          valid_pair_types=valid_pair_types)

        # ── Single-chromosome mode ──
        if chrom is not None:
            cell_matrix, max_bin = build_contact_matrix(df, chrom, bin_size)

            if cell_matrix.size == 0:
                print(f"    Skipping {fname}: no contacts on {chrom}")
                continue

            # Grow master matrix if this cell is larger
            if master_matrix is None:
                master_matrix = cell_matrix.copy()
            else:
                current_size = master_matrix.shape[0]
                if max_bin > current_size:
                    # Pad master to new size
                    padded = np.zeros((max_bin, max_bin), dtype=float)
                    padded[:current_size, :current_size] = master_matrix
                    master_matrix = padded
                # Add cell matrix (pad cell if smaller than master)
                cell_size = cell_matrix.shape[0]
                master_matrix[:cell_size, :cell_size] += cell_matrix

    if master_matrix is None:
        raise RuntimeError("No contacts were loaded — master matrix is empty.")

    print(f"\nMaster matrix shape: {master_matrix.shape}")
    print(f"Total contacts (sum/2): {int(master_matrix.sum() / 2)}")
    return master_matrix, file_list

