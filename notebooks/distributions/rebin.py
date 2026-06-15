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
from matplotlib.collections import LineCollection
from itertools import chain
import re
import pickle as pkl
import HAT
from HAT import draw


def chrom_to_number(chrom):
    """
    chr1 -> 1
    chr2 -> 2
    chrX -> 23
    chrY -> 24
    chrM/chrMT -> 25
    """
    chrom = str(chrom).replace("chr", "")

    if chrom == "X":
        return 23
    elif chrom == "Y":
        return 24
    elif chrom in {"M", "MT"}:
        return 25
    else:
        return int(chrom)


def locus_sort_key(locus):
    """
    Sort loci like chr1:0, chr1:1, chr2:0, ..., chrX:0
    """
    chrom, bin_idx = locus.split(":")
    return chrom_to_number(chrom), int(bin_idx)
        
def get_chromosomes(locus_set):
    return {l.split(':')[0] for l in locus_set}

def is_intra(locus_set):
    return len(get_chromosomes(locus_set)) == 1

def select_top_hyperedges(locus_set_counts, top_n=5):
    all_chroms = set()
    for ls in locus_set_counts:
        all_chroms.update(get_chromosomes(ls))

    selected = set()

    for chrom in sorted(all_chroms, key=lambda c: (
        int(c.replace('chr','')) 
        if c.replace('chr','').isdigit() 
        else {'X':23,'Y':24,'M':25,'MT':25}.get(c.replace('chr',''), 99)
    )):
        # Intra contacts for this chromosome
        intra = [
            (ls, cnt) for ls, cnt in locus_set_counts.items()
            if is_intra(ls) and chrom in get_chromosomes(ls)
        ]
        # Inter contacts involving this chromosome
        inter = [
            (ls, cnt) for ls, cnt in locus_set_counts.items()
            if not is_intra(ls) and chrom in get_chromosomes(ls)
        ]

        # Rank by count
        intra_ranked = sorted(intra, key=lambda x: -x[1])
        inter_ranked = sorted(inter, key=lambda x: -x[1])

        chosen_intra = []
        for ls, cnt in intra_ranked:
            if ls not in selected and len(chosen_intra) < top_n:
                chosen_intra.append(ls)

        chosen_inter = []
        for ls, cnt in inter_ranked:
            if ls not in selected and len(chosen_inter) < top_n:
                chosen_inter.append(ls)

        # Fill missing intra slots with inter if needed
        deficit = top_n - len(chosen_intra)
        if deficit > 0:
            extras = [ls for ls in chosen_inter[:deficit] if ls not in chosen_intra]
            chosen_intra.extend(extras)
            chosen_inter = [ls for ls in chosen_inter if ls not in extras]

        for ls in chosen_intra + chosen_inter:
            selected.add(ls)

    return list(selected)

def sort_hyperedges_by_earliest_locus(selected_hyperedges, locus_set_counts):
    """
    Sort hyperedges so that columns flow 'downhill':
      - Primary sort:   earliest locus in the hyperedge (genomic order)
      - Secondary sort: within a chromosome group, pure intra-chromosomal
                        hyperedges (all loci on the same chromosome) come
                        before inter-chromosomal ones
      - Tertiary sort:  all loci ranked, so same-anchor ties broken by
                        next earliest locus
      - Quaternary sort: frequency descending
    """

    def locus_rank(locus):
        chrom, idx = locus.split(':')
        chrom = chrom.replace('chr', '')
        chrom_num = (
            int(chrom) if chrom.isdigit()
            else {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(chrom, 99)
        )
        return (chrom_num, int(idx))

    def get_chrom(locus):
        return locus.split(':')[0]

    def hyperedge_sort_key(ls):
        earliest      = min(ls, key=locus_rank)
        anchor_rank   = locus_rank(earliest)
        all_ranked    = sorted(ls, key=locus_rank)
        freq          = -locus_set_counts.get(ls, 0)

        # ── Intra-chromosomal flag ────────────────────────────────────────────
        # 0 = all loci on same chromosome (sorted first within group)
        # 1 = spans multiple chromosomes  (sorted after within group)
        chroms_in_he  = set(get_chrom(l) for l in ls)
        is_inter      = 0 if len(chroms_in_he) == 1 else 1

        return (
            anchor_rank,                               # 1. earliest locus
            is_inter,                                  # 2. intra before inter
            tuple(locus_rank(l) for l in all_ranked), # 3. full locus sequence
            freq                                       # 4. frequency descending
        )

    return sorted(selected_hyperedges, key=hyperedge_sort_key)

def build_selected_incidence_df(selected_hyperedges, locus_set_counts):

    # ← Use the new sort instead of frequency-only sort
    selected_hyperedges_sorted = sort_hyperedges_by_earliest_locus(
        selected_hyperedges, locus_set_counts
    )

    all_loci = sorted(
        set(l for ls in selected_hyperedges_sorted for l in ls),
        key=lambda x: (
            int(x.split(':')[0].replace('chr', ''))
            if x.split(':')[0].replace('chr', '').isdigit()
            else {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(
                x.split(':')[0].replace('chr', ''), 99
            ),
            int(x.split(':')[1])
        )
    )

    locus_to_row = {l: i for i, l in enumerate(all_loci)}

    col_labels = [
        f"HE{i+1} (n={locus_set_counts.get(ls, 0)})"
        for i, ls in enumerate(selected_hyperedges_sorted)
    ]

    matrix = np.zeros((len(all_loci), len(selected_hyperedges_sorted)), dtype=int)

    for col_idx, ls in enumerate(selected_hyperedges_sorted):
        for locus in ls:
            matrix[locus_to_row[locus], col_idx] = 1

    incidence_df = pd.DataFrame(
        data=matrix,
        index=all_loci,
        columns=col_labels
    )

    return incidence_df


def rebin_large(df, dedup:bool, invert:bool, removal:bool):
    df_mod = df.replace(0,pd.NA)
    df_array = np.array(df)

    # --- Parse row bins only ---
    parsed = df.index.to_series().str.extract(r'(chr\w+):(\d+)')
    
    parsed = parsed.loc[
        sorted(parsed.index, key=locus_sort_key)
    ]
    
    parsed.columns = ['chr', 'pos']
    
    parsed['pos'] = parsed['pos'].astype(int)
    
    positions = parsed['pos'].to_list()
    
    bin_master =  {}
    
    for i in range(len(positions)):
        if (positions[i]//5) not in bin_master.keys():
            bin_master[(positions[i]//5)] = [positions[i]]
        elif (positions[i]//5) in bin_master.keys():
            bin_master[(positions[i]//5)].append(positions[i])
    
    
    DIVISOR = 25
    
    parsed['new_bin'] = (
        parsed['chr'] + ':' +
        (parsed['pos'] // DIVISOR).astype(str)
    )
    
    row_groups = parsed['new_bin'].values  
    
    new_IM = {}
    IM = {}
    
    for index, row in df_mod.iterrows():
        for i, val in row.items():
            if val is not pd.NA:
                if index not in IM.keys():
                    IM[index] = [i]
                elif index in IM.keys():
                    IM[index].append(i)
    
    IM = dict(sorted(IM.items()))
    
    
    for key, val in IM.items():
        for index, value in parsed.iterrows():
            if key == index:
                key_val = value['new_bin']
                if key_val not in new_IM.keys():
                    new_IM[key_val] = val
                elif key_val in new_IM.keys():
                    new_val = new_IM[key_val] + val
                    new_IM[key_val] = new_val
    
    new_IM = dict(
        sorted(
            new_IM.items(),
            key=lambda item: locus_sort_key(item[0])
        )
    )
    if dedup==True:
        # Deduplicate bins per read (column/UUID)
        deduped_IM = {}
        for locus, reads in new_IM.items():
            deduped_IM[locus] = list(set(reads))  # unique reads per locus
    else: 
        deduped_IM = new_IM
        
    if invert==True:
        # Invert: read -> set of loci
        read_to_loci = {}
        for locus, reads in deduped_IM.items():
            for read in reads:
                if read not in read_to_loci:
                    read_to_loci[read] = set()
                read_to_loci[read].add(locus)

        read_to_loci = {
            read: loci 
            for read, loci in read_to_loci.items() 
            if len(loci) >= 2
        }
        
    else:
        read_to_loci = {
            read: loci 
            for read, loci in read_to_loci.items() 
            if len(loci) >= 2
        }
    if removal == True:
        # Option A: Remove reads touching too many loci (noise/promiscuous contacts)
        MAX_LOCI = 10
        read_to_loci = {r: l for r, l in read_to_loci.items() if len(l) <= MAX_LOCI}
        
        # Option B: Remove loci appearing in very few reads (low-frequency noise)
        locus_freq = Counter(l for loci in read_to_loci.values() for l in loci)
        MIN_SUPPORT = 2
        read_to_loci = {
            r: {l for l in loci if locus_freq[l] >= MIN_SUPPORT}
            for r, loci in read_to_loci.items()
        }
        # Re-apply minimum loci filter after noise removal
        read_to_loci = {r: l for r, l in read_to_loci.items() if len(l) >= 2}
    else:
        ead_to_loci = {r: l for r, l in read_to_loci.items() if len(l) >= 2}
    
    # Collapse identical locus sets
    locus_set_counts = Counter(
        frozenset(loci) for loci in read_to_loci.values()
    )
    
    # Sort by frequency descending
    locus_set_counts = dict(
        sorted(locus_set_counts.items(), key=lambda x: -x[1])
    )
    return locus_set_counts