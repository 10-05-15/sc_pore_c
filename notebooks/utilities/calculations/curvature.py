#!/usr/bin/env python3
"""
curvature.py
==================
Compute Forman-Ricci curvatures on the poset complex induced by a pore-C
hypergraph stored as an AnnData object.

  obs  = genomic loci  (nodes of the hypergraph, cardinality-1 sets)
  var  = reads/VAs     (hyperedges, cardinality = n_bins per read)
  adata.X              = raw binary incidence matrix  (obs × var)

Incidence matrix filtering (applied before poset construction)
--------------------------------------------------------------
  1. IQR filter on locus degree  → remove outlier bins (rows)
  2. Drop reads with < 2 loci    → remove singletons created by step 1 (cols)
  3. Drop reads with > 10 loci   → remove pathological hyperedges (cols)

Algorithm
---------
  1. Build the Hasse diagram (poset complex) of the filtered hypergraph.
     Uses an inverted-index shortcut to avoid the O(n²) all-pairs subset
     check that would be infeasible at millions of reads.
  2. Extract edges (direct parent links) and triangular 2-simplices
     (i → j → k chains in the poset).
  3. Compute per-edge Forman-Ricci curvature:
         Ric(u,v) = 4 - deg(u) - deg(v) + 3·#triangles(u,v)
  4. Compute per-locus scalar curvature:
         S(i) = Σ_{edges incident to i} Ric(edge)

Usage
-----
    python curvature.py <input.h5ad> <edge_out.csv> <node_out.csv> [--chrom CHROM]

    --chrom   Process only one chromosome, e.g. --chrom chr7

Outputs
-------
    edge_out.csv
        node_u_id, node_u_label, node_u_type,
        node_v_id, node_v_label, node_v_type,
        read_label,        <- read_index of the read(s) on this edge
        triangles, edge_curvature

    node_out.csv   (genomic loci only)
        node_id, chrom_bin, chrom, bin_start, bin_end,
        degree, scalar_curvature, normalized_scalar_curvature
"""

import argparse
import time
from collections import defaultdict

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx


# logging helper 

def log(msg, t0=None):
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(msg + elapsed, flush=True)


# load & filter

def load_and_filter(path, chrom=None):
    """
    Load AnnData, optionally subset to one chromosome, then apply the
    IQR / singleton / max-size filter to adata.X.

    The original adata is kept intact for label lookups; obs_idx / var_idx
    track which rows/cols of adata correspond to each row/col of the
    returned filtered matrix H.

    Returns
    -------
    adata   : original AnnData (untouched, used only for label lookups)
    H       : filtered sparse incidence matrix  (n_loci_kept x n_reads_kept)
    obs_idx : int array, length n_loci_kept
                obs_idx[i] = row position in adata.obs for filtered locus i
    var_idx : int array, length n_reads_kept
                var_idx[j] = col position in adata.var for filtered read j
    """
    log(f"Loading {path} ...")
    adata = anndata.read_h5ad(path)
    log(f"  Raw: {adata.n_obs:,} loci x {adata.n_vars:,} reads")

    # chrom filter (optional)
    if chrom:
        log(f"  Subsetting to {chrom} ...")
        chrom_mask = adata.obs["chrom"].values == chrom
        obs_idx = np.where(chrom_mask)[0]
        H = adata.X[obs_idx, :]
        if sp.issparse(H):
            col_sums = np.asarray(H.sum(axis=0)).ravel()
        else:
            col_sums = np.asarray(H).sum(axis=0)
        col_keep = col_sums >= 2
        H       = H[:, col_keep]
        var_idx = np.where(col_keep)[0]
        log(f"  After chrom filter: {H.shape[0]:,} loci x {H.shape[1]:,} reads")
    else:
        H       = adata.X
        obs_idx = np.arange(adata.n_obs)
        var_idx = np.arange(adata.n_vars)

    if sp.issparse(H):
        H = H.tocsr()

    # IQR filter on locus degree (rows of H) 
    log("  IQR-filtering loci ...")
    s = np.asarray(H.sum(axis=1)).ravel()
    q1, q3 = np.quantile(s, [0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    row_mask = (s >= low) & (s <= high)
    H       = H[row_mask, :]
    obs_idx = obs_idx[row_mask]
    log(f"    degree range kept: [{low:.1f}, {high:.1f}]  "
        f"-> {H.shape[0]:,} loci retained")

    # drop singletons (cols with < 2 loci after row filter)
    log("  Dropping singleton reads ...")
    col_sums = np.asarray(H.sum(axis=0)).ravel()
    col_mask = col_sums >= 2
    H       = H[:, col_mask]
    var_idx = var_idx[col_mask]

    # drop gross hyperedges (cols with > 10 loci) 
    log("  Dropping large hyperedges (> 10 loci) ...")
    col_sums = np.asarray(H.sum(axis=0)).ravel()
    col_mask = col_sums <= 10
    H       = H[:, col_mask]
    var_idx = var_idx[col_mask]

    log(f"  Filtered: {H.shape[0]:,} loci x {H.shape[1]:,} reads")
    return adata, H.tocsc(), obs_idx, var_idx


# poset / Hasse diagram 

def build_poset(H, t0):
    """
    Construct the Hasse diagram of the hypergraph represented by H.

    Compact node-ID convention
    --------------------------
      0 ... n_loci-1               loci (rows of H), cardinality 1
      n_loci ... n_loci+n_reads-1  reads (cols of H), cardinality = col sum

    Returns
    -------
    parent : dict { node_id -> set of direct-parent node_ids }
    k_val  : dict { node_id -> cardinality }
    n_loci : int
    """
    n_loci, n_reads = H.shape

    log("  Building node sets from H ...", t0)
    nodes = {}    # node_id -> frozenset of locus positions (0-based in H)
    k_val = {}    # node_id -> cardinality

    # locus nodes (cardinality 1)
    for i in range(n_loci):
        nodes[i] = frozenset([i])
        k_val[i] = 1

    # read nodes
    n_active = 0
    for j in range(n_reads):
        col = H[:, j]
        loci = frozenset(
            col.nonzero()[0].tolist() if sp.issparse(col)
            else np.nonzero(np.asarray(col).ravel())[0].tolist()
        )
        if loci:
            nid = n_loci + j
            nodes[nid] = loci
            k_val[nid] = len(loci)
            n_active += 1

    log(f"  {n_loci:,} loci + {n_active:,} non-empty reads", t0)

    # Inverted index: locus position -> all node_ids that contain it.
    # Enables O(|S_i| * avg_fanout) candidate lookup instead of O(N^2).
    log("  Building inverted index ...", t0)
    locus_to_nodes = defaultdict(set)
    for nid, fset in nodes.items():
        for l in fset:
            locus_to_nodes[l].add(nid)

    by_card = defaultdict(list)
    for nid, k in k_val.items():
        by_card[k].append(nid)

    k_max = max(k_val.values())
    parent = {nid: set() for nid in nodes}

    # Process cardinalities highest -> lowest (mirrors generate_poset-2.py).
    # For each node i with locus-set S_i:
    #   - candidates = union of locus_to_nodes[l] for l in S_i
    #   - keep j where nodes[j] is a strict subset of S_i
    #   - mark i as direct parent of j; prune transitives via parent[i]
    log(f"  Computing Hasse diagram (k_max = {k_max}) ...", t0)
    for k in range(k_max, 1, -1):
        nodes_at_k = by_card[k]
        if not nodes_at_k:
            continue
        log(f"    cardinality {k:4d}: {len(nodes_at_k):>8,} nodes", t0)
        for i in nodes_at_k:
            S_i = nodes[i]
            candidates: set = set()
            for l in S_i:
                candidates |= locus_to_nodes[l]
            for j in candidates:
                if k_val[j] < k and nodes[j].issubset(S_i):
                    parent[j].add(i)
                    parent[j] -= parent[i]    # prune transitive parents

    return parent, k_val, n_loci


# graph + triangle attributes 

def build_graph(parent, k_val, t0):
    """
    Translate the Hasse diagram into a NetworkX graph.

    Two edge types (matching generate_poset-2.py output3):
      Direct : (i, j)  where j in parent[i]
      2-hop  : (i, k)  where j in parent[i] and k in parent[j]
               These complete the triangular 2-simplices (i, j, k).
    """
    log("  Extracting edges and 2-simplices ...", t0)

    G = nx.Graph()
    G.add_nodes_from(k_val.keys())

    triangles: set = set()
    for i, parents_i in parent.items():
        for j in parents_i:
            G.add_edge(i, j)
            for kk in parent.get(j, set()):
                G.add_edge(i, kk)
                triangles.add(tuple(sorted([i, j, kk])))

    log(f"  Graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges, {len(triangles):,} triangles", t0)

    # Initialize triangle-count attributes
    nx.set_node_attributes(G, 0, "triangles")
    for u, v in G.edges():
        G.edges[u, v]["triangles"] = 0

    edge_tri = defaultdict(int)
    node_tri = defaultdict(int)
    for tri in triangles:
        a, b, c = tri
        for x in (a, b, c):
            node_tri[x] += 1
        for u, v in ((a, b), (a, c), (b, c)):
            if G.has_edge(u, v):
                edge_tri[(min(u, v), max(u, v))] += 1

    for node, cnt in node_tri.items():
        if node in G.nodes:
            G.nodes[node]["triangles"] = cnt
    for u, v in G.edges():
        G.edges[u, v]["triangles"] = edge_tri.get((min(u, v), max(u, v)), 0)

    return G, len(triangles)


# curvature

def compute_curvature(G, n_triangles, t0):
    """
    Forman-Ricci per edge; scalar curvature per node.
    """
    chi = G.number_of_nodes() - G.number_of_edges() + n_triangles
    log(f"  Euler characteristic chi = {chi}", t0)

    log("  Computing edge Forman-Ricci curvatures ...", t0)
    for u, v in G.edges():
        ric = 4 - G.degree(u) - G.degree(v) + 3 * G.edges[u, v]["triangles"]
        G.edges[u, v]["RicE"] = ric

    log("  Computing scalar curvatures ...", t0)
    scalar = {}
    for i in G.nodes():
        if G.degree(i) > 0:
            s = sum(G.edges[u, v]["RicE"] for u, v in G.edges(i))
            scalar[i] = (s, s / G.degree(i))
        else:
            scalar[i] = (0, 0.0)

    return G, scalar, chi


# label helpers

def node_label(nid, n_loci, adata, obs_idx, var_idx):
    """
    Resolve a compact node ID to (label, type).

    obs_idx / var_idx map positions in the filtered H back to rows in
    adata.obs / adata.var, correcting for the IQR and chrom filtering steps.
    """
    if nid < n_loci:
        orig_row = adata.obs.iloc[obs_idx[nid]]
        return str(orig_row["chrom_bin"]), "locus"
    else:
        orig_row = adata.var.iloc[var_idx[nid - n_loci]]
        return str(orig_row["read_index"]), "read"


# output writers

# def write_edge_csv(G, n_loci, adata, obs_idx, var_idx, path):
#     log(f"  Writing edge CSV -> {path}")
#     rows = []
#     for u, v in G.edges():
#         u_lbl, u_type = node_label(u, n_loci, adata, obs_idx, var_idx)
#         v_lbl, v_type = node_label(v, n_loci, adata, obs_idx, var_idx)

#         if u_type == "read" and v_type == "locus":
#             read_lbl = u_lbl
#         elif v_type == "read" and u_type == "locus":
#             read_lbl = v_lbl
#         else:
#             read_lbl = f"{u_lbl}|{v_lbl}"   # read-read or locus-locus

#         rows.append({
#             "node_u_id":      u,
#             "node_u_label":   u_lbl,
#             "node_u_type":    u_type,
#             "node_v_id":      v,
#             "node_v_label":   v_lbl,
#             "node_v_type":    v_type,
#             "read_label":     read_lbl,
#             "triangles":      G.edges[u, v]["triangles"],
#             "edge_curvature": G.edges[u, v]["RicE"],
#         })

    # pd.DataFrame(rows).to_csv(path, index=False)
    # log(f"    {len(rows):,} edges written")


def write_node_csv(G, scalar, n_loci, adata, obs_idx, var_idx, path_edge, path_node):
    log(f"  Writing locus node CSV -> {path_edge} and {path_node}")
    rows_edges = []
    rows_nodes = []
    for nid, (s, s_norm) in scalar.items():
        if nid >= n_loci:    # should be including the hyperedges
            nid_mod = nid - n_loci
            var_row = adata.var.iloc[var_idx[nid_mod]]
            rows_edges.append({
                "read_index":                    nid_mod,
                "read_name":                     var_row.name,
                "order":                         var_row.get("order", ""),
                "read_length_bp":                var_row.get("read_length_bp", ""),
                "degree":                        G.degree(nid_mod),
                "edge_curvature":                s,
                "normalized_edge_curvature":     s_norm,
                "abs_normalized_edge_curvature": abs(s_norm),
            })
            
        else:
            obs_row = adata.obs.iloc[obs_idx[nid]]
            rows_nodes.append({
                "node_id":                       nid,
                "bin_name":                      obs_row.name,
                "chrom_bin":                     obs_row["chrom_bin"],
                "chrom":                         obs_row.get("chrom", ""),
                "bin_start":                     obs_row.get("bin_start", ""),
                "bin_end":                       obs_row.get("bin_end", ""),
                "degree":                        G.degree(nid),
                "scalar_curvature":              s,
                "normalized_scalar_curvature":   s_norm,
                "abs_normalized_edge_curvature": abs(s_norm),
            })

    pd.DataFrame(rows_edges).to_csv(path_edge, index=False)
    log(f"    {len(rows_edges):,} loci written")
    pd.DataFrame(rows_nodes).to_csv(path_node, index=False)
    log(f"    {len(rows_nodes):,} loci written")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Forman-Ricci curvature on a pore-C poset complex."
    )
    parser.add_argument("input",    help="Input .h5ad file")
    parser.add_argument("edge_csv", help="Output CSV: edge curvatures")
    parser.add_argument("node_csv", help="Output CSV: locus scalar curvatures")
    #parser.add_argument("edge_csv_extra", help="Output Csv: edge triangle calculations")
    parser.add_argument(
        "--chrom", default=None,
        help="Restrict to one chromosome, e.g. --chrom chr7  "
             "(recommended for large datasets)"
    )
    args = parser.parse_args()

    t0 = time.time()

    log("[Step 0] Loading and filtering H ...")
    adata, H, obs_idx, var_idx = load_and_filter(args.input, args.chrom)

    log("\n[Step 1] Building poset complex ...", t0)
    parent, k_val, n_loci = build_poset(H, t0)

    log("\n[Step 2] Building graph ...", t0)
    G, n_tri = build_graph(parent, k_val, t0)

    log("\n[Step 3] Computing curvatures ...", t0)
    G, scalar, chi = compute_curvature(G, n_tri, t0)

    log("\n[Step 4] Writing outputs ...", t0)
    #write_edge_csv(G, n_loci, adata, obs_idx, var_idx, args.edge_csv_extra)
    write_node_csv(G, scalar, n_loci, adata, obs_idx, var_idx, args.edge_csv, args.node_csv)

    log(f"\nFinished.  chi = {chi}", t0)


if __name__ == "__main__":
    main()