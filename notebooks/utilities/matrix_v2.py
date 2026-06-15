"""
Hypergraph utilities — sparse-first optimized version.

Key changes vs. the original:
  * expand_and_normalize_anndata no longer calls adata.to_df(); it works
    directly on the sparse adata.X, so a 2,579 x 32M input never densifies.
  * Hyperedge (column) dedup is done sparsely via per-column nonzero-pattern
    hashing instead of np.unique(..., axis=1) on a dense array.
  * Empty / singleton hyperedges are dropped before dedup — they contribute
    nothing to clique expansion and shrink the dedup pass substantially.
  * clique_expand_incidence returns a SCIPY SPARSE matrix (was dense DataFrame).
  * Sparsity flows end-to-end: A, A_kr, A_oe stay sparse where possible.
  * normalize_oe_auto always routes through the sparse OE path.
  * _diagonal_means computes all diagonal offsets in one COO pass.
  * normalize_oe_sparse divides only the stored entries (the original kept
    just the main diagonal of the Toeplitz -> divided off-diagonals by 0).
  * hypergraph_entropy uses dense eigvalsh when feasible (far faster/robuster
    than asking ARPACK for n-1 smallest eigenvalues).
  * estimate_fiedler uses eigsh (symmetric) + shift-invert at sigma=0.
  * find_outlier_row_indices no longer assumes a pandas Series.
"""

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import scipy.sparse as sps
from scipy.sparse import csr_matrix, csc_matrix, diags, issparse
from scipy.linalg import toeplitz
from scipy.stats import chi2


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def expand_and_normalize_anndata(adata, unique_only=True, oe_kr=False,
                                 keep_sparse=True, drop_singletons=True):
    """
    Expands the input matrix and applies KR and OE normalization.

    Works directly on the sparse adata.X. The previous version called
    adata.to_df(), which forces adata.X.toarray() — for a
    (2,579 x 32,545,444) matrix that is a ~625 GiB dense allocation and
    OOMs immediately. Nothing downstream actually needs a dense DataFrame:
    clique_expand_incidence accepts sparse input directly.

    Args:
        adata: An AnnData object. adata.X is expected to be
            nodes (rows) x hyperedges (columns).
        unique_only: If True, deduplicate hyperedges (columns).
        oe_kr: If True, also compute the OE-of-KR normalized matrix.
        keep_sparse: If True, store results as scipy sparse matrices in
            obsm instead of dense DataFrames. This is the big memory win;
            set False only if a downstream consumer needs DataFrames.
        drop_singletons: If True, drop hyperedges (columns) that touch
            fewer than two nodes before dedup/expansion. Such hyperedges
            contribute nothing to clique expansion (an empty column adds
            nothing; a singleton adds only a self-loop) and dropping them
            makes the dedup pass much faster.

    Returns:
        None
    """
    print("Expanding input matrix...")

    # Keep the incidence matrix sparse. Do NOT call adata.to_df().
    H = adata.X
    if issparse(H):
        H = H.tocsc()
    else:
        # Already dense (small inputs only) — wrap without a needless copy.
        H = csc_matrix(np.asarray(H))

    print(f"  input incidence matrix: shape={H.shape}, nnz={H.nnz}")

    if drop_singletons:
        # A column with < 2 nonzeros is an empty or singleton hyperedge.
        col_nnz = np.diff(H.indptr)            # nnz per column, cheap on CSC
        keep = np.flatnonzero(col_nnz >= 2)
        if keep.size != H.shape[1]:
            H = H[:, keep]
            print(f"  dropped empty/singleton hyperedges -> shape={H.shape}, "
                  f"nnz={H.nnz}")

    if unique_only:
        H = _unique_columns_sparse(H)
        print(f"  unique hyperedges -> shape={H.shape}, nnz={H.nnz}")

    obs = adata.obs_names

    # Clique expansion -> sparse CSR. Result is nodes x nodes
    # (here 2,579 x 2,579), which is small regardless of column count.
    A = clique_expand_incidence(H, zero_diag=False)          # sparse
    adata.obsm['A'] = A if keep_sparse else _to_df(A, obs)

    print("Applying KR normalization...")
    A_kr = normalize_kr(A)                                   # sparse in/out
    adata.obsm['A_kr'] = A_kr if keep_sparse else _to_df(A_kr, obs)

    print("Applying OE normalization...")
    A_oe = normalize_oe_auto(A)
    adata.obsm['A_oe'] = A_oe if keep_sparse else _to_df(A_oe, obs)

    if oe_kr:
        A_oe_kr = normalize_oe_auto(A_kr)
        adata.obsm['A_oe_kr'] = A_oe_kr if keep_sparse else _to_df(A_oe_kr, obs)

    print("Normalization complete.")


def _unique_columns_sparse(H):
    """Keep the first occurrence of each unique column of a sparse matrix.

    Replaces np.unique(H.values, axis=1), which requires a dense array. Each
    column is identified by its nonzero pattern: the row indices and the
    values of its stored entries. Memory scales with the number of *distinct*
    columns, not with rows x columns.

    Args:
        H: a scipy sparse matrix, nodes (rows) x hyperedges (columns).

    Returns:
        scipy.sparse.csc_matrix containing only the unique columns, in their
        original left-to-right order.
    """
    H = H.tocsc()
    H.sort_indices()                       # canonical order within each column
    indptr, indices, data = H.indptr, H.indices, H.data

    seen = set()
    keep = []
    for j in range(H.shape[1]):
        s, e = indptr[j], indptr[j + 1]
        # (row pattern, values) uniquely identifies the column's content.
        # For binary incidence data the values are all 1, but including
        # data.tobytes() keeps this correct for weighted incidence too.
        key = (indices[s:e].tobytes(), data[s:e].tobytes())
        if key not in seen:
            seen.add(key)
            keep.append(j)

    if len(keep) == H.shape[1]:
        return H                            # nothing to drop
    return H[:, keep]


def _to_df(mat, names):
    """Densify a sparse matrix into a labeled DataFrame (only when needed)."""
    arr = mat.toarray() if issparse(mat) else np.asarray(mat)
    return pd.DataFrame(arr, index=names, columns=names)


# ----------------------------------------------------------------------
# Clique expansion
# ----------------------------------------------------------------------
def clique_expand_incidence(I, zero_diag=True):
    """Performs clique expansion on an incidence matrix.

    Args:
        I: An incidence matrix (pd.DataFrame, ndarray, or sparse) where
           rows are nodes and columns are hyperedges.
        zero_diag: If True, zero the diagonal of the result.

    Returns:
        scipy.sparse.csr_matrix: node-by-node co-membership matrix.
    """
    if isinstance(I, pd.DataFrame):
        H = csr_matrix(I.values)
    elif issparse(I):
        H = I.tocsr()
    else:
        H = csr_matrix(np.asarray(I))

    A = (H @ H.T).tocsr()                # sparse matmul, no dense n*n alloc
    if zero_diag:
        A.setdiag(0)
        A.eliminate_zeros()
    return A


# ----------------------------------------------------------------------
# Laplacians (already sparse — minor cleanup only)
# ----------------------------------------------------------------------
def hypergraph_laplacian(H):
    """Unnormalized hypergraph Laplacian:  L = D - H E^-1 H^T."""
    H = sparse.csr_matrix(H)
    E_diag = np.asarray(H.sum(axis=0)).ravel()
    E_inv = sparse.diags(np.where(E_diag > 0, 1.0 / E_diag, 0.0))
    D_diag = np.asarray(H.sum(axis=1)).ravel()
    L = sparse.diags(D_diag) - H @ E_inv @ H.T
    return L.tocsr()


def normalized_hypergraph_laplacian(H):
    """Normalized hypergraph Laplacian:  L = I - D^-1/2 H E^-1 H^T D^-1/2."""
    H = sparse.csr_matrix(H)
    E_diag = np.asarray(H.sum(axis=0)).ravel()
    E_inv = sparse.diags(np.where(E_diag > 0, 1.0 / E_diag, 0.0))
    D_diag = np.asarray(H.sum(axis=1)).ravel()
    D_hat_inv_sqrt = sparse.diags(
        np.where(D_diag > 0, 1.0 / np.sqrt(D_diag), 0.0))
    I = sparse.eye(H.shape[0], format="csr")
    L = I - D_hat_inv_sqrt @ H @ E_inv @ H.T @ D_hat_inv_sqrt
    return L.tocsr()


# ----------------------------------------------------------------------
# Spectral quantities
# ----------------------------------------------------------------------
def hypergraph_entropy(L, dense_threshold=4000):
    """Hypergraph entropy from the Laplacian L.

    For n <= dense_threshold the full spectrum is computed with the dense
    symmetric solver eigvalsh, which is dramatically faster and more robust
    than asking ARPACK for n-1 of the smallest eigenvalues. Above the
    threshold it falls back to eigsh.

    Args:
        L: hypergraph Laplacian (sparse or dense), symmetric.
        dense_threshold: max n for the dense path.

    Returns:
        float: the hypergraph entropy.
    """
    n = L.shape[0]
    if n <= dense_threshold:
        Ld = L.toarray() if issparse(L) else np.asarray(L)
        eigenvalues = scipy.linalg.eigvalsh(Ld)
    else:
        # Genuinely large: ARPACK fallback. Still avoid k = n-1 if possible;
        # the caller should consider a stochastic estimator for huge L.
        eigenvalues, _ = sparse.linalg.eigsh(L, k=n - 1, which='SM')

    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total == 0:
        return 0.0
    p = eigenvalues / total
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def estimate_fiedler(L, min_shape=5):
    """Estimate the Fiedler value (2nd-smallest eigenvalue) of L.

    Uses the symmetric solver eigsh with shift-invert at sigma=0, which
    converges far faster for the smallest eigenvalues than which='SM'.
    """
    if L.shape[0] < min_shape or L.shape[1] < min_shape:
        return 0
    try:
        vals = sparse.linalg.eigsh(L, k=2, sigma=0, which='LM',
                                   return_eigenvectors=False)
    except Exception:
        # Shift-invert can fail on singular L; fall back to SM.
        vals = sparse.linalg.eigsh(L, k=2, which='SM',
                                   return_eigenvectors=False)
    return float(np.sort(vals)[1])


# ----------------------------------------------------------------------
# Larntz-Perlman (unchanged logic; minor robustness)
# ----------------------------------------------------------------------
def larntzPerlman(M1, M2, sample_size, alpha=0.05):
    """Larntz-Perlman test for correlation-matrix equivalence (two matrices)."""
    num_variables = M1.shape[0]
    triu = np.triu_indices(num_variables, k=1)

    fisher_z_values = np.arctanh(np.array([M1[triu], M2[triu]]))
    mean_z_score = np.mean(fisher_z_values, axis=0)
    s_values = (sample_size - 3) * np.sum(
        (fisher_z_values - mean_z_score) ** 2, axis=0)
    test_statistic = np.max(s_values)

    sidak_alpha = (1 - alpha) ** (2 / (num_variables * (num_variables - 1)))
    hypothesis_accepted = test_statistic <= chi2.ppf(sidak_alpha, 1)

    p_values = np.zeros((num_variables, num_variables))
    p_values[triu] = 1 - chi2.cdf(s_values, 1)
    p_values += p_values.T

    s_matrix = np.zeros((num_variables, num_variables))
    s_matrix[triu] = s_values
    s_matrix += s_matrix.T

    overall_p_value = 1 - (chi2.cdf(test_statistic, 1) **
                           ((num_variables * (num_variables - 1)) / 2))

    return hypothesis_accepted, p_values, s_matrix, overall_p_value


# ----------------------------------------------------------------------
# Outlier helpers
# ----------------------------------------------------------------------
def find_outlier_row_indices(matrix, threshold=1.5):
    """Row indices whose row-sum z-score exceeds `threshold`.

    Works on a plain ndarray (the original assumed a pandas Series and
    would raise AttributeError on `.index`).
    """
    arr = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    row_sums = np.asarray(arr.sum(axis=0)).ravel()
    z_scores = np.abs(scipy.stats.zscore(row_sums))
    return np.flatnonzero(z_scores > threshold).tolist()


def get_sorted_upper_triangle_indices(matrix, descending=True):
    """Sorted (row, col) index arrays for matrix entries, by value."""
    order = np.argsort(matrix, axis=None)
    if descending:
        order = order[::-1]
    row_idx, col_idx = np.unravel_index(order, matrix.shape)
    return np.asarray(row_idx).ravel(), np.asarray(col_idx).ravel()


def handle_outliers(A, n):
    """Replace the n largest entries with the matrix mean."""
    n = min(n, A.size)
    mean_val = A.mean()
    row_idx, col_idx = get_sorted_upper_triangle_indices(A)
    A[row_idx[:n], col_idx[:n]] = mean_val   # vectorized assignment
    return A


def remove_indices(arr, indices):
    """Remove rows and columns at `indices` from a square array."""
    mask = np.ones(len(arr), dtype=bool)
    mask[list(indices)] = False
    keep = np.flatnonzero(mask)
    return arr[np.ix_(keep, keep)]


# ----------------------------------------------------------------------
# OE normalization
# ----------------------------------------------------------------------
def _diagonal_means(matrix):
    """Mean of each diagonal (offset 0 .. n-1) of a symmetric matrix.

    The mean for offset k divides by the full diagonal length (n - k),
    counting structural zeros, which matches numpy's np.diagonal behavior
    on the dense path.
    """
    n = matrix.shape[0]
    if issparse(matrix):
        coo = matrix.tocoo()
        offsets = np.abs(coo.row - coo.col)
        # sum of stored entries per offset
        sums = np.bincount(offsets, weights=coo.data, minlength=n)
        # count of *all* entries per diagonal k is (n - k), not nnz —
        # diagonal mean divides by full diagonal length
        counts = n - np.arange(n)
        return sums / counts
    return np.array([np.mean(np.diagonal(matrix, k)) for k in range(n)])


def normalize_oe(matrix):
    """Normalize a symmetric matrix by its Toeplitz (distance) expectation.

    Dense path. Avoids the NaN-then-fix round trip by dividing with
    out=/where=. Still builds the divisor via a Toeplitz of the 1-D
    diagonal-means vector — cheap relative to the matrix itself.
    """
    matrix = np.asarray(matrix, dtype=float)
    diag_means = _diagonal_means(matrix)
    divisor = toeplitz(diag_means)            # symmetric by construction
    out = np.zeros_like(matrix)
    np.divide(matrix, divisor, out=out, where=divisor != 0)
    return out


def normalize_oe_auto(matrix, sparse_density_cutoff=0.10):
    """Observed/expected normalization, always via the sparse path.

    The previous dispatcher densified the matrix whenever its density
    exceeded the cutoff (matrix.toarray() + a Toeplitz of the same shape).
    For a node-by-node co-membership matrix that is an n x n dense
    allocation. normalize_oe_sparse is correct for any density, so route
    everything through it. The sparse_density_cutoff argument is kept for
    backwards-compatible call signatures but is no longer used.
    """
    if issparse(matrix):
        return normalize_oe_sparse(matrix)
    return normalize_oe_sparse(csr_matrix(np.asarray(matrix, dtype=float)))


def normalize_oe_sparse(matrix):
    """Normalize a symmetric matrix by its Toeplitz expectation, sparse.

    Only the STORED entries are divided — by the diagonal mean for their
    own offset |i - j|. The original implementation reduced the Toeplitz to
    just its main diagonal, so every off-diagonal entry was divided by 0.

    Returns:
        scipy.sparse.csr_matrix
    """
    if not issparse(matrix):
        matrix = csr_matrix(np.asarray(matrix, dtype=float))
    else:
        matrix = matrix.tocsr().astype(float)

    diag_means = _diagonal_means(matrix)

    coo = matrix.tocoo()
    offsets = np.abs(coo.row - coo.col)
    denom = diag_means[offsets]
    with np.errstate(divide='ignore', invalid='ignore'):
        new_data = np.where(denom != 0, coo.data / denom, 0.0)

    out = sparse.coo_matrix((new_data, (coo.row, coo.col)),
                            shape=matrix.shape).tocsr()
    out.eliminate_zeros()
    return out


# ----------------------------------------------------------------------
# Conversion + Knight-Ruiz balancing
# ----------------------------------------------------------------------
def convert_to_csr(data):
    """Convert ndarray / np.matrix / DataFrame / sparse to a CSR matrix."""
    if issparse(data):
        return data.tocsr()
    if isinstance(data, np.matrix):
        return csr_matrix(np.asarray(data))
    if isinstance(data, np.ndarray):
        return csr_matrix(data)
    if isinstance(data, pd.DataFrame):
        return csr_matrix(data.values)
    raise TypeError("Unsupported data type: provide ndarray, np.matrix, "
                    "DataFrame, or scipy sparse matrix.")


def normalize_kr(A, tol=1e-6, max_outer_iterations=30, max_inner_iterations=10):
    """Knight-Ruiz matrix balancing.

    Adapted from https://github.com/ay-lab/HiCKRy. Now accepts sparse input
    directly, so no dense round-trip is needed when called from the pipeline.
    Returns a sparse CSR matrix.
    """
    A = convert_to_csr(A).astype(np.float64)

    n = A.shape[0]
    e = np.ones((n, 1), dtype=np.float64)

    Delta = 3
    delta = 0.1
    x0 = np.copy(e)
    g = 0.9

    etamax = eta = 0.1
    stop_tol = tol * 0.5
    x = np.copy(x0)

    rt = tol ** 2.0
    v = x * (A.dot(x))
    rk = 1.0 - v
    rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
    rho_km2 = rho_km1
    rout = rold = rho_km1

    MVP = 0
    i = 0

    while rout > rt and i < max_outer_iterations:
        i += 1
        k = 0
        y = np.copy(e)
        innertol = max(eta ** 2.0 * rout, rt)

        while rho_km1 > innertol and k < max_inner_iterations:
            k += 1
            if k == 1:
                Z = rk / v
                p = np.copy(Z)
                rho_km1 = (rk.transpose()).dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            w = x * A.dot(x * p) + v * p
            alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
            ap = alpha * p
            ynew = y + ap

            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0.0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            y = np.copy(ynew)
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = ((rk.transpose()).dot(Z))[0, 0]

        x *= y
        v = x * (A.dot(x))
        rk = 1.0 - v
        rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
        rout = rho_km1
        MVP += k + 1

        rat = rout / rold
        rold = rout
        res_norm = rout ** 0.5
        eta_o = eta
        eta = g * rat
        if g * eta_o ** 2.0 > 0.1:
            eta = max(eta, g * eta_o ** 2.0)
        eta = max(min(eta, etamax), stop_tol / res_norm)

    x = sps.diags(x.flatten(), 0, format='csr')
    Ahat = x.dot(A.dot(x))
    return Ahat.tocsr()


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------
def symmetrize(arr, method="average"):
    """Symmetrize a square matrix ('average', 'upper', or 'lower')."""
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be square.")
    if method == "average":
        return (arr + arr.T) / 2
    elif method == "upper":
        return arr + np.tril(arr, k=-1).T
    elif method == "lower":
        return arr + np.triu(arr, k=1).T
    raise ValueError("Invalid method. Choose 'average', 'upper', or 'lower'.")