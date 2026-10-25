"""Module for constructing synthetic sparse Boolean matrices."""

from os.path import join
from typing import Callable

from numpy import arange, ndarray, ones, sqrt, where, zeros
from numpy.random import default_rng, Generator
from scipy.io import mmwrite
from scipy.sparse import coo_array, csc_array, csr_array,  vstack

from filepaths import BASE_DIR
from random_seeds import TRAIN_SEEDS, BAE_SEED
from rerank_util import ensure_boolean_values


def add_noise(M: coo_array, rng: Generator, p_flip_1: float, p_flip_0: float) -> csr_array:
    """Returns a sparse Boolean matrix generated from 'mat' by simultaneously 
    flipping 'p_flip_1' % of ones to zeros and 'p_flip_0' % of zeros to ones."""

    total_idx: int = M.shape[0] * M.shape[1]
    initial_ones: int = M.nnz

    # Flip ones to zeros with probability 'p_flip_1'.
    ones_mask: ndarray = rng.random(initial_ones) > p_flip_1
    # 'mat' where 'p_flip_1' % of ones have been flipped to zero.
    rows_1: ndarray = M.row[ones_mask]
    cols_1: ndarray = M.col[ones_mask]
    ones_after_flip: ndarray = zeros(M.shape, dtype=bool)
    ones_after_flip[rows_1, cols_1] = True
    
    # Flip zeros to ones with probability 'p_flip_0'.
    zeros_mask: ndarray = rng.random(total_idx - initial_ones) > p_flip_0
    # 'mat' where 'p_flip_0' % of zeros have been flipped to one
    initial_rows_0: ndarray; initial_cols_0: ndarray
    initial_rows_0, initial_cols_0 = where(M.toarray() == 0)
    rows_0: ndarray = initial_rows_0[zeros_mask]
    cols_0: ndarray = initial_cols_0[zeros_mask]
    zeros_after_flip: ndarray = ones(M.shape, dtype=bool)
    zeros_after_flip[rows_0, cols_0] = False

    # OR of the flipped ones and flipped zeros.
    A = csr_array(ones_after_flip + zeros_after_flip)
    ensure_boolean_values(A)
    return A


def get_column_structure(
    rng: Generator, 
    n_cols: int = 800, 
    n_sets: int = 5, 
    level: int = 3,
    density: float = 0.3, 
    sub_density: float = 0.8
) -> list[list[ndarray]]:
    """Returns a nested column-index partition tree for constructing instances 
    of nested indices data where indices vary only inside the fixed nested 
    structure."""

    col_tree: list[list[ndarray]] = []

    def recurse(prev_cols: list[ndarray], curr_lvl: int) -> None:
        """Selects sets of indices from 'prev_cols' for building instances 
        of nested indices data in which only the indices inside the nested 
        structure vary."""
        
        target_step: float = density if curr_lvl == 1 else sub_density
        p: float = sqrt(1 - (1 - target_step) ** (1 / n_sets))
        current_lvl_cols: list[ndarray] = []
        for p_cols in prev_cols:
            for _ in range(n_sets):
                n_c: int = max(1, int(len(p_cols) * p))
                current_lvl_cols.append(rng.choice(p_cols, size=n_c, replace=False))
        col_tree.append(current_lvl_cols)
        if curr_lvl < level:
            recurse(current_lvl_cols, curr_lvl + 1)

    recurse([arange(n_cols)], 1)
    return col_tree  # col_tree[-1]: n_sets**level column-index arrays


def get_instance(
    rng: Generator, 
    col_tree: list[list[ndarray]], 
    n_rows: int = 1000,
    n_cols: int = 800,
    n_sets: int = 5, 
    level: int = 3, 
    density: float = 0.3, 
    sub_density: float = 0.8,
    factors: bool = False
) -> csr_array | tuple[csr_array, list[csc_array], list[csr_array]]:
    """Samples a row structure for the given column schema and returns an
    instance of nested indices data. R_factors are identical for every 
    instance sharing 'col_tree'. L_factors (and hence the data) vary."""

    L_factors: list[csr_array] = [] 
    R_factors: list[csr_array] = []

    def recurse(prev_rows: list[ndarray], curr_lvl: int):
        """Selects sets of indices (matrices) from 'prev_rows' and uses them 
        to build factors of each level, appending them to 'L_factors' and 
        'R_factors'."""

        target_step: float = density if curr_lvl == 1 else sub_density
        p: float = sqrt(1 - (1 - target_step) ** (1 / n_sets))
        lvl_cols: list[ndarray] = col_tree[curr_lvl - 1]
        current_lvl_rows: list[ndarray] = []
        l_rows: list[int]; l_cols: list[int]; r_rows: list[int]; r_cols: list[int]
        l_rows, l_cols, r_rows, r_cols = [], [], [], []
        rank_idx: int = 0
        for p_rows in prev_rows:
            for _ in range(n_sets):
                n_r: int = max(1, int(len(p_rows) * p))
                sel_rows: ndarray = rng.choice(p_rows, size=n_r, replace=False)
                current_lvl_rows.append(sel_rows)
                for r in sel_rows:
                    l_rows.append(r); l_cols.append(rank_idx)
                for c in lvl_cols[rank_idx]:          
                    r_rows.append(rank_idx); r_cols.append(c)
                rank_idx += 1

        rank_total: int = n_sets ** curr_lvl
        L: csr_array = coo_array((ones(len(l_rows)), (l_rows, l_cols)), shape=(n_rows, rank_total)).tocsc()
        R: csr_array = coo_array((ones(len(r_rows)), (r_rows, r_cols)), shape=(rank_total, n_cols)).tocsr()
        ensure_boolean_values(L)
        ensure_boolean_values(R)
        L_factors.append(L)
        R_factors.append(R)
        if curr_lvl < level:
            recurse(current_lvl_rows, curr_lvl + 1)

    recurse([arange(n_rows)], 1)

    A = csr_array(L_factors[-1] @ R_factors[-1])
    ensure_boolean_values(A)
    return (A, L_factors, R_factors) if factors else A


def get_nested_data(
    seeds: list[int],
    n_rows: int = 1000,
    n_cols: int = 800,
    n_sets: int = 5,
    level: int = 3,
    density: float = 0.3,
    sub_density: float = 0.8,
) -> csr_array:
    """Returns vertically stacked nested indices data instances which share 
    the same fixed column schema, while their row structures are sampled from 
    'seeds'. Returned data shape is len(seeds) * n_rows x n_cols."""

    schema_rng: Generator = default_rng(BAE_SEED)
    columns: list[list[ndarray]] = get_column_structure(schema_rng)
    instances: list[csr_array] = []

    for seed in seeds:
        A: csr_array = get_instance(default_rng(seed), columns)
        instances.append(A)

    A_stack: csr_array = vstack(instances, format="csr")
    ensure_boolean_values(A_stack)

    return A_stack


def main() -> None:
    """Runs the synthetic data function and saves the result."""
    rng: Generator = default_rng(seed=None)
    func: Callable = get_nested_data
    p_flip_1: float = 0.01
    p_flip_0: float = 0.01
    #movies_per_tag: int = 150
    data: csr_array = func()
    noisy: csr_array = add_noise(data, p_flip_1, p_flip_0)
    filename: str = join(BASE_DIR, f"{func.__name__}_{p_flip_1}_{p_flip_0}.mtx") 
    mmwrite(filename, noisy)
    

if __name__ == "__main__":
    main()
