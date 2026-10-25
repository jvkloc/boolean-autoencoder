"""
Builds the nested synthetic data once, into bae/synthetic data/:

    synthetic data/
        schema.npz              R1 (5 x 800), R2 (25 x 800), R3 (125 x 800): the column
                                structure (factors) shared by every instance, from BAE_SEED
        train/id001/            instance 1, rows from TRAIN_SEEDS[0]
            clean.mtx           1000 x 800, = L3 ∘ R3
            noise_0.01.mtx      clean with 1% of ones -> 0 and 1% of zeros -> 1
            noise_0.02.mtx, noise_0.04.mtx, noise_0.08.mtx
            factors.npz         L1, L2, L3: row factors (1000 x 5, 1000 x 25, 1000 x 125), seed
        ...
        test/id001/ ...         instance 1, rows from TEST_SEEDS[0] (same factors, new rows)

Matrices are Matrix Market pattern files (scipy.io.mmread / mmwrite), as in deep BMF.

The structure is generated exactly as get_column_structure / get_instance in
synthetic_data.py (same random calls in the same order), using numpy only so the bae
environment needs no scipy. The noise uses its own generator per (instance seed, noise
level), so every noisy matrix can be rebuilt bit for bit.

Run once:  python build_synthetic.py      (finished instances are kept; a few minutes)
"""

from __future__ import annotations

import os

import numpy as np
from numpy.random import Generator, default_rng

from data import SYNTH_DIR, write_mtx
from random_seeds import BAE_SEED, TEST_SEEDS, TRAIN_SEEDS

N_ROWS, N_COLS, N_SETS, LEVEL, DENSITY, SUB_DENSITY = 1000, 800, 5, 3, 0.3, 0.8
NOISE_LEVELS = (0.01, 0.02, 0.04, 0.08)        # p_flip_1 = p_flip_0


def _step(curr_lvl: int) -> float:
    target = DENSITY if curr_lvl == 1 else SUB_DENSITY
    return np.sqrt(1 - (1 - target) ** (1 / N_SETS))


def column_structure(rng: Generator) -> list[list[np.ndarray]]:
    """Same as synthetic_data.get_column_structure."""
    col_tree = []

    def recurse(prev_cols, curr_lvl):
        p = _step(curr_lvl)
        current = []
        for p_cols in prev_cols:
            for _ in range(N_SETS):
                n_c = max(1, int(len(p_cols) * p))
                current.append(rng.choice(p_cols, size=n_c, replace=False))
        col_tree.append(current)
        if curr_lvl < LEVEL:
            recurse(current, curr_lvl + 1)

    recurse([np.arange(N_COLS)], 1)
    return col_tree


def schema_factors(col_tree) -> list[np.ndarray]:
    Rs = []
    for lvl_cols in col_tree:
        R = np.zeros((len(lvl_cols), N_COLS), dtype=bool)
        for i, cols in enumerate(lvl_cols):
            R[i, cols] = True
        Rs.append(R)
    return Rs


def instance(rng: Generator, col_tree) -> list[np.ndarray]:
    """Row factors L1..L3 of one instance; same random calls as synthetic_data.get_instance."""
    Ls = []

    def recurse(prev_rows, curr_lvl):
        p = _step(curr_lvl)
        L = np.zeros((N_ROWS, N_SETS ** curr_lvl), dtype=bool)
        current = []
        rank_idx = 0
        for p_rows in prev_rows:
            for _ in range(N_SETS):
                n_r = max(1, int(len(p_rows) * p))
                sel = rng.choice(p_rows, size=n_r, replace=False)
                current.append(sel)
                L[sel, rank_idx] = True
                rank_idx += 1
        Ls.append(L)
        if curr_lvl < LEVEL:
            recurse(current, curr_lvl + 1)

    recurse([np.arange(N_ROWS)], 1)
    return Ls


def add_noise(X: np.ndarray, rng: Generator, p_flip_1: float, p_flip_0: float) -> np.ndarray:
    """Flip each 1 to 0 with probability p_flip_1 and each 0 to 1 with probability p_flip_0."""
    u = rng.random(X.shape)
    return np.where(X, u > p_flip_1, u <= p_flip_0)


def build_one(folder: str, seed: int, col_tree, Rs):
    os.makedirs(folder, exist_ok=True)
    Ls = instance(default_rng(seed), col_tree)
    clean = (Ls[-1].astype(np.int32) @ Rs[-1].astype(np.int32)) > 0
    write_mtx(os.path.join(folder, "clean.mtx"), clean)
    for p in NOISE_LEVELS:
        rng = default_rng([seed, int(round(p * 10_000))])
        write_mtx(os.path.join(folder, f"noise_{p}.mtx"), add_noise(clean, rng, p, p))
    np.savez_compressed(os.path.join(folder, "factors.npz"), seed=np.array(str(seed)),
                        **{f"L{i}": L for i, L in enumerate(Ls, 1)})    # written last: marks done


def main():
    col_tree = column_structure(default_rng(BAE_SEED))
    Rs = schema_factors(col_tree)
    os.makedirs(os.path.join(SYNTH_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(SYNTH_DIR, "test"), exist_ok=True)
    schema = os.path.join(SYNTH_DIR, "schema.npz")
    if not os.path.exists(schema):
        np.savez_compressed(schema, **{f"R{i}": R for i, R in enumerate(Rs, 1)})
    built = 0
    for split, seeds in (("train", TRAIN_SEEDS), ("test", TEST_SEEDS)):
        for i, seed in enumerate(seeds, 1):
            folder = os.path.join(SYNTH_DIR, split, f"id{i:03d}")
            if not os.path.exists(os.path.join(folder, "factors.npz")):
                build_one(folder, seed, col_tree, Rs)
                built += 1
    print(f"{built} instance files built in {SYNTH_DIR} "
          f"({2 * len(TRAIN_SEEDS) - built} already there)")


if __name__ == "__main__":
    main()
