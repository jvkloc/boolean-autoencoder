"""
Data: every matrix is a scipy CSR Boolean matrix (rows = samples, columns = items).

Folders (next to src/):
    data/              real data: .mtx (Matrix Market, as in deep BMF), .npz, .csv/.tsv/.txt
    synthetic data/    the nested instances, built once by build_synthetic.py
    results/           experiment output

get_split(spec) returns (train, validation, test, info) for an experiment:
    {"dataset": "mammals_europe"}       a file in data/ (any supported extension); rows split
                                        once into 64% train / 16% validation / 20% test
    {"dataset": "nested", "instance": i, "noise": p}
                                        train instance i split 80% train / 20% validation,
                                        test = test instance i (same factors, new rows);
                                        p = None (clean) or 0.01, 0.02, 0.04, 0.08
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # bae/
DATA_DIR = os.path.join(ROOT, "data")
SYNTH_DIR = os.path.join(ROOT, "synthetic data")
RESULTS_DIR = os.path.join(ROOT, "results")


def to_csr(M) -> sp.csr_array:
    M = sp.csr_array(M)
    M.data = M.data != 0
    M.eliminate_zeros()
    M = M.astype(bool)
    M.sort_indices()
    return M


def write_mtx(path: str, M):
    scipy.io.mmwrite(path, sp.coo_array(to_csr(M)).astype(np.int8), field="pattern")


def read_mtx(path: str) -> sp.csr_array:
    return to_csr(scipy.io.mmread(path))


def load_matrix(path: str, transpose: bool = False, return_info: bool = False):
    """Boolean CSR matrix from .mtx, .npz (scipy sparse or numpy), .npy or a text table.
    Text tables: delimiter (; , tab or space) and header detected; columns that are not
    purely 0/1 (coordinates, ids) are set aside in info["dropped"]."""
    ext = os.path.splitext(path)[1].lower()
    info = {"columns": None, "dropped": {}}
    if ext == ".mtx" or path.endswith(".mtx.gz"):
        M = read_mtx(path)
    elif ext == ".npz":
        try:
            M = sp.load_npz(path)
        except (ValueError, KeyError):
            f = np.load(path)
            M = f[f.files[0]]
    elif ext == ".npy":
        M = np.load(path)
    else:
        with open(path, newline="") as f:
            text = f.read()
        first = text.splitlines()[0]
        delim = max([";", ",", "\t"], key=first.count)
        rows = ([line.split() for line in text.splitlines() if line.strip()] if first.count(delim) == 0
                else [r for r in csv.reader(text.splitlines(), delimiter=delim) if r])

        def num(t):
            return t.strip().lstrip("-").replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1).isdigit()

        if not all(num(t) for t in rows[0] if t.strip()):
            header, rows = [h.strip() for h in rows[0]], rows[1:]
        else:
            header = [f"col{j}" for j in range(len(rows[0]))]
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = header + [f"col{j}" for j in range(len(header), width)]
        cols = list(zip(*rows))
        keep = []
        for j, col in enumerate(cols):
            vals = {v.strip() for v in col}
            if vals <= {"0", "1", "0.0", "1.0"}:
                keep.append(j)
            elif vals != {""}:
                info["dropped"][header[j]] = [v.strip() for v in col]
        M = np.array([[cols[j][i].strip() in ("1", "1.0") for j in keep] for i in range(len(rows))])
        info["columns"] = [header[j] for j in keep]
    M = to_csr(M)
    if transpose:
        M = to_csr(M.T)
    return (M, info) if return_info else M


def find_data_file(name: str) -> str:
    if os.path.exists(name):
        return name
    hits = sorted(glob.glob(os.path.join(DATA_DIR, name + ".*")))
    if not hits:
        raise FileNotFoundError(f"no data file {name}.* in {DATA_DIR}")
    prefer = [h for h in hits if h.endswith(".mtx")]
    return (prefer or hits)[0]


def split_indices(n: int, fractions=(0.64, 0.16, 0.20), seed: int = 0):
    """Fixed random split of n rows into train / validation / test index arrays."""
    perm = np.random.default_rng(10_000 + seed).permutation(n)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    return np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + n_va]), np.sort(perm[n_tr + n_va:])


def nested_path(split: str, instance: int, noise) -> str:
    name = "clean" if noise is None else f"noise_{noise}"
    return os.path.join(SYNTH_DIR, split, f"id{instance:03d}", name + ".mtx")


def get_split(spec: dict):
    """(train, validation, test, info) as CSR matrices; see the module docstring."""
    if spec["dataset"] == "nested":
        inst, noise = spec["instance"], spec.get("noise")
        Xa = read_mtx(nested_path("train", inst, noise))
        Xte = read_mtx(nested_path("test", inst, noise))
        tr, va, _ = split_indices(Xa.shape[0], (0.8, 0.2, 0.0), seed=inst)
        info = {}
        if noise is not None:                         # error of the planted (clean) matrix
            for split, X in (("train", Xa), ("test", Xte)):
                clean = read_mtx(nested_path(split, inst, None))
                err = int((clean != X).sum())
                info[f"planted_error_per_one_{split}"] = err / max(1, int(X.sum()))
        return Xa[tr], Xa[va], Xte, info
    X = load_matrix(find_data_file(spec["dataset"]))
    tr, va, te = split_indices(X.shape[0])
    return X[tr], X[va], X[te], {}
