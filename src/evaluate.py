"""
Evaluation, in row chunks so that large data never needs a full dense copy.

errors(B, C, X)             Hamming error |X ⊕ B∘C|, also per cell and per 1 in X.
encode_rows(model, X)       Codes of all rows.
code_stats(z)               On-fraction per code bit, dead / duplicated bits.
polish_factors(B, C, X)     Exact local search on C: flip any c_lj that lowers the error.
greedy_codes(X, C)          For fixed C, each row's code chosen greedily: the reference
                            ("oracle") encoder for those factors.
evaluate(model, splits)     All of the above as one dict.
"""

from __future__ import annotations

import numpy as np

from layers import INT, intdot

CHUNK_CELLS = 1 << 24          # ~16 M cells per dense chunk


def row_chunks(n: int, m: int):
    step = max(1, CHUNK_CELLS // max(1, m))
    for a in range(0, n, step):
        yield slice(a, min(n, a + step))


def errors(B: np.ndarray, C: np.ndarray, X) -> dict:
    err = 0
    for r in row_chunks(X.shape[0], X.shape[1]):
        err += int(((intdot(B[r], C) > 0) != X[r].toarray()).sum())
    ones = int(X.sum())
    return {"error": err, "per_cell": err / max(1, X.shape[0] * X.shape[1]), "per_one": err / max(1, ones)}


def encode_rows(model, X) -> np.ndarray:
    out = [model.encode(X[r]) for r in row_chunks(X.shape[0], X.shape[1])]
    return np.concatenate(out) if out else np.zeros((0, model.k), dtype=bool)


def code_stats(z: np.ndarray) -> dict:
    if z.shape[0] == 0:
        return {}
    on = z.astype(np.float64)
    frac = on.mean(0)
    agree = (on.T @ on + (1 - on).T @ (1 - on)) / on.shape[0]
    iu = np.triu_indices(on.shape[1], 1)
    return {"on_frac": [round(float(v), 4) for v in frac],
            "dead": int(((frac < 0.01) | (frac > 0.99)).sum()),
            "dup_pairs": int((agree[iu] > 0.99).sum()),
            "distinct_codes": int(len(np.unique(z, axis=0)))}


def polish_factors(B: np.ndarray, C: np.ndarray, X, max_passes: int = 50):
    """Flips in different columns never interact, so each factor row is updated at once."""
    C = C.copy()
    total = 0
    for _ in range(max_passes):
        changed = 0
        for l in range(C.shape[0]):
            rows = np.flatnonzero(B[:, l])
            if rows.size == 0:
                continue
            gain_on = np.zeros(C.shape[1], dtype=np.int64)
            gain_off = np.zeros(C.shape[1], dtype=np.int64)
            for r in row_chunks(rows.size, C.shape[1]):
                rr = rows[r]
                S = intdot(B[rr], C)
                xi = X[rr].toarray().astype(INT)
                gain_on += ((S == 0) * (1 - 2 * xi)).sum(0)
                gain_off += ((S == 1) * (2 * xi - 1)).sum(0)
            flip = np.where(C[l], gain_off, gain_on) < 0
            n = int(flip.sum())
            if n:
                C[l] ^= flip
                changed += n
        total += changed
        if changed == 0:
            break
    return C, total


def greedy_codes(X, C: np.ndarray) -> np.ndarray:
    n, m = X.shape
    k = C.shape[0]
    B = np.zeros((n, k), dtype=bool)
    Ci = C.astype(INT)
    for r in row_chunks(n, m):
        xi = X[r].toarray().astype(INT)
        b = np.zeros((xi.shape[0], k), dtype=bool)
        covered = np.zeros(xi.shape, dtype=bool)
        for _ in range(k):
            unc = (~covered).astype(INT)
            gain = (xi * unc) @ Ci.T - ((1 - xi) * unc) @ Ci.T
            gain[b] = -1
            l = gain.argmax(1)
            add = gain[np.arange(len(l)), l] > 0
            if not add.any():
                break
            b[np.flatnonzero(add), l[add]] = True
            covered = intdot(b, C) > 0
        B[r] = b
    return B


def evaluate(model, Xtr, Xva, Xte) -> tuple[dict, np.ndarray]:
    """raw: the trained model. polished: after exact local search on C on the training
    rows (validation and test rows still go through the learned encoder). oracle: greedy
    codes for the polished C."""
    codes = {name: encode_rows(model, X) for name, X in (("train", Xtr), ("val", Xva), ("test", Xte))}
    sets = {"train": Xtr, "val": Xva, "test": Xte}
    out = {"raw": {s: errors(codes[s], model.C, X) for s, X in sets.items() if X.shape[0]}}
    C_pol, n_flips = polish_factors(codes["train"], model.C, Xtr)
    out["polished"] = {s: errors(codes[s], C_pol, X) for s, X in sets.items() if X.shape[0]}
    out["polished"]["flips"] = n_flips
    out["oracle"] = {s: errors(greedy_codes(X, C_pol), C_pol, X) for s, X in sets.items() if X.shape[0]}
    out["code"] = code_stats(codes["train"])
    out["factor_sizes"] = C_pol.sum(1).tolist()
    return out, C_pol
