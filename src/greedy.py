"""
Exact greedy flips for the BMF autoencoder: no backprop, no accumulator, integers only.

Every Boolean parameter (encoder weights and thresholds of every layer, decoder C) gets
the EXACT change of the full training error |X ⊕ X̂| if it alone is flipped. Improving
flips are applied best-first; each one after the first is re-checked against the true
error, because flips in one round can interact. The training error never increases, and
the search ends in a local optimum where no single flip helps.

Exact deltas, all parameters at once:
  1. D[r, j] = change of row r's error if output j of an encoder layer is toggled.
     Code layer: exact_flip_deltas. Hidden layer: the toggle changes the next layer's
     pre-activation by one weight row, which is applied incrementally; only rows whose
     code changes are re-scored.
  2. A flip moves the unit's pre-activation by one step (AND: ±1 where the input is on;
     XNOR and thresholds: ±2), so it toggles h_rj exactly on the rows at the threshold
     boundary. Flip delta = sum of D over those rows (a product with the layer input).
"""

from __future__ import annotations

import time

import numpy as np

from evaluate import encode_rows, errors, row_chunks
from layers import INT, intdot, is_sparse, signal_dot_T
from model import exact_flip_deltas, make_model


def _row_err(z: np.ndarray, C: np.ndarray, xd: np.ndarray) -> np.ndarray:
    return ((intdot(z, C) > 0) != xd).sum(1)


def _dense(X, rows=None):
    x = X if rows is None else X[rows]
    return x.toarray().astype(bool) if is_sparse(x) else np.asarray(x, bool)


def toggle_deltas(model, X, hs: list, l: int) -> np.ndarray:
    """D[r, j]: change of row r's error if output j of encoder layer l is toggled."""
    n = X.shape[0]
    if l == len(model.enc) - 1:
        D = np.zeros((n, model.k), dtype=np.int64)
        for r in row_chunks(n, X.shape[1]):
            _, dz = exact_flip_deltas(hs[-1][r], model.C, _dense(X, r))
            D[r] = dz
        return D
    h = hs[l]
    nxt = model.enc[l + 1]
    s_next = nxt.forward(h)
    z = hs[-1]
    D = np.zeros((n, h.shape[1]), dtype=np.int64)
    for j in range(h.shape[1]):
        on = h[:, j]
        if nxt.connective == "and":              # toggling h_j adds / removes row j of W
            delta = np.where(on[:, None], -1, 1) * nxt.weight[j].astype(INT)[None, :]
        else:                                    # e(h_j) e(w_jr) changes sign
            delta = -2 * np.where(on, 1, -1)[:, None] * np.where(nxt.weight[j], 1, -1)[None, :]
        h_next = (s_next + delta) >= 0
        z2 = h_next if l + 1 == len(model.enc) - 1 else model.encode_from(l + 1, h_next)
        changed = np.flatnonzero((z2 != z).any(1))
        if changed.size:
            xd = _dense(X, changed)
            D[changed, j] = _row_err(z2[changed], model.C, xd) - _row_err(z[changed], model.C, xd)
    return D


def all_flip_deltas(model, X):
    """[(kind, layer, delta array)]; kind "w", "t_up", "t_down" or "C". inf = not possible."""
    hs = model.encode_all(X)
    out = []
    for l, layer in enumerate(model.enc):
        a = X if l == 0 else hs[l - 1]
        s = layer.forward(a)
        D = toggle_deltas(model, X, hs, l)
        if layer.connective == "and":
            on = signal_dot_T(a, (s == -1) * D)              # w: F -> T, s += x
            off = signal_dot_T(a, (s == 0) * D)              # w: T -> F, s -= x
            dw = np.where(layer.weight, off, on)
        else:
            PD = ((s == 0) | (s == 1)) * D                   # toggles if s drops by 2
            ND = ((s == -2) | (s == -1)) * D                 # toggles if s rises by 2
            aP, aN = signal_dot_T(a, PD), signal_dot_T(a, ND)
            d_true = aP + (ND.sum(0)[None, :] - aN)          # (1 - a)^T ND = sum ND - a^T ND
            d_false = aN + (PD.sum(0)[None, :] - aP)
            dw = np.where(layer.weight, d_true, d_false)
        up = (((s == -2) | (s == -1)) * D).sum(0).astype(np.float64)
        down = (((s == 0) | (s == 1)) * D).sum(0).astype(np.float64)
        out.append(("w", l, dw.astype(np.float64)))
        out.append(("t_up", l, np.where(layer.t < layer.n_bias, up, np.inf)))
        out.append(("t_down", l, np.where(layer.t > 0, down, np.inf)))
    dC = np.zeros(model.C.shape, dtype=np.int64)
    for r in row_chunks(X.shape[0], X.shape[1]):
        d, _ = exact_flip_deltas(hs[-1][r], model.C, _dense(X, r))
        dC += d
    out.append(("C", -1, dC.astype(np.float64)))
    return out


def _apply(model, kind, l, idx):
    if kind == "C":
        model.dec.weight[idx] ^= True
    elif kind == "w":
        model.enc[l].weight[idx] ^= True
    elif kind == "t_up":
        model.enc[l].t[idx] += 1
    else:
        model.enc[l].t[idx] -= 1


def _undo(model, kind, l, idx):
    _apply(model, {"t_up": "t_down", "t_down": "t_up"}.get(kind, kind), l, idx)


def total_error(model, X) -> int:
    z = np.concatenate([model.encode(X[r]) for r in row_chunks(X.shape[0], X.shape[1])])
    return errors(z, model.C, X)["error"]


class _Checker:
    """Exact error change of one applied flip, computed only where it can matter:
    a flip of c_lj changes column j of the reconstruction; an encoder flip changes the
    codes of some rows, and only those rows are re-scored."""

    def __init__(self, model, X):
        self.model, self.X = model, X
        self.Xc = X.tocsc() if is_sparse(X) else X
        self.Z = encode_rows(model, X)

    def _col(self, j):
        c = self.Xc[:, [j]]
        return (c.toarray() if is_sparse(c) else c).ravel().astype(bool)

    def delta_C(self, l, j) -> int:
        """Called after c_lj has been flipped."""
        S_new = intdot(self.Z, self.model.C[:, [j]]).ravel()
        on = self.Z[:, l].astype(INT)
        S_old = S_new - on if self.model.C[l, j] else S_new + on
        x = self._col(j)
        return int(((S_new > 0) != x).sum() - ((S_old > 0) != x).sum())

    def delta_enc(self) -> tuple[int, np.ndarray]:
        """Called after an encoder flip; returns (delta, new codes)."""
        Z2 = encode_rows(self.model, self.X)
        ch = np.flatnonzero((Z2 != self.Z).any(1))
        if ch.size == 0:
            return 0, Z2
        xd = _dense(self.X, ch)
        return int((_row_err(Z2[ch], self.model.C, xd) - _row_err(self.Z[ch], self.model.C, xd)).sum()), Z2


def greedy_descent(model, X, verify: int = 64, max_rounds: int = 100_000,
                   time_limit: float | None = None, verbose: bool = False):
    """Exact greedy local search on all parameters, in place. Returns the history."""
    cur = total_error(model, X)
    t0 = time.time()
    rounds = flips = 0
    history = [{"round": 0, "error": cur, "sec": 0.0}]
    while rounds < max_rounds:
        rounds += 1
        groups = all_flip_deltas(model, X)
        vals = np.concatenate([d.ravel() for _, _, d in groups])
        n_imp = int((vals < 0).sum())
        if n_imp == 0:
            break
        top = np.argsort(vals, kind="stable")[:min(verify, n_imp)]
        bounds = np.cumsum([0] + [d.size for _, _, d in groups])
        check = _Checker(model, X)
        accepted = 0
        for flat in top:
            gi = int(np.searchsorted(bounds, flat, side="right") - 1)
            kind, l, d = groups[gi]
            idx = np.unravel_index(int(flat - bounds[gi]), d.shape)
            _apply(model, kind, l, idx)
            if kind == "C":
                delta, Z2 = check.delta_C(*idx), None
            else:
                delta, Z2 = check.delta_enc()
            if delta < 0:                       # exact: keep it
                cur += delta
                accepted += 1
                if Z2 is not None:
                    check.Z = Z2
            else:
                _undo(model, kind, l, idx)
        flips += accepted
        if rounds % 20 == 0 or accepted == 0:
            history.append({"round": rounds, "error": cur, "flips": flips, "sec": round(time.time() - t0, 1)})
            if verbose:
                print(f"  round {rounds:5d}  error {cur}  flips {flips}  ({time.time() - t0:.0f}s)", flush=True)
        if accepted == 0:
            break
        if time_limit and time.time() - t0 > time_limit:
            history.append({"stopped": "time_limit"})
            break
    history.append({"rounds": rounds, "flips": flips, "final_error": cur, "sec": round(time.time() - t0, 1)})
    return history


def train_greedy(Xtr, k: int, hidden=(), code_connective: str = "and", seed: int = 0,
                 verify: int = 64, time_limit: float | None = None, verbose: bool = False):
    """Greedy exact flips from the same random initialisation as the B⊕LD runs."""
    model = make_model(Xtr, k, hidden=hidden, code_connective=code_connective,
                       rng=np.random.default_rng(seed))
    history = greedy_descent(model, Xtr, verify=verify, time_limit=time_limit, verbose=verbose)
    return model, history
