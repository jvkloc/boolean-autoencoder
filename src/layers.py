"""
Boolean layers with exact integer arithmetic (numpy + scipy, no floats in the forward pass).

Data rows enter the first encoder layer as a scipy CSR matrix (Boolean), hidden
activations and codes are dense numpy Boolean arrays, weights are dense numpy Boolean
arrays [in, out]. Every pre-activation is an integer count:

    AND :  s_j = sum_i AND(x_i, w_ij)                 = x · w_j            (0 .. fan-in)
    XNOR:  s_j = sum_i e(XNOR(x_i, w_ij)), e(T)=+1, e(F)=-1
               = 4 x·w_j - 2|x| - 2|w_j| + fan-in     (-fan-in .. fan-in)

computed as integer products (sparse x dense for CSR input), so sparse data never has
to be densified. The only non-integers in the whole method are the optimizer's
accumulators and, if a configuration asks for them, the BOLD re-weighting functions
and the backprop scaling sqrt(2/m) (BOLD App. C).

Backward signals follow B⊕LD (paper Eqs. 4-8): the variation of AND w.r.t. one input is
the other input (1 / 0), of XNOR it is e(other input) (±1).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

INT = np.int32          # counts are at most the fan-in; int32 products are ~2x faster than int64


def is_sparse(x) -> bool:
    return sp.issparse(x)


def as_int(x):
    """Integer view of a Boolean matrix (sparse stays sparse)."""
    return x.astype(INT) if is_sparse(x) else np.asarray(x, dtype=INT)


def intdot(a, b) -> np.ndarray:
    """Exact integer product a @ b of two Boolean matrices as a dense array; a may be sparse."""
    return np.asarray(as_int(a) @ np.asarray(b, dtype=INT))


def row_sums(x) -> np.ndarray:
    return np.asarray(x.sum(axis=1)).ravel().astype(INT)


def signal_dot_T(x, Z) -> np.ndarray:
    """x^T Z for a Boolean x (sparse or dense) and a signal Z (integer or real)."""
    if np.issubdtype(Z.dtype, np.floating):
        xf = x.astype(np.float64) if is_sparse(x) else x.astype(np.float64)
        return np.asarray(xf.T @ Z)
    return np.asarray(as_int(x).T @ Z)


# --------------------------------------------------------------------------- #
# threshold activation and the BOLD re-weighting options (App. C.3)
# --------------------------------------------------------------------------- #
REWEIGHTS = ("tanh", "inv_sq", "exp", "window", "none")


class Threshold:
    """y = [s >= tau] with integer s and tau.

    Backward: the received signal is multiplied by a weight that decreases with the
    distance of s from the threshold. With d = s - tau for units that are on and
    d = tau - 1 - s for units that are off (d = 0, 1, 2, ...: how far the unit is
    beyond the switching point), the options are
        "tanh"    tanh'(alpha (d + 1/2)), alpha = pi / (2 sqrt(3 fan_in))   (paper's choice)
        "inv_sq"  (1 + d)^-2                                               (paper's example)
        "exp"     exp(-d)                                                  (paper's example)
        "window"  1 if d <= r else 0     (integer; r = window)
        "none"    1                      (integer)
    """

    def __init__(self, fan_in: int, tau: int, reweight: str = "tanh", window: int = 2):
        assert reweight in REWEIGHTS, reweight
        self.tau, self.reweight, self.window = tau, reweight, window
        self.alpha = np.pi / (2.0 * np.sqrt(3.0 * fan_in))
        self._s = None

    def forward(self, s: np.ndarray) -> np.ndarray:
        self._s = s
        return s >= self.tau

    def distance(self, s: np.ndarray) -> np.ndarray:
        return np.where(s >= self.tau, s - self.tau, self.tau - 1 - s)

    def backward(self, Z: np.ndarray) -> np.ndarray:
        if self.reweight == "none":
            return Z
        d = self.distance(self._s)
        if self.reweight == "window":
            return Z * (d <= self.window)
        if self.reweight == "tanh":
            w = 1.0 - np.tanh(self.alpha * (d + 0.5)) ** 2
        elif self.reweight == "inv_sq":
            w = 1.0 / (1.0 + d) ** 2
        else:
            w = np.exp(-d.astype(np.float64))
        return Z * w


# --------------------------------------------------------------------------- #
# encoder and decoder layers
# --------------------------------------------------------------------------- #
class EncoderLayer:
    """s_j = sum_i L(x_i, w_ij) + offset_j, output [s_j >= 0].

    offset_j is a Boolean thermometer of n_bias bits (the first t_j are True), so
    offset_j = 2 t_j - n_bias; one bit flip moves it by ±2. All bits of a unit get the
    same signal, so they share one accumulator and move one step at a time.
    """

    def __init__(self, in_features: int, out_features: int, connective: str = "and",
                 n_bias: int | None = None, p_init: float = 0.5, rng=None):
        assert connective in ("and", "xnor")
        rng = rng if rng is not None else np.random.default_rng()
        self.in_features, self.out_features = in_features, out_features
        self.connective = connective
        self.n_bias = in_features if n_bias is None else n_bias
        self.weight = rng.random((in_features, out_features)) < p_init
        self.t = np.full(out_features, self.n_bias // 2, dtype=INT)
        self.q_weight = self.q_thr = None
        self._x = None

    @property
    def bias_bits(self) -> np.ndarray:
        return np.arange(self.n_bias)[:, None] < self.t[None, :]

    def offset(self) -> np.ndarray:
        return 2 * self.t - self.n_bias

    def data_part(self, x) -> np.ndarray:
        xw = intdot(x, self.weight)
        if self.connective == "and":
            return xw
        return 4 * xw - 2 * row_sums(x)[:, None] - 2 * self.weight.sum(0).astype(INT)[None, :] + self.in_features

    def forward(self, x) -> np.ndarray:
        self._x = x
        return self.data_part(x) + self.offset()[None, :]

    def forward_logic(self, x) -> np.ndarray:
        """Gate-level version of forward() (slow; for the tests)."""
        x = x.toarray().astype(bool) if is_sparse(x) else x
        if self.connective == "and":
            lit = (x[:, :, None] & self.weight[None]).astype(INT)
        else:
            lit = np.where(~(x[:, :, None] ^ self.weight[None]), 1, -1)
        return lit.sum(1) + np.where(self.bias_bits, 1, -1).sum(0)

    def backward(self, Z: np.ndarray, need_input_grad: bool = True):
        """Z = δLoss/δs [B, out]. Stores q_weight, q_thr; returns δLoss/δx (or None)."""
        x = self._x
        xz = signal_dot_T(x, Z)
        if self.connective == "and":
            self.q_weight = xz
        else:                                   # e(x)^T Z = 2 x^T Z - sum_k Z_k
            self.q_weight = 2 * xz - Z.sum(0)[None, :]
        self.q_thr = Z.sum(0)
        if not need_input_grad:
            return None
        Wd = self.weight.astype(Z.dtype) if self.connective == "and" else np.where(self.weight, 1, -1).astype(Z.dtype)
        return Z @ Wd.T

    def move_threshold(self, up: np.ndarray, down: np.ndarray) -> np.ndarray:
        up = up & (self.t < self.n_bias)
        down = down & (self.t > 0)
        self.t = self.t + up.astype(INT) - down.astype(INT)
        return up | down

    def init_threshold(self, X):
        """Threshold at each unit's median pre-activation on X (unit on for ~half the rows)."""
        med = np.median(self.data_part(X), axis=0)
        self.t = np.clip(np.round((self.n_bias - med) / 2), 0, self.n_bias).astype(INT)


class DecoderLayer:
    """s_j = sum_l AND(z_l, c_lj); with threshold 1 this is OR_l (z_l AND c_lj), the
    Boolean matrix product. weight = C, [k, m]."""

    def __init__(self, in_features: int, out_features: int, p_init: float = 0.2, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.in_features, self.out_features = in_features, out_features
        self.weight = rng.random((in_features, out_features)) < p_init
        self.q_weight = None
        self._x = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._x = z
        return intdot(z, self.weight)

    def backward(self, Z: np.ndarray) -> np.ndarray:
        self.q_weight = signal_dot_T(self._x, Z)
        return Z @ self.weight.astype(Z.dtype).T
