"""
All-Boolean BMF autoencoder.

    x ∈ {0,1}^m  -> encoder layers (AND / XNOR neurons, Boolean thermometer thresholds)
                 -> code z ∈ {0,1}^k
                 -> decoder  x̂_j = OR_l (z_l AND c_lj)       (exact Boolean product)

Codes of all rows = B (n x k), decoder weights = C (k x m): X̂ = B ∘ C, and |X ⊕ X̂| is
the BMF error. Data rows are scipy CSR; everything else is dense numpy Boolean.

Training signals (B⊕LD):
  backward_errors     errors-only: +1 from false positives, -1 from false negatives,
                      0 from correct cells (integer)
  backward_exact_dec  exact integer loss change of every single flip of C for the
                      decoder weights, errors-only for the codes and the encoder
Both pass through the threshold re-weighting and the optional sqrt(2/m) scaling chosen
for the model (BOLD App. C).
"""

from __future__ import annotations

import numpy as np

from layers import INT, DecoderLayer, EncoderLayer, Threshold, intdot, is_sparse


def dense_rows(x) -> np.ndarray:
    return x.toarray().astype(bool) if is_sparse(x) else np.asarray(x, dtype=bool)


class BMFAutoencoder:
    def __init__(self, m: int, k: int, hidden=(), code_connective: str = "and",
                 p_enc: float = 0.5, p_dec: float = 0.2, reweight: str = "tanh",
                 window: int = 2, bp_scale: bool = False, rng=None):
        """hidden: list of (size, connective) for extra encoder layers before the code."""
        rng = rng if rng is not None else np.random.default_rng()
        self.m, self.k, self.bp_scale = m, k, bp_scale
        sizes = [m] + [h for h, _ in hidden] + [k]
        conns = [c for _, c in hidden] + [code_connective]
        self.enc = [EncoderLayer(a, b, connective=c, p_init=p_enc, rng=rng)
                    for a, b, c in zip(sizes[:-1], sizes[1:], conns)]
        self.enc_act = [Threshold(a, tau=0, reweight=reweight, window=window) for a in sizes[:-1]]
        self.dec = DecoderLayer(k, m, p_init=p_dec, rng=rng)
        self.dec_act = Threshold(k, tau=1, reweight=reweight, window=window)

    # -- forward -------------------------------------------------------------- #
    def encode_all(self, x) -> list[np.ndarray]:
        hs, h = [], x
        for layer, act in zip(self.enc, self.enc_act):
            h = act.forward(layer.forward(h))
            hs.append(h)
        return hs

    def encode(self, x) -> np.ndarray:
        return self.encode_all(x)[-1]

    def encode_from(self, l: int, h: np.ndarray) -> np.ndarray:
        for layer, act in zip(self.enc[l + 1:], self.enc_act[l + 1:]):
            h = act.forward(layer.forward(h))
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.dec_act.forward(self.dec.forward(z))

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

    @property
    def C(self) -> np.ndarray:
        return self.dec.weight

    # -- backward ------------------------------------------------------------- #
    def _scale(self, g: np.ndarray, out_features: int) -> np.ndarray:
        return g * np.sqrt(2.0 / out_features) if self.bp_scale else g

    def _backprop_encoder(self, g: np.ndarray):
        for l in reversed(range(len(self.enc))):
            g = self.enc[l].backward(self.enc_act[l].backward(g), need_input_grad=l > 0)
            if l > 0:                       # paper: scale by sqrt(2/m), m = the layer's output size
                g = self._scale(g, self.enc[l].out_features)

    def backward_errors(self, xd: np.ndarray, xhat: np.ndarray):
        """xd: the batch rows as a dense Boolean array."""
        Z = np.where(xhat & ~xd, 1, 0).astype(INT) - np.where(~xhat & xd, 1, 0).astype(INT)
        g = self.dec.backward(self.dec_act.backward(Z))
        self._backprop_encoder(self._scale(g, self.m))

    def backward_exact_dec(self, xd: np.ndarray, z: np.ndarray, xhat: np.ndarray):
        self.backward_errors(xd, xhat)
        dC, _ = exact_flip_deltas(z, self.C, xd)
        self.dec.q_weight = -np.where(self.C, 1, -1) * dC        # q·e(w) = -Δ: flips that lower the loss

    # -- init and state -------------------------------------------------------- #
    def init_thresholds(self, X):
        h = X
        for layer, act in zip(self.enc, self.enc_act):
            layer.init_threshold(h)
            h = act.forward(layer.forward(h))

    def state(self):
        return ([l.weight.copy() for l in self.enc], [l.t.copy() for l in self.enc], self.dec.weight.copy())

    def load_state(self, st):
        for layer, w, t in zip(self.enc, st[0], st[1]):
            layer.weight[...] = w
            layer.t = t.copy()
        self.dec.weight[...] = st[2]

    def n_params(self) -> int:
        return sum(l.weight.size + l.n_bias * l.out_features for l in self.enc) + self.dec.weight.size


def exact_flip_deltas(z: np.ndarray, C: np.ndarray, xd: np.ndarray):
    """Exact change of |x ⊕ z∘C| when one entry of C or of z is flipped (integers).
    With coverage S = z @ C: switching a factor on matters only where S = 0, switching
    it off only where S = 1. Returns (dC [k, m] summed over rows, dz [n, k])."""
    S = intdot(z, C)
    xi = xd.astype(INT)
    s0 = (S == 0) * (1 - 2 * xi)
    s1 = (S == 1) * (2 * xi - 1)
    zi, Ci = z.astype(INT), C.astype(INT)
    dC = np.where(C, zi.T @ s1, zi.T @ s0)
    dz = np.where(z, s1 @ Ci.T, s0 @ Ci.T)
    return dC, dz


def make_model(Xtr, k: int, hidden=(), code_connective: str = "and", p_enc: float = 0.5,
               p_dec: float | None = None, reweight: str = "tanh", window: int = 2,
               bp_scale: bool = False, rng=None) -> BMFAutoencoder:
    """Standard initialisation: C at the density where the initial reconstruction matches
    the data density (code bits on for ~half the rows), thresholds at the median."""
    if p_dec is None:
        d = Xtr.sum() / (Xtr.shape[0] * Xtr.shape[1])
        p_dec = min(1.0, 2 * (1 - (1 - d) ** (1 / k)))
    model = BMFAutoencoder(Xtr.shape[1], k, hidden=hidden, code_connective=code_connective,
                           p_enc=p_enc, p_dec=p_dec, reweight=reweight, window=window,
                           bp_scale=bp_scale, rng=rng)
    model.init_thresholds(Xtr)
    return model
