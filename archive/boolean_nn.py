"""
Minimal Boolean neural-network building blocks following B⊕LD
(Nguyen, Ocampo, Askri, Leconte, Tran — "BOLD: Boolean Logic Deep Learning", NeurIPS 2024).

CPU-only, no autograd for the Boolean parts: every layer has an explicit
forward() and backward(), so the logic is easy to read and to check.

Conventions
-----------
* Boolean tensors are torch.bool (True = T, False = F).
* e(.) embeds logic into numbers: e(T) = +1, e(F) = -1   (paper, Def. A.1).
  Under e, XNOR becomes multiplication (paper, Prop. A.2), so
      s_j = e(b_j) + sum_i e(xnor(x_i, w_ij))
          = e(b_j) + 2 * popcount(xnor(x, w_j)) - m.
* Backward signals Z are real numbers ("mixed-type" signals, paper Def. 3.5):
  Z = δLoss/δs, where s is the layer's integer pre-activation.

With L = XNOR the atomic variations are (paper, Eq. 4):
      δs_j/δw_ij = x_i      δs_j/δx_i = w_ij
and with a real downstream signal Z the aggregated signals (Eqs. 5-8) become
      q_w[i,j] = sum_k Z[k,j] * e(x[k,i])      (aggregate over batch)
      g_x[k,i] = sum_j Z[k,j] * e(w[i,j])      (aggregate over outputs)
      q_b[j]   = sum_k Z[k,j]
"""

from __future__ import annotations

import math
import torch


def e(b: torch.Tensor) -> torch.Tensor:
    """Embed a Boolean tensor into {-1, +1} floats."""
    return b.float() * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# Boolean linear layer (XNOR + popcount)
# --------------------------------------------------------------------------- #
class BoolLinear:
    """Fully connected Boolean layer:  s = e(b) + sum_i xnor(x_i, w_ij)  (counted in ±1).

    Weights and bias are Boolean; there are NO latent real-valued weights.
    Each Boolean parameter only has a real accumulator used by the optimizer
    (paper Eq. 10), which is reset to 0 whenever the weight flips.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 generator: torch.Generator | None = None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randint(0, 2, (in_features, out_features), generator=generator).bool()
        self.bias = torch.randint(0, 2, (out_features,), generator=generator).bool() if bias else None
        # optimisation signals filled by backward()
        self.q_weight: torch.Tensor | None = None
        self.q_bias: torch.Tensor | None = None
        self._x: torch.Tensor | None = None

    # -- forward ----------------------------------------------------------- #
    def forward_logic(self, x: torch.Tensor) -> torch.Tensor:
        """Literal XNOR + popcount implementation (slow, for reference / testing)."""
        agree = ~(x[:, :, None] ^ self.weight[None, :, :])          # xnor, shape [B, in, out]
        s = 2 * agree.sum(dim=1) - self.in_features                  # counted in ±1
        if self.bias is not None:
            s = s + e(self.bias).long()
        return s.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path, numerically identical to forward_logic (XNOR ≅ product in ±1)."""
        self._x = x
        s = e(x) @ e(self.weight)
        if self.bias is not None:
            s = s + e(self.bias)
        return s

    # -- backward ---------------------------------------------------------- #
    def backward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z = δLoss/δs (real, [B, out]). Stores q for the optimizer, returns δLoss/δx."""
        x = self._x
        self.q_weight = e(x).T @ Z                                     # [in, out]
        if self.bias is not None:
            self.q_bias = Z.sum(dim=0)                                 # [out]
        return Z @ e(self.weight).T                                    # [B, in]

    def params(self):
        """(Boolean tensor, optimisation signal) pairs for the optimizer."""
        out = [(self.weight, self.q_weight)]
        if self.bias is not None:
            out.append((self.bias, self.q_bias))
        return out


# --------------------------------------------------------------------------- #
# Threshold activation
# --------------------------------------------------------------------------- #
class BoolThreshold:
    """y = (s >= tau).  Backward re-weights the signal by tanh'(alpha * (s - tau))
    (paper, Appendix C.1) so that neurons far from the threshold get a weaker
    signal, with alpha = pi / (2*sqrt(3*m)) and m the fan-in (Eq. 24)."""

    def __init__(self, fan_in: int, tau: float = 0.0):
        self.tau = tau
        self.alpha = math.pi / (2.0 * math.sqrt(3.0 * fan_in))
        self._s: torch.Tensor | None = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        self._s = s
        return s >= self.tau

    def backward(self, Z: torch.Tensor) -> torch.Tensor:
        u = self.alpha * (self._s - self.tau)
        return Z * (1.0 - torch.tanh(u) ** 2)


# --------------------------------------------------------------------------- #
# Boolean optimizer (paper Eq. 10-11, Algorithm 1 / Algorithm 8)
# --------------------------------------------------------------------------- #
class BooleanOptimizer:
    """For each Boolean parameter w with signal q:
        m <- beta * m + eta * q
        flip w  if  m * e(w) >= 1      (i.e. xnor(m, w) = T and |m| large enough)
        reset m to 0 where flipped
        beta <- fraction of unchanged weights (per layer / tensor)
    A positive q with w = T (or negative q with w = F) says flipping w lowers the loss."""

    def __init__(self, layers, eta: float):
        self.layers = layers
        self.eta = eta
        self.state = {}  # id(tensor) -> dict(m, beta)
        self.last_flips = 0

    def step(self):
        flips = 0
        for layer in self.layers:
            for w, q in layer.params():
                st = self.state.setdefault(id(w), {"m": torch.zeros_like(q), "beta": 1.0})
                m = st["beta"] * st["m"] + self.eta * q
                flip = (m * e(w)) >= 1.0
                w ^= flip                      # in-place Boolean flip
                m[flip] = 0.0
                st["m"] = m
                st["beta"] = 1.0 - flip.float().mean().item()
                flips += int(flip.sum())
        self.last_flips = flips
        return flips
