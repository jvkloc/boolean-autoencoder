"""
B⊕LD optimizer and training loop (Nguyen et al., NeurIPS 2024, Eqs. 10-11, Alg. 1).

For every Boolean parameter w with its (integer) optimisation signal q:
    m <- beta * m + (eta / batch) * q
    flip w  if  m * e(w) >= 1,   then m <- 0 where flipped
The accumulator m is the only real-valued state of the method. Dividing eta by the batch
size keeps q itself an integer (a sum over the batch) and makes eta comparable across
batch sizes.

beta (Eq. 11: "number of unchanged weights / total number of weights" over a chosen
neighbourhood; the paper uses the layer):
    "layer"    one beta per layer (encoder layer: its weights and threshold bits; decoder: C)
    "neuron"   one beta per unit (encoder unit: its weight column and threshold;
               decoder output unit j: column j of C)
    "network"  one beta for all parameters
    0          no accumulator: every step flips on its own minibatch signal

Encoder thresholds are Boolean thermometers; each unit's threshold bits share one
accumulator and move one bit per flip.

Training signal: errors-only for the first `switch_at` fraction of the steps, then the
exact decoder signal (see model.py). The best training state seen is kept.
"""

from __future__ import annotations

import math
import time

import numpy as np

from evaluate import encode_rows, errors
from model import dense_rows, make_model

BETAS = ("layer", "neuron", "network", 0)


class BoldOptimizer:
    def __init__(self, model, eta: float, beta="layer"):
        assert beta in BETAS or beta == 0.0, beta
        self.model, self.eta, self.beta = model, eta, beta
        self.m = {}                    # accumulators, keyed by parameter name
        self.b = {}                    # current beta per parameter (scalar or broadcastable array)

    def _params(self):
        """(name, layer index, kind, signal); kind: "w" weights, "t" thresholds, "C" decoder."""
        out = []
        for l, layer in enumerate(self.model.enc):
            out.append((f"w{l}", l, "w", layer.q_weight))
            out.append((f"t{l}", l, "t", layer.q_thr))
        out.append(("C", None, "C", self.model.dec.q_weight))
        return out

    def step(self, batch_size: int) -> list[int]:
        scale = self.eta / batch_size
        flipped = {}
        for name, l, kind, q in self._params():
            beta = self.b.get(name, 1.0) if self.beta != 0 else 0.0
            m = scale * q if (name not in self.m or np.all(beta == 0.0)) else beta * self.m[name] + scale * q
            if kind == "t":
                layer = self.model.enc[l]
                f = layer.move_threshold(up=m <= -1.0, down=m >= 1.0)
            else:
                w = self.model.enc[l].weight if kind == "w" else self.model.dec.weight
                f = (m * np.where(w, 1.0, -1.0)) >= 1.0
                w ^= f
            m = np.where(f, 0.0, m)
            self.m[name] = m
            flipped[name] = f
        self._update_beta(flipped)
        n_enc = len(self.model.enc)
        return [int(flipped[f"w{l}"].sum() + flipped[f"t{l}"].sum()) for l in range(n_enc)] + [int(flipped["C"].sum())]

    def _update_beta(self, flipped: dict):
        if self.beta == 0:
            return
        n_enc = len(self.model.enc)
        if self.beta == "network":
            tot = sum(f.size for f in flipped.values())
            b = 1.0 - sum(int(f.sum()) for f in flipped.values()) / tot
            for name in flipped:
                self.b[name] = b
        elif self.beta == "layer":
            for l in range(n_enc):
                fw, ft = flipped[f"w{l}"], flipped[f"t{l}"]
                b = 1.0 - (fw.sum() + ft.sum()) / (fw.size + ft.size)
                self.b[f"w{l}"] = self.b[f"t{l}"] = b
            self.b["C"] = 1.0 - flipped["C"].mean()
        else:                                            # neuron
            for l in range(n_enc):
                fw, ft = flipped[f"w{l}"], flipped[f"t{l}"]
                b = 1.0 - (fw.sum(0) + ft) / (fw.shape[0] + 1)        # per unit j
                self.b[f"w{l}"] = b[None, :]
                self.b[f"t{l}"] = b
            self.b["C"] = (1.0 - flipped["C"].mean(0))[None, :]        # per output unit j


def train_bold(Xtr, k: int, hidden=(), code_connective: str = "and", reweight: str = "tanh",
               window: int = 2, bp_scale: bool = False, beta="layer", eta: float = 0.25,
               epochs: int = 100, batch: int = 32, switch_at: float = 0.25,
               schedule: str = "cosine", seed: int = 0, log_every: int = 10, verbose: bool = False):
    """Returns (model at its best training error, history)."""
    rng = np.random.default_rng(seed)
    model = make_model(Xtr, k, hidden=hidden, code_connective=code_connective, reweight=reweight,
                       window=window, bp_scale=bp_scale, rng=rng)
    opt = BoldOptimizer(model, eta=eta, beta=beta)
    n = Xtr.shape[0]
    steps_per_epoch = math.ceil(n / batch)
    total = epochs * steps_per_epoch
    step = 0
    best = (errors(encode_rows(model, Xtr), model.C, Xtr)["error"], model.state(), 0)
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        flips = np.zeros(len(model.enc) + 1)
        for idx in np.array_split(rng.permutation(n), steps_per_epoch):
            c = 0.5 * (1 + math.cos(math.pi * step / total)) if schedule == "cosine" else 1.0
            opt.eta = eta * c
            step += 1
            x = Xtr[idx]
            xd = dense_rows(x)
            z, xhat = model.forward(x)
            if step <= switch_at * total:
                model.backward_errors(xd, xhat)
            else:
                model.backward_exact_dec(xd, z, xhat)
            flips += opt.step(len(idx))
        tr = errors(encode_rows(model, Xtr), model.C, Xtr)
        if tr["error"] < best[0]:
            best = (tr["error"], model.state(), ep)
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            entry = {"epoch": ep, "train_per_one": round(tr["per_one"], 5),
                     "flips_per_step": [round(v, 3) for v in (flips / steps_per_epoch).tolist()],
                     "sec": round(time.time() - t0, 1)}
            history.append(entry)
            if verbose:
                print(f"  epoch {ep:4d}  train {tr['per_one']:.4f} per 1  flips/step "
                      + " ".join(f"{v:6.2f}" for v in entry["flips_per_step"]) + f"  ({entry['sec']}s)", flush=True)
    model.load_state(best[1])
    history.append({"best_epoch": best[2], "best_train_error": best[0]})
    return model, history
