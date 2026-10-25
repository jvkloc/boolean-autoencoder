"""
The 8-3-8 encoder problem with a Boolean autoencoder (B⊕LD-style training).

    x ∈ {0,1}^8 (one-hot)  ->  BoolLinear(8→3) -> threshold  ->  z ∈ {0,1}^3
    z                      ->  BoolLinear(3→8) -> + real output bias -> logits -> x̂

* Encoder and decoder weights are Boolean and trained with the Boolean optimizer
  (no latent real weights, no gradient descent on them).
* The only real-valued parameters are the 8 output biases (the "last layer in FP"
  convention from the paper). At inference they are just integer thresholds:
  x̂_p = [ s_p >= ceil(-b_p) ], so the whole trained network runs on
  XNOR + popcount + compare.
* Success = all 8 patterns reconstructed exactly, which forces the 3 latent bits
  to form a perfect 3-bit code (8 distinct codes).

Usage:
    python ae_838.py                      # 20 seeds with default settings
    python ae_838.py --seeds 50 --eta 2 --steps 3000 --batch 2
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from boolean_nn import BoolLinear, BoolThreshold, BooleanOptimizer


class BooleanAutoencoder:
    def __init__(self, n_in: int = 8, n_code: int = 3, generator=None):
        self.enc = BoolLinear(n_in, n_code, bias=True, generator=generator)
        self.act = BoolThreshold(fan_in=n_in + 1)
        self.dec = BoolLinear(n_code, n_in, bias=False, generator=generator)
        self.out_bias = torch.zeros(n_in)            # real, trained with Adam

    def encode(self, x):
        return self.act.forward(self.enc.forward(x))

    def forward(self, x):
        z = self.encode(x)
        logits = self.dec.forward(z) + self.out_bias
        return z, logits

    def backward(self, dlogits):
        """dlogits = δLoss/δlogits (real). Propagates through the whole network."""
        self.grad_out_bias = dlogits.sum(dim=0)
        g_z = self.dec.backward(dlogits)             # δLoss/δz  (real)
        g_s = self.act.backward(g_z)                 # through the threshold
        self.enc.backward(g_s)                       # fills q for encoder weights

    @torch.no_grad()
    def reconstruct_boolean(self, x):
        """Pure Boolean/integer inference: integer thresholds instead of real bias."""
        z = self.encode(x)
        s = self.dec.forward(z)
        thr = torch.ceil(-self.out_bias)
        return z, s >= thr

    @torch.no_grad()
    def calibrate_thresholds(self, X):
        """Post-training: pick each output's integer threshold by exhaustive search
        (s only takes n_code+1 values, so there are only a handful of candidates).
        Replaces the real bias with the best integer threshold on the data X."""
        z = self.encode(X)
        s = self.dec.forward(z)
        n = self.dec.in_features
        best = torch.ceil(-self.out_bias)
        for p in range(s.shape[1]):
            errs = [((s[:, p] >= t) != X[:, p]).sum().item() for t in range(-n, n + 2)]
            cur = ((s[:, p] >= best[p]) != X[:, p]).sum().item()
            t_best = int(torch.tensor(errs).argmin()) - n
            if min(errs) < cur:
                best[p] = t_best
        self.out_bias = -best          # ceil(-b) == best exactly

    @property
    def bool_layers(self):
        return [self.enc, self.dec]


def train_one(seed: int, eta: float, steps: int, batch: int, lr_bias: float, verbose=False):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X = torch.eye(8).bool()
    model = BooleanAutoencoder(generator=g)
    bopt = BooleanOptimizer(model.bool_layers, eta=eta)
    model.out_bias.requires_grad_(False)
    adam = torch.optim.Adam([model.out_bias.requires_grad_()], lr=lr_bias)

    solved_at = None
    for t in range(steps):
        idx = torch.randperm(8, generator=g)[:batch]
        x = X[idx]
        _, logits = model.forward(x)
        target = x.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.detach(), target,
                                                                    reduction="sum") / batch
        dlogits = (torch.sigmoid(logits.detach()) - target) / batch

        model.backward(dlogits)
        bopt.step()
        adam.zero_grad()
        model.out_bias.grad = model.grad_out_bias.clone()
        adam.step()

        if t % 50 == 0 or t == steps - 1:
            z, xhat = model.reconstruct_boolean(X)
            ok = torch.equal(xhat, X)
            if verbose and t % 500 == 0:
                n_codes = len({tuple(r.tolist()) for r in z.int()})
                print(f"  step {t:5d}  loss {loss.item():7.4f}  flips {bopt.last_flips:3d}  "
                      f"distinct codes {n_codes}/8  exact {ok}")
            if ok:
                solved_at = t
                break
    calibrated = False
    if solved_at is None:
        model.calibrate_thresholds(X)
        z, xhat = model.reconstruct_boolean(X)
        calibrated = torch.equal(xhat, X)
    z, xhat = model.reconstruct_boolean(X)
    bit_err = (xhat != X).float().mean().item()
    n_codes = len({tuple(r.tolist()) for r in z.int()})
    return dict(seed=seed, solved=solved_at is not None, calibrated=calibrated,
                steps=solved_at, bit_err=bit_err,
                n_codes=n_codes, model=model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--eta", type=float, default=2.0, help="Boolean accumulation factor η")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr-bias", type=float, default=0.05)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    results = []
    for s in range(a.seeds):
        r = train_one(s, a.eta, a.steps, a.batch, a.lr_bias, verbose=a.verbose)
        results.append(r)
        tag = (f"solved at step {r['steps']}" if r["solved"]
               else "solved after calib." if r["calibrated"] else "NOT solved")
        print(f"seed {s:3d}: {tag:22s} bit error {r['bit_err']:.3f}  distinct codes {r['n_codes']}/8")

    solved = [r for r in results if r["solved"]]
    n_cal = sum(r["calibrated"] for r in results)
    print(f"\n{len(solved)}/{a.seeds} seeds solved 8-3-8 exactly during training, "
          f"+{n_cal} more after integer-threshold calibration\n"
          f"(eta={a.eta}, batch={a.batch}, steps={a.steps}) in {time.time() - t0:.1f}s")
    if solved:
        med = sorted(r["steps"] for r in solved)[len(solved) // 2]
        print(f"median steps to solve: {med}")
        m = solved[0]["model"]
        X = torch.eye(8).bool()
        z, xhat = m.reconstruct_boolean(X)
        print(f"\nLearned code (seed {solved[0]['seed']}):")
        for p_ in range(8):
            print(f"  pattern {p_}  ->  z = {''.join(str(int(b)) for b in z[p_])}  ->  "
                  f"x̂ = {''.join(str(int(b)) for b in xhat[p_])}")
        print("\nEncoder weights (rows = inputs, cols = code bits):")
        print(m.enc.weight.int())
        print("Integer output thresholds:", torch.ceil(-m.out_bias).int().tolist())


if __name__ == "__main__":
    main()
