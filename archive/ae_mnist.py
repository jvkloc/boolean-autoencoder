"""
Boolean MNIST autoencoder (B⊕LD-style training), the next step after ae_838.py.

    x ∈ {0,1}^784 (binarized) -> BoolLinear(784→256) -> threshold -> h1 ∈ {0,1}^256
    h1                        -> BoolLinear(256→32)  -> threshold -> z  ∈ {0,1}^32   (code)
    z                         -> BoolLinear(32→256)  -> threshold -> h3 ∈ {0,1}^256
    h3                        -> BoolLinear(256→784) -> + real output bias -> logits -> x̂

Built on boolean_nn.py, unchanged, with the same conventions as ae_838.py:
* Boolean weights, plus Boolean biases in the hidden layers, trained with the
  Boolean optimizer. No latent real weights.
* The 784 output biases are the only real parameters, trained with Adam. At
  inference they are integer thresholds, x̂_p = [s_p >= ceil(-b_p)].
* Post-training calibration: each output's integer threshold is re-picked by
  exhaustive search, keeping the old one unless a candidate is strictly better.

Additions needed at this scale (see the MNIST results doc for why):
* Per-layer η. The output layer's signal is much larger than the hidden
  layers', so it needs its own, smaller η.
* A cosine schedule on η and on the bias learning rate (paper App. D). Without it
  the code collapses late in training even at small η.
* Per-layer flip logging, plus code-bit health: on-fraction per bit, dead bits,
  duplicated/complementary bit pairs, and distinct codes on the test set.

Usage:
    python ae_mnist.py                     # loads MNIST with torchvision into ./data
    python ae_mnist.py --data DIR          # use or create the MNIST cache in DIR instead
    python ae_mnist.py --epochs 3 --n-train 10000      # quick check
or from Python with your own loading:
    from ae_mnist import run
    run(x_train, x_test)                    # (N,28,28) or (N,784), uint8 0..255 or float 0..1

MNIST is downloaded once, on the first run, into DIR/MNIST/raw. After that it is read
from disk. Keep --data pointing at the same folder to avoid downloading it again.

After training, ae_mnist_recon.png shows test images next to their reconstructions,
the pixel errors and the 32-bit codes (needs matplotlib; skipped if it is missing).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from boolean_nn import BoolLinear, BoolThreshold, BooleanOptimizer, e

SIZES = (784, 256, 32, 256, 784)
LAYER_NAMES = ("enc1", "enc2", "dec1", "dec2")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def binarize_images(x, threshold: float = 0.5) -> torch.Tensor:
    """Images -> Boolean tensor [N, 784]: pixel on iff intensity > threshold (on a 0..1 scale)."""
    x = torch.as_tensor(np.array(x)).float().reshape(len(x), -1)
    if x.max() > 1:
        x = x / 255.0
    return x > threshold


def load_mnist(root: str = "data"):
    """Train images, test images (uint8 arrays [N, 28, 28]) and test labels, via torchvision.
    Downloaded into root/MNIST/raw on the first call only; later calls read the files there."""
    try:
        from torchvision.datasets import MNIST
    except ImportError as err:
        raise SystemExit("torchvision is needed to load MNIST: pip install torchvision") from err
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    return train.data.numpy(), test.data.numpy(), test.targets.numpy()


# --------------------------------------------------------------------------- #
# optimizer: same update as BooleanOptimizer, with per-layer η and per-layer flip counts
# --------------------------------------------------------------------------- #
class LayerwiseBooleanOptimizer(BooleanOptimizer):
    """BooleanOptimizer.step() with η set per layer (self.eta is a list aligned with layers).
    Records flips per layer in self.last_layer_flips (weights and bias together)."""

    def step(self):
        per_layer = []
        for layer, eta in zip(self.layers, self.eta):
            n = 0
            for w, q in layer.params():
                st = self.state.setdefault(id(w), {"m": torch.zeros_like(q), "beta": 1.0})
                m = st["beta"] * st["m"] + eta * q
                flip = (m * e(w)) >= 1.0
                w ^= flip
                m[flip] = 0.0
                st["m"] = m
                st["beta"] = 1.0 - flip.float().mean().item()
                n += int(flip.sum())
            per_layer.append(n)
        self.last_layer_flips = per_layer
        self.last_flips = sum(per_layer)
        return self.last_flips


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class BooleanMNISTAutoencoder:
    def __init__(self, sizes=SIZES, bp_scale: bool = True, generator=None):
        self.sizes = sizes
        n = len(sizes) - 1
        self.layers = [BoolLinear(sizes[l], sizes[l + 1], bias=(l < n - 1), generator=generator)
                       for l in range(n)]
        # fan-in includes the Boolean bias, as in ae_838.py
        self.acts = [BoolThreshold(fan_in=sizes[l] + 1) for l in range(n - 1)]
        self.out_bias = torch.zeros(sizes[-1])            # real, trained with Adam
        self.bp_scale = bp_scale                          # paper App. C.2: scale by sqrt(2/m)
        self.code_index = 1                               # z is the output of layers[1]

    def forward(self, x: torch.Tensor):
        """x: Boolean [B, 784]. Returns (hidden Boolean outputs, logits)."""
        hs = []
        h = x
        for layer, act in zip(self.layers[:-1], self.acts):
            h = act.forward(layer.forward(h))
            hs.append(h)
        s_out = self.layers[-1].forward(h)
        return hs, s_out + self.out_bias

    def backward(self, dlogits: torch.Tensor):
        """dlogits = δLoss/δlogits. Fills q for every Boolean layer."""
        self.grad_out_bias = dlogits.sum(dim=0)
        Z = dlogits
        for l in reversed(range(len(self.layers))):
            g = self.layers[l].backward(Z)                # δLoss/δ(input of layer l)
            if l == 0:
                break
            if self.bp_scale:
                g = g * math.sqrt(2.0 / self.layers[l].out_features)
            Z = self.acts[l - 1].backward(g)

    @property
    def bool_layers(self):
        return self.layers

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def output_preactivation(self, X: torch.Tensor, bs: int = 5000) -> torch.Tensor:
        out = []
        for xb in X.split(bs):
            h = xb
            for layer, act in zip(self.layers[:-1], self.acts):
                h = act.forward(layer.forward(h))
            out.append(self.layers[-1].forward(h))
        return torch.cat(out)

    @torch.no_grad()
    def encode(self, X: torch.Tensor, bs: int = 5000) -> torch.Tensor:
        out = []
        for xb in X.split(bs):
            h = xb
            for l in range(self.code_index + 1):
                h = self.acts[l].forward(self.layers[l].forward(h))
            out.append(h)
        return torch.cat(out)

    def thresholds(self) -> torch.Tensor:
        return torch.ceil(-self.out_bias.detach())

    @torch.no_grad()
    def reconstruct_boolean(self, X: torch.Tensor) -> torch.Tensor:
        """Pure Boolean/integer inference: integer thresholds instead of the real bias."""
        return self.output_preactivation(X) >= self.thresholds()

    @torch.no_grad()
    def calibrate_thresholds(self, X: torch.Tensor):
        """Same rule as ae_838.py, vectorized: for each output p try every integer
        threshold t in [-n, n+1] (n = fan-in of the output layer) and keep the current
        one unless some t gives strictly fewer errors on X."""
        s = self.output_preactivation(X)
        n = self.layers[-1].in_features
        T = X.float()
        idx = (s + n).round().long()                  # s ∈ [-n, n]  ->  0 .. 2n
        V = 2 * n + 1
        P = s.shape[1]
        h1 = torch.zeros(P, V).scatter_add_(1, idx.T, T.T)
        h0 = torch.zeros(P, V).scatter_add_(1, idx.T, (1 - T).T)
        # threshold t = k - n, k = 0 .. 2n+1: predict on iff idx >= k
        miss_on = torch.cat([torch.zeros(P, 1), h1.cumsum(1)], 1)                    # targets on, idx < k
        false_on = torch.cat([h0.flip(1).cumsum(1).flip(1), torch.zeros(P, 1)], 1)  # targets off, idx >= k
        errs = miss_on + false_on                                                    # [P, 2n+2]
        best_err, best_k = errs.min(1)
        cur = self.thresholds()
        cur_k = (cur + n).clamp(0, 2 * n + 1).long()
        cur_err = errs.gather(1, cur_k[:, None]).squeeze(1)
        new = torch.where(best_err < cur_err, (best_k - n).float(), cur)
        self.out_bias = -new                         # ceil(-b) == new exactly
        return int((new != cur).sum())


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def code_stats(z: torch.Tensor) -> dict:
    on = z.float()
    frac = on.mean(0)
    agree = (on.T @ on + (1 - on).T @ (1 - on)) / on.shape[0]
    iu = torch.triu_indices(on.shape[1], on.shape[1], 1)
    a = agree[iu[0], iu[1]]
    return {
        "on_frac": [round(v, 4) for v in frac.tolist()],
        "min": frac.min().item(), "median": frac.median().item(), "max": frac.max().item(),
        "dead": int(((frac < 0.01) | (frac > 0.99)).sum()),
        "dup_pairs": int((a > 0.99).sum()),
        "compl_pairs": int((a < 0.01).sum()),
        "distinct_codes": len(torch.unique(z, dim=0)),
    }


def pixel_error(xhat: torch.Tensor, X: torch.Tensor) -> float:
    return (xhat != X).float().mean().item()

# --------------------------------------------------------------------------- #
# figure: originals, reconstructions, errors, codes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def save_reconstruction_figure(model, X: torch.Tensor, path: str, n: int = 12,
                               labels=None, title: str = ""):
    """One column per test image. Rows: original, reconstruction (integer-threshold
    inference), pixel errors (red = missed pixel, blue = extra pixel) and the 32-bit
    code drawn as a 4x8 grid. Picks the first image of each digit when labels are
    given, otherwise the first n images."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the reconstruction figure")
        return
    if labels is not None:
        labels = torch.as_tensor(np.array(labels))
        idx = [int((labels == d).nonzero()[0]) for d in range(10) if (labels == d).any()]
        idx += [i for i in range(len(X)) if i not in idx][: max(0, n - len(idx))]
        idx = idx[:n]
    else:
        idx = list(range(min(n, len(X))))
    x = X[idx]
    xhat = model.reconstruct_boolean(x)
    z = model.encode(x)
    k = len(idx)
    fig, axes = plt.subplots(4, k, figsize=(1.15 * k, 5.4))
    axes = np.array(axes).reshape(4, k)
    for j in range(k):
        orig = x[j].reshape(28, 28).numpy()
        rec = xhat[j].reshape(28, 28).numpy()
        err = np.ones((28, 28, 3))
        err[orig & ~rec] = (0.85, 0.15, 0.15)            # missed: should be on
        err[~orig & rec] = (0.15, 0.35, 0.85)            # extra: should be off
        err[orig & rec] = (0.8, 0.8, 0.8)                # correct on-pixel
        wrong = int((orig != rec).sum())
        axes[0, j].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].imshow(rec, cmap="gray_r", vmin=0, vmax=1)
        axes[2, j].imshow(err)
        axes[2, j].set_title(f"{wrong} wrong", fontsize=8)
        axes[3, j].imshow(z[j].reshape(4, 8).numpy(), cmap="gray_r", vmin=0, vmax=1)
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    for r, name in enumerate(["original", "reconstruction", "errors", "code (32 bits)"]):
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(title or "Boolean autoencoder on binarized MNIST (test images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")



# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def run(x_train, x_test, epochs: int = 10, batch: int = 32, eta: float = 0.25,
        eta_out: float = 0.05, lr_bias: float = 0.01, schedule: str = "cosine",
        bp_scale: bool = True, seed: int = 0, log_path: str | None = "ae_mnist_log.json",
        figure_path: str | None = "ae_mnist_recon.png", test_labels=None, verbose: bool = True):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Xtr, Xte = binarize_images(x_train), binarize_images(x_test)

    model = BooleanMNISTAutoencoder(bp_scale=bp_scale, generator=g)
    eta0 = [eta] * (len(model.layers) - 1) + [eta_out]
    bopt = LayerwiseBooleanOptimizer(model.bool_layers, eta=list(eta0))
    model.out_bias.requires_grad_()
    adam = torch.optim.Adam([model.out_bias], lr=lr_bias)

    n_params = [l.weight.numel() + (l.bias.numel() if l.bias is not None else 0) for l in model.layers]
    steps_per_epoch = math.ceil(len(Xtr) / batch)
    total_steps = epochs * steps_per_epoch
    step = 0
    log = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        flips = np.zeros(len(model.layers))
        zero_steps = np.zeros(len(model.layers))
        run_loss = 0.0
        for idx in torch.randperm(len(Xtr), generator=g).split(batch):
            c = 0.5 * (1 + math.cos(math.pi * step / total_steps)) if schedule == "cosine" else 1.0
            bopt.eta = [v * c for v in eta0]
            for pg in adam.param_groups:
                pg["lr"] = lr_bias * c
            step += 1

            x = Xtr[idx]
            B = x.shape[0]
            _, logits = model.forward(x)
            target = x.float()
            logits = logits.detach()
            run_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="sum").item() / B
            dlogits = (torch.sigmoid(logits) - target) / B

            model.backward(dlogits)
            bopt.step()
            adam.zero_grad()
            model.out_bias.grad = model.grad_out_bias.clone()
            adam.step()

            f = np.array(bopt.last_layer_flips)
            flips += f
            zero_steps += (f == 0)

        with torch.no_grad():
            err = pixel_error(model.reconstruct_boolean(Xte), Xte)
            cs = code_stats(model.encode(Xte))
        entry = {
            "epoch": ep,
            "train_bce_per_image": run_loss / steps_per_epoch,
            "test_pixel_err": err,
            "flips": dict(zip(LAYER_NAMES, flips.astype(int).tolist())),
            "flips_per_step": dict(zip(LAYER_NAMES, (flips / steps_per_epoch).tolist())),
            "flip_frac": dict(zip(LAYER_NAMES, (flips / np.array(n_params)).tolist())),
            "zero_flip_step_frac": dict(zip(LAYER_NAMES, (zero_steps / steps_per_epoch).tolist())),
            "code": cs,
            "sec": time.time() - t0,
        }
        log.append(entry)
        if verbose:
            print(f"epoch {ep:3d}  bce {entry['train_bce_per_image']:7.1f}  test err {err:.4f}  "
                  "flips/step " + " ".join(f"{k} {v:5.1f}" for k, v in entry["flips_per_step"].items()) +
                  f"  | code on-frac med {cs['median']:.2f}  dead {cs['dead']}  dup {cs['dup_pairs']}  "
                  f"compl {cs['compl_pairs']}  distinct {cs['distinct_codes']}  ({entry['sec']:.0f}s)",
                  flush=True)

    # integer-threshold calibration (on the training set), as in ae_838.py
    with torch.no_grad():
        before_tr = pixel_error(model.reconstruct_boolean(Xtr), Xtr)
        before_te = pixel_error(model.reconstruct_boolean(Xte), Xte)
        changed = model.calibrate_thresholds(Xtr)
        after_tr = pixel_error(model.reconstruct_boolean(Xtr), Xtr)
        after_te = pixel_error(model.reconstruct_boolean(Xte), Xte)
        n = model.layers[-1].in_features
        thr = model.thresholds()
    calib = {"train_err_before": before_tr, "train_err_after": after_tr,
             "test_err_before": before_te, "test_err_after": after_te,
             "thresholds_changed": changed,
             "never_on_after": int((thr > n).sum()), "always_on_after": int((thr <= -n).sum())}
    if verbose:
        print("calibration:", json.dumps(calib))

    result = {"config": dict(epochs=epochs, batch=batch, eta=eta, eta_out=eta_out, lr_bias=lr_bias,
                             schedule=schedule, bp_scale=bp_scale, seed=seed),
              "baseline_constant_err_test": baseline_error(Xtr, Xte),
              "epochs": log, "calibration": calib}
    if log_path:
        with open(log_path, "w") as f:
            json.dump(result, f)
    if figure_path:
        save_reconstruction_figure(
            model, Xte, figure_path, labels=test_labels,
            title=f"Boolean autoencoder, binarized MNIST test images "
                  f"(test pixel error {after_te:.3f}, all-blank baseline {result['baseline_constant_err_test']:.3f})")
    return model, result


def baseline_error(Xtr: torch.Tensor, Xte: torch.Tensor) -> float:
    """Test pixel error of predicting each pixel's majority value from the training set."""
    return pixel_error((Xtr.float().mean(0) > 0.5).expand_as(Xte), Xte)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data",
                   help="MNIST cache folder (downloaded into DIR/MNIST/raw on first use)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eta", type=float, default=0.25, help="η for the three hidden layers")
    p.add_argument("--eta-out", type=float, default=0.05, help="η for the output layer")
    p.add_argument("--lr-bias", type=float, default=0.01)
    p.add_argument("--schedule", choices=["cosine", "none"], default="cosine")
    p.add_argument("--no-bp-scale", action="store_true", help="drop the sqrt(2/m) backprop scaling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None, help="use only the first N training images")
    p.add_argument("--log", default="ae_mnist_log.json")
    p.add_argument("--figure", default="ae_mnist_recon.png", help="reconstruction figure ('' to skip)")
    a = p.parse_args()

    xtr, xte, yte = load_mnist(a.data)
    if a.n_train:
        xtr = xtr[:a.n_train]
    run(xtr, xte, epochs=a.epochs, batch=a.batch, eta=a.eta, eta_out=a.eta_out,
        lr_bias=a.lr_bias, schedule=a.schedule, bp_scale=not a.no_bp_scale,
        seed=a.seed, log_path=a.log, figure_path=a.figure or None, test_labels=yte)


if __name__ == "__main__":
    main()
