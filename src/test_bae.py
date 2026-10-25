"""
Brute-force tests:   python test_bae.py      (or python -m pytest test_bae.py)
"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp

import evaluate
from bold import BoldOptimizer
from data import load_matrix, read_mtx, to_csr, write_mtx
from evaluate import errors, greedy_codes, polish_factors
from greedy import _apply, _undo, all_flip_deltas, greedy_descent, total_error
from layers import DecoderLayer, EncoderLayer, Threshold, intdot
from model import BMFAutoencoder, exact_flip_deltas, make_model

RNG = np.random.default_rng(0)


def rand_bool(*shape, p=0.4, rng=RNG):
    return rng.random(shape) < p


def planted(n, m, k, noise, seed):
    g = np.random.default_rng(seed)
    B, C = g.random((n, k)) < 0.3, g.random((k, m)) < 0.3
    X = (intdot(B, C) > 0) ^ (g.random((n, m)) < noise)
    return to_csr(X), B, C


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def test_forward_counts_match_gate_level_for_sparse_and_dense_input():
    x = rand_bool(7, 9)
    for conn in ("and", "xnor"):
        layer = EncoderLayer(9, 5, connective=conn, n_bias=6, rng=RNG)
        layer.t = np.array([0, 1, 3, 5, 6])
        ref = layer.forward_logic(x)
        assert np.array_equal(layer.forward(x), ref)
        assert np.array_equal(layer.forward(to_csr(x)), ref)
        assert layer.forward(to_csr(x)).dtype.kind == "i"
    dec = DecoderLayer(9, 4, rng=RNG)
    ref = (x[:, :, None] & dec.weight[None]).sum(1)
    assert np.array_equal(dec.forward(x), ref)


def test_weight_variation_is_exact():
    for conn in ("and", "xnor"):
        for sparse_in in (False, True):
            layer = EncoderLayer(6, 4, connective=conn, n_bias=4, rng=RNG)
            x = rand_bool(10, 6)
            xin = to_csr(x) if sparse_in else x
            W = RNG.integers(-3, 4, (10, 4))
            loss = lambda: int((layer.forward(xin) * W).sum())
            base = loss()
            layer.forward(xin)
            layer.backward(W)
            step = 2 if conn == "xnor" else 1
            for i in range(6):
                for j in range(4):
                    ew = 1 if layer.weight[i, j] else -1
                    layer.weight[i, j] ^= True
                    delta = loss() - base
                    layer.weight[i, j] ^= True
                    assert delta == -step * ew * layer.q_weight[i, j], (conn, sparse_in, i, j)


def test_input_variation_is_exact():
    for conn in ("and", "xnor"):
        layer = EncoderLayer(6, 4, connective=conn, n_bias=4, rng=RNG)
        x = rand_bool(3, 6)
        W = RNG.integers(-3, 4, (3, 4))
        base = int((layer.forward(x) * W).sum())
        g = layer.backward(W)
        step = 2 if conn == "xnor" else 1
        for r in range(3):
            for i in range(6):
                x2 = x.copy(); x2[r, i] ^= True
                delta = int((layer.forward(x2) * W).sum()) - base
                assert delta == -step * (1 if x[r, i] else -1) * g[r, i]


def test_threshold_moves_one_step_and_offset_matches_bits():
    layer = EncoderLayer(6, 3, n_bias=4, rng=RNG)
    layer.t = np.array([0, 2, 4])
    moved = layer.move_threshold(up=np.array([True, True, True]), down=np.zeros(3, bool))
    assert layer.t.tolist() == [1, 3, 4] and moved.tolist() == [True, True, False]
    assert np.array_equal(layer.offset(), np.where(layer.bias_bits, 1, -1).sum(0))


def test_reweighting_options():
    s = np.array([[-3, -1, 0, 1, 4]])
    for rw in ("tanh", "inv_sq", "exp", "window", "none"):
        act = Threshold(fan_in=9, tau=0, reweight=rw, window=1)
        act.forward(s)
        d = act.distance(s)
        assert d.tolist() == [[2, 0, 0, 1, 4]]                  # steps beyond the switching point
        Z = np.ones_like(s)
        out = act.backward(Z)
        if rw in ("window", "none"):
            assert out.dtype.kind == "i"                        # stays integer
        expect = {"tanh": 1 - np.tanh(act.alpha * (d + 0.5)) ** 2, "inv_sq": 1 / (1 + d) ** 2,
                  "exp": np.exp(-d), "window": (d <= 1).astype(float), "none": np.ones(d.shape)}[rw]
        assert np.allclose(out, expect), rw


def test_integer_signals_without_reweighting_and_scaling():
    X, _, _ = planted(30, 12, 3, 0.05, 1)
    for rw in ("none", "window"):
        model = make_model(X, 3, hidden=[(5, "xnor")], reweight=rw, rng=np.random.default_rng(0))
        xd = X.toarray()
        z, xhat = model.forward(X)
        model.backward_exact_dec(xd, z, xhat)
        for layer in model.enc:
            assert layer.q_weight.dtype.kind == "i" and layer.q_thr.dtype.kind == "i"
        assert model.dec.q_weight.dtype.kind == "i"


def test_decoder_is_exact_boolean_product():
    model = BMFAutoencoder(m=15, k=4, hidden=[(6, "xnor")], rng=RNG)
    X = rand_bool(50, 15, p=0.3)
    z = model.encode(to_csr(X))
    ref = np.zeros((50, 15), bool)
    for l in range(4):
        ref |= z[:, l:l + 1] & model.C[l:l + 1, :]
    assert np.array_equal(model.decode(z), ref)


def test_exact_flip_deltas_match_brute_force():
    z, C, x = rand_bool(12, 4, p=0.5), rand_bool(4, 9), rand_bool(12, 9, p=0.35)
    loss = lambda z_, C_: int(((intdot(z_, C_) > 0) != x).sum())
    base = loss(z, C)
    dC, dz = exact_flip_deltas(z, C, x)
    for l in range(4):
        for j in range(9):
            C2 = C.copy(); C2[l, j] ^= True
            assert loss(z, C2) - base == dC[l, j]
    for i in range(12):
        for l in range(4):
            z2 = z.copy(); z2[i, l] ^= True
            assert loss(z2, C) - base == dz[i, l]


# --------------------------------------------------------------------------- #
# greedy
# --------------------------------------------------------------------------- #
def test_all_flip_deltas_match_brute_force():
    for hidden in ([], [(5, "and")], [(5, "xnor")], [(5, "xnor"), (4, "and")]):
        for code_conn in ("and", "xnor"):
            g = np.random.default_rng(len(hidden) * 10 + (code_conn == "xnor"))
            X = to_csr(g.random((25, 8)) < 0.35)
            model = make_model(X, 3, hidden=hidden, code_connective=code_conn, rng=g)
            for layer in model.enc:
                layer.t = g.integers(0, layer.n_bias + 1, layer.t.shape)
            base = total_error(model, X)
            for kind, l, d in all_flip_deltas(model, X):
                for flat in range(d.size):
                    idx = np.unravel_index(flat, d.shape)
                    if np.isinf(d[idx]):
                        continue
                    _apply(model, kind, l, idx)
                    true = total_error(model, X) - base
                    _undo(model, kind, l, idx)
                    assert true == d[idx], (hidden, code_conn, kind, l, idx, true, d[idx])


def test_greedy_descent_monotone_to_local_optimum():
    X, _, _ = planted(60, 12, 3, 0.05, 2)
    model = make_model(X, 3, hidden=[(6, "xnor")], rng=np.random.default_rng(0))
    start = total_error(model, X)
    hist = greedy_descent(model, X, verify=8)
    errs = [h["error"] for h in hist if "error" in h]
    assert all(a >= b for a, b in zip(errs, errs[1:])) and errs[-1] <= start
    for _, _, d in all_flip_deltas(model, X):
        assert (d >= 0).all()


# --------------------------------------------------------------------------- #
# B⊕LD optimizer: beta options
# --------------------------------------------------------------------------- #
def _model_with_signal(q):
    model = BMFAutoencoder(m=4, k=2, rng=np.random.default_rng(1))
    model.enc[0].weight[:] = False
    model.dec.weight[:] = False
    model.enc[0].q_weight = np.full((4, 2), -q)       # q·e(w) = q > 0: flip wanted
    model.enc[0].q_thr = np.zeros(2)
    model.dec.q_weight = np.zeros((2, 4))
    return model


def test_beta_zero_has_no_memory_and_layer_beta_accumulates():
    for beta, expect in (("layer", [0, 0, 8]), (0, [0, 0, 0]), ("neuron", [0, 0, 8]), ("network", [0, 0, 8])):
        model = _model_with_signal(0.4)
        opt = BoldOptimizer(model, eta=1.0, beta=beta)
        got = []
        for _ in range(3):
            model.enc[0].weight[:] = False
            model.enc[0].q_weight = np.full((4, 2), -0.4)
            got.append(opt.step(batch_size=1)[0])
        assert got == expect, (beta, got)


def test_beta_neighbourhoods():
    model = _model_with_signal(0.0)
    model.enc[0].q_weight = np.array([[-2.0, 0], [0, 0], [0, 0], [0, 0]])   # flips w[0, 0] only
    for beta, check in (("layer", lambda b: np.isclose(b["w0"], 1 - 1 / 10)),
                        ("neuron", lambda b: np.allclose(b["w0"], [[1 - 1 / 5, 1.0]])),
                        ("network", lambda b: np.isclose(b["C"], 1 - 1 / 18))):
        model.enc[0].weight[:] = False
        opt = BoldOptimizer(model, eta=1.0, beta=beta)
        opt.step(batch_size=1)
        assert check(opt.b), (beta, opt.b)


def test_eta_is_divided_by_batch_size():
    model = _model_with_signal(1.5)
    assert BoldOptimizer(model, eta=1.0).step(batch_size=1)[0] == 8
    model = _model_with_signal(1.5)
    assert BoldOptimizer(model, eta=1.0).step(batch_size=2)[0] == 0


# --------------------------------------------------------------------------- #
# evaluation and data
# --------------------------------------------------------------------------- #
def test_chunked_errors_equal_full():
    X, B, C = planted(70, 13, 4, 0.1, 3)
    full = int(((intdot(B, C) > 0) != X.toarray()).sum())
    old = evaluate.CHUNK_CELLS
    evaluate.CHUNK_CELLS = 30                         # forces many small chunks
    try:
        assert errors(B, C, X)["error"] == full
        Bg = greedy_codes(X, C)
        C1, _ = polish_factors(B, C, X)
    finally:
        evaluate.CHUNK_CELLS = old
    assert np.array_equal(Bg, greedy_codes(X, C))
    assert np.array_equal(C1, polish_factors(B, C, X)[0])


def test_polish_reaches_local_optimum():
    X, B, _ = planted(80, 20, 4, 0.1, 3)
    C0 = rand_bool(4, 20, p=0.5)
    C1, _ = polish_factors(B, C0, X)
    e1 = errors(B, C1, X)["error"]
    assert e1 <= errors(B, C0, X)["error"]
    for l in range(4):
        for j in range(20):
            C2 = C1.copy(); C2[l, j] ^= True
            assert errors(B, C2, X)["error"] >= e1


def test_matrix_formats_roundtrip():
    tmp = tempfile.mkdtemp()
    M = rand_bool(5, 4, p=0.5)
    write_mtx(f"{tmp}/m.mtx", M)
    sp.save_npz(f"{tmp}/m.npz", sp.csr_array(M))
    np.savetxt(f"{tmp}/m.txt", M.astype(int), fmt="%d")
    with open(f"{tmp}/m.csv", "w") as f:
        f.write("loc;" + ";".join(f"sp{j}" for j in range(4)) + "\n")
        for i, r in enumerate(M.astype(int)):
            f.write(f"L{i};" + ";".join(map(str, r)) + "\n")
    for name in ("m.mtx", "m.npz", "m.txt", "m.csv"):
        X = load_matrix(os.path.join(tmp, name))
        assert sp.issparse(X) and X.dtype == bool and np.array_equal(X.toarray(), M), name
    assert np.array_equal(read_mtx(f"{tmp}/m.mtx").toarray(), M)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
