"""
Sanity tests for the Boolean layer and optimizer.

Run:  python test_bool_layer.py      (or: python -m pytest test_bool_layer.py)

The key test is an exact check of the Boolean variation: for a loss that is
linear in the pre-activation s, flipping a single weight changes the loss by
exactly  -2 * e(w) * q_w ; we verify that by brute force, flipping every weight
one at a time and recomputing the loss.
"""

import torch
from boolean_nn import BoolLinear, BoolThreshold, BooleanOptimizer, e

torch.manual_seed(0)


def test_forward_logic_matches_fast_path():
    layer = BoolLinear(13, 7)
    x = torch.randint(0, 2, (5, 13)).bool()
    assert torch.equal(layer.forward_logic(x), layer.forward(x))


def test_preactivation_range_and_parity():
    # s = e(b) + sum of m (±1) terms: |s| <= m+1 and s has parity of m+1
    m = 8
    layer = BoolLinear(m, 3)
    x = torch.randint(0, 2, (64, m)).bool()
    s = layer.forward(x)
    assert s.abs().max() <= m + 1
    assert torch.all((s.long() - (m + 1)) % 2 == 0)   # odd for m = 8 -> never exactly 0


def _linear_loss(layer, x, C):
    return (layer.forward(x) * C).sum()


def test_weight_variation_is_exact_for_linear_loss():
    layer = BoolLinear(6, 4)
    x = torch.randint(0, 2, (10, 6)).bool()
    C = torch.randn(10, 4)                     # loss = sum C * s   =>   Z = C
    base = _linear_loss(layer, x, C)
    layer.forward(x)
    layer.backward(C)
    q = layer.q_weight
    for i in range(6):
        for j in range(4):
            layer.weight[i, j] ^= True
            delta = _linear_loss(layer, x, C) - base
            layer.weight[i, j] ^= True
            expected = -2.0 * e(layer.weight[i, j]) * q[i, j]
            assert torch.isclose(delta, expected, atol=1e-4), (i, j, delta, expected)


def test_bias_variation_is_exact_for_linear_loss():
    layer = BoolLinear(6, 4)
    x = torch.randint(0, 2, (10, 6)).bool()
    C = torch.randn(10, 4)
    base = _linear_loss(layer, x, C)
    layer.forward(x)
    layer.backward(C)
    for j in range(4):
        layer.bias[j] ^= True
        delta = _linear_loss(layer, x, C) - base
        layer.bias[j] ^= True
        assert torch.isclose(delta, -2.0 * e(layer.bias[j]) * layer.q_bias[j], atol=1e-4)


def test_input_variation_is_exact_for_linear_loss():
    layer = BoolLinear(6, 4)
    x = torch.randint(0, 2, (3, 6)).bool()
    C = torch.randn(3, 4)
    base = _linear_loss(layer, x, C)
    layer.forward(x)
    g = layer.backward(C)
    for k in range(3):
        for i in range(6):
            x2 = x.clone()
            x2[k, i] ^= True
            delta = _linear_loss(layer, x2, C) - base
            assert torch.isclose(delta, -2.0 * e(x[k, i]) * g[k, i], atol=1e-4)


def test_threshold_activation():
    act = BoolThreshold(fan_in=8)
    s = torch.tensor([[-3.0, -1.0, 1.0, 3.0]])
    assert torch.equal(act.forward(s), torch.tensor([[False, False, True, True]]))
    Z = torch.ones_like(s)
    back = act.backward(Z)
    assert torch.all(back > 0) and torch.all(back <= 1)
    assert back[0, 1] > back[0, 0]             # closer to threshold -> stronger signal


def test_optimizer_flips_only_when_it_lowers_the_loss():
    torch.manual_seed(1)
    layer = BoolLinear(6, 4)
    x = torch.randint(0, 2, (10, 6)).bool()
    C = torch.randn(10, 4)
    for _ in range(30):                        # repeated steps on a linear loss must be monotone
        before = _linear_loss(layer, x, C)
        layer.forward(x)
        layer.backward(C)
        opt = BooleanOptimizer([layer], eta=10.0) if _ == 0 else opt
        opt.step()
        after = _linear_loss(layer, x, C)
        assert after <= before + 1e-4
    # at the optimum of a linear loss every weight sign equals -sign(q)
    layer.forward(x)
    layer.backward(C)
    assert torch.all(e(layer.weight) * layer.q_weight <= 1e-6)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
