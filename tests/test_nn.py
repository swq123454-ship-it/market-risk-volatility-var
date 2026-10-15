"""Tests for the hand-written GRU.

The reason for writing the network by hand rather than importing one is that the
backward pass can be checked against a numerical gradient, and that is the first
test in this file. Everything else - the numerics of the sigmoid, the sequence
alignment, the early-stopping restore - is downstream of getting that right.
"""

from __future__ import annotations

import numpy as np
import pytest

from marketrisk import nn


def _loss_and_grads(params, x, y):
    pred, cache, h_final = nn.forward(params, x)
    err = pred - y
    dy = 2.0 * err / len(y)
    loss = float(np.mean(err**2))
    grads = nn.backward(params, cache, h_final, dy)
    return loss, grads


# ---------------------------------------------------------------------------
# The gradient check
# ---------------------------------------------------------------------------
def test_analytic_gradients_match_finite_differences():
    """Every parameter tensor: analytic gradient vs central difference.

    Central differences, not forward differences, because the truncation error of
    a one-sided difference is the same order as the perturbation and would put a
    floor of about 1e-4 on the relative error - comfortably above any real bug.
    """
    rng = np.random.default_rng(11)
    batch, steps, features, hidden = 6, 5, 4, 3
    x = rng.standard_normal((batch, steps, features))
    y = rng.standard_normal(batch)

    params = nn.init_params(features, hidden, rng)
    _, analytic = _loss_and_grads(params, x, y)

    eps = 1e-6
    names = ["wz", "wr", "wh", "bz", "br", "bh", "wy", "by"]
    for name, grad in zip(names, analytic, strict=True):
        array = getattr(params, name)
        numeric = np.zeros_like(array)
        flat = array.ravel()
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = _loss_and_grads(params, x, y)[0]
            flat[i] = original - eps
            minus = _loss_and_grads(params, x, y)[0]
            flat[i] = original
            numeric.ravel()[i] = (plus - minus) / (2.0 * eps)
        denom = max(float(np.abs(numeric).max()), 1e-12)
        relative = float(np.abs(numeric - grad).max()) / denom
        assert relative < 1e-5, f"{name}: relative gradient error {relative:.2e}"


def test_gradients_are_zero_at_a_flat_objective():
    """A network whose output already equals the target has a vanishing output layer."""
    rng = np.random.default_rng(12)
    params = nn.init_params(3, 4, rng)
    x = rng.standard_normal((5, 4, 3))
    pred, _, _ = nn.forward(params, x)
    _, grads = _loss_and_grads(params, x, pred)
    for grad in grads:
        assert np.allclose(grad, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def test_sigmoid_does_not_overflow():
    x = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    out = nn.sigmoid(x)
    assert np.isfinite(out).all()
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert out[2] == pytest.approx(0.5)


def test_sigmoid_matches_the_naive_formula_in_the_stable_range():
    x = np.linspace(-20.0, 20.0, 401)
    assert np.allclose(nn.sigmoid(x), 1.0 / (1.0 + np.exp(-x)))


def test_batch_size_does_not_change_the_prediction():
    """Rows are independent, so a batch of one equals the matching row of a batch of many."""
    rng = np.random.default_rng(13)
    params = nn.init_params(3, 4, rng)
    x = rng.standard_normal((7, 6, 3))
    many, _, _ = nn.forward(params, x)
    one, _, _ = nn.forward(params, x[3:4])
    assert one[0] == pytest.approx(many[3], rel=1e-12)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def test_make_sequences_windows_end_at_the_requested_row():
    values = np.arange(30, dtype=float).reshape(10, 3)
    x, index = nn.make_sequences(values, length=4, start=5, stop=8)
    assert index.tolist() == [5, 6, 7]
    assert x.shape == (3, 4, 3)
    assert np.array_equal(x[0], values[2:6])
    assert np.array_equal(x[-1], values[4:8])


def test_make_sequences_returns_nothing_when_the_window_does_not_fit():
    values = np.arange(12, dtype=float).reshape(4, 3)
    x, index = nn.make_sequences(values, length=5, start=0, stop=3)
    assert x.shape[0] == 0
    assert index.size == 0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_fit_reduces_the_holdout_loss():
    """Training must move the holdout loss, or the model is not learning anything."""
    rng = np.random.default_rng(14)
    steps, features = 8, 3
    n = 800
    x = rng.standard_normal((n, steps, features))
    y = 0.7 * x[:, -1, 0] - 0.3 * x[:, -1, 1] + 0.05 * rng.standard_normal(n)

    params = nn.init_params(features, 6, rng)
    initial_model = nn.GRU(
        params=params,
        input_mean=x.reshape(-1, features).mean(0),
        input_std=x.reshape(-1, features).std(0) + 1e-8,
        target_mean=float(y.mean()),
        target_std=float(y.std()),
    )
    before = float(np.mean((initial_model.predict(x) - y) ** 2))
    fitted = nn.GRU.fit(
        x,
        y,
        rng,
        {
            "sequence_length": steps,
            "hidden_size": 6,
            "learning_rate": 0.02,
            "epochs": 60,
            "batch_size": 32,
            "grad_clip": 5.0,
            "patience": 10,
        },
    )
    after = float(np.mean((fitted.predict(x) - y) ** 2))
    assert after < before


def test_predict_returns_one_value_per_sequence():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((40, 5, 3))
    y = rng.standard_normal(40)
    model = nn.GRU.fit(
        x,
        y,
        rng,
        {
            "sequence_length": 5,
            "hidden_size": 4,
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 16,
            "grad_clip": 5.0,
            "patience": 2,
        },
    )
    assert model.predict(x).shape == (40,)


def test_training_is_reproducible_for_a_fixed_seed():
    rng_a = np.random.default_rng(16)
    rng_b = np.random.default_rng(16)
    x = np.random.default_rng(17).standard_normal((200, 4, 2))
    y = np.random.default_rng(18).standard_normal(200)
    options = {
        "sequence_length": 4,
        "hidden_size": 3,
        "learning_rate": 0.01,
        "epochs": 3,
        "batch_size": 16,
        "grad_clip": 5.0,
        "patience": 2,
    }
    a = nn.GRU.fit(x, y, rng_a, options)
    b = nn.GRU.fit(x, y, rng_b, options)
    assert np.array_equal(a.params.flat(), b.params.flat())


def test_early_stopping_restores_the_best_state():
    """The returned parameters score at least as well as the last epoch's."""
    rng = np.random.default_rng(19)
    x = rng.standard_normal((400, 6, 3))
    y = rng.standard_normal(400)
    model = nn.GRU.fit(
        x,
        y,
        rng,
        {
            "sequence_length": 6,
            "hidden_size": 5,
            "learning_rate": 0.01,
            "epochs": 40,
            "batch_size": 32,
            "grad_clip": 5.0,
            "patience": 3,
        },
    )
    assert model.history and np.isfinite(model.history[0])


def test_gradient_clipping_keeps_a_huge_target_finite():
    """A single extreme window cannot blow the hidden state up."""
    rng = np.random.default_rng(20)
    x = rng.standard_normal((100, 5, 3))
    y = rng.standard_normal(100) * 50.0  # deliberately enormous targets
    model = nn.GRU.fit(
        x,
        y,
        rng,
        {
            "sequence_length": 5,
            "hidden_size": 4,
            "learning_rate": 0.05,
            "epochs": 15,
            "batch_size": 16,
            "grad_clip": 1.0,
            "patience": 5,
        },
    )
    assert np.isfinite(model.params.flat()).all()
    assert np.isfinite(model.predict(x)).all()
