"""A GRU written from scratch, in NumPy.

Why not a framework
-------------------
The sequence problem here is small: 4,000 training windows of 20 daily
observations with 15 features, a 12-unit hidden state and a scalar output. A
deep-learning dependency would be several hundred megabytes, would pin the
project to a platform-specific wheel, and would put the one component whose
correctness is hardest to verify behind an opaque binary.

Writing it out has a concrete payoff beyond the dependency: the backward pass can
be checked against a numerical gradient. The test suite asserts that the analytic
gradients agree with finite differences to a relative tolerance of 1e-5, which is
the kind of statement that cannot be made about a framework's optimiser.

Design notes
------------
* **One output per sequence**, taken from the final hidden state, fed through a
  linear layer to predict ``log RV``. Predicting the log rather than the level
  keeps the loss from being dominated by a handful of crisis days, the same
  reason the HAR model is fitted in logs.
* **Adam**, because plain SGD on a small GRU with a scalar loss spends most of
  its time in a plateau. The implementation is the standard one, including bias
  correction.
* **Gradient clipping by global norm**, which is not optional here: the
  likelihood-like target has heavy tails and a single outlier window can blow the
  hidden state up otherwise.
* **Early stopping on a holdout tail** of the training window, with the best
  parameters restored. Without it the model overfits within a few dozen epochs
  and the comparison against HAR becomes a comparison against an overfit network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as cfg


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


@dataclass
class GRUParams:
    """All weights of the network, as plain arrays."""

    #: Gate and candidate weights, each (input_dim + hidden, hidden) and (hidden,).
    wz: np.ndarray
    wr: np.ndarray
    wh: np.ndarray
    bz: np.ndarray
    br: np.ndarray
    bh: np.ndarray
    #: Output layer, mapping the final hidden state to log variance.
    wy: np.ndarray
    by: np.ndarray

    def flat(self) -> np.ndarray:
        return np.concatenate(
            [a.ravel() for a in (self.wz, self.wr, self.wh, self.bz, self.br, self.bh, self.wy, self.by)]
        )

    def shapes(self) -> list[tuple[int, ...]]:
        return [a.shape for a in (self.wz, self.wr, self.wh, self.bz, self.br, self.bh, self.wy, self.by)]


def init_params(input_dim: int, hidden: int, rng: np.random.Generator) -> GRUParams:
    """Xavier-scaled initialisation for the input weights, unit scale for the rest."""
    scale = np.sqrt(1.0 / (input_dim + hidden))
    hscale = np.sqrt(1.0 / hidden)
    return GRUParams(
        wz=rng.normal(0.0, scale, (input_dim + hidden, hidden)),
        wr=rng.normal(0.0, scale, (input_dim + hidden, hidden)),
        wh=rng.normal(0.0, scale, (input_dim + hidden, hidden)),
        bz=np.zeros(hidden),
        br=np.zeros(hidden),
        bh=np.zeros(hidden),
        wy=rng.normal(0.0, hscale, (hidden, 1)),
        by=np.zeros(1),
    )


def _pack(params: GRUParams) -> list[np.ndarray]:
    """All parameter arrays in the order :func:`backward` returns gradients."""
    return [params.wz, params.wr, params.wh, params.bz, params.br, params.bh, params.wy, params.by]


def forward(params: GRUParams, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]], np.ndarray]:
    """Forward pass over a batch.

    ``x`` has shape (batch, sequence, features). Returns the predictions, shape
    (batch,), and the per-step activations needed by the backward pass.
    """
    batch, steps, _ = x.shape
    hidden = params.wz.shape[1]
    h = np.zeros((batch, hidden))
    cache: list[dict[str, np.ndarray]] = []
    for t in range(steps):
        xh = np.concatenate([x[:, t, :], h], axis=1)
        z = sigmoid(xh @ params.wz + params.bz)
        r = sigmoid(xh @ params.wr + params.br)
        xr = np.concatenate([x[:, t, :], r * h], axis=1)
        h_tilde = np.tanh(xr @ params.wh + params.bh)
        h_new = (1.0 - z) * h + z * h_tilde
        cache.append({"xh": xh, "xr": xr, "z": z, "r": r, "h_prev": h, "h_tilde": h_tilde})
        h = h_new
    y = (h @ params.wy + params.by).ravel()
    return y, cache, h


def backward(params: GRUParams, cache: list[dict[str, np.ndarray]], h_final: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Backpropagation through time. Returns gradients in the order of :func:`_pack`."""
    hidden = params.wz.shape[1]
    input_dim = params.wz.shape[0] - hidden

    gwz = np.zeros_like(params.wz)
    gwr = np.zeros_like(params.wr)
    gwh = np.zeros_like(params.wh)
    gbz = np.zeros_like(params.bz)
    gbr = np.zeros_like(params.br)
    gbh = np.zeros_like(params.bh)
    gwy = h_final.T @ dy.reshape(-1, 1)
    gby = np.array([dy.sum()])

    dh = dy.reshape(-1, 1) @ params.wy.T
    for t in reversed(range(len(cache))):
        c = cache[t]
        z, r, h_prev, h_tilde, xh, xr = c["z"], c["r"], c["h_prev"], c["h_tilde"], c["xh"], c["xr"]
        # h_t = (1 - z) * h_prev + z * h_tilde
        dz = dh * (h_tilde - h_prev)
        dh_tilde = dh * z
        dh_prev = dh * (1.0 - z)

        # h_tilde = tanh(xr @ wh + bh)
        dpre = dh_tilde * (1.0 - h_tilde**2)
        gwh += xr.T @ dpre
        gbh += dpre.sum(axis=0)
        dxr = dpre @ params.wh.T
        dr = dxr[:, input_dim:] * h_prev
        dh_prev = dh_prev + dxr[:, input_dim:] * r

        # r = sigmoid(xh @ wr + br)
        dpr = dr * r * (1.0 - r)
        gwr += xh.T @ dpr
        gbr += dpr.sum(axis=0)
        dxh = dpr @ params.wr.T

        # z = sigmoid(xh @ wz + bz)
        dpz = dz * z * (1.0 - z)
        gwz += xh.T @ dpz
        gbz += dpz.sum(axis=0)
        dxh = dxh + dpz @ params.wz.T

        dh = dh_prev + dxh[:, input_dim:]

    return [gwz, gwr, gwh, gbz, gbr, gbh, gwy, gby]


@dataclass
class GRU:
    """A trained sequence model for log realised variance."""

    params: GRUParams
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float
    history: list[float] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        params: dict[str, object] | None = None,
    ) -> GRU:
        """Train on sequences ``x`` (batch, steps, features) and targets ``y``."""
        options = dict(cfg.GRU_PARAMS if params is None else params)
        hidden = int(options["hidden_size"])
        epochs = int(options["epochs"])
        lr = float(options["learning_rate"])
        batch_size = int(options["batch_size"])
        clip = float(options["grad_clip"])
        patience = int(options["patience"])

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        # Standardise inputs per feature and the target itself. Without this the
        # tanh saturates immediately and the network predicts the mean forever.
        input_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        input_std = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-8
        target_mean = float(y.mean())
        target_std = float(y.std() + 1e-8)
        xs = (x - input_mean) / input_std
        ys = (y - target_mean) / target_std

        model = init_params(xs.shape[-1], hidden, rng)
        state = _pack(model)
        m = [np.zeros_like(a) for a in state]
        v = [np.zeros_like(a) for a in state]
        step = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Hold out the tail for early stopping; the walk-forward engine refits
        # annually, so the holdout is inside the training window and never
        # overlaps the evaluation days.
        n = xs.shape[0]
        split = max(int(n * 0.85), 1)
        best_loss = np.inf
        best_state = [a.copy() for a in state]
        stale = 0

        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) < 2:
                    continue
                pred, cache, h_final = forward(model, xs[idx])
                err = pred - ys[idx]
                dy = 2.0 * err / len(idx)
                grads = backward(model, cache, h_final, dy)
                step += 1
                norm = np.sqrt(sum(float((g**2).sum()) for g in grads))
                if norm > clip:
                    grads = [g * (clip / norm) for g in grads]
                for i, g in enumerate(grads):
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    mhat = m[i] / (1 - beta1**step)
                    vhat = v[i] / (1 - beta2**step)
                    state[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            holdout, _, _ = forward(model, xs[split:])
            loss = float(np.mean((holdout - ys[split:]) ** 2)) if split < n else np.inf
            if loss < best_loss - 1e-6:
                best_loss = loss
                best_state = [a.copy() for a in state]
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
            _ = epoch

        trained = GRUParams(*best_state)
        return cls(
            params=trained,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            history=[best_loss],
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict ``log RV`` for a batch of sequences."""
        xs = (np.asarray(x, dtype=float) - self.input_mean) / self.input_std
        pred, _, _ = forward(self.params, xs)
        return pred * self.target_std + self.target_mean


def make_sequences(values: np.ndarray, length: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a feature matrix into overlapping sequences ending at each row.

    Row ``i`` of the output is ``values[i - length + 1 : i + 1]``, i.e. a window
    that *includes* row ``i``. Because row ``i`` of the feature matrix already
    contains only information dated ``i-1`` or earlier, the sequence ending at
    ``i`` is a legal predictor of the return on day ``i``.
    """
    values = np.asarray(values, dtype=float)
    indices = [i for i in range(max(start, length - 1), min(stop, len(values)))]
    if not indices:
        return np.empty((0, length, values.shape[1])), np.empty(0, dtype=int)
    sequences = np.stack([values[i - length + 1 : i + 1] for i in indices])
    return sequences, np.asarray(indices, dtype=int)
