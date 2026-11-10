"""Group-averaged networks and float64 box enclosures.

All box computations in this module are float64 enclosures, not validated.
The interval formulas are mathematical enclosures in exact arithmetic.
No directed rounding is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


def sine_signs(dimension: int) -> np.ndarray:
    """The four sign vectors in the order 1, sigma, rho, sigma rho."""
    if dimension < 1:
        raise ValueError("The representation dimension must be positive")
    rho = np.where(np.arange(dimension) % 2 == 0, 1.0, -1.0)
    return np.stack((np.ones(dimension), -np.ones(dimension), rho, -rho))


class EquivariantMLP(nn.Module):
    """Average a two-hidden-layer tanh network over the given sign action."""

    def __init__(self, input_dim: int, output_dim: int, width: int = 64):
        super().__init__()
        self.input_dim, self.output_dim, self.width = input_dim, output_dim, width
        self.register_buffer("input_signs", torch.tensor(sine_signs(input_dim),
                                                       dtype=torch.float64))
        self.register_buffer("output_signs", torch.tensor(sine_signs(output_dim),
                                                        dtype=torch.float64))
        self.net = nn.Sequential(
            nn.Linear(input_dim, width, dtype=torch.float64), nn.Tanh(),
            nn.Linear(width, width, dtype=torch.float64), nn.Tanh(),
            nn.Linear(width, output_dim, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float64:
            raise TypeError("Model inputs must be float64")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.input_dim)
        transformed = flat[None] * self.input_signs[:, None, :]
        values = self.net(transformed)
        averaged = (values * self.output_signs[:, None, :]).mean(dim=0)
        return averaged.reshape(*original_shape, self.output_dim)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Return complete group orbits, preserving the input's split assignment."""
    signs = torch.as_tensor(sine_signs(x.shape[-1]), dtype=x.dtype, device=x.device)
    return (x[None] * signs[:, None, :]).reshape(-1, x.shape[-1])


def representatives(x: np.ndarray) -> np.ndarray:
    """Choose a reproducible representative by the first nonzero odd/even modes."""
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    for parity in (0, 1):
        block = result[:, parity::2]
        if not block.shape[1]:
            continue
        nonzero = block != 0
        first = np.argmax(nonzero, axis=1)
        value = block[np.arange(len(block)), first]
        sign = np.where(value < 0, -1.0, 1.0)
        result[:, parity::2] *= sign[:, None]
    return result


def ball_samples(n: int, dimension: int, radius: float, seed: int):
    """Half volume-uniform interior and half uniform boundary samples."""
    if n % 2 or radius <= 0:
        raise ValueError("Use an even sample count and a positive radius")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, dimension))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    scales = np.ones(n)
    scales[:n // 2] = rng.random(n // 2) ** (1.0 / dimension)
    return representatives(radius * scales[:, None] * direction), np.r_[
        np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)]


def _linear_bounds(lo, hi, weight, bias):
    positive, negative = np.maximum(weight, 0), np.minimum(weight, 0)
    return lo @ positive.T + hi @ negative.T + bias, (
        hi @ positive.T + lo @ negative.T + bias)


def _derivative_linear(lo, hi, weight):
    positive, negative = np.maximum(weight, 0), np.minimum(weight, 0)
    return np.einsum("oi,bid->bod", positive, lo) + np.einsum(
        "oi,bid->bod", negative, hi), np.einsum(
        "oi,bid->bod", positive, hi) + np.einsum("oi,bid->bod", negative, lo)


def _positive_interval_product(lo, hi, positive_lo, positive_hi):
    terms = np.stack((lo * positive_lo, lo * positive_hi,
                      hi * positive_lo, hi * positive_hi))
    return terms.min(axis=0), terms.max(axis=0)


@dataclass
class NetworkEnclosure:
    """Numpy copy of a fixed network, with natural and mean-value enclosures."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    input_signs: np.ndarray
    output_signs: np.ndarray

    @classmethod
    def from_model(cls, model: EquivariantMLP):
        layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        return cls(
            [layer.weight.detach().cpu().numpy().copy() for layer in layers],
            [layer.bias.detach().cpu().numpy().copy() for layer in layers],
            model.input_signs.detach().cpu().numpy().copy(),
            model.output_signs.detach().cpu().numpy().copy(),
        )

    def _base(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if i + 1 < len(self.weights):
                x = np.tanh(x)
        return x

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros((*x.shape[:-1], self.output_signs.shape[1]))
        for sin, sout in zip(self.input_signs, self.output_signs):
            result += self._base(x * sin) * sout
        return result / len(self.input_signs)

    def _base_bounds(self, lo, hi):
        batch, dimension = lo.shape
        jlo = np.broadcast_to(np.eye(dimension), (batch, dimension, dimension)).copy()
        jhi = jlo.copy()
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            lo, hi = _linear_bounds(lo, hi, weight, bias)
            jlo, jhi = _derivative_linear(jlo, jhi, weight)
            if i + 1 < len(self.weights):
                ylo, yhi = np.tanh(lo), np.tanh(hi)
                abs_max = np.maximum(np.abs(ylo), np.abs(yhi))
                abs_min = np.where((lo <= 0) & (hi >= 0), 0,
                                   np.minimum(np.abs(ylo), np.abs(yhi)))
                derivative_lo = np.maximum(0, 1 - abs_max ** 2)
                derivative_hi = np.maximum(0, 1 - abs_min ** 2)
                jlo, jhi = _positive_interval_product(
                    jlo, jhi, derivative_lo[..., None], derivative_hi[..., None])
                lo, hi = ylo, yhi
        return lo, hi, jlo, jhi

    def enclose(self, lower, upper):
        """Enclose boxes (batch, dimension), without the caller's padding."""
        lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        if lower.ndim != 2 or lower.shape != upper.shape:
            raise ValueError("Bounds must have matching (batch, dimension) shapes")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Nonfinite input bounds")
        if np.any(lower > upper):
            raise ValueError("Reversed box bounds")
        output_dim = self.output_signs.shape[1]
        vlo = np.zeros((len(lower), output_dim))
        vhi = np.zeros_like(vlo)
        jlo = np.zeros((len(lower), output_dim, lower.shape[1]))
        jhi = np.zeros_like(jlo)
        for sin, sout in zip(self.input_signs, self.output_signs):
            xlo = np.minimum(lower * sin, upper * sin)
            xhi = np.maximum(lower * sin, upper * sin)
            a, b, ja, jb = self._base_bounds(xlo, xhi)
            vlo += np.minimum(a * sout, b * sout)
            vhi += np.maximum(a * sout, b * sout)
            signs = sout[:, None] * sin[None, :]
            jlo += np.minimum(ja * signs, jb * signs)
            jhi += np.maximum(ja * signs, jb * signs)
        count = len(self.input_signs)
        vlo, vhi, jlo, jhi = vlo / count, vhi / count, jlo / count, jhi / count
        center, radius = (lower + upper) / 2, (upper - lower) / 2
        midvalue = self(center)
        displacement = np.einsum("bod,bd->bo", np.maximum(np.abs(jlo), np.abs(jhi)), radius)
        result_lo = np.maximum(vlo, midvalue - displacement)
        result_hi = np.minimum(vhi, midvalue + displacement)
        if np.any(result_lo > result_hi) or not np.isfinite(result_lo).all():
            raise ArithmeticError("Inconsistent or nonfinite float64 enclosure")
        return result_lo, result_hi


def evaluate(model: nn.Module, values: np.ndarray, batch_size: int = 4096):
    """Evaluate a fixed model without retaining an autograd graph."""
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            inputs = torch.as_tensor(values[start:start + batch_size], dtype=torch.float64)
            parts.append(model(inputs).cpu().numpy())
    if not parts:
        return np.empty((0, model.output_dim), dtype=np.float64)
    return np.concatenate(parts)


def fit_models(data, seed: int, config: dict, progress: Callable | None = None):
    """Run the fixed two-stage training schedule, without model selection."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    n, d, width = config["modes"], config["latent_dim"], config["width"]
    encoder, decoder = EquivariantMLP(n, d, width), EquivariantMLP(d, n, width)
    latent = EquivariantMLP(d, d, width)
    mask = data["split"] == 0
    train_x = torch.as_tensor(data["XA"][mask], dtype=torch.float64)
    train_y = torch.as_tensor(data["YA"][mask], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 1000)
    history = []

    def optimize(parameters, updates, loss_function, stage):
        optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, updates, eta_min=config["minimum_learning_rate"])
        last = {}
        for step in range(updates):
            ids = torch.randint(len(train_x), (config["batch_size"],), generator=generator)
            optimizer.zero_grad()
            loss, terms = loss_function(ids)
            if not torch.isfinite(loss):
                raise ArithmeticError("Training loss became nonfinite")
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % 250 == 0 or step + 1 == updates:
                last = {"stage": stage, "step": step + 1,
                        **{key: float(value.detach()) for key, value in terms.items()}}
                history.append(last)
                if progress:
                    progress(last)
        return last

    def reconstruction_loss(ids):
        x = augment(train_x[ids])
        z = encoder(x)
        decoded = decoder(z)
        rec = ((decoded - x) ** 2).mean()
        cycle = ((encoder(decoded) - z) ** 2).mean()
        return rec + config["cycle_weight"] * cycle, {"reconstruction": rec, "cycle": cycle}

    optimize(list(encoder.parameters()) + list(decoder.parameters()),
             config["autoencoder_updates"], reconstruction_loss, "autoencoder")
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad_(False)
    with torch.no_grad():
        encoded_x, encoded_y = encoder(train_x), encoder(train_y)

    def latent_loss(ids):
        x, y = augment(encoded_x[ids]), augment(encoded_y[ids])
        residual = ((latent(x) - y) ** 2).mean()
        return residual, {"latent_mse": residual}

    optimize(latent.parameters(), config["latent_updates"], latent_loss, "latent")
    for model in (encoder, decoder, latent):
        model.eval()
    return encoder, decoder, latent, history


def model_metrics(encoder, decoder, latent, x, y):
    z, target = evaluate(encoder, x), evaluate(encoder, y)
    reconstruction = evaluate(decoder, z)
    prediction = evaluate(latent, z)
    residual = np.linalg.norm(prediction - target, axis=1)
    orbit_targets = target[None] * sine_signs(target.shape[1])[:, None, :]
    residual_mod = np.linalg.norm(prediction[None] - orbit_targets, axis=-1).min(axis=0)
    return {
        "count": len(x),
        "relative_reconstruction_mse": float(
            np.sum((reconstruction - x) ** 2) / np.sum(x ** 2)),
        "latent_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "latent_residual_max": float(residual.max()),
        "latent_residual_mod_max": float(residual_mod.max()),
        "cycle_max": float(np.linalg.norm(evaluate(encoder, reconstruction) - z, axis=1).max()),
        "latent_standard_deviation": z.std(axis=0).tolist(),
    }


def fit_frozen_latent(data, saved_state, seed, config, progress=None):
    """Replay the original latent stage, then run the fixed-duration retry."""
    torch.set_default_dtype(torch.float64)

    def initialize():
        torch.manual_seed(seed)
        n, d, width = config["modes"], config["latent_dim"], config["width"]
        encoder, decoder = EquivariantMLP(n, d, width), EquivariantMLP(d, n, width)
        latent = EquivariantMLP(d, d, width)
        encoder.load_state_dict(saved_state["encoder"])
        decoder.load_state_dict(saved_state["decoder"])
        for parameter in list(encoder.parameters()) + list(decoder.parameters()):
            parameter.requires_grad_(False)
        return encoder.eval(), decoder.eval(), latent

    encoder, decoder, original_latent = initialize()
    mask = data["split"] == 0
    with torch.no_grad():
        x = torch.as_tensor(data["XA"][mask], dtype=torch.float64)
        y = torch.as_tensor(data["YA"][mask], dtype=torch.float64)
        encoded_x, encoded_y = encoder(x), encoder(y)
        cycle_mse = float(((encoder(decoder(encoded_x)) - encoded_x) ** 2).mean())

    def train(latent, updates, stage):
        generator = torch.Generator().manual_seed(seed + 1000)
        for _ in range(config["autoencoder_updates"]):
            torch.randint(len(x), (config["batch_size"],), generator=generator)
        optimizer = torch.optim.Adam(latent.parameters(), lr=config["learning_rate"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, updates, eta_min=config["minimum_learning_rate"])
        history = []
        for step in range(updates):
            ids = torch.randint(len(x), (config["batch_size"],), generator=generator)
            optimizer.zero_grad()
            loss = ((latent(augment(encoded_x[ids])) - augment(encoded_y[ids])) ** 2).mean()
            if not torch.isfinite(loss):
                raise ArithmeticError("Retry training loss became nonfinite")
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % 250 == 0 or step + 1 == updates:
                row = {"stage": stage, "step": step + 1, "latent_mse": float(loss.detach()),
                       "frozen_cycle_mse": cycle_mse}
                history.append(row)
                if progress:
                    progress(row)
        return history

    replay_history = train(original_latent, config["original_latent_updates"], "replay")
    differences = {key: float((value - saved_state["latent"][key]).abs().max())
                   for key, value in original_latent.state_dict().items()}
    if any(value != 0 for value in differences.values()):
        raise ArithmeticError(f"Original latent training did not reproduce: {differences}")
    _, _, latent = initialize()
    history = train(latent, config["latent_updates"], "latent_retry")
    for key, model in (("encoder", encoder), ("decoder", decoder)):
        if any(not torch.equal(value, saved_state[key][name])
               for name, value in model.state_dict().items()):
            raise AssertionError("A frozen encoder or decoder parameter changed")
    return encoder, decoder, latent.eval(), history, replay_history, differences

