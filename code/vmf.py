"""Differentiable von Mises-Fisher utilities for a hyperspherical VAE.

The sampler follows the rejection-reparameterization construction used by
hyperspherical VAEs. The KL is analytic against a uniform hyperspherical
prior. scipy.special.ive is evaluated on CPU, with its exact derivative
provided to PyTorch autograd; all model tensors and sampling remain on the
original device.
"""

import math

import numpy as np
import torch
from scipy import special


class _Ive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, order, x):
        x_np = x.detach().cpu().numpy()
        value_np = special.ive(order, x_np)
        next_np = special.ive(order + 1.0, x_np)
        value = torch.as_tensor(value_np, dtype=x.dtype, device=x.device)
        next_value = torch.as_tensor(next_np, dtype=x.dtype, device=x.device)
        ctx.order = order
        ctx.save_for_backward(x, value, next_value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        x, value, next_value = ctx.saved_tensors
        safe_x = x.clamp_min(torch.finfo(x.dtype).tiny)
        # d/dx [exp(-x) I_v(x)] for x > 0.
        derivative = next_value + (ctx.order / safe_x - 1.0) * value
        return None, grad_output * derivative


def ive(order, x):
    return _Ive.apply(float(order), x)


def sample_vmf(loc, concentration, eps=1e-7, max_steps=10000):
    """Draw a reparameterized sample from vMF(loc, concentration).

    Args:
        loc: unit vectors with shape (batch, event_dim).
        concentration: positive values with shape (batch, 1).
    """
    if loc.ndim != 2 or concentration.shape != (loc.shape[0], 1):
        raise ValueError(
            "Expected loc=(batch, event_dim) and concentration=(batch, 1), "
            f"got {tuple(loc.shape)} and {tuple(concentration.shape)}"
        )

    batch_size, event_dim = loc.shape
    if event_dim < 2:
        raise ValueError("vMF requires event_dim >= 2")

    kappa = concentration.squeeze(-1).clamp_min(eps)
    dim_minus_one = float(event_dim - 1)
    b = dim_minus_one / (
        torch.sqrt(4.0 * kappa.square() + dim_minus_one**2) + 2.0 * kappa
    )
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + dim_minus_one * torch.log1p(-x0.square())

    w = torch.zeros_like(kappa)
    pending = torch.ones(batch_size, dtype=torch.bool, device=loc.device)
    beta = torch.distributions.Beta(
        torch.tensor(dim_minus_one / 2.0, dtype=loc.dtype, device=loc.device),
        torch.tensor(dim_minus_one / 2.0, dtype=loc.dtype, device=loc.device),
    )

    for _ in range(max_steps):
        if not bool(pending.any()):
            break
        indices = pending.nonzero(as_tuple=False).squeeze(-1)
        z = beta.rsample((indices.numel(),))
        b_i = b[indices]
        candidate = (1.0 - (1.0 + b_i) * z) / (
            1.0 - (1.0 - b_i) * z
        )
        score = (
            kappa[indices] * candidate
            + dim_minus_one * torch.log1p(-x0[indices] * candidate)
            - c[indices]
        )
        accept = score >= torch.log(
            torch.rand_like(score).clamp_min(torch.finfo(score.dtype).tiny)
        )
        accepted_indices = indices[accept]
        w[accepted_indices] = candidate[accept]
        pending[accepted_indices] = False
    else:
        raise RuntimeError("vMF rejection sampler did not converge")

    tangent = torch.randn(
        batch_size, event_dim - 1, dtype=loc.dtype, device=loc.device
    )
    tangent = torch.nn.functional.normalize(tangent, dim=-1, eps=eps)
    base_sample = torch.cat(
        [
            w.unsqueeze(-1),
            torch.sqrt((1.0 - w.square()).clamp_min(eps)).unsqueeze(-1)
            * tangent,
        ],
        dim=-1,
    )

    e1 = torch.zeros_like(loc)
    e1[:, 0] = 1.0
    reflection = e1 - loc
    reflection = reflection / reflection.norm(dim=-1, keepdim=True).clamp_min(eps)
    return base_sample - 2.0 * (
        base_sample * reflection
    ).sum(dim=-1, keepdim=True) * reflection


def vmf_kl_uniform(concentration, event_dim):
    """KL(vMF || Uniform(S^(event_dim-1))) for each batch element."""
    if event_dim < 2:
        raise ValueError("event_dim must be >= 2")

    original_dtype = concentration.dtype
    kappa = concentration.squeeze(-1).double().clamp_min(1e-8)
    order = event_dim / 2.0 - 1.0
    ive_order = ive(order, kappa).clamp_min(torch.finfo(kappa.dtype).tiny)
    ive_next = ive(order + 1.0, kappa)
    mean_resultant_length = ive_next / ive_order

    log_bessel = torch.log(ive_order) + kappa
    log_normalizer = (
        order * torch.log(kappa)
        - (event_dim / 2.0) * math.log(2.0 * math.pi)
        - log_bessel
    )
    log_surface_area = (
        math.log(2.0)
        + (event_dim / 2.0) * math.log(math.pi)
        - math.lgamma(event_dim / 2.0)
    )
    kl = kappa * mean_resultant_length + log_normalizer + log_surface_area
    return kl.clamp_min(0.0).to(original_dtype)
