"""Straight-through estimator for the latent weight path (handoff section 18).

The forward pass uses the reconstructed quantized weight; gradients pass through
as identity to the latent BF16 weight::

    ste_weight = latent_weight + (reconstructed_weight - latent_weight).detach()

so ``d(ste_weight)/d(latent_weight) = 1`` and the quantizer is treated as a
constant in the backward pass.

Two variants are provided. :func:`quantized_weight_ste` is the additive idiom
written verbatim in handoff section 18; its forward value equals
``reconstructed`` only to within floating-point rounding, because
``lat + (rec - lat)`` is not exactly ``rec`` in binary floating point.
:func:`quantized_weight_ste_exact` uses an explicit ``autograd.Function`` so the
forward value is bit-exact. The numerical difference is ~1 ulp and does not
affect training; the exact variant exists so that "forward == reconstructed" is
a testable property rather than an approximation.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "StraightThrough",
    "quantized_weight_ste",
    "quantized_weight_ste_exact",
    "ste_matmul",
]


def quantized_weight_ste(
    latent_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse a reconstructed weight into the graph with an identity gradient.

    :param latent_weight: the trainable full-precision weight
    :param reconstructed_weight: quantized/dequantized weight, same shape
    :return: tensor whose forward value is ``reconstructed_weight`` and whose
        gradient with respect to ``latent_weight`` is the identity
    """
    if latent_weight.shape != reconstructed_weight.shape:
        raise ValueError(
            f"shape mismatch: latent {tuple(latent_weight.shape)} vs "
            f"reconstructed {tuple(reconstructed_weight.shape)}"
        )
    reconstructed = reconstructed_weight.to(latent_weight.dtype)
    return latent_weight + (reconstructed - latent_weight).detach()


def quantized_weight_ste_exact(
    latent_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
) -> torch.Tensor:
    """Bit-exact straight-through weight.

    Identical semantics to :func:`quantized_weight_ste`, but the forward value is
    exactly ``reconstructed_weight`` rather than within 1 ulp of it.

    :param latent_weight: the trainable full-precision weight
    :param reconstructed_weight: quantized/dequantized weight, same shape
    :return: tensor whose forward value is exactly ``reconstructed_weight``
    """
    if latent_weight.shape != reconstructed_weight.shape:
        raise ValueError(
            f"shape mismatch: latent {tuple(latent_weight.shape)} vs "
            f"reconstructed {tuple(reconstructed_weight.shape)}"
        )
    reconstructed = reconstructed_weight.to(latent_weight.dtype).detach()
    return StraightThrough.apply(latent_weight, reconstructed)


def ste_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear forward using the straight-through weight.

    :param x: input activations
    :param weight: a weight already passed through :func:`quantized_weight_ste`
    :param bias: optional bias
    """
    return torch.nn.functional.linear(x, weight, bias)


class StraightThrough(torch.autograd.Function):
    """Explicit autograd identity-override version of the STE.

    ``forward`` returns ``reconstructed``; ``backward`` passes the incoming
    gradient straight to the latent weight.
    """

    @staticmethod
    def forward(
        ctx: Any,
        latent: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, latent
        return reconstructed

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_outputs[0], None
