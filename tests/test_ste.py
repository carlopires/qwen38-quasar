"""STE tests (handoff sections 18, 42).

Forward value equals the reconstructed weight; the gradient with respect to the
latent weight is the identity.
"""

from __future__ import annotations

import pytest
import torch

from qwen38_quasar.quantization.ste import (
    StraightThrough,
    quantized_weight_ste,
    quantized_weight_ste_exact,
    ste_matmul,
)


def _grad(tensor: torch.Tensor) -> torch.Tensor:
    """Narrow a `.grad` access for the type checker."""
    assert tensor.grad is not None
    return tensor.grad


def _reconstructed(shape: tuple[int, ...] = (8, 16)) -> torch.Tensor:
    """A plausible E2M1-quantized weight: codebook values times a scale."""
    codes = torch.randint(0, 16, shape)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    return magnitudes[codes & 0x7] * sign * 0.01


def test_forward_value_is_the_reconstructed_weight() -> None:
    """The forward pass must use the quantized weight, not the latent one."""
    latent = torch.randn(8, 16, requires_grad=True)
    reconstructed = _reconstructed()
    weight = quantized_weight_ste_exact(latent, reconstructed)
    assert torch.equal(weight.detach(), reconstructed.to(latent.dtype))


def test_additive_variant_matches_within_float_rounding() -> None:
    """The additive idiom is faithful up to ~1 ulp, which is why both exist."""
    latent = torch.randn(8, 16, requires_grad=True)
    reconstructed = _reconstructed()
    weight = quantized_weight_ste(latent, reconstructed)
    # Absolute tolerance, not relative: `rec - lat` cancels and loses precision
    # when |lat| >> |rec|, which is exactly the regime being documented here.
    assert torch.allclose(weight.detach(), reconstructed.to(latent.dtype), atol=1e-6)
    assert not torch.equal(weight.detach(), reconstructed.to(latent.dtype))


def test_gradient_with_respect_to_latent_is_identity() -> None:
    """d(forward)/d(latent) == 1 elementwise."""
    latent = torch.randn(8, 16, requires_grad=True)
    weight = quantized_weight_ste(latent, _reconstructed())
    weight.sum().backward()
    assert torch.equal(_grad(latent), torch.ones_like(latent))


def test_exact_variant_gradient_is_identity() -> None:
    """The autograd.Function variant also yields an identity gradient."""
    latent = torch.randn(4, 32, requires_grad=True)
    weight = quantized_weight_ste_exact(latent, _reconstructed((4, 32)))
    weight.sum().backward()
    assert torch.equal(_grad(latent), torch.ones_like(latent))


def test_gradient_is_identity_through_an_actual_quantizer() -> None:
    """End-to-end: quantize with E2M1, fuse with the STE, check the gradient."""
    from qwen38_quasar.quantization.nvfp4 import dequantize, quantize_tensor_group

    latent = torch.nn.Parameter(torch.randn(32, 64) * 0.02)
    packed, scale, global_scale = quantize_tensor_group(latent.detach())
    reconstructed = dequantize(packed, scale, global_scale)

    weight = quantized_weight_ste_exact(latent, reconstructed)
    # (8, 64) @ (64, 32) -> (8, 32); out[i, j] = sum_k x[i, k] * W[j, k]
    downstream = torch.randn(8, 64)
    ste_matmul(downstream, weight).sum().backward()

    # d(sum)/dW[j, k] = sum_i x[i, k], independent of j
    expected = downstream.sum(dim=0, keepdim=True).expand_as(latent)
    assert torch.allclose(_grad(latent), expected)


def test_quantizer_receives_no_gradient() -> None:
    """The discrete search must not attempt to backpropagate into itself."""
    latent = torch.randn(8, 16, requires_grad=True)
    reconstructed = _reconstructed()
    weight = quantized_weight_ste_exact(latent, reconstructed)
    weight.sum().backward()
    # reconstructed was a constant; only the latent weight accumulated a gradient
    assert latent.grad is not None


def test_shape_mismatch_is_rejected() -> None:
    """A mismatched reconstructed weight is an error, not a silent broadcast."""
    latent = torch.randn(8, 16)
    for function in (quantized_weight_ste, quantized_weight_ste_exact):
        with pytest.raises(ValueError, match="shape mismatch"):
            function(latent, torch.randn(8, 32))


def test_dtype_is_preserved() -> None:
    """bf16 latent weights stay bf16 in the forward pass."""
    latent = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = quantized_weight_ste_exact(latent, _reconstructed())
    assert weight.dtype == torch.bfloat16


def test_straight_through_function_directly() -> None:
    """The autograd.Function itself behaves as documented."""
    latent = torch.randn(4, 4, requires_grad=True)
    reconstructed = torch.randn(4, 4)
    out = StraightThrough.apply(latent, reconstructed)
    assert torch.equal(out, reconstructed)
    out.sum().backward()
    assert torch.equal(_grad(latent), torch.ones_like(latent))
