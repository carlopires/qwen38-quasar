"""Activation quantization tests (handoff sections 20, 42).

Compares the dynamic-local NVFP4 activation fake quantizer against a
``compressed_tensors`` oracle built from the same contract.
"""

from __future__ import annotations

import pytest
import torch

from qwen38_quasar.quantization.activation import (
    ACTIVATION_GROUP_SIZE,
    ActivationQuantConfig,
    activation_global_scale,
    fake_quantize_activations,
)
from qwen38_quasar.quantization.nvfp4 import FP8_E4M3_MAX, round_to_fp8_e4m3

pytest.importorskip("compressed_tensors", reason="compressed-tensors is the oracle")

from compressed_tensors.quantization import (  # noqa: E402
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize  # noqa: E402


def _oracle(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation via compressed_tensors' own QDQ path.

    The scale is FP8-rounded and then widened back to float32 before being
    handed over: compressed_tensors' QDQ divides by the scale directly and so
    cannot accept a float8 scale, whereas the stored form is FP8. Rounding to
    FP8 and widening reproduces exactly what the deployed checkpoints store.
    """
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=ACTIVATION_GROUP_SIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        dynamic="local",
    )
    n_groups = x.shape[-1] // ACTIVATION_GROUP_SIZE
    grouped = x.unflatten(-1, (n_groups, ACTIVATION_GROUP_SIZE))
    ideal = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 6.0
    scale = round_to_fp8_e4m3(ideal).squeeze(-1).to(torch.float32)

    return fake_quantize(
        x,
        scale=scale,
        zero_point=torch.zeros_like(scale, dtype=torch.int32),
        args=args,
    )


def test_config_matches_the_reference_contract() -> None:
    """The declared contract mirrors the public QUASAR checkpoint."""
    config = ActivationQuantConfig()
    assert config.num_bits == 4
    assert config.group_size == 16
    assert config.symmetric is True
    assert config.dynamic == "local"
    assert config.scale_dtype == "torch.float8_e4m3fn"


def test_rejects_incompatible_configs() -> None:
    """Unsupported contracts fail loudly rather than silently mis-quantizing."""
    with pytest.raises(ValueError, match="num_bits"):
        ActivationQuantConfig(num_bits=8)
    with pytest.raises(ValueError, match="symmetric"):
        ActivationQuantConfig(symmetric=False)
    with pytest.raises(ValueError, match="dynamic"):
        ActivationQuantConfig(dynamic="static")


def test_shape_and_dtype_are_preserved() -> None:
    """fake_quantize_activations is a drop-in replacement in the forward pass."""
    x = torch.randn(4, 8, 64, dtype=torch.bfloat16)
    out = fake_quantize_activations(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_matches_compressed_tensors_oracle() -> None:
    """Agrees with compressed-tensors' own QDQ for the same contract."""
    torch.manual_seed(0)
    for shape in ((16, 64), (2, 32, 128)):
        x = torch.randn(*shape)
        mine = fake_quantize_activations(x).to(torch.float32)
        theirs = _oracle(x).to(torch.float32)
        assert torch.allclose(mine, theirs, rtol=1e-3, atol=1e-4), f"mismatch for {shape}"


def test_handles_indivisible_last_dimension() -> None:
    """A width that is not a multiple of group_size is padded, not truncated."""
    x = torch.randn(4, 20)
    out = fake_quantize_activations(x)
    assert out.shape == x.shape


def test_quantization_error_is_bounded_by_the_group_scale() -> None:
    """Every element is within half the largest E2M1 step of its group scale.

    E2M1 steps are non-uniform: the widest is 4.0 -> 6.0, so the worst-case
    nearest-code distance is 1.0 (at the 5.0 midpoint), i.e. one full scale step,
    plus a little slack for the FP8 rounding of the scale itself.
    """
    torch.manual_seed(1)
    x = torch.randn(32, 256)
    out = fake_quantize_activations(x)
    grouped_in = x.unflatten(-1, (16, 16))
    grouped_out = out.unflatten(-1, (16, 16))
    step = (grouped_in.abs().amax(dim=-1, keepdim=True) / 6.0).clamp(min=1e-12)
    error = (grouped_out - grouped_in).abs()
    assert (error <= step * 1.15).all()


def test_scale_is_rounded_into_fp8_before_use() -> None:
    """The deployed scale is an FP8 value, not the ideal float."""
    x = torch.randn(1, 16) * 3.0
    out = fake_quantize_activations(x)
    group_max = x.abs().max()
    ideal = group_max / 6.0
    fp8 = round_to_fp8_e4m3(ideal.reshape(1)).to(torch.float32)
    # reconstructed magnitudes must be E2M1 values times the FP8 scale
    ratio = (out / fp8).flatten()
    allowed = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    positive = ratio.abs()
    assert torch.isin(positive, allowed).all()


def test_zero_tensor_stays_zero() -> None:
    """An all-zero activation block reconstructs to zero, with no NaNs."""
    x = torch.zeros(2, 32)
    out = fake_quantize_activations(x)
    assert torch.count_nonzero(out) == 0
    assert not torch.isnan(out).any()


def test_gradients_flow_through_the_activation_quantizer() -> None:
    """The activation path is differentiable in the straight-through sense."""
    x = torch.randn(4, 32, requires_grad=True)
    out = fake_quantize_activations(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) > 0


def test_activation_global_scale_convention() -> None:
    """Tensor-level activation global scale uses the reciprocal convention."""
    scale = activation_global_scale(2.0)
    assert torch.allclose(scale, torch.tensor([FP8_E4M3_MAX * 6.0 / 2.0]))


def test_global_scale_reduces_scale_range_pressure() -> None:
    """Supplying a global scale keeps the fake quantization stable."""
    x = torch.randn(8, 64) * 0.001
    global_scale = activation_global_scale(x.abs().max())
    out = fake_quantize_activations(x, global_scale=global_scale.reshape(1))
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
