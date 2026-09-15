"""Dynamic-local NVFP4 activation fake quantization (handoff section 20).

The first milestone is true W4A4. Activations are quantized with the same
contract as the public QUASAR checkpoint::

    num_bits = 4
    type = float
    strategy = tensor_group
    group_size = 16
    symmetric = true
    scale_dtype = torch.float8_e4m3fn
    dynamic = "local"

``dynamic = "local"`` means the scale is computed from the current activation
tensor at runtime rather than from calibration statistics, so the activation
quantizer needs no stored parameters. Training must quantize activations in the
forward pass; training in BF16 and exporting as W4A4 would invalidate the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen38_quasar.quantization.e2m1 import round_to_e2m1
from qwen38_quasar.quantization.nvfp4 import (
    E2M1_MAX,
    FP8_E4M3_MAX,
    group_view,
    round_to_fp8_e4m3,
)

__all__ = [
    "ACTIVATION_GROUP_SIZE",
    "ActivationQuantConfig",
    "fake_quantize_activations",
]

ACTIVATION_GROUP_SIZE = 16

_EPS = 1e-12


@dataclass(frozen=True)
class ActivationQuantConfig:
    """Activation quantization contract matching the reference checkpoint."""

    num_bits: int = 4
    group_size: int = ACTIVATION_GROUP_SIZE
    symmetric: bool = True
    dynamic: str = "local"
    scale_dtype: str = "torch.float8_e4m3fn"

    def __post_init__(self) -> None:
        if self.num_bits != 4:
            raise ValueError("NVFP4 activation quantization requires num_bits = 4")
        if not self.symmetric:
            raise ValueError("E2M1 activation quantization is symmetric only")
        if self.dynamic != "local":
            raise ValueError("milestone 1 uses dynamic='local' activations")


def fake_quantize_activations(
    x: torch.Tensor,
    config: ActivationQuantConfig | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize-then-dequantize activations under the NVFP4 W4A4 contract.

    The last dimension is grouped into ``group_size`` blocks. Each block gets a
    scale derived from its own absolute maximum, and that scale is rounded into
    FP8 E4M3 before being used to reconstruct -- the same discipline the weight
    path follows, so that training sees the deployable quantizer.

    Nonlinearity is applied on the quantized value, so gradients flow through via
    the reconstruction (a straight-through style path) rather than being zeroed.

    :param x: activation tensor, any shape with a last dimension divisible by
        ``group_size``
    :param config: activation quantization configuration
    :param global_scale: optional tensor-level FP32 scale
    :return: fake-quantized activations, same shape and dtype as ``x``
    """
    config = config or ActivationQuantConfig()
    dtype = x.dtype
    original_shape = x.shape

    if x.shape[-1] % config.group_size != 0:
        # Fall back to padding so that odd widths do not silently quantize wrong.
        pad = config.group_size - (x.shape[-1] % config.group_size)
        x = torch.nn.functional.pad(x, (0, pad))

    work = x.to(torch.float32)
    grouped = group_view(work, config.group_size)

    group_max = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=_EPS)
    ideal_scale = group_max / E2M1_MAX

    stored_scale = round_to_fp8_e4m3(
        ideal_scale if global_scale is None else ideal_scale * global_scale.reshape(())
    )
    deployed = stored_scale.to(torch.float32)
    if global_scale is not None:
        deployed = deployed / global_scale.reshape(())

    codes = (grouped / deployed.clamp(min=_EPS)).clamp(-E2M1_MAX, E2M1_MAX)
    reconstructed = round_to_e2m1(codes) * deployed

    out = reconstructed.flatten(-2, -1)
    if out.shape != original_shape:
        out = out[..., : original_shape[-1]]
    return out.reshape(original_shape).to(dtype)


def activation_global_scale(
    max_abs: torch.Tensor | float,
    fp8_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """Tensor-level activation global scale, in the stored reciprocal convention.

    Provided for the static case; milestone 1 uses ``dynamic = "local"`` and does
    not need it.

    :param max_abs: tensor absolute maximum
    :return: fp32 tensor of shape ``(1,)``
    """
    value = torch.as_tensor(max_abs, dtype=torch.float32).clamp(min=torch.finfo(torch.float32).tiny)
    return (fp8_max * E2M1_MAX / value).reshape(1)
