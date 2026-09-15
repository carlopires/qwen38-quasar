"""NVFP4 storage format: global scale, FP8 group scales, packing.

Conventions are *derived* from ``compressed_tensors`` (see
``compressed_tensors.quantization.utils.helpers.generate_gparam``,
``calculate_qparams`` and ``lifecycle.forward_helpers._dequantize``) rather than
assumed. Handoff section 17 explicitly forbids guessing whether the stored
tensor holds a scale or a reciprocal scale.

Storage, per quantized Linear module:

``weight_packed``        uint8 ``(out, in // 2)``, two E2M1 codes per byte
``weight_scale``         fp8_e4m3fn ``(out, in // group_size)``
``weight_global_scale``  fp32 scalar ``(1,)``
``input_global_scale``   fp32 scalar ``(1,)`` (present when activations are static)

Dequantization::

    effective_scale = weight_scale / weight_global_scale
    weight          = e2m1(weight_packed) * effective_scale

The global scale is therefore stored in **reciprocal** convention::

    weight_global_scale = FP8_E4M3_MAX * E2M1_MAX / max(|W|)
                        = 448 * 6 / max(|W|)

and the per-group scale is stored pre-multiplied by it::

    weight_scale = effective_scale * weight_global_scale

so that a group whose magnitude equals the tensor maximum maps to 448.0, the
top of the FP8 E4M3 range. This matches
``generate_gparam`` / ``calculate_qparams`` exactly.
"""

from __future__ import annotations

import torch

from qwen38_quasar.quantization.e2m1 import decode, encode

__all__ = [
    "E2M1_MAX",
    "FP8_E4M3_MAX",
    "Fp8ScaleSaturation",
    "compute_global_scale",
    "dequantize",
    "group_view",
    "pack_e2m1",
    "quantize_tensor_group",
    "round_to_fp8_e4m3",
    "unpack_e2m1",
]

FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
_FP8_TINY = float(torch.finfo(torch.float8_e4m3fn).tiny)


def compute_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """Tensor-level FP32 global scale, in the reciprocal storage convention.

    Mirrors ``compressed_tensors.generate_gparam``: derived from the tensor
    absolute maximum, with the same NaN/Inf fallback to 1.0 (meaning "no global
    scaling"), which is the safe value for an all-zero tensor.

    :param weight: full-precision weight tensor
    :return: fp32 tensor of shape ``(1,)``
    """
    max_abs = weight.abs().amax().to(torch.float32)
    max_abs = max_abs.clamp(min=torch.finfo(torch.float32).tiny)
    global_scale = FP8_E4M3_MAX * E2M1_MAX / max_abs
    global_scale = torch.nan_to_num(
        global_scale, nan=1.0, posinf=1.0, neginf=1.0
    )
    return global_scale.to(torch.float32).reshape(1)


def round_to_fp8_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """Round scale values to FP8 E4M3.

    Uses the native cast, which rounds to nearest even, matching
    ``calculate_qparams``' ``round_to_quantized_type_dtype``.

    :param scale: float tensor
    :return: fp8_e4m3fn tensor of the same shape
    """
    return scale.to(torch.float8_e4m3fn)


def group_view(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reshape the last dimension of ``x`` into ``(..., n_groups, group_size)``.

    :param x: tensor whose last dimension is divisible by ``group_size``
    :param group_size: number of elements per quantization group
    :raises ValueError: if the last dimension is not divisible
    """
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} is not divisible by group_size {group_size}"
        )
    return x.unflatten(-1, (x.shape[-1] // group_size, group_size))


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 codes two-per-byte, matching the ``compressed_tensors`` layout.

    Even columns occupy the low nibble, odd columns the high nibble.

    :param codes: uint8 tensor with an even number of columns
    :return: uint8 tensor with half the columns
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError("tensor must have an even number of columns for nvfp4 packing")
    lo = codes[..., 0::2].to(torch.uint8)
    hi = codes[..., 1::2].to(torch.uint8)
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Unpack two E2M1 codes per byte into float values.

    :param packed: uint8 tensor
    :param dtype: output dtype (defaults to float32)
    :return: float tensor with twice the columns
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed tensor, got {packed.dtype}")
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    values = decode(interleaved)
    if dtype is not None:
        values = values.to(dtype)
    return values


def dequantize(
    codes_or_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct full-precision weights from stored NVFP4 components.

    ``weight_scale`` is broadcast over each group; ``weight_global_scale`` is
    divided out (reciprocal storage convention).

    :param codes_or_packed: uint8 packed weights, or a float tensor already
        holding E2M1 values
    :param weight_scale: per-group scale, shape ``(out, in // group_size)``
    :param weight_global_scale: fp32 scalar, or None for no global scaling
    :return: reconstructed weight, same shape as the unpacked weight
    """
    if codes_or_packed.dtype == torch.uint8:
        values = unpack_e2m1(codes_or_packed, dtype=torch.float32)
    else:
        values = codes_or_packed.to(torch.float32)

    # ``values`` is (out, in) and ``weight_scale`` is (out, n_groups); rebuild the
    # group axis on the last dimension so the scale broadcasts over each group.
    n_groups = weight_scale.shape[-1]
    group_size = values.shape[-1] // n_groups
    grouped = values.unflatten(-1, (n_groups, group_size))
    scale = weight_scale.to(torch.float32).unsqueeze(-1)

    if weight_global_scale is not None:
        scale = scale / weight_global_scale.to(torch.float32).reshape(())

    return (grouped * scale).flatten(-2, -1)


def quantize_tensor_group(
    weight: torch.Tensor,
    group_size: int = 16,
    weight_global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordinary (non-QUASAR) NVFP4 weight quantization.

    Round-to-nearest E2M1 assignment with a max-based per-group scale. This is
    the baseline that QUASAR's candidate search must not lose to (handoff
    section 42, ``test_quasar_search.py``).

    :param weight: full-precision weight, shape ``(out, in)``
    :param group_size: quantization group size
    :param weight_global_scale: precomputed global scale, or None to compute one
    :return: ``(packed, weight_scale, weight_global_scale)``
    """
    if weight_global_scale is None:
        weight_global_scale = compute_global_scale(weight)

    grouped = group_view(weight.to(torch.float32), group_size)
    group_max = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=_FP8_TINY)

    effective_scale = group_max / E2M1_MAX
    stored_scale = round_to_fp8_e4m3(effective_scale * weight_global_scale)
    weight_scale = stored_scale.squeeze(-1)

    codes = encode(grouped / effective_scale)
    packed = pack_e2m1(codes.reshape(weight.shape[0], weight.shape[1]))

    return packed, weight_scale, weight_global_scale


class Fp8ScaleSaturation:
    """Tracks how often FP8 group scales saturate or underflow.

    Handoff section 23 requires FP8 scale saturation statistics to be logged
    during training.
    """

    def __init__(self) -> None:
        self.total = 0
        self.saturated = 0
        self.underflowed = 0

    def update(self, weight_scale: torch.Tensor) -> None:
        """Accumulate statistics from a tensor of ideal (pre-rounding) scales.

        :param weight_scale: float tensor of scale values about to be rounded
        """
        scale = weight_scale.to(torch.float32)
        self.total += scale.numel()
        self.saturated += int((scale.abs() > FP8_E4M3_MAX).sum().item())
        self.underflowed += int(((scale != 0) & (scale.abs() < _FP8_TINY)).sum().item())

    def as_dict(self) -> dict[str, float | int]:
        """Return the counters plus derived rates."""
        denom = self.total or 1
        return {
            "total": self.total,
            "saturated": self.saturated,
            "underflowed": self.underflowed,
            "saturation_rate": self.saturated / denom,
            "underflow_rate": self.underflowed / denom,
        }
