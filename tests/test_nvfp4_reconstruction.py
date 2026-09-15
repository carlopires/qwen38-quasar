"""NVFP4 storage-format tests (handoff sections 16, 17, 42).

Covers the global-scale convention, packing layout, and the requirement that the
FP8-rounded group scale participates in scoring.

The storage conventions are cross-checked against ``compressed_tensors``, which
is the format oracle the exported checkpoint must satisfy.
"""

from __future__ import annotations

import pytest
import torch

from qwen38_quasar.quantization.nvfp4 import (
    E2M1_MAX,
    FP8_E4M3_MAX,
    Fp8ScaleSaturation,
    compute_global_scale,
    dequantize,
    group_view,
    pack_e2m1,
    quantize_tensor_group,
    round_to_fp8_e4m3,
    unpack_e2m1,
)

pytest.importorskip("compressed_tensors", reason="compressed-tensors is the format oracle")

from compressed_tensors.compressors.nvfp4.helpers import (  # noqa: E402
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)


def test_global_scale_matches_derived_convention() -> None:
    """weight_global_scale == FP8_MAX * E2M1_MAX / max|W| (reciprocal convention)."""
    weight = torch.randn(32, 64)
    expected = FP8_E4M3_MAX * E2M1_MAX / weight.abs().max()
    assert torch.allclose(compute_global_scale(weight), expected.reshape(1))


def test_global_scale_matches_compressed_tensors() -> None:
    """Agrees with compressed_tensors.generate_gparam on random tensors."""
    from compressed_tensors.quantization.utils.helpers import generate_gparam

    for scale in (1e-3, 1.0, 17.0):
        weight = torch.randn(16, 32) * scale
        mine = compute_global_scale(weight).item()
        theirs = float(generate_gparam(weight.min(), weight.max()).item())
        assert mine == pytest.approx(theirs, rel=1e-6)


def test_global_scale_survives_an_all_zero_tensor() -> None:
    """Degenerate input falls back to 1.0 rather than NaN/Inf."""
    assert float(compute_global_scale(torch.zeros(8, 8)).item()) == 1.0


def test_pack_layout_matches_oracle() -> None:
    """Even columns in the low nibble, odd columns in the high nibble."""
    # a valid 2-D fp4 tensor with an even column count
    row = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    )
    fp4 = row.repeat(4, 1)
    mine = pack_e2m1(_encode_values(fp4))
    theirs = pack_fp4_to_uint8(fp4.clone())
    assert mine.dtype == torch.uint8
    assert torch.equal(mine, theirs)


def _encode_values(values: torch.Tensor) -> torch.Tensor:
    from qwen38_quasar.quantization.e2m1 import encode

    return encode(values)


def test_pack_unpack_round_trip() -> None:
    """Codes survive pack/unpack, including signed zeros."""
    from qwen38_quasar.quantization.e2m1 import encode

    values = torch.tensor([0.0, -0.0, 6.0, -6.0, 0.5, -1.5]).repeat(8, 1)
    codes = encode(values)
    assert torch.equal(unpack_e2m1(pack_e2m1(codes), dtype=torch.float32), values)


def test_unpack_matches_oracle() -> None:
    """Our unpacking agrees with compressed_tensors' unpacking."""

    row = torch.tensor([0.0, -0.0, 6.0, -6.0, 0.5, -1.5, 3.0, -4.0])
    fp4 = row.repeat(2, 1)
    packed = pack_fp4_to_uint8(fp4.clone())
    mine = unpack_e2m1(packed, dtype=torch.float32)
    theirs = unpack_fp4_from_uint8(packed, *fp4.shape, dtype=torch.float32)
    assert torch.equal(mine, theirs)


def test_stored_shapes_match_compressed_tensors_expectations() -> None:
    """weight_packed (out, in//2), weight_scale (out, in//group_size) fp8, global scalar."""
    weight = torch.randn(64, 256)
    packed, scale, global_scale = quantize_tensor_group(weight, group_size=16)
    assert packed.shape == (64, 128)
    assert packed.dtype == torch.uint8
    assert scale.shape == (64, 16)
    assert scale.dtype == torch.float8_e4m3fn
    assert global_scale.shape == (1,)
    assert global_scale.dtype == torch.float32


def test_dequantize_is_the_inverse_of_storage() -> None:
    """Storage -> dequantize reproduces the reconstruction the trainer sees."""
    weight = torch.randn(32, 64)
    packed, scale, global_scale = quantize_tensor_group(weight, group_size=16)
    reconstructed = dequantize(packed, scale, global_scale)
    assert reconstructed.shape == weight.shape

    # Re-derive independently from the stored components.
    values = unpack_e2m1(packed, dtype=torch.float32)
    grouped = group_view(values, 16)
    expected = grouped * (scale.to(torch.float32).unsqueeze(-1) / global_scale.reshape(()))
    assert torch.allclose(reconstructed, expected.flatten(-2, -1))


def test_effective_scale_maps_group_max_to_e2m1_max() -> None:
    """A group at the tensor maximum uses the full E2M1 range and the top of FP8."""
    weight = torch.randn(8, 16) * 0.01
    weight[0, 0] = 5.0  # this group now holds the tensor maximum
    packed, scale, global_scale = quantize_tensor_group(weight, group_size=16)
    del packed
    effective = scale.to(torch.float32) / global_scale.reshape(())
    # group 0 of row 0 holds the max, so its stored FP8 scale should be ~448
    assert float(scale[0, 0]) == pytest.approx(FP8_E4M3_MAX, rel=0.05)
    assert float(effective[0, 0]) == pytest.approx(5.0 / E2M1_MAX, rel=0.05)


def test_group_view_rejects_indivisible() -> None:
    """A non-divisible last dimension is an error, not a silent truncation."""
    with pytest.raises(ValueError, match="not divisible"):
        group_view(torch.randn(4, 17), 16)


def test_fp8_scale_rounding_is_bounded() -> None:
    """Rounding a scale into FP8 keeps it within the representable range."""
    scale = torch.tensor([0.0, 1e-9, 1.0, 448.0, 1e4])
    rounded = round_to_fp8_e4m3(scale).to(torch.float32)
    assert rounded[2] == pytest.approx(1.0)
    assert rounded[3] == pytest.approx(448.0)
    assert rounded[4] <= FP8_E4M3_MAX


def test_fp8_saturation_statistics() -> None:
    """Saturation/underflow counters behave as documented."""
    stats = Fp8ScaleSaturation()
    stats.update(torch.tensor([1.0, 2.0, 1e9]))
    stats.update(torch.tensor([1e-12, 3.0]))
    result = stats.as_dict()
    assert result["total"] == 5
    assert result["saturated"] == 1
    assert result["underflowed"] == 1
    assert result["saturation_rate"] == pytest.approx(0.2)
