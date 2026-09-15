"""E2M1 codebook tests (handoff section 14).

The oracle is ``compressed_tensors.quantization.utils.fp4_utils.cast_to_fp4``.
That function dispatches to a Triton kernel on CUDA and to a torch fallback on
CPU; the two agree on all tested points. We test against the torch path, which
operates in float32 and is therefore the stricter comparison for our bf16/fp32
inputs.
"""

from __future__ import annotations

import pytest
import torch

from qwen38_quasar.quantization.e2m1 import (
    CODEBOOK,
    CODEBOOK_MAGNITUDES,
    MIDPOINTS,
    NUM_CODES,
    decode,
    encode,
    round_to_e2m1,
)

oracle_fp4 = pytest.importorskip(
    "compressed_tensors.quantization.utils.fp4_utils",
    reason="compressed-tensors is required as the E2M1 oracle",
).cast_to_fp4


def test_codebook_is_the_e2m1_set() -> None:
    """All 16 codes, as the documented E2M1 value set."""
    assert len(CODEBOOK) == NUM_CODES
    assert CODEBOOK_MAGNITUDES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for magnitude in CODEBOOK_MAGNITUDES:
        assert magnitude in CODEBOOK
        assert -magnitude in CODEBOOK


def test_every_code_round_trips() -> None:
    """Encode/decode round trip covers all 16 codes and is lossless."""
    codes = torch.arange(NUM_CODES, dtype=torch.uint8)
    values = decode(codes)
    assert torch.equal(encode(values), codes)
    assert torch.equal(decode(encode(values)), values)


def test_zero_is_exact_and_signed() -> None:
    """Both zero codes survive, and -0.0 keeps its sign."""
    positive = round_to_e2m1(torch.tensor([0.0]))
    negative = round_to_e2m1(torch.tensor([-0.0]))
    assert positive.item() == 0.0
    assert negative.item() == 0.0
    assert not torch.signbit(positive).item()
    assert torch.signbit(negative).item()
    assert int(encode(torch.tensor([0.0])).item()) == 0b0000
    assert int(encode(torch.tensor([-0.0])).item()) == 0b1000


def test_sign_is_preserved() -> None:
    """Negation of the input negates the output."""
    x = torch.linspace(-6.5, 6.5, 4097)
    assert torch.equal(round_to_e2m1(-x), -round_to_e2m1(x))


def test_saturation() -> None:
    """Magnitudes beyond 6.0 saturate to +-6.0."""
    x = torch.tensor([6.0, 6.0001, 7.0, 100.0, -6.0001, -7.0, -1e6])
    expected = torch.tensor([6.0, 6.0, 6.0, 6.0, -6.0, -6.0, -6.0])
    assert torch.equal(round_to_e2m1(x), expected)


@pytest.mark.parametrize(
    ("midpoint", "expected_index"),
    [
        # Round-half-to-even on the *code index*, verified against the oracle:
        # 0.25 -> 0 (idx 0), 0.75 -> 1.0 (idx 2), 1.25 -> 1.0 (idx 2),
        # 1.75 -> 2.0 (idx 4), 2.5 -> 2.0 (idx 4), 3.5 -> 4.0 (idx 6),
        # 5.0 -> 4.0 (idx 6).
        (0.25, 0),
        (0.75, 2),
        (1.25, 2),
        (1.75, 4),
        (2.5, 4),
        (3.5, 6),
        (5.0, 6),
    ],
)
def test_midpoint_ties_round_to_even_index(midpoint: float, expected_index: int) -> None:
    """Exact midpoints resolve to the even code index."""
    value = round_to_e2m1(torch.tensor([midpoint])).item()
    assert value == CODEBOOK_MAGNITUDES[expected_index]


def test_midpoints_match_the_oracle_exactly() -> None:
    """Every tie boundary, and both sides of it, match the oracle."""
    base = torch.tensor(MIDPOINTS, dtype=torch.float64)
    offsets = torch.tensor([0.0, 1e-6, -1e-6], dtype=torch.float64)
    values = (base[:, None] + offsets[None, :]).reshape(-1)
    signed = torch.cat([values, -values]).float()
    assert torch.equal(round_to_e2m1(signed), oracle_fp4(signed.clone()))


def test_matches_oracle_on_a_dense_grid() -> None:
    """Dense sweep of the whole representable range, both signs."""
    grid = torch.cat(
        [
            torch.linspace(-7.0, 7.0, 400001),
            torch.linspace(-0.01, 0.01, 20001),
            torch.linspace(5.0, 6.0, 10001),
            torch.tensor(list(CODEBOOK) + [7.0, 100.0, -100.0]),
        ]
    )
    mine = round_to_e2m1(grid)
    theirs = oracle_fp4(grid.clone().float())
    mismatches = int((mine != theirs).sum())
    assert mismatches == 0, f"{mismatches} mismatches vs oracle"


def test_every_output_is_a_codebook_value() -> None:
    """Rounding never produces a non-E2M1 value."""
    grid = torch.linspace(-8.0, 8.0, 100001)
    allowed = torch.tensor(CODEBOOK)
    assert torch.isin(round_to_e2m1(grid), allowed).all()


def test_bfloat16_input_round_trips() -> None:
    """bf16 weights, the actual training dtype, encode/decode correctly."""
    grid = torch.linspace(-6.5, 6.5, 4097, dtype=torch.bfloat16)
    codes = encode(grid)
    values = decode(codes).to(torch.bfloat16)
    # decoding is exact for E2M1 values, all of which are bf16-representable
    assert torch.equal(values, round_to_e2m1(grid).to(torch.bfloat16))
    assert torch.equal(encode(values), codes)


def test_multidimensional_shapes() -> None:
    """Shapes are preserved, including a 3-D grouped view."""
    x = torch.randn(4, 8, 16)
    codes = encode(x)
    assert codes.shape == x.shape
    assert codes.dtype == torch.uint8
    assert decode(codes).shape == x.shape
