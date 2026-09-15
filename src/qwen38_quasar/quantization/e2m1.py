"""E2M1 (NVFP4) codebook primitives.

Independent implementation from the QUASAR paper and the NVFP4 format, validated
against ``compressed_tensors`` as an oracle (see ``tests/test_e2m1.py``).

E2M1 is a 4-bit float with 1 sign bit, 2 exponent bits and 1 mantissa bit, giving
8 magnitudes and 16 codes. Bit layout (MSB first): ``s e e m``.

Codes are stored as ``uint8`` in the low nibble, sign-magnitude:

    code 0b0000 -> +0.0      code 0b1000 -> -0.0
    code 0b0001 -> +0.5      code 0b1001 -> -0.5
    ...

Rounding rule, established by probing the oracle: nearest magnitude, with exact
midpoints resolved to the **even code index** (round-half-to-even on the index).
Any magnitude above 6.0 saturates to 6.0.
"""

from __future__ import annotations

import torch

__all__ = [
    "CODEBOOK_MAGNITUDES",
    "CODEBOOK",
    "MIDPOINTS",
    "NUM_CODES",
    "decode",
    "encode",
    "round_to_e2m1",
]

#: The 8 non-negative E2M1 magnitudes, indexed by the 3-bit unsigned field.
CODEBOOK_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: All 16 signed E2M1 values, indexed by code.
CODEBOOK: tuple[float, ...] = tuple(
    [m for m in CODEBOOK_MAGNITUDES] + [-m for m in CODEBOOK_MAGNITUDES]
)

#: Midpoints between adjacent magnitudes; the tie boundaries.
MIDPOINTS: tuple[float, ...] = tuple(
    (CODEBOOK_MAGNITUDES[i] + CODEBOOK_MAGNITUDES[i + 1]) / 2.0
    for i in range(len(CODEBOOK_MAGNITUDES) - 1)
)

NUM_CODES = 16
MAX_MAGNITUDE = CODEBOOK_MAGNITUDES[-1]


def _magnitude_index(x: torch.Tensor) -> torch.Tensor:
    """Return the E2M1 magnitude index for each element of ``x`` (unclipped input).

    ``x`` is interpreted as a magnitude. The result is in ``[0, 7]``.
    """
    mid = torch.tensor(MIDPOINTS, dtype=x.dtype, device=x.device)

    # `bucketize(..., right=True)` returns the count of midpoints <= x, i.e. the
    # index of the smaller candidate magnitude for a non-tie, and the *upper*
    # candidate for an exact tie.
    idx = torch.bucketize(x, mid, right=True)

    # Exact midpoint: choose the even index of the two tied candidates.
    # `idx - 1` is the lower candidate. Prefer whichever of {idx-1, idx} is even.
    lower = idx - 1
    on_mid = (lower >= 0) & (lower < mid.numel())
    tie = on_mid & (lower >= 0) & (x == mid[lower.clamp(0, mid.numel() - 1)])
    lower_is_even = (lower % 2) == 0
    idx = torch.where(tie & lower_is_even, lower, idx)

    return idx.clamp(0, len(CODEBOOK_MAGNITUDES) - 1)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the nearest E2M1 representable value.

    Midpoint ties round to the even code index. Magnitudes above 6.0 saturate.
    The sign of zero is preserved.

    :param x: float tensor of any shape
    :return: float tensor of the same shape and dtype, holding E2M1 values
    """
    dtype = x.dtype
    sign = torch.signbit(x)
    magnitude = x.abs()

    mag = torch.tensor(CODEBOOK_MAGNITUDES, dtype=dtype, device=x.device)
    index = _magnitude_index(magnitude)
    rounded = mag[index]

    # Preserve -0.0: signbit-restore rather than multiply by ±1.
    return torch.where(sign, -rounded, rounded)


def encode(x: torch.Tensor) -> torch.Tensor:
    """Encode float values as E2M1 codes (``uint8`` in the low nibble).

    :param x: float tensor
    :return: ``uint8`` tensor of codes with the same shape
    """
    sign = torch.signbit(x)
    index = _magnitude_index(x.abs())
    code = index.to(torch.uint8) | (sign.to(torch.uint8) << 3)
    return code


def decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes to float values.

    :param codes: integer tensor whose low nibble holds the code
    :param dtype: output dtype (defaults to ``float32``)
    :return: float tensor of the same shape
    """
    values = torch.tensor(CODEBOOK, dtype=torch.float32, device=codes.device)
    code = (codes.to(torch.long) & 0xF).reshape(-1)
    out = values[code].reshape(codes.shape)
    return out
