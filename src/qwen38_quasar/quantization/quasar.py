"""QUASAR loss-aware reconstruction.

Implemented independently from the QUASAR paper (arXiv 2608.13966), Algorithm 1
and the NVFP4 procedure. No official source repository was public at
implementation time; see ``drafts/execution-plan.md`` risk R8.

The objective is a saliency-weighted reconstruction error::

    S_hat = sum_i h_i (r_i - w_i)^2

where ``w`` are the latent full-precision weights, ``r`` the reconstructed
(quantized then dequantized) weights, and ``h`` the online saliency, taken from
AdamW's second moment ``exp_avg_sq``.

The procedure is applied per group, per weight, refreshed after each optimizer
update:

1. for each clipping candidate ``alpha`` in ``{0.30, 0.35, ..., 1.00}``, derive a
   group scale ``alpha * group_max / E2M1_MAX`` and assign E2M1 codes by rounding
   ``w / scale``;
2. refit the group scale by saliency-weighted least squares, holding the codes
   fixed::

       s* = sum_i h_i q_i w_i / sum_i h_i q_i^2

3. round ``s* * global_scale`` into FP8 E4M3;
4. reconstruct using the **rounded** scale
   (``weight_scale / weight_global_scale``);
5. score the saliency-weighted error of that reconstruction;
6. keep, per group independently, the candidate with the lowest error.

Step 4 is deliberate and required by handoff section 16: the deployable FP8 scale
must participate in candidate scoring, because the ideal floating-point scale is
not what the checkpoint stores.

Selection is **per group** -- different groups of the same weight may choose
different clipping factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen38_quasar.quantization.e2m1 import decode, encode
from qwen38_quasar.quantization.nvfp4 import (
    E2M1_MAX,
    Fp8ScaleSaturation,
    compute_global_scale,
    group_view,
    pack_e2m1,
    round_to_fp8_e4m3,
)

__all__ = [
    "CANDIDATE_MIN",
    "CANDIDATE_MAX",
    "CANDIDATE_STEP",
    "QuasarConfig",
    "QuasarResult",
    "candidate_grid",
    "reconstruct_grouped",
    "saliency_from_optimizer",
    "select_scales",
]

CANDIDATE_MIN = 0.30
CANDIDATE_MAX = 1.00
CANDIDATE_STEP = 0.05

_EPS = 1e-12


def candidate_grid(
    candidate_min: float = CANDIDATE_MIN,
    candidate_max: float = CANDIDATE_MAX,
    candidate_step: float = CANDIDATE_STEP,
) -> torch.Tensor:
    """Build the clipping-candidate grid.

    Defaults reproduce the paper's reported candidates: 0.30 to 1.00 step 0.05,
    i.e. 15 candidates.

    :return: 1-D fp32 tensor of candidate factors
    """
    n = round((candidate_max - candidate_min) / candidate_step) + 1
    return torch.linspace(candidate_min, candidate_max, n, dtype=torch.float32)


@dataclass(frozen=True)
class QuasarConfig:
    """Configuration for QUASAR reconstruction."""

    group_size: int = 16
    candidate_min: float = CANDIDATE_MIN
    candidate_max: float = CANDIDATE_MAX
    candidate_step: float = CANDIDATE_STEP
    saliency: str = "adamw_exp_avg_sq"
    #: Local implementation choice, not specified by the paper: how to bootstrap
    #: saliency before AdamW has populated ``exp_avg_sq``.
    saliency_bootstrap: str = "ones"
    #: Whether to refit the group scale by weighted least squares. Set False to
    #: reproduce the plain max-based scale, which makes the alpha=1.0 candidate
    #: exactly the ordinary NVFP4 baseline.
    refit: bool = True

    def candidates(self) -> torch.Tensor:
        """Return the candidate grid for this configuration."""
        return candidate_grid(self.candidate_min, self.candidate_max, self.candidate_step)


@dataclass
class QuasarResult:
    """Result of a QUASAR reconstruction pass over one weight tensor."""

    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    reconstructed: torch.Tensor
    #: Index into the candidate grid chosen for each group, ``(out, n_groups)``.
    chosen: torch.Tensor
    #: Per-group saliency-weighted reconstruction error, ``(out, n_groups)``.
    error: torch.Tensor
    #: Per-group error of the ordinary max-based baseline, same rounding rules.
    baseline_error: torch.Tensor
    saturation: dict[str, float | int]

    @property
    def mean_error(self) -> float:
        """Mean per-group saliency-weighted reconstruction error."""
        return float(self.error.mean())

    @property
    def mean_baseline_error(self) -> float:
        """Mean per-group error of the ordinary baseline."""
        return float(self.baseline_error.mean())

    def candidate_histogram(self) -> dict[int, int]:
        """Candidate-factor histogram, required by handoff section 23."""
        flat = self.chosen.flatten()
        counts = torch.bincount(flat, minlength=1)
        return {int(i): int(c) for i, c in enumerate(counts.tolist()) if c}


def saliency_from_optimizer(
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
    bootstrap: str = "ones",
    eps: float = _EPS,
) -> torch.Tensor:
    """Extract QUASAR saliency from an AdamW optimizer state.

    QUASAR uses the optimizer's EMA of squared gradients, i.e. AdamW's
    ``exp_avg_sq``, used directly -- **not** its square root (handoff section 15).

    The first optimizer step has no populated state. The paper does not specify
    the startup behaviour, so it is an explicit local implementation choice
    exposed via ``bootstrap``:

    ``"ones"``   unweighted reconstruction until state exists (default)
    ``"zeros"``  return zero saliency for the parameter
    ``"raise"``  refuse to run before state exists

    :param optimizer: the training optimizer
    :param param: the latent weight parameter
    :param bootstrap: behaviour when ``exp_avg_sq`` is missing or all-zero
    :param eps: constant added to the saliency
    :return: saliency tensor shaped like the parameter
    """
    state = optimizer.state.get(param, {})
    exp_avg_sq = state.get("exp_avg_sq")

    missing = not torch.is_tensor(exp_avg_sq)
    if not missing and float(exp_avg_sq.abs().sum()) == 0.0:
        missing = True

    if missing:
        if bootstrap == "raise":
            raise RuntimeError(
                "AdamW exp_avg_sq is unavailable; QUASAR saliency cannot be bootstrapped"
            )
        if bootstrap == "zeros":
            return torch.zeros_like(param.detach())
        return torch.full_like(param.detach(), eps)

    return exp_avg_sq.detach().to(param.dtype) + eps


def _weighted_scale(
    w: torch.Tensor,
    q: torch.Tensor,
    h: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Saliency-weighted least-squares group scale with codes held fixed.

    For a symmetric quantizer with codes ``q``::

        s* = sum_i h_i q_i w_i / sum_i h_i q_i^2

    Groups whose codes are all zero cannot inform a scale; those fall back to the
    max-based scale rather than being pinned to zero.

    :param w: grouped latent weights ``(..., n_groups, group_size)``
    :param q: grouped E2M1 code values, same shape
    :param h: grouped saliency, same shape
    :return: grouped ideal scale ``(..., n_groups, 1)``
    """
    num = (h * q * w).sum(dim=-1, keepdim=True)
    den = (h * q * q).sum(dim=-1, keepdim=True)
    fallback = w.abs().amax(dim=-1, keepdim=True) / E2M1_MAX
    scale = torch.where(den > eps, num / den.clamp(min=eps), fallback)
    return scale.abs().clamp(min=_EPS)


def reconstruct_grouped(
    grouped_values: torch.Tensor,
    grouped_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reconstruct grouped weights from code values and stored scales.

    :param grouped_values: ``(..., n_groups, group_size)`` E2M1 code values
    :param grouped_scale: ``(..., n_groups, 1)`` stored (FP8-rounded) scales
    :param group_size: elements per group, validated for clarity
    :return: grouped reconstruction, same shape as ``grouped_values``
    """
    if grouped_values.shape[-1] != group_size:
        raise ValueError(
            f"grouped last dimension {grouped_values.shape[-1]} != group_size {group_size}"
        )
    return grouped_values * grouped_scale.expand_as(grouped_values)


def select_scales(
    weight: torch.Tensor,
    saliency: torch.Tensor | None = None,
    config: QuasarConfig | None = None,
) -> QuasarResult:
    """Run the QUASAR candidate search over a weight tensor.

    Vectorized over groups and candidates: one Python-level loop over the (15)
    candidate factors, with all per-group work vectorized. Vectorizing further
    would not help; looping per group would be catastrophic.

    :param weight: latent full-precision weight, shape ``(out, in)``
    :param saliency: per-element saliency ``h``, same shape, or None for uniform
    :param config: QUASAR configuration
    :return: a :class:`QuasarResult` holding the stored tensors and diagnostics
    """
    config = config or QuasarConfig()
    group_size = config.group_size
    dtype = torch.float32

    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")

    w_full = weight.detach().to(dtype)
    if saliency is None:
        h_full = torch.ones_like(w_full)
    else:
        if saliency.shape != weight.shape:
            raise ValueError(
                f"saliency shape {tuple(saliency.shape)} does not match "
                f"weight shape {tuple(weight.shape)}"
            )
        h_full = saliency.detach().to(dtype)

    w = group_view(w_full, group_size)
    h = group_view(h_full, group_size)

    out_features, in_features = weight.shape
    n_groups = in_features // group_size
    device = w.device

    global_scale = compute_global_scale(weight)
    gs_scalar = global_scale.reshape(())
    saturation = Fp8ScaleSaturation()

    group_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=_EPS)

    def score(codes: torch.Tensor, ideal_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deploy-round the scale, reconstruct with it, and score per group.

        ``codes`` are E2M1 *codes* and must be decoded to values before use:
        casting the raw code integers to float would score the wrong quantizer.
        """
        values = decode(codes)
        stored = round_to_fp8_e4m3(ideal_scale * gs_scalar)
        deployed = stored.to(dtype) / gs_scalar
        recon = reconstruct_grouped(values, deployed, group_size)
        err = (h * (recon - w) ** 2).sum(dim=-1)
        return err, deployed

    # Ordinary NVFP4 baseline, evaluated under identical FP8 rounding rules so
    # the comparison in tests is apples-to-apples.
    base_scale = group_max / E2M1_MAX
    base_codes = encode(w / base_scale)
    baseline_error, _ = score(base_codes, base_scale)

    candidates = config.candidates().tolist()

    def evaluate(factor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run steps 1-5 for one clipping candidate: codes, ideal scale, error."""
        alpha = torch.full_like(group_max, float(factor))
        raw_scale = (alpha * group_max / E2M1_MAX).clamp(min=_EPS)

        # 1. code assignment induced by this clipping candidate
        codes = encode(w / raw_scale)

        # 2. saliency-weighted least-squares refit of the group scale, using the
        #    E2M1 *values* of the assigned codes
        ideal = _weighted_scale(w, decode(codes), h) if config.refit else raw_scale
        saturation.update(ideal * gs_scalar)

        # 3-5. FP8-round, reconstruct with the deployed scale, score
        err, _ = score(codes, ideal)
        return codes, ideal, err

    # The first candidate seeds the running best, avoiding optional state.
    best_codes, best_scale, best_error = evaluate(candidates[0])
    best_index = torch.zeros((out_features, n_groups), dtype=torch.long, device=device)

    for index, factor in enumerate(candidates[1:], start=1):
        codes, ideal, err = evaluate(factor)
        improved = err < best_error
        broadcast = improved.unsqueeze(-1)
        best_error = torch.where(improved, err, best_error)
        best_codes = torch.where(broadcast, codes, best_codes)
        best_scale = torch.where(broadcast, ideal, best_scale)
        best_index = torch.where(improved, torch.full_like(best_index, index), best_index)

    # Final stored tensors, built from the deployed FP8 scale.
    stored_scale = round_to_fp8_e4m3(best_scale * gs_scalar).squeeze(-1)
    deployed = stored_scale.to(dtype).unsqueeze(-1) / gs_scalar
    final_recon = reconstruct_grouped(decode(best_codes), deployed, group_size)

    packed = pack_e2m1(best_codes.reshape(out_features, in_features))

    return QuasarResult(
        weight_packed=packed,
        weight_scale=stored_scale,
        weight_global_scale=global_scale,
        reconstructed=final_recon.reshape(out_features, in_features),
        chosen=best_index,
        error=best_error,
        baseline_error=baseline_error,
        saturation=saturation.as_dict(),
    )
