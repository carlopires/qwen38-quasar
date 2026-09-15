"""QUASAR search tests (handoff sections 15, 16, 19, 42).

The central requirement is that the deployable FP8-rounded group scale
participates in candidate scoring, and that the search never loses to the
ordinary NVFP4 baseline under equivalent scale rules.
"""

from __future__ import annotations

import torch

from qwen38_quasar.quantization.e2m1 import decode, encode
from qwen38_quasar.quantization.nvfp4 import (
    FP8_E4M3_MAX,
    compute_global_scale,
    group_view,
    round_to_fp8_e4m3,
)
from qwen38_quasar.quantization.quasar import (
    CANDIDATE_MAX,
    CANDIDATE_MIN,
    CANDIDATE_STEP,
    QuasarConfig,
    candidate_grid,
    saliency_from_optimizer,
    select_scales,
)

GROUP = 16


def _reference_group_error(
    w: torch.Tensor,
    h: torch.Tensor,
    factor: float,
    global_scale: torch.Tensor,
    refit: bool = True,
) -> float:
    """Independent per-group reference implementation of one candidate.

    Deliberately written from the spec rather than by calling into the module
    under test, so that it can detect a scoring mistake.
    """
    group_max = w.abs().max().clamp(min=1e-12)
    raw_scale = (group_max * factor / 6.0).clamp(min=1e-12)
    codes = encode(w / raw_scale)
    q = decode(codes)

    if refit:
        numerator = (h * q * w).sum()
        denominator = (h * q * q).sum()
        ideal = (
            (numerator / denominator).abs().clamp(min=1e-12)
            if float(denominator) > 1e-12
            else raw_scale
        )
    else:
        ideal = raw_scale

    stored = round_to_fp8_e4m3(ideal * global_scale.reshape(()))
    deployed = stored.to(torch.float32) / global_scale.reshape(())
    return float((h * (q * deployed - w) ** 2).sum())


def test_candidate_grid_matches_the_paper() -> None:
    """0.30 to 1.00 step 0.05 -> 15 candidates, inclusive at both ends."""
    grid = candidate_grid()
    assert grid.numel() == 15
    assert grid[0] == torch.tensor(CANDIDATE_MIN)
    assert grid[-1] == torch.tensor(CANDIDATE_MAX)
    steps = (grid[1:] - grid[:-1]).abs()
    assert torch.allclose(steps, torch.full_like(steps, CANDIDATE_STEP), atol=1e-6)


def test_selection_is_the_argmin_of_the_deployed_error() -> None:
    """The chosen candidate is the argmin of the FP8-rounded, deployed error.

    This is the direct test of handoff section 16: scoring must use the stored
    FP8 scale, not the ideal floating-point scale.
    """
    torch.manual_seed(11)
    w = torch.randn(1, GROUP) * 0.05
    w[0, 5] = 1.7  # an outlier, so clipping candidates genuinely differ
    h = torch.ones_like(w)

    result = select_scales(w, h)
    global_scale = compute_global_scale(w)

    errors = [
        _reference_group_error(w.flatten(), h.flatten(), float(f), global_scale)
        for f in candidate_grid().tolist()
    ]
    best = min(range(len(errors)), key=lambda i: errors[i])
    assert int(result.chosen.item()) == best


def test_scoring_uses_rounded_scale_not_the_ideal_scale() -> None:
    """Scoring the ideal scale would sometimes pick a different candidate.

    Demonstrates that the distinction is real, which is why it is tested.
    """
    torch.manual_seed(3)
    disagreed = 0
    trials = 40
    for _ in range(trials):
        w = torch.randn(1, GROUP) * 0.05
        w[0, int(torch.randint(0, GROUP, (1,)).item())] *= 25.0
        h = torch.ones_like(w)
        global_scale = compute_global_scale(w)

        deployed = [
            _reference_group_error(w.flatten(), h.flatten(), float(f), global_scale)
            for f in candidate_grid().tolist()
        ]
        ideal_only = [
            _reference_group_error(w.flatten(), h.flatten(), float(f), torch.ones(1))
            for f in candidate_grid().tolist()
        ]
        if min(range(len(deployed)), key=deployed.__getitem__) != min(
            range(len(ideal_only)), key=ideal_only.__getitem__
        ):
            disagreed += 1
    assert disagreed > 0, "expected FP8 rounding to change the winner at least sometimes"


def test_never_worse_than_the_ordinary_baseline() -> None:
    """Handoff section 42: QUASAR must not lose to the ordinary baseline."""
    for seed in range(12):
        torch.manual_seed(seed)
        w = torch.randn(32, 128) * 0.03
        # occasionally plant outliers so clipping becomes interesting
        if seed % 3 == 0:
            w[seed % 32, 8:16] *= 20.0
        h = torch.rand_like(w).square() + 1e-3

        result = select_scales(w, h)
        assert (result.error <= result.baseline_error + 1e-12).all(), (
            f"seed {seed}: QUASAR regression vs baseline"
        )


def test_baseline_is_recovered_exactly_without_refit() -> None:
    """With the grid pinned to alpha=1.0 and no refit, QUASAR == the baseline.

    Proves the search machinery itself introduces no bias, independently of the
    refit step.
    """
    torch.manual_seed(5)
    w = torch.randn(16, 64) * 0.05
    h = torch.rand_like(w) + 1e-3
    config = QuasarConfig(candidate_min=1.0, candidate_max=1.0, refit=False)
    result = select_scales(w, h, config)
    assert torch.allclose(result.error, result.baseline_error, rtol=1e-6, atol=1e-12)


def test_refit_never_hurts_relative_to_the_max_based_scale() -> None:
    """For identical codes, the weighted least-squares scale is optimal."""
    torch.manual_seed(7)
    w = torch.randn(16, 64) * 0.04
    h = torch.rand_like(w).square() + 1e-3
    with_refit = select_scales(w, h, QuasarConfig(refit=True))
    without = select_scales(w, h, QuasarConfig(refit=False))
    assert with_refit.mean_error <= without.mean_error * (1 + 1e-9)


def test_selection_is_per_group() -> None:
    """Different groups of the same weight may pick different factors."""
    torch.manual_seed(9)
    w = torch.randn(4, 64) * 0.02
    w[0, 0:16] *= 40.0  # only the first group of row 0 has outliers
    h = torch.ones_like(w)
    result = select_scales(w, h)
    chosen = result.chosen
    assert chosen.shape == (4, 4)
    # the outlier group should prefer clipping (a lower factor) than its neighbours
    assert int(chosen[0, 0]) < int(chosen[0, 1])
    assert chosen.unique().numel() > 1


def test_saliency_shapes_the_fit() -> None:
    """High-saliency elements dominate the least-squares scale.

    With saliency concentrated on a single element, the refit scale should
    be the scale that makes that element's reconstruction nearest.
    """
    torch.manual_seed(13)
    w = torch.randn(1, GROUP) * 0.05
    h = torch.ones_like(w)
    h[0, 7] = 1e6  # one dominant element

    result = select_scales(w, h, QuasarConfig(candidate_min=1.0, candidate_max=1.0))
    q = _group_values(result)
    scale = result.weight_scale.to(torch.float32)[0, 0] / result.weight_global_scale.reshape(())
    reconstructed = q * scale
    # The dominant element's reconstruction should be closer than with uniform
    # saliency. Codes are identical in both runs (they do not depend on
    # saliency at alpha=1.0), so this isolates the scale fit.
    uniform = select_scales(
        w, torch.ones_like(w), QuasarConfig(candidate_min=1.0, candidate_max=1.0)
    )
    uniform_scale = uniform.weight_scale.to(torch.float32)[
        0, 0
    ] / uniform.weight_global_scale.reshape(())
    uniform_reconstructed = q * uniform_scale
    assert (reconstructed[7] - w[0, 7]).abs() <= (uniform_reconstructed[7] - w[0, 7]).abs() + 1e-9


def _group_values(result) -> torch.Tensor:
    """Decoded E2M1 values of group 0 of row 0."""
    from qwen38_quasar.quantization.nvfp4 import unpack_e2m1

    return unpack_e2m1(result.weight_packed, dtype=torch.float32)[0, :GROUP]


def test_candidate_histogram_accounts_for_every_group() -> None:
    """Every group is assigned exactly one candidate."""
    torch.manual_seed(17)
    w = torch.randn(8, 128) * 0.03
    result = select_scales(w, torch.ones_like(w))
    histogram = result.candidate_histogram()
    assert sum(histogram.values()) == 8 * (128 // GROUP)
    assert all(0 <= k < 15 for k in histogram)


def test_saturation_statistics_are_reported() -> None:
    """Handoff section 23 requires FP8 scale saturation statistics."""
    torch.manual_seed(19)
    w = torch.randn(8, 64) * 0.02
    result = select_scales(w, torch.ones_like(w))
    assert result.saturation["total"] == 15 * 8 * (64 // GROUP)
    assert 0.0 <= result.saturation["saturation_rate"] <= 1.0


def test_determinism() -> None:
    """Identical inputs give bit-identical stored tensors."""
    torch.manual_seed(23)
    w = torch.randn(16, 64) * 0.03
    h = torch.rand_like(w)
    a = select_scales(w, h)
    b = select_scales(w, h)
    assert torch.equal(a.weight_packed, b.weight_packed)
    assert torch.equal(a.weight_scale, b.weight_scale)
    assert torch.equal(a.chosen, b.chosen)
    assert torch.equal(a.reconstructed, b.reconstructed)


def test_uniform_saliency_equals_no_saliency() -> None:
    """Passing None is equivalent to uniform saliency."""
    torch.manual_seed(29)
    w = torch.randn(8, 32) * 0.03
    explicit = select_scales(w, torch.ones_like(w))
    implicit = select_scales(w, None)
    assert torch.equal(explicit.weight_packed, implicit.weight_packed)
    assert torch.equal(explicit.chosen, implicit.chosen)


def test_global_scale_uses_the_whole_tensor() -> None:
    """Global scaling is tensor-level, not per group."""
    torch.manual_seed(31)
    w = torch.randn(8, 64) * 0.02
    result = select_scales(w, None)
    expected = FP8_E4M3_MAX * 6.0 / w.abs().max()
    assert torch.allclose(result.weight_global_scale, expected.reshape(1), rtol=1e-5)


def test_saliency_from_optimizer_uses_exp_avg_sq_directly() -> None:
    """Saliency is exp_avg_sq, not its square root (handoff section 15)."""
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.randn(4, 4)
    optimizer.step()

    state = optimizer.state[parameter]["exp_avg_sq"]
    expected = state.detach() + 1e-12
    actual = saliency_from_optimizer(optimizer, parameter)
    assert torch.allclose(actual, expected)
    # explicitly not the square root
    assert not torch.allclose(actual, state.sqrt())


def test_saliency_bootstrap_is_an_explicit_local_choice() -> None:
    """Unpopulated state follows the documented bootstrap policy."""
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    ones = saliency_from_optimizer(optimizer, parameter, bootstrap="ones")
    assert torch.allclose(ones, torch.full_like(ones, 1e-12))

    zeros = saliency_from_optimizer(optimizer, parameter, bootstrap="zeros")
    assert torch.count_nonzero(zeros) == 0

    try:
        saliency_from_optimizer(optimizer, parameter, bootstrap="raise")
    except RuntimeError as error:
        assert "exp_avg_sq" in str(error)
    else:  # pragma: no cover
        raise AssertionError("bootstrap='raise' should have raised")


def test_saliency_applies_to_the_scale_fit() -> None:
    """A constant saliency scale leaves the chosen candidate unchanged."""
    torch.manual_seed(37)
    w = torch.randn(8, 64) * 0.03
    base = select_scales(w, torch.ones_like(w))
    scaled = select_scales(w, torch.full_like(w, 7.5))
    assert torch.equal(base.weight_packed, scaled.weight_packed)


def test_saliency_shape_mismatch_is_rejected() -> None:
    """A mismatched saliency tensor is an error, not silent broadcast."""
    try:
        select_scales(torch.randn(4, 32), torch.randn(4, 16))
    except ValueError as error:
        assert "saliency shape" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched saliency should raise")


def test_group_view_used_for_scales() -> None:
    """Stored scales are one per group along the input dimension."""
    w = torch.randn(8, 96)
    result = select_scales(w, None)
    assert result.weight_scale.shape == (8, 96 // GROUP)
    assert group_view(w, GROUP).shape == (8, 96 // GROUP, GROUP)
