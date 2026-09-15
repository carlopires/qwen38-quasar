"""Distillation-loss tests (handoff sections 23, 42).

The KL must be exact forward KL over response tokens, and the response mask must
exclude both prompt and padding positions.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from qwen38_quasar.distill.loss import (
    assert_exact_kl,
    greedy_agreement,
    logit_metrics,
    response_mask,
    soft_cross_entropy,
    teacher_student_kl,
)


def _logits(batch: int = 2, seq: int = 6, vocab: int = 32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(batch, seq, vocab, generator=generator)
    student = torch.randn(batch, seq, vocab, generator=generator)
    return teacher, student


def test_response_mask_covers_only_response_tokens() -> None:
    """Prompt positions are context, not loss targets."""
    labels = torch.zeros(3, 10, dtype=torch.long)
    lengths = torch.tensor([0, 4, 7])
    mask = response_mask(labels, lengths)
    assert mask.sum(dim=1).tolist() == [10, 6, 3]
    assert not mask[1, :4].any()
    assert mask[1, 4:].all()


def test_response_mask_excludes_padding() -> None:
    """Padding is masked out even after the prompt.

    Row 0: prompt is [1, 2, 3], so the response is [0, 0, 0] -- all padding, hence
    0 surviving tokens. Row 1: prompt is [4, 5, 6, 7], response is [8, 0], of which
    only the 8 survives. Total: exactly one token.
    """
    labels = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 8, 0]])
    lengths = torch.tensor([3, 4])
    mask = response_mask(labels, lengths, pad_token_id=0)
    assert mask.sum().item() == 1
    assert not mask[0].any()
    assert mask[1, 4]
    assert not mask[1, 5]
    assert not mask[1, 3]  # position 3 is still prompt


def test_forward_kl_matches_the_textbook_formula() -> None:
    """KL(p_t || p_s) computed independently."""
    teacher, student = _logits()
    mask = response_mask(torch.zeros(2, 6, dtype=torch.long), torch.tensor([2, 3]))

    log_p_teacher = functional.log_softmax(teacher[mask].float(), dim=-1)
    log_p_student = functional.log_softmax(student[mask].float(), dim=-1)
    expected = (log_p_teacher.exp() * (log_p_teacher - log_p_student)).sum(dim=-1).mean()

    actual = teacher_student_kl(teacher, student, mask)
    assert actual == pytest.approx(float(expected), rel=1e-6)


def test_kl_is_asymmetric_so_direction_matters() -> None:
    """Forward and reverse KL differ; we use forward."""
    teacher, student = _logits(seed=7)
    mask = torch.ones(2, 6, dtype=torch.bool)
    forward = teacher_student_kl(teacher, student, mask)
    reverse = teacher_student_kl(student, teacher, mask)
    assert forward != reverse


def test_kl_is_zero_for_identical_distributions() -> None:
    """A student equal to the teacher has zero loss."""
    teacher, _ = _logits()
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert float(teacher_student_kl(teacher, teacher, mask)) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_non_negative() -> None:
    """Forward KL cannot be negative."""
    for seed in range(5):
        teacher, student = _logits(seed=seed)
        mask = torch.ones(2, 6, dtype=torch.bool)
        assert float(teacher_student_kl(teacher, student, mask)) >= -1e-6


def test_vocabulary_chunking_is_exact() -> None:
    """Chunked accumulation reproduces the unchunked result exactly.

    Each chunk is normalised against the full-vocabulary log-sum-exp, so this is
    an implementation detail, not an approximation.
    """
    teacher, student = _logits(vocab=128)
    mask = torch.ones(2, 6, dtype=torch.bool)
    reference = teacher_student_kl(teacher, student, mask)
    for chunk in (1, 7, 16, 64, 128):
        chunked = teacher_student_kl(
            teacher, student, mask, chunk_vocab=True, max_vocab_chunk=chunk
        )
        assert chunked == pytest.approx(float(reference), rel=1e-5), f"chunk={chunk}"


def test_temperature_scales_both_sides() -> None:
    """Temperature is applied to teacher and student alike."""
    teacher, student = _logits()
    mask = torch.ones(2, 6, dtype=torch.bool)
    scaled = teacher_student_kl(teacher, student, mask, temperature=2.0)
    expected = teacher_student_kl(teacher / 2, student / 2, mask)
    assert float(scaled) == pytest.approx(float(expected), rel=1e-6)


def test_empty_mask_returns_a_zero_gradient_tensor() -> None:
    """An all-prompt batch contributes zero loss but stays differentiable."""
    teacher, student = _logits()
    student.requires_grad_(True)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    loss = teacher_student_kl(teacher, student, mask)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert student.grad is not None


def test_shape_mismatch_is_rejected() -> None:
    """Teacher and student logits must align."""
    teacher, student = _logits()
    with pytest.raises(ValueError, match="shapes differ"):
        teacher_student_kl(teacher, student[:, :5], torch.ones(2, 6, dtype=torch.bool))


def test_gradients_flow_to_the_student_only() -> None:
    """Only the student receives gradient.

    The teacher is held under ``no_grad`` by the trainer (handoff section 23:
    ``eval()``, ``requires_grad = False``), which is the contract that matters.
    The KL is not mathematically constant in the teacher logits, so the freezing
    is the caller's responsibility and is asserted here explicitly.
    """
    teacher, student = _logits()
    student.requires_grad_(True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        teacher_logits = teacher.clone()
    teacher_student_kl(teacher_logits, student, mask).backward()
    assert student.grad is not None
    assert teacher_logits.grad is None


def test_frozen_teacher_receives_no_gradient() -> None:
    """A teacher with requires_grad=False never accumulates gradient."""
    teacher, student = _logits()
    teacher.requires_grad_(False)
    student.requires_grad_(True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    teacher_student_kl(teacher, student, mask).backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_approximate_chunking_is_refused() -> None:
    """Renormalising within a chunk would be approximate, so it errors."""
    with pytest.raises(ValueError, match="approximate"):
        assert_exact_kl(1000, 128, full_vocab_normalization=False)


def test_chunk_guard_allows_exact_chunking() -> None:
    """Exact chunking is permitted and needs no declaration."""
    assert_exact_kl(1000, None)
    assert_exact_kl(1000, 128)
    assert_exact_kl(1000, 128, full_vocab_normalization=True)
    with pytest.raises(ValueError, match="positive"):
        assert_exact_kl(1000, 0)


def test_greedy_agreement() -> None:
    """Agreement is 1.0 for identical logits and drops for perturbed ones."""
    teacher, _ = _logits()
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert greedy_agreement(teacher, teacher, mask) == 1.0
    perturbed = teacher.clone()
    perturbed[..., 0] += 1e3
    assert greedy_agreement(teacher, perturbed, mask) == 0.0


def test_logit_metrics_report_max_error_and_agreement() -> None:
    """The diagnostics required by the fidelity gate."""
    teacher, _ = _logits()
    mask = torch.ones(2, 6, dtype=torch.bool)
    metrics = logit_metrics(teacher, teacher, mask)
    assert metrics["max_logit_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["top1_agreement"] == 1.0

    shifted = teacher + 0.5
    assert logit_metrics(teacher, shifted, mask)["max_logit_error"] == pytest.approx(0.5, abs=1e-5)


def test_soft_cross_entropy_masks_positions() -> None:
    """Cross-entropy matches the reference over the same positions."""
    logits = torch.randn(2, 5, 16)
    targets = torch.randint(0, 16, (2, 5))
    mask = torch.tensor([[False, True, True, False, False]] * 2)
    actual = soft_cross_entropy(logits, targets, mask)
    expected = functional.cross_entropy(logits[mask], targets[mask])
    assert float(actual) == pytest.approx(float(expected), rel=1e-6)
