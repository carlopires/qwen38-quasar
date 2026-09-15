"""QAD distillation losses (handoff section 23).

The baseline objective is forward KL from the frozen teacher to the quantized
student, evaluated on teacher-response token positions with the prompt present as
context::

    L = KL(p_teacher || p_student)

Forward (not reverse) KL is used because the teacher's distribution is the
authority being preserved. Response positions are selected by masking.

Handoff section 25 forbids silently substituting a top-k or otherwise approximate
KL to fit memory; :func:`assert_exact_kl` exists so that any such fallback has to
be deliberate and visible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

__all__ = [
    "assert_exact_kl",
    "greedy_agreement",
    "logit_metrics",
    "response_mask",
    "soft_cross_entropy",
    "teacher_student_kl",
]

_LOGIT_CLAMP = 1e-12


def assert_exact_kl(
    vocab_size: int,
    max_vocab_chunk: int | None = None,
    full_vocab_normalization: bool = True,
) -> None:
    """Guard against an accidental approximate-KL fallback.

    Chunking over the vocabulary is *not* inherently approximate: each chunk is
    normalised against the full-vocabulary log-sum-exp, so the sum is identical to
    the unchunked result (and a test asserts that). What would be approximate is
    renormalising within a chunk, or truncating to a top-k vocabulary. This guard
    rejects the latter and requires chunked callers to declare full-vocabulary
    normalisation.

    :param vocab_size: the model vocabulary size
    :param max_vocab_chunk: chunk size actually used, or None for unchunked
    :param full_vocab_normalization: whether each chunk is normalised against the
        full vocabulary
    :raises ValueError: if the objective would no longer be exact
    """
    if max_vocab_chunk is None:
        return
    if max_vocab_chunk < 1:
        raise ValueError(f"max_vocab_chunk must be positive, got {max_vocab_chunk}")
    if not full_vocab_normalization:
        raise ValueError(
            "chunked KL without full-vocabulary normalisation would be approximate. "
            "If exact KL does not fit the available hardware, report that explicitly "
            "instead of degrading the objective (handoff section 25)."
        )
    if max_vocab_chunk > vocab_size:
        # Harmless, but signal the caller's intent is off.
        return


def response_mask(
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Build the loss mask over teacher-response positions.

    Prompt positions are context only and padding is excluded, so the objective
    covers response tokens exactly.

    :param labels: token ids, shape ``(batch, seq)``
    :param prompt_lengths: number of prompt tokens per row, shape ``(batch,)``
    :param pad_token_id: padding id to exclude, if any
    :return: bool mask, shape ``(batch, seq)``
    """
    batch, seq = labels.shape
    positions = torch.arange(seq, device=labels.device).unsqueeze(0).expand(batch, seq)
    mask = positions >= prompt_lengths.unsqueeze(1)
    if pad_token_id is not None:
        mask = mask & (labels != pad_token_id)
    return mask


def teacher_student_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    chunk_vocab: bool = False,
    max_vocab_chunk: int | None = None,
) -> torch.Tensor:
    """Forward KL from teacher to student over masked positions.

    ``KL(p_t || p_s) = sum_v p_t (log p_t - log p_s)``

    Computed at the same precision for both sides, with log-softmax applied to the
    full vocabulary (optionally in chunks that are each normalised against the
    full vocabulary, which keeps the result exact while reducing peak memory).

    :param teacher_logits: ``(batch, seq, vocab)``
    :param student_logits: ``(batch, seq, vocab)``
    :param mask: bool mask over response positions, ``(batch, seq)``
    :param temperature: softmax temperature
    :param chunk_vocab: accumulate log-softmax in vocabulary chunks
    :param max_vocab_chunk: chunk size; each chunk is normalised against the full
        vocabulary, so the result stays exact
    :return: scalar mean KL over masked positions
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher/student logit shapes differ: {tuple(teacher_logits.shape)} vs "
            f"{tuple(student_logits.shape)}"
        )

    vocab_size = teacher_logits.shape[-1]
    assert_exact_kl(vocab_size, max_vocab_chunk if chunk_vocab else None)

    if not bool(mask.any()):
        return student_logits.sum() * 0.0

    teacher = teacher_logits[mask].to(torch.float32)
    student = student_logits[mask].to(torch.float32)

    if temperature != 1.0:
        teacher = teacher / temperature
        student = student / temperature

    teacher_log_probs = functional.log_softmax(teacher, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    if not chunk_vocab:
        student_log_probs = functional.log_softmax(student, dim=-1)
        return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()

    # Exact chunked accumulation: subtract the full-vocabulary log-sum-exp, which
    # makes each chunk's contribution identical to the unchunked result.
    chunk = max_vocab_chunk or vocab_size
    lse_student = torch.logsumexp(student, dim=-1, keepdim=True)
    total = torch.zeros(student.shape[0], dtype=torch.float32, device=student.device)
    for start in range(0, vocab_size, chunk):
        stop = min(start + chunk, vocab_size)
        student_chunk_log_probs = student[:, start:stop] - lse_student
        total += (
            teacher_probs[:, start:stop]
            * (teacher_log_probs[:, start:stop] - student_chunk_log_probs)
        ).sum(dim=-1)
    return total.mean()


@torch.no_grad()
def greedy_agreement(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Fraction of masked positions where teacher and student argmax agree.

    Logged per handoff section 23 as a cheap quality signal not requiring the full
    KL computation.

    :param teacher_logits: ``(batch, seq, vocab)``
    :param student_logits: ``(batch, seq, vocab)``
    :param mask: bool mask over response positions
    :return: agreement in ``[0, 1]``
    """
    if not bool(mask.any()):
        return 0.0
    teacher_argmax = teacher_logits[mask].argmax(dim=-1)
    student_argmax = student_logits[mask].argmax(dim=-1)
    return float((teacher_argmax == student_argmax).float().mean())


@torch.no_grad()
def logit_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Diagnostics required by handoff sections 23 and 28.

    :return: maximum absolute logit error and top-1 agreement over masked positions
    """
    if not bool(mask.any()):
        return {"max_logit_error": 0.0, "top1_agreement": 0.0}
    teacher = teacher_logits[mask].to(torch.float32)
    student = student_logits[mask].to(torch.float32)
    return {
        "max_logit_error": float((teacher - student).abs().max()),
        "top1_agreement": float((teacher.argmax(dim=-1) == student.argmax(dim=-1)).float().mean()),
    }


def soft_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy against hard targets.

    Not part of the baseline objective (handoff section 23 explicitly excludes
    supervised cross-entropy on ground-truth labels unless a verified reference
    requires it). Provided for the PTQ control comparison in handoff section 40.

    :param logits: ``(batch, seq, vocab)``
    :param target_ids: ``(batch, seq)``
    :param mask: bool mask over positions to score
    :return: scalar mean cross-entropy
    """
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected = logits[mask].to(torch.float32)
    targets = target_ids[mask]
    return functional.cross_entropy(selected, targets)
