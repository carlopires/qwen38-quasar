"""Provenance pinning (handoff sections 12, 43).

Every source is resolved to an immutable revision before any formal experiment.
The lock file records what was actually used, not what was intended.

The revisions below were resolved on 2026-09-15 and are asserted to still match
when ``scripts/bootstrap_sources.sh`` runs; drift is reported rather than
silently absorbed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "EFFICIENTTHINK_REPO",
    "QUASAR_REFERENCE_REPO",
    "QWEN_OFFICIAL_REPO",
    "OPEN_PERFECTBLEND_REPO",
    "NINFER_REPO",
    "QUASAR_PAPER",
    "ProvenanceLock",
    "SourcePin",
    "build_lock",
    "git_sha",
    "read_lock",
    "write_lock",
]

EFFICIENTTHINK_REPO = (
    "nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2"
)
QUASAR_REFERENCE_REPO = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
QWEN_OFFICIAL_REPO = "Qwen/Qwen3.8-27B"
OPEN_PERFECTBLEND_REPO = "mlabonne/open-perfectblend"
NINFER_REPO = "carlopires/ninfer-rtx5090-mobile"

QUASAR_PAPER = {
    "title": (
        "QUASAR: Lowering the Loss Floor of Quantization-Aware Training "
        "with Loss-Aware Reconstruction"
    ),
    "arxiv": "2608.13966",
    "abs_url": "https://arxiv.org/abs/2608.13966",
    "html_url": "https://arxiv.org/html/2608.13966",
    "retrieved": "2026-09-15",
    "note": (
        "No official source-code repository was public at implementation time. "
        "The algorithm is implemented independently from the paper."
    ),
}


@dataclass
class SourcePin:
    """An immutable reference to a source artifact."""

    kind: Literal["model", "dataset", "repo", "paper"]
    repo: str
    revision: str
    role: str = ""
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {k: v for k, v in asdict(self).items() if v not in ("", {}, None)}


@dataclass
class ProvenanceLock:
    """The full set of pinned sources for a run or for the repository."""

    generated: str
    sources: dict[str, SourcePin]

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "generated": self.generated,
            "sources": {name: pin.as_dict() for name, pin in self.sources.items()},
        }

    def revision_of(self, name: str) -> str:
        """Return the pinned revision for a source.

        :param name: source key
        :raises KeyError: if the source is not pinned
        """
        return self.sources[name].revision


def git_sha(path: str | Path, remote: str | None = None) -> str:
    """Resolve a git revision for a local checkout.

    :param path: repository path
    :param remote: optional remote name whose ``HEAD`` should be used instead of
        the checked-out commit (useful when the working tree is on a branch)
    :return: the 40-character SHA
    """
    if remote:
        reference = f"refs/remotes/{remote}/master"
        try:
            out = subprocess.run(
                ["git", "-C", str(path), "rev-parse", reference],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except subprocess.CalledProcessError:
            pass

    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def build_lock(
    *,
    efficientthink: str,
    quasar_reference: str,
    open_perfectblend: str,
    ninfer: str,
    qwen_official: str | None = None,
    generated: str = "",
) -> ProvenanceLock:
    """Assemble the provenance lock from already-resolved revisions.

    :param efficientthink: HF commit for the EfficientThink BF16 checkpoint
    :param quasar_reference: HF commit for the QUASAR schema oracle
    :param open_perfectblend: HF commit for the QAD dataset
    :param ninfer: git SHA for NInfer
    :param qwen_official: optional HF commit for vanilla Qwen3.8
    :param generated: ISO timestamp
    :return: the populated lock
    """
    sources: dict[str, SourcePin] = {
        "efficientthink": SourcePin(
            kind="model",
            repo=EFFICIENTTHINK_REPO,
            revision=efficientthink,
            role="teacher + student initialization (BF16/, not GGUF)",
        ),
        "quasar_reference": SourcePin(
            kind="model",
            repo=QUASAR_REFERENCE_REPO,
            revision=quasar_reference,
            role="schema oracle only",
            note=(
                "Never used to initialize the student. Only config.json is read, to "
                "verify quantization_config, compressed-tensors metadata, tensor "
                "naming and ignored modules."
            ),
        ),
        "open_perfectblend": SourcePin(
            kind="dataset",
            repo=OPEN_PERFECTBLEND_REPO,
            revision=open_perfectblend,
            role="QAD baseline dataset (milestone 1 only)",
        ),
        "ninfer": SourcePin(
            kind="repo",
            repo=NINFER_REPO,
            revision=ninfer,
            role=".ninfer conversion, runtime, serving",
            note="Local checkout is ~/code/ninfer with the upstream remote named 'carlo'.",
        ),
        "quasar_paper": SourcePin(
            kind="paper",
            repo="arXiv:2608.13966",
            revision=QUASAR_PAPER["retrieved"],
            role="algorithmic authority",
            extra=QUASAR_PAPER,
        ),
    }
    if qwen_official:
        sources["qwen_official"] = SourcePin(
            kind="model",
            repo=QWEN_OFFICIAL_REPO,
            revision=qwen_official,
            role="optional equality check for non-quantized tensors",
        )

    return ProvenanceLock(generated=generated, sources=sources)


def write_lock(lock: ProvenanceLock, path: str | Path) -> None:
    """Write the lock file as JSON.

    :param lock: the lock to write
    :param path: destination path
    """
    Path(path).write_text(json.dumps(lock.as_dict(), indent=2, sort_keys=True) + "\n")


def read_lock(path: str | Path) -> ProvenanceLock:
    """Read a lock file back.

    :param path: source path
    :return: the parsed lock
    """
    raw = json.loads(Path(path).read_text())
    sources = {name: SourcePin(**pin) for name, pin in raw["sources"].items()}
    return ProvenanceLock(generated=raw["generated"], sources=sources)


def resolve_hf_revision(
    repo: str,
    kind: str = "model",
    attempts: int = 4,
    backoff: float = 1.5,
    timeout: float = 30.0,
) -> str:
    """Resolve the current commit SHA of a Hugging Face repository.

    Retries with exponential backoff: the Hub intermittently returns 404/429 for
    requests that succeed on a second attempt, and a transient blip must not
    abort provenance pinning.

    :param repo: repository id, e.g. ``owner/name``
    :param kind: ``"model"`` or ``"dataset"``
    :param attempts: total number of attempts
    :param backoff: multiplier applied between attempts
    :param timeout: per-request timeout in seconds
    :return: the commit SHA reported by the Hub
    :raises RuntimeError: if every attempt fails
    """
    import time
    import urllib.error
    import urllib.request

    prefix = "datasets/" if kind == "dataset" else "models/"
    url = f"https://huggingface.co/api/{prefix}{repo}"
    headers = {
        "User-Agent": "qwen38-quasar/0.0.1",
        "Accept": "application/json",
    }

    last_error: Exception | None = None
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            sha = payload.get("sha")
            if not sha:
                raise RuntimeError(f"no 'sha' field in response from {url}")
            return str(sha)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(delay)
                delay *= backoff

    raise RuntimeError(
        f"could not resolve revision for {repo} after {attempts} attempts: {last_error}"
    ) from last_error
