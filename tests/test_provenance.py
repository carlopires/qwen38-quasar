"""Provenance tests (handoff sections 12, 43).

The lock file is the contract that no formal experiment runs against a moving
revision, so it is tested like any other artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen38_quasar.provenance import (
    OPEN_PERFECTBLEND_REPO,
    QUASAR_PAPER,
    SourcePin,
    build_lock,
    read_lock,
    write_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "provenance.lock.json"

#: Revisions resolved on 2026-09-15 and recorded in drafts/execution-plan.md.
EXPECTED = {
    "efficientthink": "882e1af3b8f5844f3b3163b132e343c06e91908f",
    "quasar_reference": "cfd1460322b9d8367a4ab13564a2182d50852d60",
    "open_perfectblend": "af60f3c18201652a83a93f46fcfee1b646ba3df7",
}


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text())


@pytest.mark.skipif(not LOCK_PATH.exists(), reason="provenance lock not generated yet")
def test_lock_pins_every_required_source() -> None:
    """All sources required by handoff section 43 are present."""
    sources = _lock()["sources"]
    for name in (
        "efficientthink",
        "quasar_reference",
        "open_perfectblend",
        "ninfer",
        "quasar_paper",
    ):
        assert name in sources, f"{name} is not pinned"


@pytest.mark.skipif(not LOCK_PATH.exists(), reason="provenance lock not generated yet")
def test_revisions_are_immutable_shas() -> None:
    """Every model/dataset source is pinned to a 40-character commit SHA."""
    sources = _lock()["sources"]
    for name in ("efficientthink", "quasar_reference", "open_perfectblend", "ninfer"):
        revision = sources[name]["revision"]
        assert len(revision) == 40, f"{name} revision is not a full SHA: {revision}"
        int(revision, 16)  # hexadecimal
        assert not revision.startswith("main"), f"{name} is pinned to a moving ref"


@pytest.mark.skipif(not LOCK_PATH.exists(), reason="provenance lock not generated yet")
def test_revisions_match_the_recorded_plan() -> None:
    """Resolved revisions agree with what the execution plan recorded."""
    sources = _lock()["sources"]
    for name, expected in EXPECTED.items():
        assert sources[name]["revision"] == expected, f"{name} drifted from the plan"


@pytest.mark.skipif(not LOCK_PATH.exists(), reason="provenance lock not generated yet")
def test_quasar_reference_is_marked_as_an_oracle_only() -> None:
    """The reference checkpoint must never be used to initialize the student."""
    pin = _lock()["sources"]["quasar_reference"]
    assert pin["role"] == "schema oracle only"
    assert "never" in pin["note"].lower()


@pytest.mark.skipif(not LOCK_PATH.exists(), reason="provenance lock not generated yet")
def test_efficientthink_role_is_teacher_and_initialization() -> None:
    """Teacher and student initialization are the same BF16 checkpoint."""
    pin = _lock()["sources"]["efficientthink"]
    assert "teacher" in pin["role"]
    assert "student initialization" in pin["role"]
    assert "GGUF" in pin["role"]


def test_build_lock_uses_the_reference_repo_ids() -> None:
    """Repo ids are the canonical ones, not transposed."""
    lock = build_lock(
        efficientthink="a" * 40,
        quasar_reference="b" * 40,
        open_perfectblend="c" * 40,
        ninfer="d" * 40,
        generated="2026-09-15T00:00:00+00:00",
    )
    assert lock.sources["open_perfectblend"].repo == OPEN_PERFECTBLEND_REPO
    assert lock.sources["efficientthink"].repo.startswith("nerkyor/Qwen3.8-27B-EfficientThink")
    assert lock.sources["quasar_reference"].repo == "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
    assert lock.revision_of("ninfer") == "d" * 40


def test_optional_qwen_official_pin() -> None:
    """The vanilla Qwen pin is optional and only for equality checks."""
    without = build_lock(
        efficientthink="a" * 40,
        quasar_reference="b" * 40,
        open_perfectblend="c" * 40,
        ninfer="d" * 40,
    )
    assert "qwen_official" not in without.sources

    with_pin = build_lock(
        efficientthink="a" * 40,
        quasar_reference="b" * 40,
        open_perfectblend="c" * 40,
        ninfer="d" * 40,
        qwen_official="e" * 40,
    )
    assert with_pin.sources["qwen_official"].revision == "e" * 40


def test_lock_round_trips_through_disk(tmp_path: Path) -> None:
    """write_lock/read_lock preserve the pins."""
    lock = build_lock(
        efficientthink="a" * 40,
        quasar_reference="b" * 40,
        open_perfectblend="c" * 40,
        ninfer="d" * 40,
        generated="2026-09-15T00:00:00+00:00",
    )
    destination = tmp_path / "provenance.lock.json"
    write_lock(lock, destination)
    restored = read_lock(destination)
    assert restored.generated == lock.generated
    assert set(restored.sources) == set(lock.sources)
    assert restored.revision_of("efficientthink") == "a" * 40


def test_missing_source_raises() -> None:
    """Asking for an unpinned source is an error, not a default."""
    lock = build_lock(
        efficientthink="a" * 40,
        quasar_reference="b" * 40,
        open_perfectblend="c" * 40,
        ninfer="d" * 40,
    )
    with pytest.raises(KeyError):
        lock.revision_of("qwen_official")


def test_paper_pin_records_the_independent_implementation_choice() -> None:
    """The paper pin must state that no official code was available."""
    assert QUASAR_PAPER["arxiv"] == "2608.13966"
    assert "independently" in QUASAR_PAPER["note"]
    assert "No official source-code repository" in QUASAR_PAPER["note"]


def test_source_pin_omits_empty_fields() -> None:
    """The serialized pin stays clean."""
    pin = SourcePin(kind="model", repo="x/y", revision="a" * 40)
    assert pin.as_dict() == {"kind": "model", "repo": "x/y", "revision": "a" * 40}
