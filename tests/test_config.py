"""Configuration and parameter-manifest tests (handoff section 24).

Every parameter must be labelled ``public_reference`` or ``local_choice``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen38_quasar.config import (
    PARAMETER_PROVENANCE,
    Config,
    ConfigError,
    build_manifest,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL = """
[model]
source = "nerkyor/Qwen3.8-27B-EfficientThink"
subdir = "BF16"

[quantization]
group_size = 16

[training]
learning_rate = 1e-6
max_steps = 2446
adam_betas = [0.9, 0.95]

[data]
dataset = "mlabonne/open-perfectblend"
"""


@pytest.fixture
def config(tmp_path: Path) -> Config:
    path = tmp_path / "test.toml"
    path.write_text(MINIMAL)
    return load_config(path)


def test_load_config_reads_toml(config: Config) -> None:
    """Dotted lookup and section access work."""
    assert config.get("model.subdir") == "BF16"
    assert config.get("training.max_steps") == 2446
    assert config.section("quantization") == {"group_size": 16}


def test_missing_config_raises(tmp_path: Path) -> None:
    """A missing file is an error, not an empty config."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_malformed_toml_raises(tmp_path: Path) -> None:
    """Invalid TOML is reported with the path."""
    path = tmp_path / "bad.toml"
    path.write_text("[model\nsource = 1")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_require_reports_the_missing_key(config: Config) -> None:
    """Required settings fail loudly."""
    assert config.require("training.learning_rate") == 1e-6
    with pytest.raises(ConfigError, match="teacher.dtype"):
        config.require("teacher.dtype")


def test_missing_section_raises(config: Config) -> None:
    """A malformed section is an error."""
    with pytest.raises(ConfigError, match="teacher"):
        config.section("teacher")


def test_manifest_labels_public_reference_values(config: Config) -> None:
    """Published recipe values are marked public_reference."""
    manifest = build_manifest(config)
    by_key = {entry["key"]: entry for entry in manifest["parameters"]}
    for key in ("training.learning_rate", "training.max_steps", "training.objective"):
        if key in by_key:
            assert by_key[key]["provenance"] == "public_reference"
    assert by_key["quantization.group_size"]["provenance"] == "public_reference"
    assert by_key["data.dataset"]["provenance"] == "public_reference"


def test_manifest_labels_local_choices(config: Config) -> None:
    """Values no public source specifies are marked local_choice."""
    manifest = build_manifest(config)
    by_key = {entry["key"]: entry for entry in manifest["parameters"]}
    assert by_key["training.adam_betas"]["provenance"] == "local_choice"


def test_manifest_notes_explain_the_label(config: Config) -> None:
    """Each labelled value carries a reason."""
    manifest = build_manifest(config)
    for entry in manifest["parameters"]:
        assert entry["note"], f"{entry['key']} has no provenance note"
        assert entry["provenance"] in ("public_reference", "local_choice")


def test_manifest_reports_unlabelled_values(tmp_path: Path) -> None:
    """Unlabelled keys are surfaced, not silently dropped."""
    path = tmp_path / "extra.toml"
    path.write_text(MINIMAL + "\n[experimental]\nshiny_knob = 7\n")
    manifest = build_manifest(load_config(path))
    assert "experimental.shiny_knob" in manifest["unlabelled"]
    assert manifest["counts"]["unlabelled"] >= 1


def test_manifest_tolerates_nested_tables(tmp_path: Path) -> None:
    """Nested tables flatten correctly."""
    path = tmp_path / "nested.toml"
    path.write_text("[a.b.c]\nd = 1\n")
    manifest = build_manifest(load_config(path))
    assert "a.b.c.d" in manifest["unlabelled"]


def test_every_label_maps_to_a_known_provenance() -> None:
    """The label table itself is well formed."""
    for key, (provenance, note) in PARAMETER_PROVENANCE.items():
        assert provenance in ("public_reference", "local_choice"), key
        assert note, f"{key} needs a note"
        assert "." in key, f"{key} should be a dotted path"


def test_quasar_saliency_bootstrap_is_a_local_choice() -> None:
    """The first-step saliency policy is ours, and labelled as such."""
    provenance, note = PARAMETER_PROVENANCE["quasar.saliency_bootstrap"]
    assert provenance == "local_choice"
    assert "paper does not specify" in note


@pytest.mark.skipif(
    not (REPO_ROOT / "configs" / "efficientthink-quasar-w4a4.toml").exists(),
    reason="baseline config not written yet",
)
def test_baseline_config_has_no_unlabelled_parameters() -> None:
    """The shipped baseline config must be fully accounted for."""
    config = load_config(REPO_ROOT / "configs" / "efficientthink-quasar-w4a4.toml")
    manifest = build_manifest(config)
    assert manifest["unlabelled"] == []
    assert manifest["counts"]["public_reference"] > 0
