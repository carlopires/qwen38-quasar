"""Export-config tests (handoff sections 27, 42).

``tests/test_export_config.py`` must "assert exact compressed-tensors
config/schema compatibility". The oracle is the ``quantization_config`` captured
from ``QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4``, stored as a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qwen38_quasar.export.compressed_tensors import (
    QUASAR_IGNORE_PATTERNS,
    ExportManifest,
    QuantizedTensorSet,
    build_quantization_config,
    compare_with_reference,
    module_quantized_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "tests" / "data" / "quasar-reference-config.json"

pytestmark = pytest.mark.skipif(
    not REFERENCE_PATH.exists(),
    reason="pinned QUASAR reference config fixture is absent",
)


@pytest.fixture(scope="module")
def reference_config() -> dict:
    return json.loads(REFERENCE_PATH.read_text())["quantization_config"]


def test_schema_is_identical_to_the_reference_checkpoint(reference_config: dict) -> None:
    """Our quantization_config matches the oracle exactly, ignoring producer."""
    ours = build_quantization_config()
    assert compare_with_reference(ours, reference_config) == []


def test_producer_is_ours_not_the_reference(reference_config: dict) -> None:
    """We do not misrepresent the checkpoint as the reference producer's work."""
    ours = build_quantization_config()
    assert ours["producer"] != reference_config["producer"]
    assert ours["producer"]["name"] == "qwen38-quasar"


def test_top_level_fields(reference_config: dict) -> None:
    """The identifying fields are exactly as required by handoff section 27."""
    ours = build_quantization_config()
    assert ours["quant_method"] == "compressed-tensors"
    assert ours["quantization_status"] == "compressed"
    assert ours["format"] == "nvfp4-pack-quantized"
    assert ours["global_compression_ratio"] is None
    assert ours["kv_cache_scheme"] is None
    assert ours["sparsity_config"] == {}
    assert ours["transform_config"] == {}
    assert set(ours) == set(reference_config)


def test_weight_quantization_is_static_and_activation_is_dynamic() -> None:
    """The asymmetry between the two dynamic flags is intentional."""
    group = build_quantization_config()["config_groups"]["group_0"]
    assert group["weights"]["dynamic"] is False
    assert group["input_activations"]["dynamic"] == "local"


@pytest.mark.parametrize("which", ["weights", "input_activations"])
def test_nvfp4_parameters(which: str) -> None:
    """Both weight and activation specs match the W4A4 contract."""
    spec = build_quantization_config()["config_groups"]["group_0"][which]
    assert spec["num_bits"] == 4
    assert spec["type"] == "float"
    assert spec["strategy"] == "tensor_group"
    assert spec["group_size"] == 16
    assert spec["symmetric"] is True
    assert spec["scale_dtype"] == "torch.float8_e4m3fn"
    assert spec["actorder"] is None
    assert spec["block_structure"] is None
    assert spec["zp_dtype"] is None
    assert spec["observer_kwargs"] == {}


def test_targets_are_linear_modules() -> None:
    """Targets are ``["Linear"]``, which excludes the embedding implicitly."""
    group = build_quantization_config()["config_groups"]["group_0"]
    assert group["targets"] == ["Linear"]
    assert group["output_activations"] is None
    assert group["format"] == "nvfp4-pack-quantized"


def test_ignore_patterns_are_verbatim(reference_config: dict) -> None:
    """The ignore list is copied exactly, including lm_head."""
    ours = build_quantization_config()
    assert ours["ignore"] == reference_config["ignore"]
    assert ours["ignore"] == QUASAR_IGNORE_PATTERNS
    assert "lm_head" in ours["ignore"]
    assert "embed_tokens" not in " ".join(ours["ignore"])


def test_comparison_detects_differences(reference_config: dict) -> None:
    """The comparison used in the schema test can actually fail."""
    tampered = build_quantization_config()
    tampered["config_groups"]["group_0"]["weights"]["group_size"] = 32
    differences = compare_with_reference(tampered, reference_config)
    assert any("group_size" in d for d in differences)

    tampered = build_quantization_config()
    tampered["ignore"] = ["lm_head"]
    assert any("ignore" in d for d in compare_with_reference(tampered, reference_config))

    tampered = build_quantization_config()
    del tampered["format"]
    assert any("format" in d for d in compare_with_reference(tampered, reference_config))


def _tensor_set(**overrides: torch.Tensor) -> QuantizedTensorSet:
    packed = torch.zeros((8, 32), dtype=torch.uint8)
    scale = torch.ones((8, 4), dtype=torch.float8_e4m3fn)
    global_scale = torch.ones(1, dtype=torch.float32)
    fields = {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_global_scale": global_scale,
    }
    fields.update(overrides)
    return QuantizedTensorSet(**fields)


def test_tensor_set_accepts_the_canonical_shapes() -> None:
    """(out, in//2) uint8, (out, in//16) fp8, fp32 scalar."""
    _tensor_set().check("gate_proj")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weight_packed": torch.zeros((8, 32), dtype=torch.int32)}, "uint8"),
        ({"weight_packed": torch.zeros(256, dtype=torch.uint8)}, "2-D"),
        ({"weight_scale": torch.ones((8, 4))}, "float8_e4m3fn"),
        ({"weight_scale": torch.ones(4, dtype=torch.float8_e4m3fn)}, "2-D"),
        ({"weight_scale": torch.ones((4, 4), dtype=torch.float8_e4m3fn)}, "rows"),
        ({"weight_scale": torch.ones((8, 8), dtype=torch.float8_e4m3fn)}, "groups"),
        ({"weight_global_scale": torch.ones(2)}, "scalar"),
        ({"weight_global_scale": torch.ones(1, dtype=torch.bfloat16)}, "float32"),
    ],
)
def test_tensor_set_rejects_malformed_shapes(overrides: dict, message: str) -> None:
    """Malformed stored tensors are rejected before a checkpoint is written."""
    with pytest.raises(ValueError, match=message):
        _tensor_set(**overrides).check("gate_proj")


def test_input_global_scale_is_validated() -> None:
    """input_global_scale, when present, must be a float32 scalar."""
    _tensor_set(input_global_scale=torch.ones(1)).check("gate_proj")
    with pytest.raises(ValueError, match="float32"):
        _tensor_set(input_global_scale=torch.ones(1, dtype=torch.bfloat16)).check("x")
    with pytest.raises(ValueError, match="scalar"):
        _tensor_set(input_global_scale=torch.ones(4)).check("x")


def test_state_dict_naming_matches_the_reference_schema() -> None:
    """Stored tensor names are the compressed-tensors names."""
    entries = _tensor_set(input_global_scale=torch.ones(1)).as_state_dict(
        "model.layers.0.mlp.gate_proj"
    )
    assert set(entries) == {
        "model.layers.0.mlp.gate_proj.weight_packed",
        "model.layers.0.mlp.gate_proj.weight_scale",
        "model.layers.0.mlp.gate_proj.weight_global_scale",
        "model.layers.0.mlp.gate_proj.input_global_scale",
    }


def test_state_dict_omits_input_global_scale_when_absent() -> None:
    """Dynamic activations do not need a stored input_global_scale."""
    entries = _tensor_set().as_state_dict("m")
    assert "m.input_global_scale" not in entries
    assert "m.weight" not in entries


def test_module_quantized_state_extraction() -> None:
    """Extraction works on a module carrying the expected attributes."""
    module = torch.nn.Module()
    module.register_buffer("weight_packed", torch.zeros((4, 8), dtype=torch.uint8))
    module.register_buffer("weight_scale", torch.ones((4, 1), dtype=torch.float8_e4m3fn))
    module.register_buffer("weight_global_scale", torch.ones(1))
    state = module_quantized_state(module)
    state.check("probe")


def test_module_quantized_state_rejects_unquantized_modules() -> None:
    """An unquantized module is an error, not a silently missing tensor."""
    with pytest.raises(AttributeError, match="not quantized"):
        module_quantized_state(torch.nn.Linear(4, 4))


def test_export_manifest_round_trip(tmp_path: Path) -> None:
    """The manifest records the counts needed to audit an export."""
    manifest = ExportManifest(
        target_count=496,
        quantized_count=496,
        source_model="EfficientThink",
        source_revision="882e1af3b8f5844f3b3163b132e343c06e91908f",
        quantization_config=build_quantization_config(),
    )
    destination = tmp_path / "EVAL_MANIFEST.json"
    manifest.write(destination)
    restored = json.loads(destination.read_text())
    assert restored["target_count"] == 496
    assert restored["quantized_count"] == 496
    assert restored["quantization_config"]["format"] == "nvfp4-pack-quantized"
