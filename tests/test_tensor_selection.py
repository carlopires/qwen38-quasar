"""Tensor-selection tests (handoff sections 21, 42).

Builds the *real* Qwen3.8 architecture from the pinned EfficientThink config with
reduced dimensions, so the layer structure is authentic while the run is cheap.
This proves the expected all-W4A4 baseline selects exactly 496 transformer Linear
modules -- one of the two structural facts verified in the execution plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen38_quasar.models.patch_linear import (
    EXPECTED_TRANSFORMER_LINEARS,
    GDN_SUFFIXES,
    MLP_SUFFIXES,
    REFERENCE_IGNORE_PATTERNS,
    SELF_ATTENTION_SUFFIXES,
    assert_target_count,
    classify_linear,
    collect_target_linears,
    freeze_non_targets,
    is_ignored,
    targeting_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "tests" / "data" / "efficientthink-config.json"

pytestmark = pytest.mark.skipif(
    not CONFIG_PATH.exists(),
    reason="pinned EfficientThink config.json fixture is absent",
)


@pytest.fixture(scope="module")
def qwen38_tiny():
    """The real architecture, shrunk so it fits in a test."""
    raw = json.loads(CONFIG_PATH.read_text())
    text_config = raw["text_config"]
    text_config.update(
        {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 256,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 512,
            "mtp_num_hidden_layers": 1,
        }
    )
    raw.pop("vision_config", None)
    raw["language_model_only"] = True
    raw["eos_token_id"] = 1
    raw["bos_token_id"] = 0
    config = AutoConfig.for_model(**raw)
    return AutoModelForCausalLM.from_config(config)


def test_architecture_is_the_expected_one(qwen38_tiny) -> None:
    """The fixture really is the Qwen3.8 architecture, not a stand-in."""
    assert type(qwen38_tiny).__name__ == "Qwen3_5ForCausalLM"
    layer_types = qwen38_tiny.config.layer_types
    assert len(layer_types) == 64
    assert layer_types.count("linear_attention") == 48
    assert layer_types.count("full_attention") == 16


def test_selects_exactly_496_transformer_linears(qwen38_tiny) -> None:
    """The all-W4A4 baseline target count is 496, no more and no less."""
    targets = collect_target_linears(qwen38_tiny)
    assert len(targets) == EXPECTED_TRANSFORMER_LINEARS == 496


def test_lm_head_is_a_linear_and_is_excluded(qwen38_tiny) -> None:
    """lm_head is an nn.Linear; only the explicit name rule keeps it out."""
    assert isinstance(qwen38_tiny.lm_head, torch.nn.Linear)
    named = dict(qwen38_tiny.named_modules())
    assert "lm_head" in named
    assert classify_linear("lm_head") is None
    assert "lm_head" in REFERENCE_IGNORE_PATTERNS
    names = {name for name, _ in collect_target_linears(qwen38_tiny)}
    assert "lm_head" not in names


def test_layout_matches_the_documented_breakdown(qwen38_tiny) -> None:
    """240 GDN + 64 attention + 192 MLP = 496."""
    report = targeting_report(qwen38_tiny)
    assert report.counts == {"linear_attn": 240, "self_attn": 64, "mlp": 192}
    assert report.total == 496


def test_gdn_control_linears_are_included(qwen38_tiny) -> None:
    """in_proj_a and in_proj_b are quantized in the baseline (all 496 are W4A4)."""
    names = {name for name, _ in collect_target_linears(qwen38_tiny)}
    for suffix in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
        assert any(n.endswith(suffix) for n in names)
        assert suffix in GDN_SUFFIXES


def test_non_linear_modules_are_excluded(qwen38_tiny) -> None:
    """Conv1d, RMSNorm and activation modules are never targeted."""
    selected = {name for name, _ in collect_target_linears(qwen38_tiny)}
    for module_name, module in qwen38_tiny.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            assert module_name not in selected
    for name in (
        "model.layers.0.linear_attn.conv1d",
        "model.layers.0.linear_attn.norm",
        "model.layers.0.input_layernorm",
        "model.layers.0.mlp.act_fn",
    ):
        assert name not in selected
    # q_norm/k_norm exist only on full-attention layers and are not Linears
    assert not any(n.endswith(("self_attn.q_norm", "self_attn.k_norm")) for n in selected)


def test_attention_output_gate_is_fused_not_a_separate_linear(qwen38_tiny) -> None:
    """attn_output_gate is folded into q_proj: still exactly 4 attention Linears."""
    attention = [
        name
        for name, _ in collect_target_linears(qwen38_tiny)
        if ".self_attn." in name and ".layers.3." in name
    ]
    assert len(attention) == 4
    assert "gate" not in " ".join(attention)
    assert len(SELF_ATTENTION_SUFFIXES) == 4


def test_mlp_has_three_projections_per_layer(qwen38_tiny) -> None:
    """gate/up/down for all 64 layers."""
    assert len(MLP_SUFFIXES) == 3
    mlp = [n for n, _ in collect_target_linears(qwen38_tiny) if ".mlp." in n]
    assert len(mlp) == 64 * 3


def test_ignore_patterns_match_the_reference_checkpoint() -> None:
    """Ignore patterns are the reference checkpoint's, verbatim."""
    assert REFERENCE_IGNORE_PATTERNS == (
        "lm_head",
        r"re:.*visual.*",
        r"re:.*mtp.*",
        r"re:.*embed_vision.*",
        r"re:.*embed_audio.*",
        r"re:.*vision_embedder.*",
    )


@pytest.mark.parametrize(
    "name",
    [
        "lm_head",
        "model.visual.blocks.0.attn.qkv",
        "model.mtp.layers.0.mlp.gate_proj",
        "model.embed_vision.proj",
        "model.embed_audio.proj",
        "model.vision_embedder.patch_embed",
    ],
)
def test_ignored_names_are_rejected(name: str) -> None:
    """Vision, MTP and audio-embedder modules are excluded."""
    assert is_ignored(name)
    assert classify_linear(name) is None


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.63.mlp.down_proj",
    ],
)
def test_valid_names_are_accepted(name: str) -> None:
    """Canonical transformer Linear names classify correctly."""
    assert not is_ignored(name)
    assert classify_linear(name) is not None


def test_names_outside_a_decoder_layer_are_rejected() -> None:
    """A Linear must live inside a decoder layer to be targeted."""
    assert classify_linear("model.self_attn.q_proj") is None
    assert classify_linear("model.embed_tokens") is None
    assert classify_linear("lm_head") is None


def test_wrong_count_fails_loudly() -> None:
    """A target set other than 496 is a hard error, not a warning."""
    with pytest.raises(RuntimeError, match="expected exactly 496"):
        assert_target_count([(f"t{i}", None) for i in range(495)])
    with pytest.raises(RuntimeError, match="expected exactly 496"):
        assert_target_count([(f"t{i}", None) for i in range(497)])
    # the correct count is accepted
    assert_target_count([(f"t{i}", None) for i in range(496)])


def test_freeze_non_targets_leaves_only_latent_weights_trainable(qwen38_tiny) -> None:
    """Handoff section 23: only targeted latent weights are trainable."""
    targets = collect_target_linears(qwen38_tiny)
    trainable = freeze_non_targets(qwen38_tiny, targets)

    assert len(trainable) == 496
    assert "lm_head.weight" not in trainable
    assert "model.embed_tokens.weight" not in trainable
    assert not any("norm" in name for name in trainable)
    assert not any("conv1d" in name for name in trainable)

    trainable_set = set(trainable)
    for name, parameter in qwen38_tiny.named_parameters():
        assert parameter.requires_grad is (name in trainable_set)


def test_every_target_weight_is_two_dimensional(qwen38_tiny) -> None:
    """Every targeted module is a matmul weight, so grouping is well defined."""
    for name, module in collect_target_linears(qwen38_tiny):
        assert module.weight.dim() == 2, name


def test_report_round_trips_to_json(qwen38_tiny) -> None:
    """The startup report is serializable for the run records."""
    report = targeting_report(qwen38_tiny)
    payload = json.dumps(report.as_dict())
    restored = json.loads(payload)
    assert restored["total"] == 496
    assert len(restored["targeted_names"]) == 496
    assert restored["counts"]["linear_attn"] == 240
