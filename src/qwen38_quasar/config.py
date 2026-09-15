"""Configuration loading and the parameter-provenance manifest.

Handoff section 24 requires every training parameter to be labelled as either
``public_reference`` (a value taken from the QUASAR paper, the EfficientThink
model card, or the public QUASAR checkpoint) or ``local_choice`` (a value we
picked because no public reference specifies it).

That labelling is not documentation-only: :func:`build_manifest` emits it as a
machine-readable artifact that ships with every run, so a reviewer can tell at a
glance which numbers are borrowed and which are ours.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ConfigError",
    "Config",
    "ParameterRecord",
    "build_manifest",
    "load_config",
    "PARAMETER_PROVENANCE",
]

Provenance = Literal["public_reference", "local_choice"]

#: Labels for every configurable value, with the reason it carries that label.
#: Kept next to the loader so that adding a config key without labelling it is a
#: visible omission rather than a silent default.
PARAMETER_PROVENANCE: dict[str, tuple[Provenance, str]] = {
    # --- model / teacher ---
    "model.source": ("public_reference", "handoff section 2, BF16 sibling not GGUF"),
    "model.subdir": ("public_reference", "the BF16/ subdirectory of the source repo"),
    "teacher.same_as_student_initialization": (
        "public_reference",
        "handoff section 3: teacher == student initialization == EfficientThink BF16",
    ),
    "teacher.dtype": ("public_reference", "bf16 source checkpoint"),
    "teacher.frozen": ("public_reference", "handoff section 23: no teacher gradients"),
    # --- quantization ---
    "quantization.format": ("public_reference", "reference checkpoint format field"),
    "quantization.weight_bits": ("public_reference", "reference checkpoint num_bits"),
    "quantization.activation_bits": ("public_reference", "reference checkpoint num_bits"),
    "quantization.group_size": ("public_reference", "reference checkpoint group_size"),
    "quantization.scale_dtype": ("public_reference", "reference checkpoint scale_dtype"),
    "quantization.dynamic_activation": (
        "public_reference",
        "reference checkpoint input_activations.dynamic == 'local'",
    ),
    "quantization.dynamic_weight": (
        "public_reference",
        "reference checkpoint weights.dynamic == false",
    ),
    "quantization.all_transformer_linears": (
        "public_reference",
        "handoff section 21: all 496 transformer Linear modules",
    ),
    # --- quasar ---
    "quasar.candidate_min": ("public_reference", "paper reports clipping candidates from 0.30"),
    "quasar.candidate_max": ("public_reference", "paper reports clipping candidates to 1.00"),
    "quasar.candidate_step": ("public_reference", "paper reports a 0.05 step"),
    "quasar.saliency": ("public_reference", "handoff section 15: AdamW exp_avg_sq"),
    "quasar.saliency_bootstrap": (
        "local_choice",
        "the paper does not specify first-step saliency; we use ones",
    ),
    "quasar.refit": (
        "public_reference",
        "paper fits the dequantization scale by saliency-weighted least squares",
    ),
    # --- training ---
    "training.objective": ("public_reference", "handoff section 23: forward KL"),
    "training.learning_rate": ("public_reference", "published QUASAR recipe: 1e-6"),
    "training.global_batch_size": ("public_reference", "published QUASAR recipe: 32"),
    "training.max_steps": ("public_reference", "published QUASAR recipe: 2446 steps"),
    "training.sequence_length": ("local_choice", "4096 chosen for the first baseline"),
    "training.optimizer": ("public_reference", "handoff section 15 assumes AdamW"),
    "training.adam_betas": ("local_choice", "not specified by any public reference"),
    "training.adam_epsilon": ("local_choice", "not specified by any public reference"),
    "training.weight_decay": ("local_choice", "not specified by any public reference"),
    "training.warmup_steps": ("local_choice", "not specified by any public reference"),
    "training.lr_schedule": ("local_choice", "not specified by any public reference"),
    "training.grad_clip": ("local_choice", "not specified by any public reference"),
    "training.gradient_accumulation": ("local_choice", "derived from device count and batch size"),
    "training.seed": ("local_choice", "not specified by any public reference"),
    "training.activation_checkpointing": ("local_choice", "memory strategy, handoff section 25"),
    # --- data ---
    "data.dataset": ("public_reference", "handoff section 4.3"),
    "data.teacher_max_new_tokens": ("local_choice", "generation limit"),
    "data.teacher_temperature": (
        "local_choice",
        "handoff section 22: deterministic generation for reproducibility",
    ),
    "data.teacher_top_p": ("local_choice", "unused when temperature is 0"),
    "data.teacher_do_sample": ("local_choice", "deterministic first run"),
    "data.max_prompts": ("local_choice", "pilot size"),
}


class ConfigError(Exception):
    """Raised when a configuration is missing or inconsistent."""


@dataclass
class Config:
    """A loaded experiment configuration."""

    path: Path
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested value by dotted path.

        :param dotted: e.g. ``"training.learning_rate"``
        :param default: returned when the path is absent
        """
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section, erroring if it is absent.

        :param name: section name
        :raises ConfigError: if the section is missing or not a table
        """
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"missing or malformed [{name}] section in {self.path}")
        return value

    def require(self, dotted: str) -> Any:
        """Return a required value, erroring if it is absent.

        :param dotted: dotted path
        :raises ConfigError: if the value is missing
        """
        value = self.get(dotted, None)
        if value is None:
            raise ConfigError(f"required setting '{dotted}' is missing from {self.path}")
        return value


def load_config(path: str | Path) -> Config:
    """Load a TOML experiment configuration.

    :param path: path to a ``.toml`` file
    :return: the parsed configuration
    :raises ConfigError: if the file is missing or unparseable
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    try:
        raw = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {config_path}: {error}") from error
    return Config(path=config_path, raw=raw)


@dataclass(frozen=True)
class ParameterRecord:
    """One configuration value and where it came from."""

    key: str
    value: Any
    provenance: Provenance
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "key": self.key,
            "value": self.value,
            "provenance": self.provenance,
            "note": self.note,
        }


def build_manifest(config: Config) -> dict[str, Any]:
    """Build the parameter-provenance manifest for a configuration.

    Every label in :data:`PARAMETER_PROVENANCE` that is present in the config is
    emitted. Values present in the config but *unlabelled* are reported under
    ``unlabelled`` rather than silently ignored, so the manifest cannot quietly
    become incomplete.

    :param config: a loaded configuration
    :return: the manifest, ready to be written next to a run
    """
    records: list[ParameterRecord] = []
    seen: set[str] = set()

    for key, (provenance, note) in PARAMETER_PROVENANCE.items():
        value = config.get(key, None)
        if value is None:
            continue
        seen.add(key)
        records.append(ParameterRecord(key, value, provenance, note))

    unlabelled = sorted(_flatten(config.raw) - seen)

    public = [r for r in records if r.provenance == "public_reference"]
    local = [r for r in records if r.provenance == "local_choice"]

    return {
        "config_path": str(config.path),
        "counts": {
            "public_reference": len(public),
            "local_choice": len(local),
            "unlabelled": len(unlabelled),
        },
        "parameters": [r.as_dict() for r in records],
        "unlabelled": unlabelled,
    }


def _flatten(node: dict[str, Any], prefix: str = "") -> set[str]:
    """Flatten nested tables into dotted keys (leaves only)."""
    keys: set[str] = set()
    for name, value in node.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            keys |= _flatten(value, f"{key}.")
        else:
            keys.add(key)
    return keys
