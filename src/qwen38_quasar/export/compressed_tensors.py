"""Export to a standard Hugging Face ``compressed-tensors`` NVFP4 checkpoint.

The output must be loadable without any custom inference code, so the
``quantization_config`` is built to match the public
``QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`` checkpoint exactly. That checkpoint is
used as a *schema oracle*: we copy its structure and never its weights
(handoff section 11).

Captured from revision ``cfd1460322b9d8367a4ab13564a2182d50852d60``:

``quant_method``         ``compressed-tensors``
``quantization_status``  ``compressed``
``format``               ``nvfp4-pack-quantized``
``targets``              ``["Linear"]``
``weights``              4-bit float, ``tensor_group``, group_size 16, symmetric,
                         ``scale_dtype`` ``torch.float8_e4m3fn``, ``dynamic`` false
``input_activations``    same, but ``dynamic`` ``"local"``
``ignore``               ``lm_head`` plus the vision/MTP regexes

The two ``dynamic`` flags differ deliberately: weight scales are stored
(statically computed at export), activation scales are computed from the live
activation tensor at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "QUASAR_IGNORE_PATTERNS",
    "QuantizedTensorSet",
    "build_quantization_config",
    "compare_with_reference",
    "module_quantized_state",
]

#: Ignore patterns from the reference checkpoint, byte-identical.
QUASAR_IGNORE_PATTERNS: list[str] = [
    "lm_head",
    "re:.*visual.*",
    "re:.*mtp.*",
    "re:.*embed_vision.*",
    "re:.*embed_audio.*",
    "re:.*vision_embedder.*",
]

PRODUCER_NAME = "qwen38-quasar"


def build_quantization_config(producer: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the ``quantization_config`` block for the exported checkpoint.

    :param producer: optional producer block; defaults to this project
    :return: a config dict ready to be embedded in ``config.json``
    """
    producer_block = producer or {"name": PRODUCER_NAME, "version": "0.1.0"}

    group: dict[str, Any] = {
        "format": "nvfp4-pack-quantized",
        "input_activations": {
            "actorder": None,
            "block_structure": None,
            "dynamic": "local",
            "group_size": 16,
            "num_bits": 4,
            "observer": "static_minmax",
            "observer_kwargs": {},
            "scale_dtype": "torch.float8_e4m3fn",
            "strategy": "tensor_group",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
        "output_activations": None,
        "targets": ["Linear"],
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": 16,
            "num_bits": 4,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": "torch.float8_e4m3fn",
            "strategy": "tensor_group",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
    }

    return {
        "config_groups": {"group_0": group},
        "format": "nvfp4-pack-quantized",
        "global_compression_ratio": None,
        "ignore": list(QUASAR_IGNORE_PATTERNS),
        "kv_cache_scheme": None,
        "producer": producer_block,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


@dataclass
class QuantizedTensorSet:
    """The four tensors stored per quantized Linear module."""

    weight_packed: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale: torch.Tensor | None = None

    def as_state_dict(self, prefix: str) -> dict[str, torch.Tensor]:
        """Render as Hugging Face state-dict entries.

        :param prefix: module name prefix, e.g. ``model.layers.0.mlp.gate_proj``
        :return: name -> tensor mapping
        """
        entries = {
            f"{prefix}.weight_packed": self.weight_packed,
            f"{prefix}.weight_scale": self.weight_scale,
            f"{prefix}.weight_global_scale": self.weight_global_scale,
        }
        if self.input_global_scale is not None:
            entries[f"{prefix}.input_global_scale"] = self.input_global_scale
        return entries

    def check(self, name: str = "module") -> None:
        """Validate shapes and dtypes against the schema oracle.

        :param name: module name, for error messages
        :raises ValueError: if any tensor is not in the expected form
        """
        packed = self.weight_packed
        if packed.dtype != torch.uint8:
            raise ValueError(f"{name}: weight_packed must be uint8, got {packed.dtype}")
        if packed.dim() != 2:
            raise ValueError(f"{name}: weight_packed must be 2-D, got {packed.dim()}")

        scale = self.weight_scale
        if scale.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{name}: weight_scale must be float8_e4m3fn, got {scale.dtype}")
        if scale.dim() != 2:
            raise ValueError(f"{name}: weight_scale must be 2-D, got {scale.dim()}")
        if scale.shape[0] != packed.shape[0]:
            raise ValueError(
                f"{name}: weight_scale rows {scale.shape[0]} != weight_packed rows "
                f"{packed.shape[0]}"
            )
        if scale.shape[1] * 16 != packed.shape[1] * 2:
            raise ValueError(
                f"{name}: weight_scale groups {scale.shape[1]} inconsistent with "
                f"packed width {packed.shape[1]} for group_size 16"
            )

        global_scale = self.weight_global_scale
        if global_scale.dtype != torch.float32:
            raise ValueError(
                f"{name}: weight_global_scale must be float32, got {global_scale.dtype}"
            )
        if global_scale.numel() != 1:
            raise ValueError(
                f"{name}: weight_global_scale must be a scalar, got {global_scale.numel()}"
            )

        if self.input_global_scale is not None:
            if self.input_global_scale.dtype != torch.float32:
                raise ValueError(f"{name}: input_global_scale must be float32")
            if self.input_global_scale.numel() != 1:
                raise ValueError(f"{name}: input_global_scale must be a scalar")


def module_quantized_state(
    module: nn.Module,
    input_global_scale: torch.Tensor | None = None,
) -> QuantizedTensorSet:
    """Extract the stored NVFP4 tensors from a patched module.

    :param module: a module carrying ``weight_packed``, ``weight_scale`` and
        ``weight_global_scale`` attributes
    :param input_global_scale: optional static activation global scale
    :return: the tensor set
    """
    missing = [
        attribute
        for attribute in ("weight_packed", "weight_scale", "weight_global_scale")
        if not hasattr(module, attribute)
    ]
    if missing:
        raise AttributeError(f"module is not quantized; missing {missing}")

    # getattr with a runtime check keeps this honest: attribute presence was
    # verified above, and the tensors are validated by QuantizedTensorSet.check.
    def _tensor(attribute: str) -> torch.Tensor:
        value = getattr(module, attribute)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{attribute} must be a Tensor, got {type(value).__name__}")
        return value

    return QuantizedTensorSet(
        weight_packed=_tensor("weight_packed"),
        weight_scale=_tensor("weight_scale"),
        weight_global_scale=_tensor("weight_global_scale"),
        input_global_scale=input_global_scale,
    )


def compare_with_reference(
    ours: dict[str, Any],
    reference: dict[str, Any],
) -> list[str]:
    """Compare a ``quantization_config`` against the schema oracle.

    The ``producer`` block is expected to differ and is not compared.

    :param ours: the config we built
    :param reference: the reference checkpoint's ``quantization_config``
    :return: list of human-readable differences; empty means schema-identical
    """
    differences: list[str] = []

    for key in sorted(set(ours) | set(reference)):
        if key == "producer":
            continue
        if key not in ours:
            differences.append(f"missing key: {key}")
        elif key not in reference:
            differences.append(f"unexpected key: {key}")
        elif ours[key] != reference[key]:
            differences.append(f"{key}: ours={ours[key]!r} reference={reference[key]!r}")

    return differences


@dataclass
class ExportManifest:
    """Record describing an exported checkpoint (handoff section 27)."""

    target_count: int
    quantized_count: int
    source_model: str
    source_revision: str
    quantization_config: dict[str, Any] = field(default_factory=dict)
    ignored_modules: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        """Write the manifest as JSON.

        :param path: destination file
        """
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "target_count": self.target_count,
            "quantized_count": self.quantized_count,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "ignored_modules": self.ignored_modules,
            "quantization_config": self.quantization_config,
        }
