"""Runtime capability detection: which device, and which model can it serve."""

from __future__ import annotations

from maintidoc.runtime.devices import (
    DeviceInfo,
    DeviceKind,
    ResolvedModel,
    detect_devices,
    estimate_parameters,
    estimate_vram_gb,
    resolve_model,
    resolve_runtime,
    select_device,
)

__all__ = [
    "DeviceInfo",
    "DeviceKind",
    "ResolvedModel",
    "detect_devices",
    "estimate_parameters",
    "estimate_vram_gb",
    "resolve_model",
    "resolve_runtime",
    "select_device",
]
