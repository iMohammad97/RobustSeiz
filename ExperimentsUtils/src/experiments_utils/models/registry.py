from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ModelFn = Callable[..., list[int]]
McdFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Benchmark-facing description of a model adapter (checked on the host without imports)."""

    module: str  # adapter module under experiments_utils.models, exposing detect(eeg, data)
    montage: str = "unipolar"  # input montage: "unipolar" (19-ch CAR) or "bipolar" (18-ch double banana)
    mcd: bool = False  # adapter exposes compute_agreement(eeg, data)


_SPECS: dict[str, ModelSpec] = {
    "SeizureTransformer": ModelSpec("seizure_transformer", montage="unipolar", mcd=True),
    "EEGWaveNet": ModelSpec("eegwavenet", montage="bipolar", mcd=False),
    "EEGNet": ModelSpec("eegnet", montage="bipolar", mcd=True),
    "EventNet": ModelSpec("eventnet", montage="unipolar", mcd=False),
}

_REGISTRY: dict[str, ModelFn] = {}
_MCD_REGISTRY: dict[str, McdFn] = {}


def _load_model_adapter(name: str) -> None:
    if name in _REGISTRY:
        return
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"Model adapter {name!r} is not loaded.")
    adapter = importlib.import_module(f"experiments_utils.models.{spec.module}")
    _REGISTRY[name] = adapter.detect
    if spec.mcd:
        _MCD_REGISTRY[name] = adapter.compute_agreement


def _known_models() -> set[str]:
    return set(_SPECS) | set(_REGISTRY)


def _read_active_model_file(project_root: Path | None = None) -> str | None:
    root = project_root or Path(__file__).resolve().parents[4]
    active_file = root / "Model" / "active_model"
    if active_file.is_file():
        name = active_file.read_text(encoding="utf-8-sig").strip()
        return name or None
    return None


def get_active_model_name(project_root: Path | None = None) -> str:
    name = os.getenv("ACTIVE_MODEL", "").strip() or _read_active_model_file(project_root)
    if not name:
        name = "SeizureTransformer"
    if name not in _known_models():
        known = ", ".join(sorted(_known_models()))
        raise ValueError(f"Unknown model {name!r}. Registered models: {known}")
    return name


def get_model_spec(project_root: Path | None = None) -> ModelSpec:
    return _SPECS[get_active_model_name(project_root)]


def get_model_fn(project_root: Path | None = None) -> ModelFn:
    name = get_active_model_name(project_root)
    _load_model_adapter(name)
    return _REGISTRY[name]


def model_supports_mcd(project_root: Path | None = None) -> bool:
    return get_model_spec(project_root).mcd


def get_mcd_fn(project_root: Path | None = None) -> McdFn | None:
    name = get_active_model_name(project_root)
    if not _SPECS[name].mcd:
        return None
    _load_model_adapter(name)
    return _MCD_REGISTRY.get(name)


def register_model(
    name: str, fn: ModelFn, mcd_fn: McdFn | None = None, montage: str = "unipolar"
) -> None:
    _SPECS[name] = ModelSpec(module="", montage=montage, mcd=mcd_fn is not None)
    _REGISTRY[name] = fn
    if mcd_fn is not None:
        _MCD_REGISTRY[name] = mcd_fn
    else:
        _MCD_REGISTRY.pop(name, None)


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_known_models()))


def mcd_supported_models() -> tuple[str, ...]:
    return tuple(sorted(name for name, spec in _SPECS.items() if spec.mcd))
