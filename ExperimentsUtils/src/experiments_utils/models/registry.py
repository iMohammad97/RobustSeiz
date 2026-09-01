from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

ModelFn = Callable[..., list[int]]
McdFn = Callable[..., np.ndarray]

# Model names known to the benchmark (adapters loaded lazily on first use).
_BUILTIN_MODELS: set[str] = {"SeizureTransformer"}
# Models that declare MCD support (checked on host without importing model packages).
_MCD_MODELS: set[str] = {"SeizureTransformer"}

_REGISTRY: dict[str, ModelFn] = {}
_MCD_REGISTRY: dict[str, McdFn] = {}


def _load_model_adapter(name: str) -> None:
    if name in _REGISTRY:
        return
    if name == "SeizureTransformer":
        from experiments_utils.models import seizure_transformer

        _REGISTRY[name] = seizure_transformer.detect
        _MCD_REGISTRY[name] = seizure_transformer.compute_agreement
        return
    if name not in _REGISTRY:
        raise ValueError(f"Model adapter {name!r} is not loaded.")


def _known_models() -> set[str]:
    return _BUILTIN_MODELS | set(_REGISTRY.keys())


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


def get_model_fn(project_root: Path | None = None) -> ModelFn:
    name = get_active_model_name(project_root)
    _load_model_adapter(name)
    return _REGISTRY[name]


def model_supports_mcd(project_root: Path | None = None) -> bool:
    return get_active_model_name(project_root) in _MCD_MODELS


def get_mcd_fn(project_root: Path | None = None) -> McdFn | None:
    name = get_active_model_name(project_root)
    if name not in _MCD_MODELS:
        return None
    _load_model_adapter(name)
    return _MCD_REGISTRY.get(name)


def register_model(name: str, fn: ModelFn, mcd_fn: McdFn | None = None) -> None:
    _REGISTRY[name] = fn
    _BUILTIN_MODELS.add(name)
    if mcd_fn is not None:
        _MCD_REGISTRY[name] = mcd_fn
        _MCD_MODELS.add(name)
    else:
        _MCD_REGISTRY.pop(name, None)
        _MCD_MODELS.discard(name)


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_known_models()))


def mcd_supported_models() -> tuple[str, ...]:
    return tuple(sorted(_MCD_MODELS))
