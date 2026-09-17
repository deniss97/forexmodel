"""Сохранение/загрузка артефактов обучения.

Модель без списка признаков (в том же порядке) и без конфига, по которому эти
признаки считались, бесполезна — в ноутбуке это уже приводило к тому, что
`feature_cols` в `save_catboost_package` бралась из соседней ячейки и могла не
соответствовать модели. Поэтому здесь всё сохраняется одним бандлом.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config
from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["save_bundle", "load_bundle", "save_json", "TrainingArtifacts"]


class TrainingArtifacts:
    """Пути внутри каталога прогона (artifacts/<run_name>/)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def primary(self) -> Path:
        return self.path("primary_catboost.pkl")

    @property
    def nn(self) -> Path:
        return self.path("nn_cnn_bilstm.pkl")

    @property
    def meta(self) -> Path:
        return self.path("meta_catboost.pkl")

    @property
    def config(self) -> Path:
        return self.path("config.yaml")

    @property
    def features(self) -> Path:
        return self.path("features.json")

    @property
    def metrics(self) -> Path:
        return self.path("metrics.json")

    def ensure(self) -> "TrainingArtifacts":
        self.root.mkdir(parents=True, exist_ok=True)
        return self


def save_bundle(obj: Any, path: Path | str, cfg: Optional[Config] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"object": obj, "config": asdict(cfg) if cfg is not None else None}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    log.info("Сохранено: %s", path)
    return path


def load_bundle(path: Path | str) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Артефакт не найден: {path}. Сначала запустите обучение (cli train).")
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["object"] if isinstance(payload, dict) and "object" in payload else payload


def save_json(data: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    log.info("Сохранено: %s", path)
    return path
