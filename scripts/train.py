"""Обучение моделей.

    python scripts/train.py -c configs/default.yaml
    python scripts/train.py -c configs/default.yaml --set nn.enabled=false
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (путь до пакета)

from forexmodel.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
