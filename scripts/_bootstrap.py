"""Добавляет корень репозитория в sys.path.

Нужен только чтобы `python scripts/train.py` работал без установки пакета
(`pip install -e .`). Если пакет установлен — импорт и так найдётся.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
