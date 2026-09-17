"""Единая настройка логирования.

В ноутбуке всё печаталось через print(), из-за чего при падении в середине
перебора параметров невозможно было понять, на каком шаге это случилось.
Здесь один logger на пакет; скрипты вызывают `setup_logging()` один раз.

В каждой строке есть время от старта процесса (`+MM:SS`) — по логу сразу видно,
сколько заняли данные, сколько обучение, и где прогон встал.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

_FORMAT = "%(asctime)s | +%(elapsed)-8s | %(levelname)-7s | %(name)-30s | %(message)s"
_DATEFMT = "%H:%M:%S"

_STARTED = time.perf_counter()
_LOG_FILES: List[Path] = []


class _ElapsedFilter(logging.Filter):
    """Добавляет в запись `elapsed` — время от старта процесса."""

    def filter(self, record: logging.LogRecord) -> bool:
        seconds = int(time.perf_counter() - _STARTED)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        record.elapsed = f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"
        return True


def setup_logging(level: int | str = logging.INFO, log_file: Optional[Path | str] = None) -> Optional[Path]:
    """Настраивает вывод в stdout и (опционально) в файл. Возвращает путь к файлу."""
    root = logging.getLogger("forexmodel")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    # фильтр вешается на ОБРАБОТЧИКИ, а не на логгер: фильтры логгера
    # не применяются к записям дочерних логгеров (forexmodel.models.* и т.п.),
    # а они как раз и составляют весь лог
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    stream.addFilter(_ElapsedFilter())
    root.addHandler(stream)

    if log_file is None:
        return None
    return add_log_file(log_file)


def add_log_file(log_file: Path | str) -> Path:
    """Добавляет файловый обработчик к уже настроенному логированию.

    Нужно потому, что путь к логу зависит от `paths.run_name` из конфига, а
    конфиг читается уже с логированием (его предупреждения тоже надо видеть).
    """
    root = logging.getLogger("forexmodel")
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    handler.addFilter(_ElapsedFilter())
    root.addHandler(handler)
    _LOG_FILES.append(log_file)
    return log_file


def log_files() -> List[Path]:
    """Куда пишется лог этого процесса — чтобы напомнить об этом в конце прогона."""
    return list(_LOG_FILES)


def get_logger(name: str) -> logging.Logger:
    """`get_logger(__name__)` в каждом модуле."""
    if not name.startswith("forexmodel"):
        name = f"forexmodel.{name}"
    return logging.getLogger(name)


def section(logger: logging.Logger, title: str, char: str = "-", width: int = 78) -> None:
    """Подзаголовок внутри этапа: в длинном логе глазу нужны зацепки."""
    logger.info("")
    logger.info("%s %s", char * 3, f"{title} " + char * max(width - len(title) - 5, 3))
