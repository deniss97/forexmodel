"""Прогресс, тайминги и ETA для долгих шагов пайплайна.

Зачем отдельный модуль: в логах обучения важны три вещи — что идёт сейчас,
какая доля сделана и сколько ещё ждать. CatBoost пишет свой `verbose` прямо в
stdout (в файл лога это не попадает и ETA там нет), torch-цикл по эпохам вообще
ничего не сообщает до конца эпохи, а walk-forward CV молчит по нескольку минут
на фолд. Здесь всё это приведено к одному виду:

    [2/6] CNN-BiLSTM ............... 45% [#####.....]  18/40 эпох | прошло 1м 12с | осталось ~1м 28с

Тайминги этапов складываются в `artifacts/<run>/timings.json`, поэтому со
второго прогона в начале печатается ожидаемое общее время по прошлому запуску.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "fmt_duration",
    "Progress",
    "StageTracker",
    "CatBoostProgress",
]

_BAR_WIDTH = 14


def fmt_duration(seconds: Optional[float]) -> str:
    """Человекочитаемая длительность: 8.4с / 1м 23с / 2ч 05м."""
    if seconds is None or seconds != seconds or seconds < 0:  # None/NaN
        return "н/д"
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f}мс"
    if seconds < 10:
        return f"{seconds:.1f}с"
    if seconds < 60:
        return f"{seconds:.0f}с"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}м {sec:02d}с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}ч {minutes:02d}м"


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


# ----------------------------------------------------------------------
# прогресс внутри одного шага
# ----------------------------------------------------------------------
class Progress:
    """Счётчик с ETA, который сам решает, когда писать в лог.

    Логирует не чаще `min_interval` секунд (плюс первый и последний шаг), иначе
    на 2000 итерациях CatBoost лог превращается в простыню. `eta_prefix="≤"`
    для шагов с early stopping: там общее число итераций — это верхняя граница,
    и обещать точное время нельзя.
    """

    def __init__(
        self,
        total: Optional[int],
        label: str,
        unit: str = "ит",
        logger: Optional[logging.Logger] = None,
        min_interval: float = 10.0,
        level: int = logging.INFO,
        eta_prefix: str = "~",
        every_n: Optional[int] = None,
        min_total: int = 0,
    ) -> None:
        self.total = int(total) if total else None
        self.label = label
        self.unit = unit
        self.log = logger or log
        self.min_interval = min_interval
        self.level = level
        self.eta_prefix = eta_prefix
        self.every_n = int(every_n) if every_n else None
        # короткие циклы не стоят ни одной строки в логе: 200 сигналов бэктеста
        # обрабатываются мгновенно, и прогресс по ним — только шум
        self.muted = bool(min_total) and self.total is not None and self.total < min_total

        self.started = time.perf_counter()
        self.done_count = 0
        self._last_emit = 0.0
        self._last_emitted_count = -1

    # -- основное API ---------------------------------------------------
    def update(self, step: int = 1, note: str = "") -> None:
        self.set(self.done_count + step, note)

    def set(self, done: int, note: str = "") -> None:
        self.done_count = int(done)
        if self.muted:
            return
        now = time.perf_counter()
        is_last = self.total is not None and self.done_count >= self.total
        if self.done_count == self._last_emitted_count:
            return
        on_step = bool(self.every_n) and self.done_count % self.every_n == 0
        if not (is_last or on_step or self.done_count <= 1) and now - self._last_emit < self.min_interval:
            return
        self._emit(note)

    def finish(self, note: str = "") -> None:
        """Финальная строка с фактическим временем (ETA уже не нужен)."""
        if self.muted:
            return
        elapsed = time.perf_counter() - self.started
        rate = self.done_count / elapsed if elapsed > 0 else 0.0
        parts = [f"{self.label}: готово", f"{self.done_count} {self.unit}", f"за {fmt_duration(elapsed)}"]
        if rate and self.done_count >= 2:
            parts.append(f"{self._fmt_rate(rate)}")
        if note:
            parts.append(note)
        self.log.log(self.level, " | ".join(parts))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    # -- внутреннее -----------------------------------------------------
    def _fmt_rate(self, rate: float) -> str:
        if rate >= 1:
            return f"{rate:.1f} {self.unit}/с"
        return f"в среднем {fmt_duration(1 / rate)} на шаг"

    def _emit(self, note: str = "") -> None:
        elapsed = self.elapsed
        rate = self.done_count / elapsed if elapsed > 0 else 0.0

        if self.total:
            fraction = self.done_count / self.total
            head = f"{self.label}: {fraction * 100:5.1f}% [{_bar(fraction)}] {self.done_count}/{self.total} {self.unit}"
        else:
            head = f"{self.label}: {self.done_count} {self.unit}"

        parts = [head, f"прошло {fmt_duration(elapsed)}"]
        # по одному шагу скорость оценивать нельзя: первый батч включает прогрев
        # и даёт ETA, отличающийся от правды на порядки
        reliable = self.done_count >= 2 and elapsed >= 0.5
        if self.total and reliable and self.done_count < self.total:
            parts.append(f"осталось {self.eta_prefix}{fmt_duration((self.total - self.done_count) / rate)}")
        if reliable:
            parts.append(self._fmt_rate(rate))
        if note:
            parts.append(note)

        self.log.log(self.level, " | ".join(parts))
        self._last_emit = time.perf_counter()
        self._last_emitted_count = self.done_count


class CatBoostProgress:
    """Callback для CatBoost: прогресс с ETA вместо его собственного stdout.

    CatBoost вызывает `after_iteration(info)` после каждого дерева; `info`
    несёт номер итерации и историю метрик по learn/validation. Возврат True
    означает «продолжать» — early stopping живёт внутри CatBoost и на это
    не влияет.
    """

    def __init__(
        self,
        total_iterations: int,
        label: str = "CatBoost",
        logger: Optional[logging.Logger] = None,
        min_interval: float = 10.0,
        every_n: Optional[int] = None,
    ) -> None:
        self.progress = Progress(
            total_iterations,
            label=label,
            unit="дер",
            logger=logger or log,
            min_interval=min_interval,
            eta_prefix="≤",  # early stopping может оборвать раньше
            every_n=every_n,
        )
        self._seen = 0

    def after_iteration(self, info) -> bool:
        # считаем вызовы сами: нумерация `info.iteration` в разных версиях
        # CatBoost то с нуля, то с единицы, а вызов приходит ровно раз на дерево
        self._seen += 1
        self.progress.set(self._seen, self._metrics_note(getattr(info, "metrics", None)))
        return True

    @staticmethod
    def _metrics_note(metrics) -> str:
        if not metrics:
            return ""
        chunks: List[str] = []
        for pool_name, alias in (("learn", "learn"), ("validation", "val"), ("validation_0", "val")):
            pool = metrics.get(pool_name)
            if not pool:
                continue
            for metric_name, history in pool.items():
                if history:
                    chunks.append(f"{alias}:{metric_name}={history[-1]:.4f}")
                break  # первой метрики достаточно, остальное — шум в логе
        return " ".join(chunks)


# ----------------------------------------------------------------------
# этапы пайплайна
# ----------------------------------------------------------------------
@dataclass
class StageRecord:
    name: str
    seconds: float


@dataclass
class StageTracker:
    """Нумерованные этапы с заголовками, итоговой таблицей и общим ETA.

    Ожидаемое время берётся из `timings.json` прошлого прогона — придумывать
    прогноз на пустом месте смысла нет, а «в прошлый раз это заняло 6м 40с»
    отвечает на вопрос «сколько ждать» честно.
    """

    title: str
    stages: Sequence[str]
    logger: logging.Logger = field(default_factory=lambda: log)
    timings_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.started = time.perf_counter()
        self.records: List[StageRecord] = []
        self.previous: Dict[str, float] = self._load_previous()
        self._index = 0
        self._announce()

    # -- API ------------------------------------------------------------
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._index += 1
        expected = self.previous.get(name)
        header = f"ЭТАП {self._index}/{len(self.stages)} · {name}"

        info = [f"прошло всего {fmt_duration(time.perf_counter() - self.started)}"]
        if expected:
            info.append(f"в прошлый прогон этот этап занял {fmt_duration(expected)}")
            remaining = self._expected_remaining()
            if remaining:
                info.append(f"до конца ~{fmt_duration(remaining)}")

        self.logger.info("")
        self.logger.info("=" * 78)
        self.logger.info("  %s", header)
        self.logger.info("  %s", " | ".join(info))
        self.logger.info("=" * 78)

        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            self.logger.error("ЭТАП %d/%d · %s — УПАЛ через %s", self._index, len(self.stages), name,
                              fmt_duration(time.perf_counter() - t0))
            raise
        spent = time.perf_counter() - t0
        self.records.append(StageRecord(name=name, seconds=spent))
        self.logger.info("--> этап «%s» завершён за %s", name, fmt_duration(spent))

    def finish(self, save: bool = True) -> str:
        """Итоговая таблица времени; возвращает её же текстом (для отчёта)."""
        total = time.perf_counter() - self.started
        width = max((len(r.name) for r in self.records), default=10)
        lines = ["", f"ВРЕМЯ ПО ЭТАПАМ — {self.title}", "-" * (width + 30)]
        for i, rec in enumerate(self.records, 1):
            share = rec.seconds / total * 100 if total else 0.0
            lines.append(f"  {i}. {rec.name:<{width}}  {fmt_duration(rec.seconds):>9}  {share:4.0f}%")
        lines.append("-" * (width + 30))
        lines.append(f"  {'ИТОГО':<{width + 3}}  {fmt_duration(total):>9}")
        text = "\n".join(lines)
        self.logger.info(text)

        if save and self.timings_path is not None:
            self._save({r.name: round(r.seconds, 2) for r in self.records} | {"__total__": round(total, 2)})
        return text

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self.started

    # -- внутреннее -----------------------------------------------------
    def _announce(self) -> None:
        self.logger.info("")
        self.logger.info("#" * 78)
        self.logger.info("# %s", self.title)
        self.logger.info("# этапов: %d — %s", len(self.stages), " -> ".join(self.stages))
        total_expected = self.previous.get("__total__")
        if total_expected:
            self.logger.info("# прошлый такой прогон занял %s — примерно столько и ждать", fmt_duration(total_expected))
        else:
            self.logger.info("# оценки общего времени нет (первый прогон) — ETA появится внутри длинных этапов")
        self.logger.info("#" * 78)

    def _expected_remaining(self) -> Optional[float]:
        """Сколько осталось по прошлым таймингам: текущий этап + все следующие."""
        remaining = 0.0
        known = False
        for name in list(self.stages)[self._index - 1 :]:
            value = self.previous.get(name)
            if value:
                remaining += value
                known = True
        return remaining if known else None

    def _load_previous(self) -> Dict[str, float]:
        if self.timings_path is None or not Path(self.timings_path).exists():
            return {}
        try:
            data = json.loads(Path(self.timings_path).read_text(encoding="utf-8"))
            return {k: float(v) for k, v in data.items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _save(self, data: Dict[str, float]) -> None:
        path = Path(self.timings_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Тайминги этапов сохранены: %s (из них берётся ETA следующего прогона)", path)
