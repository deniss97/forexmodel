import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_signals as ls  # noqa: E402


def test_live_dataset_keeps_unlabelled_bars(monkeypatch):
    """Свежая свеча ещё без метки triple-barrier: с разметкой её выбросил бы test, и вход опоздал бы на дни (30.10)."""
    seen = {}

    def fake_build(cfg, with_labels=True):
        seen["with_labels"] = with_labels
        raise StopIteration

    monkeypatch.setattr(ls, "build_dataset", fake_build)
    live = yaml.safe_load((ls.ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    spec = live["instruments"]["gazp"]
    if not (ls.ROOT / spec["csv"]).exists():
        pytest.skip("нет минуток GAZP")
    with pytest.raises(StopIteration):
        ls.analyse("gazp", spec, live, None)
    assert seen["with_labels"] is False


def test_wait_close_only_for_unfinished_hour():
    """Запуск в HH:00: ждём минуту HH:59 только если торги шли до конца часа; конец сессии (23:49) и ночь — не ждём."""
    import fetch_tinvest as ft
    hour = pd.Timestamp("2026-11-03 15:00")
    lasts = {"SBER": pd.Timestamp("2026-11-03 14:58"), "GAZP": pd.Timestamp("2026-11-03 14:59"),
             "LKOH": pd.Timestamp("2026-11-03 14:49")}
    assert ft.waiting(lasts, hour) == ["SBER"]
    assert ft.waiting({"SBER": pd.Timestamp("2026-11-02 23:49")}, pd.Timestamp("2026-11-03 00:00")) == []
