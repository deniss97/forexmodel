"""Диагностика «входов на излёте» по уже сохранённым сделкам.

С этого стоит начинать любую итерацию: если winrate падает с ростом
extension_atr, значит модель систематически входит в конце движения, и имеет
смысл сперва поставить грубое правило `simulation.max_extension_atr`, а уже
потом переобучать.

    python scripts/diagnose.py -c configs/default.yaml --split sim
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from forexmodel.config import load_config
from forexmodel.evaluation.diagnostics import exit_reason_breakdown, late_entry_diagnostic
from forexmodel.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Диагностика поздних входов")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--split", default="sim")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    reports = cfg.reports_path()

    trades_path = reports / f"trades_{args.split}.csv"
    signals_path = reports / f"signals_{args.split}.csv"
    for path in (trades_path, signals_path):
        if not path.exists():
            raise FileNotFoundError(f"Нет файла {path} — сначала запустите бэктест")

    trades = pd.read_csv(trades_path, parse_dates=["signal_dt", "open_dt", "close_dt"])
    signals = pd.read_csv(signals_path, parse_dates=["time"])

    late_entry_diagnostic(trades, signals, atr_col=cfg.atr_col)
    exit_reason_breakdown(trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
