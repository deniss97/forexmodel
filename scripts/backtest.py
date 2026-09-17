"""Бэктест обученных моделей.

    python scripts/backtest.py -c configs/default.yaml --split sim
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from forexmodel.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["backtest", *sys.argv[1:]]))
