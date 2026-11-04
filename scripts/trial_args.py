"""Аргументы live_signals.py для пробы P42 по configs/account.yaml; проба выключена — пустая строка.

    live_signals.py --tag trial $(python scripts/trial_args.py)     # так зовёт scripts/queues/live_hourly.sh
"""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "account.yaml"


def trial_args(cfg: dict) -> str:
    stocks = (cfg.get("strategies") or {}).get("P42")
    if not isinstance(stocks, list) or not stocks:
        return ""
    out = ["--only", *[str(s).lower() for s in stocks], "--impulse-atr", f"{float(cfg.get('trial_impulse_atr', 1.5)):g}"]
    if not cfg.get("trial_gate", True):
        out.append("--no-gate")
    return " ".join(out)


if __name__ == "__main__":
    print(trial_args(yaml.safe_load(CFG.read_text(encoding="utf-8"))))
