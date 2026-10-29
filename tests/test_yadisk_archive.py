"""Архив минуток на Яндекс Диске (scripts/yadisk_archive.py) и дописка ISS (fetch_moex_iss.update) — без сети и Диска."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_moex_iss as iss  # noqa: E402
import yadisk_archive as ya  # noqa: E402

COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def minutes(start: str, n: int) -> pd.DataFrame:
    t = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"open": 1.0, "close": 1.0, "high": 1.0, "low": 1.0, "value": 1.0, "volume": 1,
                         "begin": t.strftime("%Y-%m-%d %H:%M:%S"), "end": (t + pd.Timedelta(seconds=59)).strftime("%Y-%m-%d %H:%M:%S")})[COLS]


def test_years_and_partitions_round_trip(tmp_path):
    f = tmp_path / "SBER_1m_live.csv"
    pd.concat([minutes("2025-12-31 23:50", 10), minutes("2026-01-02 10:00", 25)]).to_csv(f, index=False)
    ys = ya.years(f, chunk=7)                                       # маленькие куски — год рвётся между ними
    assert ys == {"2025": {"rows": 10, "last": "2025-12-31 23:59:00"}, "2026": {"rows": 25, "last": "2026-01-02 10:24:00"}}
    out = ya.write_years(f, {"2026"}, tmp_path / "stage", chunk=7)
    back = pd.read_csv(out["2026"])
    assert list(back.columns) == COLS and len(back) == 25 and back["begin"].is_monotonic_increasing


def test_plan_uploads_only_new_or_grown_years(tmp_path):
    f = tmp_path / "GAZP_1m_live.csv"
    pd.concat([minutes("2025-06-01 10:00", 5), minutes("2026-03-01 10:00", 5)]).to_csv(f, index=False)
    done = {"GAZP": {"2025": {"rows": 5, "last": "2025-06-01 10:04:00"}, "2026": {"rows": 3, "last": "2026-03-01 10:02:00"}}}
    todo = ya.plan(done, {"GAZP": f})
    assert todo["GAZP"][2] == {"2026"}                              # 2025 уже на Диске, в 2026 появились минуты
    done["GAZP"]["2026"]["last"] = "2026-03-01 10:04:00"
    assert ya.plan(done, {"GAZP": f}) == {}


def test_iss_update_appends_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.setattr(iss, "LIVE", tmp_path)
    monkeypatch.setattr(iss, "ROOT", tmp_path)
    f = tmp_path / "SBER_1m_live.csv"
    minutes("2026-10-28 10:00", 3).to_csv(f, index=False)
    monkeypatch.setattr(iss, "fetch", lambda t, d: minutes("2026-10-28 10:01", 4))   # пересекается с хвостом
    iss.update("SBER")
    d = pd.read_csv(f)
    assert len(d) == 5 and d["begin"].is_unique and d["begin"].iloc[-1] == "2026-10-28 10:04:00"
