"""Архив данных forexmodel на Яндекс Диске (подключён 29.10: rclone, удалённый диск yadisk:, монтирование ~/yadisk).

Горячие данные живого контура — минутки 19 бумаг, которые дописываются каждый час, — остаются на сервере: Яндекс Диск
не умеет дописывать в файл (rclone отправлял бы файл в 180 МБ заново после каждой записи), а при сбое связи контур
остался бы без данных посреди торгов. Раз в сутки (cron, после вечерней сессии) они уходят в архив:
    yadisk:forexmodel/минутки/<БУМАГА>/<год>.csv.gz
прошлые годы — один раз, текущий — когда в нём появились новые минуты. Загрузка — через API (rclone copyto сверяет
хэш), а не через монтирование; что уже выгружено — reports/live/yadisk_manifest.json. Сбой — уведомление в группу.
Новые большие выгрузки истории (python scripts/moex_data.py … --yadisk) сразу уходят на Диск через ~/bin/yd, на
сервере остаётся ссылка.

    python scripts/yadisk_archive.py              # раз в сутки — scripts/queues/yadisk_nightly.sh
    python scripts/yadisk_archive.py --dry-run    # что было бы выгружено
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import notify
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "reports" / "live" / "yadisk_manifest.json"
STAGE = ROOT / "reports" / "_yadisk_stage"          # временные .csv.gz, удаляются после загрузки
RCLONE = Path.home() / "bin" / "rclone"
REMOTE = "yadisk:forexmodel/минутки"
SOCK = Path(f"/run/user/{os.getuid()}/rclone-yadisk.sock")


def hot_files() -> dict[str, Path]:
    """Минутки живого контура: S42 — data/raw/live/<T>_1m_live.csv, остальные бумаги — data/raw/moex/<T>_1m.csv."""
    out = {p.name.split("_")[0]: p for p in sorted((RAW / "moex").glob("*_1m.csv"))}
    out.update({p.name.split("_")[0]: p for p in sorted((RAW / "live").glob("*_1m_live.csv"))})
    return out


def years(path: Path, chunk: int = 500_000) -> dict[str, dict]:
    """Год → число минут и последняя минута (потоком, без загрузки файла целиком)."""
    out: dict[str, dict] = {}
    for c in pd.read_csv(path, usecols=["begin"], chunksize=chunk):
        y = c["begin"].str[:4]
        for year, g in c.groupby(y):
            r = out.setdefault(year, {"rows": 0, "last": ""})
            r["rows"] += len(g)
            r["last"] = max(r["last"], g["begin"].iloc[-1])
    return out


def write_years(path: Path, want: set[str], stage: Path, chunk: int = 500_000) -> dict[str, Path]:
    """Нужные годы файла → stage/<год>.csv.gz (с заголовком), потоком."""
    stage.mkdir(parents=True, exist_ok=True)
    files, head = {}, pd.read_csv(path, nrows=0).columns
    for c in pd.read_csv(path, chunksize=chunk):
        for year, g in c.groupby(c["begin"].str[:4]):
            if year not in want:
                continue
            f = stage / f"{year}.csv.gz"
            new = f not in files.values()
            with gzip.open(f, "at" if not new else "wt", encoding="utf-8", newline="") as fh:
                g[head].to_csv(fh, index=False, header=new)
            files[year] = f
    return files


def upload(local: Path, remote: str) -> None:
    subprocess.run([str(RCLONE), "copyto", str(local), remote, "--stats", "0", "--retries", "3"], check=True,
                   capture_output=True, text=True, timeout=3600)


def plan(manifest: dict, files: dict[str, Path]) -> dict[str, tuple[Path, dict, set[str]]]:
    """Бумага → (файл, годы в файле, годы к выгрузке): нового года нет в манифесте или его последняя минута сдвинулась."""
    todo = {}
    for t, p in files.items():
        ys = years(p)
        done = manifest.get(t, {})
        want = {y for y, r in ys.items() if done.get(y, {}).get("last") != r["last"]}
        if want:
            todo[t] = (p, ys, want)
    return todo


def run(dry: bool) -> tuple[int, list[str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    todo = plan(manifest, hot_files())
    errors, n = [], 0
    for t, (p, ys, want) in todo.items():
        print(f"[{t}] {p.relative_to(ROOT)}: к выгрузке {', '.join(sorted(want))}", flush=True)
        if dry:
            n += len(want)
            continue
        stage = STAGE / t
        try:
            for year, f in sorted(write_years(p, want, stage).items()):
                upload(f, f"{REMOTE}/{t}/{year}.csv.gz")
                manifest.setdefault(t, {})[year] = {**ys[year], "gz_mb": round(f.stat().st_size / 2 ** 20, 2),
                                                    "uploaded": pd.Timestamp.now().isoformat(timespec="seconds")}
                MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                n += 1
        except Exception as e:                 # сеть, Диск, место — остальные бумаги всё равно пробуем
            msg = e.stderr.strip().splitlines()[-1] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
            errors.append(f"{t}: {msg[:200]}")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    if n and not dry and SOCK.exists():         # mount не видит загруженного через API, пока не сброшен кэш каталогов
        subprocess.run([str(RCLONE), "rc", "--unix-socket", str(SOCK), "vfs/forget"], capture_output=True, timeout=60)
    return n, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    n, errors = run(a.dry_run)
    print(f"[архив] {'было бы выгружено' if a.dry_run else 'выгружено'} файлов: {n}; ошибок: {len(errors)}", flush=True)
    if errors and not a.dry_run:
        notify.send("⚠️ Архив минуток на Яндекс Диск не обновлён:\n" + "\n".join(errors[:10]) + "\nКонтур работает: данные на сервере целы.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
