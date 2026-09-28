"""Направление движения диффузионной моделью TokDiT против CatBoost и MLP; реальные и синтетические данные.

Постановка одинакова для всех моделей. Окно — L часовых баров контекста и H
баров вперёд. Путь окна — 100·log(close) относительно close последнего бара
контекста, так что в момент прогноза путь равен нулю, а в конце горизонта
равен доходности в %. Цель — знак пути в конце горизонта.

  * TokDiT (../TokDiT, ветка d.a.lanovenko-ablation_study; nano, учится с нуля
    на CPU) видит путь, у которого скрыты последние H точек (TailMask), и
    сэмплирует S продолжений. P(up) — доля сэмплов, закончившихся выше нуля.
  * CatBoost и MLP получают L−1 доходностей контекста, делённых на их std
    внутри окна, и log этой std. Так у них та же информация, что у TokDiT
    после его нормализации.

Утечек нет:
  * нормировка — только по контексту окна: у TokDiT статистики ZNorm
    считаются по наблюдённым позициям (скрытый хвост в них не входит), у
    CatBoost и MLP — std доходностей контекста;
  * окна обучения целиком, включая горизонт, заканчиваются до начала года
    теста; контекст окна теста может уходить в прошлое, это законно;
  * генератор синтетики не видит реальных данных вовсе (фиксированные
    пресеты), поэтому test в нём не протекает.

Синтетика (`--synthetic`) подмешивается только в обучение, проверка — только
на реальном будущем:
  * `v21:<пресет>` — SamplerV21 из TokDiT (realistic, wide, fitted...); ряд
    считается путём;
  * `finstress:<тип>:<набор>` — финансовые генераторы (garch, har, regime...);
    значения считаются доходностями, путь — их накопленная сумма;
  * `--synthetic-only` — обучение только на синтетике.
Масштаб синтетики не важен: все признаки и нормировки инвариантны к масштабу.

Метрики по году и в целом: AUC P(up) против знака, IC (Спирмен) ожидаемого
хода против фактического, торговля по сигналу. Сделка открывается на доле
`--tops` самых уверенных прогнозов периода (порог — квантиль |P(up) − 0.5| за
период; метки в нём не участвуют, но распределение прогнозов периода известно
заранее — для проверки наличия информации это допустимо), держится H баров,
позиции не перекрываются, комиссия 0.04% за круг. Это не боевая симуляция.

    python scripts/tokdit_direction.py --years 2019 2026 --name silver_tokdit
    python scripts/tokdit_direction.py --years 2019 2026 --synthetic finstress:garch:3 --name silver_syn_garch
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKDIT = Path(os.environ.get("TOKDIT_PATH", ROOT.parent / "TokDiT"))
COMMISSION = 0.04


# ---------------------------------------------------------------- данные

def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path, parse_dates=["time"])
    return df[["time", "close"]].dropna().reset_index(drop=True)


def real_windows(df: pd.DataFrame, L: int, H: int):
    """Все окна: пути (N, L+H), время прогноза (последний бар контекста)."""
    lp = 100.0 * np.log(df["close"].to_numpy(dtype=np.float64))
    T = L + H
    idx = np.arange(len(lp) - T + 1)
    paths = lp[idx[:, None] + np.arange(T)[None, :]]
    paths = paths - paths[:, L - 1: L]           # ноль в момент прогноза
    t_pred = df["time"].to_numpy()[idx + L - 1]
    t_end = df["time"].to_numpy()[idx + T - 1]
    return paths.astype(np.float32), t_pred, t_end


def synthetic_paths(spec: str, n: int, L: int, H: int, seed: int) -> np.ndarray:
    """N синтетических путей длины L+H в той же форме, что real_windows."""
    import torch

    torch.manual_seed(seed)
    T = L + H
    kind, _, rest = spec.partition(":")
    out = []
    step = 4096
    for i in range(0, n, step):
        b = min(step, n - i)
        if kind == "v21":
            from sampler.synthetic import CONFIGS_V21, SamplerV21

            s = SamplerV21(CONFIGS_V21[rest or "realistic"], seq_len=T)
            x = s.generate(s.sample_plan(b))[..., 1].double().numpy()
        elif kind == "finstress":
            from sampler.finstress import level_sampler

            dyn, _, lvl = rest.partition(":")
            s = level_sampler(dyn, int(lvl or 3), seq_len=T, n_channels=1)
            r = s.generate(s.sample_plan(b))[..., 1].double().numpy()
            x = np.cumsum(r, axis=1)
        else:
            raise ValueError(f"неизвестная синтетика {spec!r}")
        out.append(x)
    x = np.concatenate(out)
    x = x[np.isfinite(x).all(axis=1)]
    x = x - x[:, L - 1: L]
    # единый масштаб, как у реального пути: std шага контекста = 1 (признаки к масштабу инвариантны,
    # а TokDiT нормирует по контексту — это только чтобы не было огромных чисел)
    sd = np.diff(x[:, :L], axis=1).std(axis=1, keepdims=True)
    ok = sd[:, 0] > 1e-9
    return (x[ok] / sd[ok]).astype(np.float32)


def tabular(paths: np.ndarray, L: int) -> np.ndarray:
    """Признаки CatBoost/MLP: доходности контекста / их std, log std. Только контекст."""
    r = np.diff(paths[:, :L], axis=1)
    sd = r.std(axis=1, keepdims=True) + 1e-9
    return np.hstack([r / sd, np.log(sd)]).astype(np.float32)


def target_up(paths: np.ndarray) -> np.ndarray:
    return (paths[:, -1] > 0).astype(np.int64)


# ---------------------------------------------------------------- модели

def fit_tokdit(train_paths: np.ndarray, L: int, H: int, args, seed: int):
    import torch
    from model.tokdit import AsinhSpec, TokDiT, ZNormSpec
    from model.zoo import tokdit_config

    torch.manual_seed(seed)
    cfg = tokdit_config(args.tokdit_size).model_copy(update={"normalizer": [ZNormSpec(), AsinhSpec()]})
    model = TokDiT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    T = L + H
    grid = torch.arange(T, dtype=torch.float32)
    mask = torch.zeros(1, T, 2, dtype=torch.bool)
    mask[:, L:, 1] = True                           # скрыт хвост значений; время не маскируется
    model.init_time_scale(grid)
    data = torch.from_numpy(train_paths)
    rng = np.random.default_rng(seed)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.from_numpy(rng.integers(0, len(data), args.batch))
        v = data[idx]
        x = torch.stack([grid.expand(len(idx), T), v], dim=-1)
        loss = model.loss(x, mask.expand(len(idx), T, 2))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"    tokdit шаг {step}: loss {float(loss.detach()):.4f} ({time.time() - t0:.0f} с)", flush=True)
    model.eval()
    return model


def predict_tokdit(model, paths: np.ndarray, L: int, H: int, args) -> tuple[np.ndarray, np.ndarray]:
    """P(up) и ожидаемый ход в конце горизонта. Хвост путей обнуляется — модель его не видит."""
    import torch

    T = L + H
    grid = torch.arange(T, dtype=torch.float32)
    p_up, mean = [], []
    with torch.no_grad():
        for i in range(0, len(paths), args.pred_batch):
            v = torch.from_numpy(paths[i: i + args.pred_batch].copy())
            v[:, L:] = 0.0                          # будущее не передаётся вовсе
            b = len(v)
            x = torch.stack([grid.expand(b, T), v], dim=-1)
            mask = torch.zeros(b, T, 2, dtype=torch.bool)
            mask[:, L:, 1] = True
            s = model.sample(x, mask, n_samples=args.samples, n_sampler_steps=args.sampler_steps,
                             chunk_size=args.chunk_size)[..., -1, 1]          # (S, B) — конец горизонта
            p_up.append((s > 0).float().mean(0).numpy())
            mean.append(s.mean(0).numpy())
    return np.concatenate(p_up), np.concatenate(mean)


def split_fit_val(n_real: int, n_total: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Ранняя остановка — на последних 15% РЕАЛЬНЫХ окон; зазор `gap` окон (L+H) отделяет их от
    обучающих, чтобы перекрывающиеся окна не попадали в обе части. Синтетика (после реальных) — в fit."""
    cut = int(n_real * 0.85)
    fit = np.r_[np.arange(max(cut - gap, 0)), np.arange(n_real, n_total)]
    return fit, np.arange(cut, n_real)


def fit_catboost(X, y, seed, fit_idx, val_idx):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=6, random_seed=seed,
                           verbose=False, early_stopping_rounds=100)
    m.fit(X[fit_idx], y[fit_idx], eval_set=(X[val_idx], y[val_idx]))
    return m


def fit_mlp(X, y, seed, fit_idx, val_idx, epochs=30):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-6      # статистики — по обучающей части
    Xt = torch.from_numpy((X[fit_idx] - mu) / sd)
    yt = torch.from_numpy(y[fit_idx].astype(np.float32))
    Xv = torch.from_numpy((X[val_idx] - mu) / sd)
    yv = torch.from_numpy(y[val_idx].astype(np.float32))
    cut = len(fit_idx)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(cut)
        for i in range(0, cut, 512):
            b = perm[i: i + 512]
            opt.zero_grad()
            lossf(net(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv).squeeze(1), yv))
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy((Xn - mu) / sd)).squeeze(1)).numpy()
    return predict


# ---------------------------------------------------------------- оценка

def trade(p_up: np.ndarray, ret: np.ndarray, H: int, margin: float) -> np.ndarray:
    """Нетто-результаты сделок: вход при |p−0.5| ≥ margin (и p ≠ 0.5), держим H прогнозов, без перекрытия."""
    out, free_at = [], 0
    for i in range(len(p_up)):
        conf = abs(p_up[i] - 0.5)
        if i < free_at or conf == 0 or conf < margin:
            continue
        out.append(np.sign(p_up[i] - 0.5) * ret[i] - COMMISSION)
        free_at = i + H
    return np.asarray(out)


def summarize(model: str, label: str, p_up, exp_ret, ret, H, tops) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    y = (ret > 0).astype(int)
    row = {"модель": model, "год": label, "n": len(ret),
           "auc": roc_auc_score(y, p_up) if 0 < y.sum() < len(y) else float("nan"),
           "ic": spearmanr(exp_ret, ret).statistic}
    conf = np.abs(np.asarray(p_up) - 0.5)
    for q in tops:
        # доля самых уверенных прогнозов периода: порог — квантиль уверенности (метки не используются)
        t = trade(p_up, ret, H, float(np.quantile(conf, 1 - q)) if q < 1 else 0.0)
        row[f"сделок@{q}"] = len(t)
        row[f"нетто@{q}"] = float(t.mean()) if len(t) else float("nan")
        row[f"t@{q}"] = float(t.mean() / (t.std() / np.sqrt(len(t)))) if len(t) > 1 and t.std() > 0 else float("nan")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="reports/_cache/silver_1h.pkl", help="часовые бары: time, close")
    ap.add_argument("--years", nargs=2, type=int, required=True)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=["tokdit", "catboost", "mlp"])
    ap.add_argument("--synthetic", default=None, help="v21:<пресет> | finstress:<тип>:<набор>")
    ap.add_argument("--synthetic-ratio", type=float, default=1.0, help="синтетических окон на одно реальное")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--tokdit-size", default="nano")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--sampler-steps", type=int, default=25)
    ap.add_argument("--pred-batch", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=2048, help="внутренний батч сэмплирования (память)")
    ap.add_argument("--test-stride", type=int, default=1, help="прогноз на каждом k-м баре теста")
    ap.add_argument("--train-years", type=int, default=None, help="обучение только на последних N годах")
    ap.add_argument("--tops", nargs="+", type=float, default=[1.0, 0.3, 0.1],
                    help="доли самых уверенных прогнозов, по которым торговать")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(TOKDIT))
    L, H = args.context, args.horizon
    df = load_hourly(ROOT / args.data)
    paths, t_pred, t_end = real_windows(df, L, H)
    out_dir = ROOT / "reports" / "tokdit"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, preds = [], []
    for year in range(args.years[0], args.years[1] + 1):
        start, end = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        tr = t_end < start                           # горизонт окна обучения целиком до теста
        if args.train_years:
            tr &= t_pred >= np.datetime64(f"{year - args.train_years}-01-01")
        te = np.flatnonzero((t_pred >= start) & (t_pred < end))[:: args.test_stride]
        if not len(te):
            continue
        trp = paths[tr]
        n_real = len(trp)
        if args.synthetic:
            n_syn = int(len(trp) * args.synthetic_ratio)
            syn = synthetic_paths(args.synthetic, n_syn, L, H, seed=args.seed + year)
            # синтетике — масштаб случайного реального окна train (std шага контекста), чтобы признак
            # log std и значения TokDiT были в том же диапазоне; реальные пути не меняются
            sd = np.diff(trp[:, :L], axis=1).std(axis=1)
            syn = syn * np.random.default_rng(args.seed + year).choice(sd, len(syn))[:, None].astype(np.float32)
            # при --synthetic-only реальные окна train идут только в раннюю остановку CB/MLP; TokDiT их не видит
            trp = np.concatenate([trp, syn])
            print(f"  синтетика {args.synthetic}: {len(syn)} путей, обучение на {len(trp)}", flush=True)
        te_paths = paths[te]
        ret = te_paths[:, -1].astype(np.float64)
        print(f"[{args.name}] {year}: обучение {len(trp)} окон, тест {len(te)}", flush=True)
        # порядок обучающих окон — хронологический (реальные), для ранней остановки CB/MLP это holdout = конец
        for mname in args.models:
            t0 = time.time()
            fit_idx, val_idx = split_fit_val(n_real, len(trp), gap=L + H)
            if args.synthetic_only:
                fit_idx = np.arange(n_real, len(trp))
            if mname == "tokdit":
                model = fit_tokdit(trp[n_real:] if args.synthetic_only else trp, L, H, args, args.seed)
                p_up, exp_ret = predict_tokdit(model, te_paths, L, H, args)
            elif mname == "catboost":
                Xtr = tabular(trp, L)
                m = fit_catboost(Xtr, target_up(trp), args.seed, fit_idx, val_idx)
                p_up = m.predict_proba(tabular(te_paths, L))[:, 1]
                exp_ret = p_up - 0.5
            elif mname == "mlp":
                predict = fit_mlp(tabular(trp, L), target_up(trp), args.seed, fit_idx, val_idx)
                p_up = predict(tabular(te_paths, L))
                exp_ret = p_up - 0.5
            else:
                raise ValueError(mname)
            # при прореживании теста одна позиция занимает ceil(H / stride) прогнозов
            row = summarize(mname, str(year), p_up, exp_ret, ret, -(-H // args.test_stride), args.tops)
            row["сек"] = round(time.time() - t0)
            row["p_std"] = float(np.std(p_up))
            rows.append(row)
            preds.append(pd.DataFrame({"модель": mname, "год": year, "time": t_pred[te], "p_up": p_up,
                                       "exp_ret": exp_ret, "ret": ret}))
            print("  " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                  flush=True)
        pd.DataFrame(rows).to_csv(out_dir / f"{args.name}.csv", index=False)
        pd.concat(preds).to_csv(out_dir / f"{args.name}_preds.csv", index=False)

    table = pd.DataFrame(rows)
    allp = pd.concat(preds)
    print(f"\nИТОГ · {args.name}")
    print(table.round(4).to_string(index=False))
    for mname, g in allp.groupby("модель"):
        # сделки по годам по отдельности (без перекрытия внутри года), затем вместе
        s = summarize(mname, "все", g["p_up"].to_numpy(), g["exp_ret"].to_numpy(), g["ret"].to_numpy(),
                      -(-H // args.test_stride), args.tops)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
