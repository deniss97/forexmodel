"""CNN + BiLSTM на последовательностях признаков.

Исправленные баги из ноутбука:

  1. `scaler.fit_transform` вызывался ДО train/val split — валидация участвовала
     в скейлинге, и её отчёт был оптимистичен. Теперь scaler фитится только на
     train-части.
  2. `create_sequences` брал окно `data[i:i+seq_len]` с меткой `labels[i+seq_len]`,
     то есть модель НЕ видела текущий бар и отставала на час относительно
     CatBoost. В ансамбле «agreement» это сдвигало момент согласия ещё позже.
     Теперь окно заканчивается на баре метки: `data[i-seq_len+1 : i+1]`.
  3. Обучение шло фиксированные 20 эпох без early stopping, а переобучение
     начиналось с первой (ValLoss 0.94 -> 1.13) — забиралась модель последней
     эпохи. Теперь early stopping с восстановлением лучших весов.
  4. Обучение шло с seq_len=20, а инференс вызывался с seq_len=10. Теперь
     seq_len хранится внутри бандла и на инференсе не задаётся вручную.
  5. Добавлены веса классов (разметка несбалансирована) и фиксация сидов.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import NNConfig
from ..evaluation.metrics import classification_summary
from ..features.selection import assert_features_present
from ..logging_utils import get_logger, section
from ..progress import Progress, fmt_duration

log = get_logger(__name__)

__all__ = ["CNNBiLSTM", "NNModel", "train_nn", "predict_nn", "make_sequences"]


def _torch():
    import torch

    return torch


def _build_module_cls():
    """nn.Module определяется лениво, чтобы импорт пакета не требовал torch."""
    import torch.nn as nn

    class CNNBiLSTM(nn.Module):
        def __init__(self, input_dim: int, num_classes: int, cfg: NNConfig):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, cfg.conv_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(cfg.conv_channels)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(
                input_size=cfg.conv_channels,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            import torch

            x = x.permute(0, 2, 1)
            x = self.relu(self.bn1(self.conv1(x)))
            x = x.permute(0, 2, 1)
            _, (h_n, _) = self.lstm(x)
            h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
            return self.fc(h_cat)

    return CNNBiLSTM


CNNBiLSTM = None  # заполняется при первом обучении/загрузке (см. _module_cls)


def _module_cls():
    global CNNBiLSTM
    if CNNBiLSTM is None:
        CNNBiLSTM = _build_module_cls()
    return CNNBiLSTM


@dataclass
class NNModel:
    state_dict: Dict[str, Any]
    scaler: Any
    features: List[str]
    label_map: Dict[int, int]
    seq_len: int
    cfg: NNConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    _model: Any = None

    @property
    def inv_label_map(self) -> Dict[int, int]:
        return {v: k for k, v in self.label_map.items()}

    def module(self):
        """Собирает torch-модуль из state_dict (лениво, один раз)."""
        if self._model is None:
            cls = _module_cls()
            model = cls(input_dim=len(self.features), num_classes=len(self.label_map), cfg=self.cfg)
            model.load_state_dict(self.state_dict)
            model.eval()
            self._model = model
        return self._model


def make_sequences(
    data: np.ndarray, seq_len: int, labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Окна, ЗАКАНЧИВАЮЩИЕСЯ на баре метки (модель видит текущий бар).

    Возвращает (X, y, positions), где positions — индексы баров, к которым
    относится предсказание.
    """
    n = len(data)
    if n < seq_len:
        raise ValueError(f"Недостаточно данных: {n} строк при seq_len={seq_len}")

    positions = np.arange(seq_len - 1, n)
    # sliding_window_view даёт (n-seq_len+1, features, seq_len) без копирования
    windows = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
    X = np.ascontiguousarray(windows.transpose(0, 2, 1))
    y = labels[positions] if labels is not None else None
    return X, y, positions


def _device(cfg: NNConfig):
    torch = _torch()
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def train_nn(
    df: pd.DataFrame,
    features: List[str],
    cfg: NNConfig,
    embargo: int = 0,
    target_col: str = "label",
) -> NNModel:
    torch = _torch()
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    assert_features_present(df, features)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    df = df.reset_index(drop=True)
    labels_unique = sorted(df[target_col].dropna().unique())
    label_map = {int(v): i for i, v in enumerate(labels_unique)}
    y_enc = df[target_col].map(label_map).to_numpy()

    raw = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    # --- split ДО скейлинга ---
    n = len(df)
    n_val = int(n * cfg.test_size)
    train_end = n - n_val - embargo
    if train_end <= cfg.seq_len:
        raise ValueError(f"Недостаточно данных для seq_len={cfg.seq_len} (train={train_end})")

    scaler = StandardScaler().fit(raw[:train_end])
    scaled = scaler.transform(raw).astype(np.float32)

    section(log, "CNN-BiLSTM: подготовка")
    X_tr, y_tr, _ = make_sequences(scaled[:train_end], cfg.seq_len, y_enc[:train_end])
    X_va, y_va, _ = make_sequences(scaled[n - n_val - cfg.seq_len + 1 :], cfg.seq_len, y_enc[n - n_val - cfg.seq_len + 1 :])
    log.info(
        "Последовательностей: train=%d | val=%d | seq_len=%d | признаков=%d",
        len(X_tr),
        len(X_va),
        cfg.seq_len,
        len(features),
    )

    device = _device(cfg)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
    yv = torch.tensor(y_va, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg.batch_size, shuffle=True)

    model = _module_cls()(input_dim=len(features), num_classes=len(label_map), cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Устройство: %s | параметров: %s | батч %d -> %d батчей на эпоху | эпох максимум %d (patience %d)",
        device,
        f"{n_params:,}".replace(",", " "),
        cfg.batch_size,
        len(loader),
        cfg.epochs,
        cfg.early_stopping_patience,
    )

    counts = Counter(y_tr.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(label_map) * counts.get(i, 1)) for i in range(len(label_map))], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=2)

    best_loss, best_state, bad_epochs, best_epoch = np.inf, None, 0, 0

    section(log, f"CNN-BiLSTM: обучение, до {cfg.epochs} эпох")
    epochs_bar = Progress(cfg.epochs, label="CNN-BiLSTM", unit="эпох", logger=log, min_interval=0.0, eta_prefix="≤")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        # прогресс внутри эпохи: на CPU одна эпоха идёт десятки секунд, и без
        # этого лог молчит так, что прогон не отличить от зависшего
        batches = Progress(
            len(loader),
            label=f"  эпоха {epoch + 1}/{cfg.epochs}",
            unit="батч",
            logger=log,
            min_interval=20.0,
        )
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            batches.update(note=f"loss={loss.item():.4f}")
        train_loss /= max(len(Xt), 1)

        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = criterion(logits, yv).item()
            val_pred = logits.argmax(dim=1).cpu().numpy()

        scheduler.step(val_loss)
        acc = float((val_pred == y_va).mean())

        improved = val_loss < best_loss - 1e-5
        if improved:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        epochs_bar.set(
            epoch + 1,
            "train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | %s | lr=%.2g"
            % (
                train_loss,
                val_loss,
                acc,
                "лучшая" if improved else f"без улучшения {bad_epochs}/{cfg.early_stopping_patience}",
                optimizer.param_groups[0]["lr"],
            ),
        )

        if bad_epochs >= cfg.early_stopping_patience:
            log.info(
                "Early stopping: %d эпох без улучшения. Лучший val_loss=%.4f на эпохе %d",
                bad_epochs,
                best_loss,
                best_epoch,
            )
            break

    epochs_bar.finish(f"лучший val_loss={best_loss:.4f} на эпохе {best_epoch}")
    log.info("Среднее время на эпоху: %s", fmt_duration(epochs_bar.elapsed / max(epochs_bar.done_count, 1)))

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Восстановлены веса лучшей эпохи (%d), а не последней", best_epoch)

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).argmax(dim=1).cpu().numpy()

    inv = {v: k for k, v in label_map.items()}
    metrics = classification_summary(
        y_true=np.array([inv[i] for i in y_va]),
        y_pred=np.array([inv[i] for i in val_pred]),
        title="CNN-BiLSTM val",
    )

    return NNModel(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        scaler=scaler,
        features=list(features),
        label_map=label_map,
        seq_len=cfg.seq_len,
        cfg=cfg,
        metrics=metrics,
    )


def predict_nn(bundle: NNModel, df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Предсказания NN. seq_len берётся из бандла — рассинхрон обучения и
    инференса (20 против 10 в ноутбуке) теперь невозможен."""
    torch = _torch()
    assert_features_present(df, bundle.features)

    raw = df[bundle.features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scaled = bundle.scaler.transform(raw).astype(np.float32)
    X, _, positions = make_sequences(scaled, bundle.seq_len)

    model = bundle.module()
    device = _device(bundle.cfg)
    model = model.to(device)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()

    inv = bundle.inv_label_map
    out = pd.DataFrame({time_col: df[time_col].to_numpy()[positions]})
    for idx, label in inv.items():
        out[f"proba_{label}"] = proba[:, idx]
    out["y_pred"] = [inv[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1)
    return out
