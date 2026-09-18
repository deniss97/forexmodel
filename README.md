# forexmodel

ML-пайплайн для внутридневной торговли одной бумагой (по умолчанию LKOH):

```
минутные котировки → часовые бары → признаки → triple-barrier разметка
        → CatBoost (primary) + CNN-BiLSTM → мета-модель (фильтр поздних входов)
        → минутная симуляция сделок → отчёт и диагностика
```

Это переработанная версия ноутбука `notebooks/ML_ForexModelGPT_pipe6_clear_new.ipynb`:
59 ячеек разложены по модулям, а найденные в разборе утечки и баги исправлены
(таблица «Что изменилось» ниже). Ноутбук и файл патчей оставлены в `notebooks/`
как справочный материал — рабочий код в них больше не живёт.

**Результаты прогонов и план дальнейших шагов — в [RESULTS.md](RESULTS.md).**
Прогнано на двух инструментах: LKOH (комиссия 0.15%) и серебро XAGUSD (0.04%).
На LKOH преимущество модели (+0.02…+0.05% на сделку до издержек) в разы меньше
комиссии, и фильтр ожидаемой ценности вынужденно сжимает выборку до 3–5 сделок.
На серебре, где комиссия составляет 2.3% целевого движения вместо 26.7%, этот
фильтр перестаёт отсекать сигналы и сделок становятся сотни — но прибыли это
не приносит: за 2025 год результат до комиссии равен −0.038%. Разбор и что
делать дальше — там же.

Конфигурации: [configs/default.yaml](configs/default.yaml) (LKOH),
[configs/silver.yaml](configs/silver.yaml) (серебро).

---

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
pip install -e .              # опционально: чтобы работал импорт из любой папки
```

Положите минутный CSV (колонки `begin,open,high,low,close,value`) в `data/raw/`
и укажите путь в `configs/default.yaml` → `data.csv_path`.

`pandas_ta`, `MetaTrader5` и `optuna` больше не нужны: ADX/DI считаются внутри
`forexmodel/features/indicators.py`.

## Быстрый старт

```bash
# 1. проверить данные, признаки и нарезку выборок (без обучения)
python -m forexmodel.cli data -c configs/default.yaml

# 2. обучить primary + NN + мета-модель, сохранить в artifacts/<run_name>/
python -m forexmodel.cli train -c configs/default.yaml

# 3. прогнать бэктест на sim-периоде, сохранить сделки и отчёт в reports/<run_name>/
python -m forexmodel.cli backtest -c configs/default.yaml --split sim

# всё сразу
python -m forexmodel.cli run -c configs/default.yaml

# перебор параметров
python -m forexmodel.cli sweep -c configs/default.yaml \
    -p labeling.horizon=8,10,15 -p labeling.tp_atr=1.25,1.5,2.0

# график сделок (HTML)
python -m forexmodel.cli chart -c configs/default.yaml --split sim
```

Любой параметр переопределяется из командной строки без правки YAML:

```bash
python -m forexmodel.cli train -c configs/default.yaml --set catboost.depth=4 --set nn.enabled=false
```

Эквивалентные точки входа — `python scripts/train.py`, `scripts/backtest.py`,
`scripts/diagnose.py` (для тех, кому удобнее запускать файл, а не модуль).

Пороги входа и правила выхода к обучению отношения не имеют, поэтому перебираются
на готовых моделях за секунды — без `sweep`, который переобучает всё заново:

```bash
# пороги мета-модели, EV-фильтр, тренд-фильтр
python scripts/compare_filters.py -c configs/default.yaml --thresholds 0.55 0.50 0.45

# правила выхода: ширина стопа, момент включения трейлинга, горизонт удержания
python scripts/compare_exits.py -c configs/silver.yaml --set simulation.signal_source=cb
python scripts/compare_exits.py -c configs/silver.yaml --sl-sweep 0.75 1 1.5 2 3
```

Из ноутбука/REPL:

```python
from forexmodel import load_config, build_dataset, run_training, run_backtest

cfg = load_config("configs/default.yaml")
ds = build_dataset(cfg)
trained = run_training(cfg, dataset=ds)
result = run_backtest(cfg, dataset=ds, primary=trained.primary,
                      nn_model=trained.nn, meta_model=trained.meta)
result.report
```

Готовый пример — `notebooks/pipeline_demo.ipynb`.

---

## Что смотреть после прогона

Каждая команда в конце печатает блок «ГОТОВО» со путями ко всем файлам, так что
искать вручную не нужно. Раскладка такая:

```
reports/<run_name>/
  chart_<split>.html        свечи + фон тренда + все сделки (plotly, интерактив)
  equity_<split>.html       кривая капитала, просадка, распределение PnL,
                            суммарный PnL по причинам выхода
  trades_worst_<split>.html разбор 12 худших сделок: панель на сделку с минутным
  trades_best_<split>.html  путём цены, линией входа, уровнем стопа и уровнем
                            включения трейлинга — видно, выбило ли откатом
                            или движение реально пошло против
  summary_<split>.txt       вся сводка текстом: метрики, диагностика, примеры сделок
  trades_<split>.csv        каждая сделка: вход/выход/PnL/причина выхода
  report_<split>.csv        метрики прогона одной строкой (удобно склеивать прогоны)
  signals_<split>.csv       бары с вероятностями моделей и итоговым сигналом
  diag_late_entry_<split>.csv   PnL в разрезе растяжения на входе (ATR)
  diag_exit_reasons_<split>.csv разбивка по причинам выхода
  logs/<команда>_<дата>.log     полный лог прогона

artifacts/<run_name>/
  primary_catboost.pkl / nn_cnn_bilstm.pkl / meta_catboost.pkl
  features.json, metrics.json, config.yaml   чем именно обучалось
  train_summary.txt         итог обучения + время по этапам
  timings_*.json            тайминги этапов -> ETA следующего прогона
```

Графики строятся автоматически в конце `backtest`/`run`. Команда
`python -m forexmodel.cli chart -c configs/default.yaml --split sim`
нужна, только если хочется перерисовать их по уже сохранённым CSV.

## Логи: статус и «сколько ещё ждать»

Лог пишется и в консоль, и в файл (`reports/<run_name>/logs/`) — отключается
флагом `--no-log-file`, путь переопределяется `--log-file`. В каждой строке есть
время от старта процесса, поэтому по логу видно, куда ушло время:

```
04:31:26 | +00:19    | INFO | forexmodel.models.nn_cnn_bilstm | Устройство: cpu | параметров: 166 211 | батч 128 -> 168 батчей на эпоху
```

Долгие шаги сообщают прогресс и ETA, а не молчат до конца:

```
==============================================================================
  ЭТАП 3/6 · OOF-предсказания
  прошло всего 29с | в прошлый прогон этот этап занял 4м 10с | до конца ~6м 30с
==============================================================================
  фолд 2/5:  43.2% [######........] 864/2000 дер | прошло 18с | осталось ≤24с | 47.1 дер/с | learn:MultiClass=1.0741
CNN-BiLSTM:  45.0% [######........] 18/40 эпох | прошло 2м 54с | осталось ≤3м 33с | train_loss=1.0905 | val_loss=1.0964 | без улучшения 1/5
```

* `≤` вместо `~` там, где работает early stopping: общее число итераций — это
  верхняя граница, и обещать точное время нельзя;
* ETA целого прогона берётся из `timings_*.json` прошлого запуска — со второго
  раза в начале каждого этапа видно, сколько осталось до конца;
* в конце обучения и бэктеста печатается таблица «ВРЕМЯ ПО ЭТАПАМ» — сразу
  понятно, что именно тормозит;
* прогресс CatBoost идёт через callback в логгер, а не в stdout, поэтому он
  попадает и в файл лога (`catboost.verbose` задаёт шаг в деревьях, время между
  строками ограничено сверху отдельно — простыни на 2000 итераций не будет).

---

## Структура

```
configs/
  default.yaml              рабочая конфигурация (ATR-разметка, трейлинг, мета-модель)
  baseline_notebook.yaml    конфигурация «как в ноутбуке» — точка отсчёта для сравнения
forexmodel/
  config.py                 типизированный конфиг + валидация (опечатки падают сразу)
  logging_utils.py          логирование вместо print
  data/
    loader.py               чтение CSV, ресемплинг (объём сохраняется)
    splits.py               train/test/sim, проверка непересечения, embargo
  features/
    indicators.py           SMA/EMA/ATR/RSI/MACD/BB/ADX (свои, без pandas_ta)
    technical.py            индикаторы + их безразмерные производные
    extension.py            признаки «где мы внутри движения» (растяжение, ER, объём)
    trend.py                тренд-фильтр 4H: early (опережающий) и confirm — оба каузальные
    selection.py            отбор признаков, отсев абсолютных ценовых уровней
    builder.py              сборка всех признаков на непрерывном ряде
  labeling/
    core.py                 быстрый поиск первого касания по минутным барам
    first_touch.py          симметричная разметка ±threshold (как в ноутбуке)
    atr_barriers.py         асимметричная triple-barrier разметка в ATR (рабочая)
    uniqueness.py           веса де Прадо для пересекающихся меток
  models/
    cv.py                   purged walk-forward и хронологический split с embargo
    catboost_primary.py     primary-модель (3 класса) + инференс с полными вероятностями
    nn_cnn_bilstm.py        CNN-BiLSTM с early stopping и корректным выравниванием окон
    meta.py                 OOF → мета-метки через симуляцию → мета-модель → фильтр
    persistence.py          сохранение/загрузка бандлов (модель + фичи + конфиг)
  simulation/
    signals.py              ансамбль, пороги, фильтр по ожидаемой ценности
    exits.py                фиксированные барьеры и трейлинг
    simulator.py            минутная симуляция
    report.py               сводка по сделкам
  evaluation/
    metrics.py              полярная точность, корзины уверенности
    diagnostics.py          диагностика поздних входов, проверка выборок
  viz/charts.py             HTML-график: свечи + фон тренда + сделки
  pipelines/                dataset → train → backtest → sweep
  cli.py                    командный интерфейс
tests/                      pytest: утечки, разметка, симулятор, конфиг
```

Артефакты обучения пишутся в `artifacts/<run_name>/`, результаты бэктеста —
в `reports/<run_name>/`; обе папки в `.gitignore`.

---

## Что изменилось относительно ноутбука

### Утечки и «околоутечки»

| Проблема в ноутбуке | Где исправлено |
|---|---|
| `rsi_z` нормировался по mean/std **всей** выборки | `features/technical.py` — rolling-окно `features.rsi_z_window` |
| `scaler.fit_transform` до train/val split в NN | `models/nn_cnn_bilstm.py` — scaler фитится только на train |
| holdout CatBoost резался встык (метки смотрят вперёд) | `models/cv.py::chronological_split` + `embargo` |
| мета-модель обучалась с `eval_set=(X_hold, y_hold)` и на нём же отчитывалась | `models/meta.py` — отдельный val для early stopping |
| `add_trend_filter_enhanced` мёрджил 4H/daily по началу бина | функция удалена; остались только каузальные фильтры в `features/trend.py` |
| `df_min_test` целиком лежал внутри `df_min_train` | `data/splits.py` — пересечение выборок падает при загрузке конфига |
| абсолютные уровни цены (`close_lag_*`, `sma_*`, `bb_up`, ...) в признаках | `features/selection.py` — `dimensionless_only: true` |
| признаки считались на каждой выборке отдельно (warm-up/NaN в начале sim) | `pipelines/dataset.py` — фичи на непрерывном ряде, нарезка после |

### Баги, портившие результат симуляции

| Проблема | Где исправлено |
|---|---|
| мета-модель не подключена: `final_signal` не использовался симулятором | `simulation/signals.py` — режим `signal_source: meta` |
| барьеры симуляции не совпадали с барьерами разметки (комиссия в цене входа, TP реально +0.65% вместо +0.5%, комиссия на выходе не бралась) | `simulation/simulator.py` — барьеры от сырой цены, комиссия за круг вычитается из PnL |
| NN обучалась с `seq_len=20`, а инференс шёл с `seq_len=10` | `seq_len` хранится в бандле модели |
| окно NN не включало текущий бар (отставание на час от CatBoost) | `make_sequences`: `data[i-seq_len+1 : i+1]` |
| NN переобучалась с первой эпохи, бралась модель последней | early stopping + восстановление лучших весов |
| при касании TP и SL в одном баре побеждал TP | `simulation/exits.py` — побеждает стоп (консервативно) |

### Ответ на «вход на излёте»

1. **Асимметричные ATR-барьеры** (`labeling/atr_barriers.py`, `labeling.mode: atr_asym`).
   При `tp_atr > sl_atr` бар на излёте движения не успевает дойти до цели раньше
   стопа и получает класс 1 или противоположный — модель вынуждена отличать
   раннюю фазу от поздней. При симметричных ±0.5% эта информация из данных
   не извлекается в принципе.
2. **Признаки растяжения** (`features/extension.py`): `ext_from_low/high_*`,
   `runup_*`, efficiency ratio Кауфмана, `candle_streak`, «возраст» 24-барного
   экстремума, ранг `bb_width`, **относительный объём** (в ноутбуке объём терялся
   дважды — при чтении CSV и при ресемплинге).
3. **Опережающий тренд-фильтр** (`features/trend.py`, `trend.mode: early`):
   наклон регрессии 4H в ATR + ADX в коридоре `adx_min < ADX < adx_max` и растущий.
   Верхняя граница по ADX — прямой запрет входа в перегретый тренд.
   `trend_age_4h` отдаётся модели как признак.
4. **Мета-модель как детектор поздних входов** (`models/meta.py`): в её признаки
   обязательно включаются признаки растяжения.
5. **Трейлинг вместо фиксированного TP** (`simulation/exits.py`,
   `simulation.exit_mode: trailing`): стоп 1 ATR, после хода на 1 ATR — трейл
   1.5 ATR от лучшей цены.
6. **Сначала измерить**: `evaluation/diagnostics.py::late_entry_diagnostic`
   группирует PnL по растяжению на момент входа (`python scripts/diagnose.py`).
   Если winrate падает с ростом растяжения — включите
   `simulation.max_extension_atr: 1.5` ещё до переобучения.

### Прочее

* веса уникальности де Прадо для пересекающихся меток (`labeling/uniqueness.py`);
* CatBoost: early stopping, `l2_leaf_reg`, `depth` 8 → 6;
* вход по ожидаемой ценности `p*TP − (1−p)*SL − комиссия > 0` вместо порога
  «уверенность ≥ 0.5» (`simulation/signals.py::expected_value`);
* метрика `polar_edge` = полярная точность минус база «угадай по частоте»:
  именно она показывает, есть ли у модели преимущество (в ноутбуке было ~0);
* минутные циклы по `iterrows()` заменены на numpy-срезы — разметка и симуляция
  считаются на порядки быстрее.

---

## Как отлаживать и расширять

* **Добавить признак** → `features/technical.py` или `features/extension.py`.
  Если он в абсолютных ценовых единицах — внесите его имя в
  `features/selection.py::absolute_price_columns`, иначе он утечёт в модель как
  маркер эпохи. Проверка: `pytest tests/test_leakage.py`.
* **Добавить правило выхода** → `simulation/exits.py` + ветка в
  `simulation/simulator.py::_resolve_exit` + значение в `SimulationConfig.exit_mode`.
* **Добавить правило входа** → `simulation/signals.py::build_signal_column`.
* **Поменять разметку** → `labeling/`, не забыв согласовать барьеры с
  `simulation.exit_mode` (модель должна учиться на тех же уровнях, на которых торгует).
* **Сравнить со старым пайплайном** → прогоните `configs/baseline_notebook.yaml`
  и `configs/default.yaml` на одних данных.
* Логи: `--log-level DEBUG --log-file reports/run.log`.

## Тесты

```bash
pytest -q
```

Покрыты: каузальность признаков и тренд-фильтров (сравнение расчёта на полном
ряде и на префиксе), корректность разметки на вручную заданных траекториях,
веса уникальности, барьеры/комиссия/трейлинг в симуляторе, правила ансамбля и
фильтр ожидаемой ценности, валидация конфига и непересечение выборок.

## Ограничения

* Проверялись только синтетические данные: на реальном CSV пайплайн запускается
  пользователем (в окружении, где стоят catboost/torch).
* `data.base_timeframe` и `simulation.open_delay_minutes` нужно менять
  согласованно: задержка входа должна равняться длине бара + 1 минута, если
  `labeling.use_next_open: true`.
* Комиссия задаётся одним числом за круг; проскальзывание не моделируется.
