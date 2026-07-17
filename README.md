# VisionGuard AI — продуктовый трек Detection

Единое веб-приложение для трёх задач: повреждения ЛЭП, переломы на X-ray и
автотранспорт на аэроснимках. Для каждой задачи пользователь выбирает быстрый
или точный профиль и получает bounding boxes, классы, уверенность и размеченный файл.

## Каталог задач и моделей

| Задача | Датасет | Быстрая модель | Точная модель |
|---|---|---|---|
| Повреждения ЛЭП | Реестр ФГАУ ЦИТ | `powerline_yolo11n.pt` | `powerline_yolo11s.pt` |
| Переломы | Bone Fracture Detection | `fracture_yolov8_fast.pt` | `fracture_yolov8l.pt` |
| Автотранспорт | VisDrone / Open Images V7 | `vehicle_yolo11n.pt` | `vehicle_yolo11s.pt` |

Команды обучения находятся в `scripts/train.py`, конфигурации — в `configs/`,
а notebook с шестью отдельными ячейками запуска — в
`notebooks/train_product_tracks.ipynb`.

## Результат

- лучшая представленная модель: **mAP@0.5 = 0.902**;
- общий максимум **F1 = 0.87** при `confidence = 0.439`;
- 8 классов объектов ЛЭП;
- шесть checkpoints: YOLO11n и YOLO11s отдельно для каждой из трёх задач;
- FastAPI backend и Streamlit frontend;
- локальный запуск, Docker и конфигурация для Streamlit Community Cloud;
- notebook с двумя отдельными ячейками обучения.

Приложенные метрики относятся к предоставленному запуску лучшей модели. Они не
переносятся автоматически на fallback-веса или на вторую модель.

## Архитектура

```text
Пользователь → Streamlit → FastAPI → PredictionService → YOLO11n / YOLO11s
                              ↓
                 JSON + изображение с разметкой
```

Streamlit умеет работать и в embedded-режиме без отдельного HTTP-процесса — это
удобно для бесплатного облачного размещения. Наличие и готовность FastAPI можно
проверить в `/docs` и `/health`.

## Быстрый старт

Требуется Python 3.10–3.12.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Терминал 1 — backend:

```bash
python run_api.py
```

Терминал 2 — frontend через API:

```bash
$env:POWERGUARD_API_URL="http://localhost:8000"
streamlit run app.py
```

Или один процесс в embedded-режиме:

```bash
streamlit run app.py
```

API: `http://localhost:8000/docs`, UI: `http://localhost:8501`.

## Веса

Итоговые файлы не хранятся в Git из-за размера. После обучения положите их сюда:

```text
models/powerline_yolo11n.pt
models/powerline_yolo11s.pt
models/fracture_yolov8_fast.pt
models/fracture_yolov8l.pt
models/vehicle_yolo11n.pt
models/vehicle_yolo11s.pt
```

Пути можно переопределить:

```bash
$env:POWERLINE_FAST_MODEL_PATH="C:\weights\powerline_fast.pt"
$env:FRACTURE_ACCURATE_MODEL_PATH="C:\weights\fracture_accurate.pt"
$env:VEHICLE_FAST_MODEL_PATH="C:\weights\vehicle_fast.pt"
```

Если custom-весов нет, по умолчанию загружаются базовые `yolo11n.pt` и
`yolo11s.pt`. Это **только проверка приложения**: COCO-классы не совпадают с
классами ЛЭП. Для строгого режима задайте
`ALLOW_PRETRAINED_FALLBACK=false` — тогда API честно сообщит об отсутствующих
весах.

## Обучение

Откройте [notebooks/train_models.ipynb](notebooks/train_models.ipynb) в Colab,
Kaggle или Jupyter. Notebook содержит:

1. аудит структуры и распределения классов;
2. отдельную ячейку обучения YOLO11n;
3. отдельную ячейку обучения YOLO11s;
4. валидацию, сравнительную таблицу и экспорт весов.

Ожидаемая структура датасета Ultralytics:

```text
data/lep/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

В обсуждении проекта отмечалась неполная распаковка исходного архива: вместо
заявленных 7 988 изображений у части участников получалось около 5 042. Поэтому
до обучения обязательно сравните фактическое число файлов, проверьте пары
image/label и зафиксируйте версию датасета в отчёте.

## Метрики и анализ

Для object detection выбраны:

- `mAP@0.5` — основная метрика критерия проекта и интегральная оценка качества;
- `mAP@0.5:0.95` — более строгая оценка локализации;
- precision/recall — цена ложных тревог и пропусков;
- F1-confidence — выбор рабочего порога интерфейса;
- latency — сравнение быстрого и точного профилей.

По PR-кривой средний AP@0.5 равен 0.902. Сильные классы: `bad_insulator`
(0.979), `safety_sign+` (0.977), `nest` (0.948). Слабее распознаются
`polymer_insulators` (0.771) и `vibration_damper` (0.792). Нормализованная
матрица показывает заметные пропуски в background для виброгасителей (0.32) и
гнёзд (0.22). Вероятная причина — малый размер объекта, сложный фон и дисбаланс.
Следующие эксперименты: oversampling слабых классов, crop/tiling (SAHI),
увеличение разрешения, hard-negative mining и проверка разбиения на утечки.

Исходные графики находятся в `assets/metrics/`.

## API

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@tower.jpg" \
  -F "model_id=accurate" \
  -F "confidence=0.44" \
  -F "iou=0.45"
```

Endpoints:

- `GET /health` — состояние сервиса;
- `GET /api/v1/tasks` — три задачи, классы и ссылки на датасеты;
- `GET /api/v1/models?task_id=vehicle` — модели задачи и источник весов;
- `POST /api/v1/predict` — инференс изображения до 20 МБ.

## Docker

```bash
docker compose up --build
```

Сервисы: UI на `8501`, API на `8000`. Смонтируйте свои `.pt` в папку `models/`.

## Проверки

```bash
pip install -r requirements-dev.txt
pytest -q
python -m compileall detection_app app.py run_api.py
```

## Размещение

Для Streamlit Community Cloud выберите `app.py`, Python 3.11 и добавьте веса
через разрешённое внешнее хранилище или Git LFS. При одном процессе оставьте
`POWERGUARD_API_URL` пустым. Для полноценного раздельного контура разверните API
на Render/Railway/VPS и задайте URL в секретах окружения Streamlit.

Перед публикацией замените шаблонные контакты и ссылки в
`docs/submission_checklist.md`, запишите короткую демонстрацию по сценарию из
`docs/presentation_script.md` и проверьте лицензию датасета/весов.

## Ограничения

MVP не является системой промышленной безопасности и не заменяет осмотр
инженером. Нужны полевые испытания, мониторинг дрейфа, журналирование версий
модели и ручная проверка критических находок.
