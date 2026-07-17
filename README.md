# Detection Product Track

A unified web application for three computer vision tasks: power line damage detection, bone fracture detection in X-ray images, and vehicle detection in aerial imagery.

For each task, the user can select either a fast or an accurate model profile and receive bounding boxes, predicted classes, confidence scores, and an annotated output image.

## Task and Model Catalog

| Task | Dataset | Fast Model | Accurate Model |
|---|---|---|---|
| Power Line Damage Detection | FGAU CIT Registry | `powerline_yolo11n.pt` | `powerline_yolo11s.pt` |
| Bone Fracture Detection | Bone Fracture Detection | `fracture_yolov8_fast.pt` | `fracture_yolov8l.pt` |
| Vehicle Detection | VisDrone / Open Images V7 | `vehicle_yolo11n.pt` | `vehicle_yolo11s.pt` |

Training commands are available in `scripts/train.py`, configuration files are stored in `configs/`, and the notebook with six separate training cells is located at:

```text
notebooks/train_product_tracks.ipynb
```

## Results

- Best presented model: **mAP@0.5 = 0.902**
- Maximum overall **F1 score = 0.87** at `confidence = 0.439`
- 8 power line object classes
- Six checkpoints: separate YOLO11n and YOLO11s models for each of the three tasks
- FastAPI backend and Streamlit frontend
- Support for local execution, Docker, and Streamlit Community Cloud deployment
- Notebook with separate training cells for each model configuration

The included metrics correspond to the provided run of the best-performing model. They do not automatically apply to fallback weights or to the alternative model profile.

## Architecture

```text
User → Streamlit → FastAPI → PredictionService → YOLO11n / YOLO11s
                             ↓
                  JSON + annotated image
```

Streamlit can also operate in embedded mode without a separate HTTP backend process. This is convenient for free cloud deployment.

FastAPI availability and service status can be checked through:

```text
/docs
/health
```

## Quick Start

Python 3.10–3.12 is required.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Terminal 1 — start the backend:

```bash
python run_api.py
```

Terminal 2 — start the frontend through the API:

```bash
$env:POWERGUARD_API_URL="http://localhost:8000"
streamlit run app.py
```

Alternatively, run the application as a single process in embedded mode:

```bash
streamlit run app.py
```

API documentation:

```text
http://localhost:8000/docs
```

User interface:

```text
http://localhost:8501
```

## Model Weights

Final model weight files are not stored in Git because of their size. After training, place them in the following locations:

```text
models/powerline_yolo11n.pt
models/powerline_yolo11s.pt
models/fracture_yolov8_fast.pt
models/fracture_yolov8l.pt
models/vehicle_yolo11n.pt
models/vehicle_yolo11s.pt
```

The model paths can be overridden through environment variables:

```bash
$env:POWERLINE_FAST_MODEL_PATH="C:\weights\powerline_fast.pt"
$env:FRACTURE_ACCURATE_MODEL_PATH="C:\weights\fracture_accurate.pt"
$env:VEHICLE_FAST_MODEL_PATH="C:\weights\vehicle_fast.pt"
```

If custom weights are unavailable, the application downloads the base `yolo11n.pt` and `yolo11s.pt` models by default.

This fallback is intended only for testing the application. COCO classes do not match the power line inspection classes.

For strict mode, set:

```bash
$env:ALLOW_PRETRAINED_FALLBACK="false"
```

In strict mode, the API reports missing custom weights instead of loading pretrained fallback models.

## Metrics and Analysis

The following metrics were selected for object detection evaluation:

- `mAP@0.5` — the main project metric and an overall measure of detection quality
- `mAP@0.5:0.95` — a stricter measure of localization accuracy
- Precision and recall — evaluation of false alarms and missed detections
- F1-confidence curve — selection of the operating confidence threshold
- Latency — comparison between the fast and accurate model profiles

According to the precision-recall curve, the mean AP@0.5 is 0.902.

The strongest classes are:

- `bad_insulator`: 0.979
- `safety_sign+`: 0.977
- `nest`: 0.948

The weaker-performing classes are:

- `polymer_insulators`: 0.771
- `vibration_damper`: 0.792

The normalized confusion matrix shows a noticeable number of missed detections assigned to the background class for vibration dampers, with a value of 0.32, and nests, with a value of 0.22.

Possible causes include small object size, complex backgrounds, and class imbalance.

Potential next experiments include:

- Oversampling underrepresented classes
- Image cropping and tiling with SAHI
- Increasing input resolution
- Hard-negative mining
- Checking the train-validation split for data leakage

The original metric plots are stored in:

```text
assets/metrics/
```

## API

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@tower.jpg" \
  -F "model_id=accurate" \
  -F "confidence=0.44" \
  -F "iou=0.45"
```

Available endpoints:

- `GET /health` — service health status
- `GET /api/v1/tasks` — available tasks, classes, and dataset links
- `GET /api/v1/models?task_id=vehicle` — models available for a selected task and the source of their weights
- `POST /api/v1/predict` — image inference for files up to 20 MB

## Docker

```bash
docker compose up --build
```

The services are available at:

- UI: port `8501`
- API: port `8000`

Mount custom `.pt` model weights into the `models/` directory.

## Tests and Validation

```bash
pip install -r requirements-dev.txt
pytest -q
python -m compileall detection_app app.py run_api.py
```

## Deployment

For Streamlit Community Cloud:

1. Select `app.py` as the application entry point.
2. Use Python 3.11.
3. Store model weights in an approved external storage service or use Git LFS.
4. Leave `POWERGUARD_API_URL` empty when running in single-process embedded mode.

For a separate frontend and backend architecture, deploy the API to Render, Railway, or a VPS, then add the API URL to the Streamlit environment secrets.

Before publication:

- Replace placeholder contacts and links in `docs/submission_checklist.md`
- Record a short demonstration using the script in `docs/presentation_script.md`
- Verify the licenses of the datasets and model weights

## Limitations

This MVP is not an industrial safety system and does not replace inspection by a qualified engineer.

Before production use, the system would require:

- Field testing
- Model drift monitoring
- Model version logging
- Validation on representative operational data
- Manual review of critical detections
