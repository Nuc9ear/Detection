from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .inference import InferenceError, prediction_service
from .schemas import HealthResponse, ModelInfo, PredictionResponse, TaskInfo


app = FastAPI(
    title="VisionGuard AI API",
    description="API for power-line damage, fracture, and vehicle detection.",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="VisionGuard AI", version=__version__)


@app.get("/api/v1/tasks", response_model=list[TaskInfo], tags=["models"])
def list_tasks() -> list[TaskInfo]:
    return prediction_service.list_tasks()


@app.get("/api/v1/models", response_model=list[ModelInfo], tags=["models"])
def list_models(task_id: str | None = None) -> list[ModelInfo]:
    return prediction_service.list_models(task_id)


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form("powerline_fast"),
    confidence: float = Form(0.44, ge=0.01, le=0.99),
    iou: float = Form(0.45, ge=0.01, le=0.99),
    device: str = Form("auto"),
) -> PredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported.")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The maximum file size is 20 MB.")
    try:
        return prediction_service.predict(content, model_id, confidence, iou, device)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
