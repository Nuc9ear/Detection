from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    task_id: str
    title: str
    description: str
    speed_label: str
    image_size: int
    weights_source: str
    ready: bool


class TaskInfo(BaseModel):
    id: str
    title: str
    short_title: str
    description: str
    classes: list[str]
    dataset_url: str
    disclaimer: str


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    task_id: str
    task_title: str
    model_id: str
    model_title: str
    weights_source: str
    inference_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]
    annotated_image: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
