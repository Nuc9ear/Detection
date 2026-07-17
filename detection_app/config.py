from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    id: str
    title: str
    short_title: str
    description: str
    classes: tuple[str, ...]
    dataset_url: str
    disclaimer: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    task_id: str
    title: str
    description: str
    custom_weights: Path
    fallback_weights: str
    image_size: int
    speed_label: str


TASK_SPECS: dict[str, TaskSpec] = {
    "powerline": TaskSpec(
        id="powerline",
        title="Power-line damage",
        short_title="Power lines",
        description="Components and potential damage on overhead power lines.",
        classes=(
            "vibration_damper", "festoon_insulators", "traverse", "nest",
            "safety_sign+", "bad_insulator", "damaged_insulator", "polymer_insulators",
        ),
        dataset_url="https://registry.cit.gov.ru/datasets/1613b970-58b2-4146-9e15-9d62d03b3f82#description",
        disclaimer="Results must be reviewed by a qualified power-line engineer.",
    ),
    "fracture": TaskSpec(
        id="fracture",
        title="Fractures in X-ray images",
        short_title="Fractures",
        description="Localization of upper-extremity fractures in X-ray images.",
        classes=(
            "fracture", "elbow positive", "fingers positive", "forearm fracture",
            "humerus fracture", "humerus", "shoulder fracture", "wrist positive",
        ),
        dataset_url="https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project",
        disclaimer="This educational result is not a diagnosis and must be reviewed by a physician.",
    ),
    "vehicle": TaskSpec(
        id="vehicle",
        title="Vehicles in aerial imagery",
        short_title="Vehicles",
        description="Vehicle detection in drone imagery and urban camera footage.",
        classes=(
            "bicycle", "car", "van", "truck", "tricycle",
            "awning-tricycle", "bus", "motor",
        ),
        dataset_url="https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset",
        disclaimer="Quality depends on capture altitude, object density, and occlusion.",
    ),
}


def _weights(env_name: str, filename: str) -> Path:
    return Path(os.getenv(env_name, ROOT / "models" / filename))


MODEL_SPECS: dict[str, ModelSpec] = {
    "powerline_fast": ModelSpec(
        "powerline_fast", "powerline", "Fast model", "Rapid review of power-line imagery.",
        _weights("POWERLINE_FAST_MODEL_PATH", "powerline_yolo11n.pt"), "yolo11n.pt", 640, "maximum speed",
    ),
    "powerline_accurate": ModelSpec(
        "powerline_accurate", "powerline", "Accurate model", "Detailed inspection of power-line components.",
        _weights("POWERLINE_ACCURATE_MODEL_PATH", "powerline_yolo11s.pt"), "yolo11s.pt", 960, "higher accuracy",
    ),
    "fracture_fast": ModelSpec(
        "fracture_fast", "fracture", "Fast model", "Binary localization of fracture regions.",
        _weights("FRACTURE_FAST_MODEL_PATH", "fracture_yolov8_fast.pt"), "yolo11n.pt", 640, "maximum speed",
    ),
    "fracture_accurate": ModelSpec(
        "fracture_accurate", "fracture", "Accurate model", "Localization and classification by anatomical region.",
        _weights("FRACTURE_ACCURATE_MODEL_PATH", "fracture_yolov8l.pt"), "yolo11s.pt", 640, "higher accuracy",
    ),
    "vehicle_fast": ModelSpec(
        "vehicle_fast", "vehicle", "Fast model", "Rapid traffic-flow analysis.",
        _weights("VEHICLE_FAST_MODEL_PATH", "vehicle_yolo11n.pt"), "yolo11n.pt", 640, "maximum speed",
    ),
    "vehicle_accurate": ModelSpec(
        "vehicle_accurate", "vehicle", "Accurate model", "Small, densely packed objects in aerial imagery.",
        _weights("VEHICLE_ACCURATE_MODEL_PATH", "vehicle_yolo11s.pt"), "yolo11s.pt", 1280, "higher accuracy",
    ),
}


def allow_fallback() -> bool:
    return os.getenv("ALLOW_PRETRAINED_FALLBACK", "true").lower() in {"1", "true", "yes"}
