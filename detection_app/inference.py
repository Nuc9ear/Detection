from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import MODEL_SPECS, TASK_SPECS, ModelSpec, allow_fallback
from .schemas import Detection, ModelInfo, PredictionResponse, TaskInfo


class InferenceError(RuntimeError):
    """A user-facing inference error."""


class PredictionService:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._locks = {model_id: threading.Lock() for model_id in MODEL_SPECS}

    @staticmethod
    def _weights_for(spec: ModelSpec) -> tuple[str, str, bool]:
        if spec.custom_weights.is_file():
            return str(spec.custom_weights), "custom", True
        if allow_fallback():
            return spec.fallback_weights, "pretrained-fallback", True
        return str(spec.custom_weights), "missing", False

    def list_tasks(self) -> list[TaskInfo]:
        return [
            TaskInfo(
                id=task.id,
                title=task.title,
                short_title=task.short_title,
                description=task.description,
                classes=list(task.classes),
                dataset_url=task.dataset_url,
                disclaimer=task.disclaimer,
            )
            for task in TASK_SPECS.values()
        ]

    def list_models(self, task_id: str | None = None) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        for spec in MODEL_SPECS.values():
            if task_id and spec.task_id != task_id:
                continue
            _, source, ready = self._weights_for(spec)
            result.append(
                ModelInfo(
                    id=spec.id,
                    task_id=spec.task_id,
                    title=spec.title,
                    description=spec.description,
                    speed_label=spec.speed_label,
                    image_size=spec.image_size,
                    weights_source=source,
                    ready=ready,
                )
            )
        return result

    def _get_model(self, model_id: str) -> tuple[Any, ModelSpec, str]:
        if model_id not in MODEL_SPECS:
            raise InferenceError(f"Unknown model: {model_id}")
        spec = MODEL_SPECS[model_id]
        weights, source, ready = self._weights_for(spec)
        if not ready:
            raise InferenceError(
                f"Weights were not found: {spec.custom_weights}. "
                "Add the file or enable ALLOW_PRETRAINED_FALLBACK."
            )

        if model_id not in self._models:
            with self._locks[model_id]:
                if model_id not in self._models:
                    try:
                        from ultralytics import YOLO

                        self._models[model_id] = YOLO(weights)
                    except Exception as exc:  # import, download, or invalid weights
                        raise InferenceError(
                            "Could not load the model. Check the Ultralytics installation, "
                            "weight-file access, and network connection for fallback weights."
                        ) from exc
        return self._models[model_id], spec, source

    @staticmethod
    def _read_image(content: bytes) -> Image.Image:
        if not content:
            raise InferenceError("The uploaded file is empty.")
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
            return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InferenceError("The file is not a valid image.") from exc

    def predict(
        self,
        content: bytes,
        model_id: str,
        confidence: float = 0.44,
        iou: float = 0.45,
        device: str = "auto",
    ) -> PredictionResponse:
        image = self._read_image(content)
        model, spec, source = self._get_model(model_id)
        selected_device = None if device == "auto" else device

        started = time.perf_counter()
        try:
            results = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=spec.image_size,
                device=selected_device,
                verbose=False,
            )
            result = results[0]
        except Exception as exc:
            raise InferenceError("An error occurred during model inference.") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        names = result.names
        allowed_classes = set(TASK_SPECS[spec.task_id].classes)
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else names[class_id]
                )
                class_name = {"motorcycle": "motor"}.get(str(class_name), str(class_name))
                if class_name not in allowed_classes:
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=round(float(box.conf.item()), 4),
                        bbox=[round(float(v), 2) for v in box.xyxy[0].tolist()],
                    )
                )

        plotted_bgr = result.plot()
        annotated = Image.fromarray(plotted_bgr[:, :, ::-1])
        output = io.BytesIO()
        annotated.save(output, format="JPEG", quality=90, optimize=True)

        return PredictionResponse(
            task_id=spec.task_id,
            task_title=TASK_SPECS[spec.task_id].title,
            model_id=model_id,
            model_title=spec.title,
            weights_source=source,
            inference_ms=round(elapsed_ms, 1),
            image_width=image.width,
            image_height=image.height,
            detections=detections,
            annotated_image=base64.b64encode(output.getvalue()).decode("ascii"),
        )


prediction_service = PredictionService()
