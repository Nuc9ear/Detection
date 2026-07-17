import io

import pytest
from PIL import Image

from detection_app.inference import InferenceError, PredictionService


def test_read_image_accepts_png() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), "white").save(stream, format="PNG")
    image = PredictionService._read_image(stream.getvalue())
    assert image.size == (16, 12)
    assert image.mode == "RGB"


def test_read_image_rejects_invalid_bytes() -> None:
    with pytest.raises(InferenceError):
        PredictionService._read_image(b"broken")

