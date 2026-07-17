from fastapi.testclient import TestClient

from detection_app.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_expose_two_profiles() -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert len(response.json()) == 6
    assert {item["task_id"] for item in response.json()} == {
        "powerline", "fracture", "vehicle"
    }


def test_each_task_exposes_two_models() -> None:
    tasks = client.get("/api/v1/tasks").json()
    assert len(tasks) == 3
    for task in tasks:
        models = client.get("/api/v1/models", params={"task_id": task["id"]}).json()
        assert len(models) == 2
        assert all(model["task_id"] == task["id"] for model in models)


def test_predict_rejects_non_image() -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
