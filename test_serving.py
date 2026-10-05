import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from blastograde.api.main import app, grader_dependency
from blastograde.data.synth import render_blastocyst
from blastograde.inference.grader import EmbryoGrader, InvalidImageError


@pytest.fixture(scope="module")
def grader(checkpoint_path):
    return EmbryoGrader.load(checkpoint_path)


@pytest.fixture(scope="module")
def client(grader):
    app.dependency_overrides[grader_dependency] = lambda: grader
    yield TestClient(app)
    app.dependency_overrides.clear()


def _png(seed: int = 0, colour: bool = False) -> bytes:
    image = render_blastocyst(np.random.default_rng(seed), 3, 0, 0, 160).image
    if colour:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.imencode(".png", image)[1].tobytes()


def test_grader_output_contract(grader):
    result = grader.grade_bytes(_png(), heatmaps=True)
    assert result["gardner_grade"][0] in "123456"
    assert abs(sum(result["expansion"]["probabilities"].values()) - 1) < 1e-2
    assert set(result["heatmaps"]) <= {"expansion", "icm", "te"}
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result["heatmaps"]["expansion"]), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[2] == 3
    if result["inner_cell_mass"] is None:
        assert len(result["gardner_grade"]) == 1 and not result["good_quality"]


def test_grader_handles_rectangular_colour_images(grader):
    wide = np.random.default_rng(0).integers(0, 255, (120, 300, 3), dtype=np.uint8)
    assert "gardner_grade" in grader.grade_array(wide)


def test_grader_rejects_tiny_images(grader):
    with pytest.raises(InvalidImageError):
        grader.grade_array(np.zeros((20, 20), np.uint8))


def test_api_grade_and_batch(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["backbone"] == "resnet18"
    single = client.post("/grade?heatmaps=true", files={"file": ("e1.png", _png(1), "image/png")})
    assert single.status_code == 200 and single.json()["filename"] == "e1.png" and "heatmaps" in single.json()
    batch = client.post("/grade/batch", files=[("files", ("a.png", _png(2), "image/png")), ("files", ("b.png", _png(3, True), "image/png")),
                                              ("files", ("bad.png", b"not an image", "image/png"))])
    body = batch.json()
    assert body["count"] == 3 and "error" in body["results"][-1]


def test_api_rejects_bad_uploads(client):
    assert client.post("/grade", files={"file": ("x.png", b"garbage", "image/png")}).status_code == 422
    assert client.post("/grade", files={"file": ("x.png", b"", "image/png")}).status_code == 400
