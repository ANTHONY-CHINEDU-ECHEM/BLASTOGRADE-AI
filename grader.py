"""Serving facade: image in, Gardner grade with confidence and heatmaps out."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from blastograde import EXPANSION_NAMES, MIN_GRADABLE_EXPANSION_INDEX, TASKS
from blastograde.config import load_config, resolve
from blastograde.data.dataset import preprocess, to_grayscale
from blastograde.engine.metrics import gardner_string
from blastograde.engine.train import load_checkpoint, pick_device
from blastograde.explain.gradcam import GradCAM, overlay

REVIEW_THRESHOLD = 0.60
MIN_SIDE = 64


class InvalidImageError(ValueError):
    """Raised when an upload cannot be interpreted as a usable embryo image."""


class EmbryoGrader:
    """Grade blastocyst images with a trained multi task network."""

    def __init__(self, model, checkpoint: dict, device: torch.device):
        self.model, self.checkpoint, self.device = model, checkpoint, device
        self.image_size = checkpoint["image_size"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EmbryoGrader":
        path = Path(path) if path else resolve(load_config().artifacts.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}. Run `make all` to train one.")
        device = pick_device()
        model, checkpoint = load_checkpoint(path, device)
        return cls(model, checkpoint, device)

    def grade_array(self, image: np.ndarray, heatmaps: bool = False) -> dict:
        """Grade a decoded image (grayscale or BGR)."""
        if image is None or image.ndim not in (2, 3) or min(image.shape[:2]) < MIN_SIDE:
            raise InvalidImageError(f"Image must be at least {MIN_SIDE} pixels on its shorter side.")
        gray = to_grayscale(image)
        side = min(gray.shape)
        top, left = (gray.shape[0] - side) // 2, (gray.shape[1] - side) // 2
        gray = gray[top:top + side, left:left + side]           # centre crop to a square field of view
        tensor = preprocess(gray, self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = {task: out.softmax(dim=1)[0].cpu().numpy() for task, out in self.model(tensor).items()}
        index = {task: int(p.argmax()) for task, p in probs.items()}
        gradable = index["expansion"] >= MIN_GRADABLE_EXPANSION_INDEX

        def describe(task: str) -> dict:
            classes = TASKS[task]
            return {"grade": classes[index[task]], "confidence": round(float(probs[task].max()), 4),
                    "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs[task])}}

        result = {
            "gardner_grade": gardner_string(index["expansion"], index["icm"], index["te"]),
            "expansion": {**describe("expansion"), "stage": EXPANSION_NAMES[TASKS["expansion"][index["expansion"]]]},
            "inner_cell_mass": describe("icm") if gradable else None,
            "trophectoderm": describe("te") if gradable else None,
            "good_quality": bool(gradable and index["icm"] <= 1 and index["te"] <= 1),
        }
        used = ["expansion", "icm", "te"] if gradable else ["expansion"]
        lowest = min(float(probs[t].max()) for t in used)
        result["needs_review"] = lowest < REVIEW_THRESHOLD
        result["lowest_confidence"] = round(lowest, 4)
        if heatmaps:
            cam = GradCAM(self.model, self.checkpoint["cam_stage"])
            display = cv2.resize(gray, (self.image_size * 2, self.image_size * 2), interpolation=cv2.INTER_CUBIC)
            result["heatmaps"] = {}
            for task in used:
                heat, _ = cam(tensor, task)
                ok, png = cv2.imencode(".png", overlay(display, heat[0]))
                result["heatmaps"][task] = base64.b64encode(png.tobytes()).decode("ascii")
            cam.close()
        return result

    def grade_bytes(self, data: bytes, heatmaps: bool = False) -> dict:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError("The file could not be decoded as an image. Send a PNG, JPEG or TIFF.")
        return self.grade_array(image, heatmaps)

    def grade_file(self, path: str | Path, heatmaps: bool = False) -> dict:
        return self.grade_bytes(Path(path).read_bytes(), heatmaps)


@lru_cache(maxsize=1)
def get_grader() -> EmbryoGrader:
    return EmbryoGrader.load()
