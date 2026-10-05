"""Grad CAM for a multi head network, implemented with forward and backward hooks."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from blastograde.data.dataset import MEAN, STD


class GradCAM:
    """Class activation maps for any head of :class:`BlastocystGrader`.

    Each head gets its own heatmap, because the question differs by head: the
    inner cell mass grade should be driven by the cell cluster, while the
    trophectoderm grade should be driven by the cavity lining.
    """

    def __init__(self, model, stage_index: int):
        self.model, self.stage_index = model, stage_index
        self._activation = None
        self._hook = model.stages[stage_index].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._activation = output
        if output.requires_grad:
            output.retain_grad()

    def close(self) -> None:
        self._hook.remove()

    def __call__(self, batch: torch.Tensor, task: str, class_index: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Return heatmaps in [0, 1] with shape (n, h, w) and the logits of the chosen head."""
        self.model.eval()
        batch = batch.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(batch)[task]
            if class_index is None:
                class_index = logits.argmax(dim=1)
            score = logits.gather(1, class_index.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        activation, gradient = self._activation.detach(), self._activation.grad.detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.cpu().numpy(), logits.detach()


def tensor_to_gray(tensor: torch.Tensor) -> np.ndarray:
    """Invert preprocessing for display."""
    image = tensor.squeeze().cpu().numpy() * STD + MEAN
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over a grayscale image. Returns a BGR uint8 image."""
    heat = cv2.resize(cam.astype(np.float32), gray.shape[::-1], interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    weight = (alpha * heat)[..., None]
    return (base * (1 - weight) + colour * weight).astype(np.uint8)


def peak_location(cam: np.ndarray) -> tuple[float, float]:
    """Location of the heatmap maximum as fractions of width and height (cell centre)."""
    row, col = np.unravel_index(int(cam.argmax()), cam.shape)
    return (col + 0.5) / cam.shape[1], (row + 0.5) / cam.shape[0]
