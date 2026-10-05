"""Torch dataset, preprocessing and augmentation for blastocyst images.

The dataset reads any folder that contains a ``labels.csv`` with the columns
``image``, ``expansion``, ``icm``, ``te`` and ``split``. The synthetic
generator writes that layout, and a real annotated image set can be used by
producing the same file (see ``docs/real_data.md``).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from blastograde import MIN_GRADABLE_EXPANSION_INDEX, TASKS

IGNORE_INDEX = -100
MEAN, STD = 0.5, 0.25


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Accept grayscale, BGR or BGRA input and return single channel uint8."""
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
    """Deterministic preprocessing shared by evaluation and serving."""
    gray = to_grayscale(image)
    if gray.shape[0] != size or gray.shape[1] != size:
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return ((tensor - MEAN) / STD).unsqueeze(0)


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label preserving augmentation.

    A blastocyst has no canonical orientation, so arbitrary rotation and
    flips are safe. Scale and shift stay small because overall size carries
    information about expansion.
    """
    size = gray.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(0, 360), rng.uniform(0.94, 1.06))
    matrix[:, 2] += rng.uniform(-0.05, 0.05, 2) * size
    out = cv2.warpAffine(gray, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    out = out.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def encode_labels(frame: pd.DataFrame, suffix: str = "") -> dict[str, np.ndarray]:
    """Convert grade strings to class indices, masking ungradable structures."""
    encoded = {}
    for task, classes in TASKS.items():
        column = frame[f"{task}{suffix}"].fillna("").astype(str).str.strip().str.upper()
        encoded[task] = column.map({c: i for i, c in enumerate(classes)}).fillna(IGNORE_INDEX).astype(int).to_numpy().copy()
    early = encoded["expansion"] < MIN_GRADABLE_EXPANSION_INDEX
    encoded["icm"][early] = IGNORE_INDEX
    encoded["te"][early] = IGNORE_INDEX
    return encoded


def load_labels(root: str | Path, split: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(Path(root) / "labels.csv", dtype={"expansion": str, "expansion_ref": str}, keep_default_na=False)
    return frame[frame["split"] == split].reset_index(drop=True) if split else frame


class BlastocystDataset(Dataset):
    """Images with three grade targets (expansion, inner cell mass, trophectoderm)."""

    def __init__(self, root: str | Path, split: str, image_size: int, train: bool = False, seed: int = 0):
        self.root, self.image_size, self.train = Path(root), image_size, train
        self.frame = load_labels(root, split)
        self.targets = encode_labels(self.frame)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        image = cv2.imread(str(self.root / self.frame.loc[index, "image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.root / self.frame.loc[index, "image"])
        if image.shape[0] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.train:
            image = augment(image, self.rng)
        targets = {task: torch.tensor(values[index]) for task, values in self.targets.items()}
        return preprocess(image, self.image_size), targets
