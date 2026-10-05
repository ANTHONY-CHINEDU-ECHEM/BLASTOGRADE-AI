from __future__ import annotations

import pytest
import torch

from blastograde import TASKS, __version__
from blastograde.data.synth import generate_dataset
from blastograde.models.network import BlastocystGrader


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("embryos")
    labels = generate_dataset(root, 60, render_size=128, stored_size=96, annotator_noise={"expansion": 0.1, "icm": 0.1, "te": 0.1}, seed=1)
    return root, labels


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory):
    """An untrained checkpoint in the production format, enough to exercise serving code."""
    torch.manual_seed(0)
    model = BlastocystGrader("resnet18")
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.save({"state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                "backbone": "resnet18", "dropout": 0.25, "image_size": 96, "cam_stage": model.pick_cam_stage(96),
                "tasks": TASKS, "version": __version__, "epoch": 0, "val_mean_kappa": 0.0}, path)
    return path
