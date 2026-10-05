import numpy as np
import pytest
import torch

from blastograde.data.dataset import IGNORE_INDEX
from blastograde.engine.metrics import gardner_string, is_good_quality, task_metrics
from blastograde.engine.train import multitask_loss
from blastograde.explain.gradcam import GradCAM, overlay, peak_location
from blastograde.models.network import BlastocystGrader

WEIGHTS = {"expansion": 1.0, "icm": 1.0, "te": 1.0}


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_forward_shapes(backbone):
    model = BlastocystGrader(backbone).eval()
    out = model(torch.randn(2, 1, 96, 96))
    assert out["expansion"].shape == (2, 6) and out["icm"].shape == (2, 3) and out["te"].shape == (2, 3)


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError):
        BlastocystGrader("vgg99")


def test_loss_ignores_ungradable_structures():
    torch.manual_seed(0)
    outputs = {"expansion": torch.randn(4, 6, requires_grad=True), "icm": torch.randn(4, 3, requires_grad=True),
               "te": torch.randn(4, 3, requires_grad=True)}
    ignore = torch.full((4,), IGNORE_INDEX)
    loss, parts = multitask_loss(outputs, {"expansion": torch.tensor([0, 1, 0, 1]), "icm": ignore, "te": ignore}, WEIGHTS, 0.0)
    loss.backward()
    assert parts["icm"] == 0.0 and parts["te"] == 0.0 and torch.isfinite(loss)
    assert outputs["icm"].grad.abs().sum() == 0 and outputs["expansion"].grad.abs().sum() > 0


def test_gradcam_maps_are_normalised_per_head():
    model = BlastocystGrader("resnet18").eval()
    stage = model.pick_cam_stage(128)
    cam = GradCAM(model, stage)
    batch = torch.randn(2, 1, 128, 128)
    for task in ["expansion", "icm", "te"]:
        heat, logits = cam(batch, task)
        assert heat.shape == (2, 8, 8) and heat.min() >= 0 and heat.max() <= 1 + 1e-6
        assert logits.shape[0] == 2
    cam.close()
    x, y = peak_location(heat[0])
    assert 0 < x < 1 and 0 < y < 1
    blended = overlay(np.zeros((128, 128), np.uint8), heat[0])
    assert blended.shape == (128, 128, 3)


def test_cam_stage_respects_minimum_resolution():
    model = BlastocystGrader("resnet18")
    assert model.pick_cam_stage(224) == 4   # 7 by 7 at the last stage
    assert model.pick_cam_stage(128) == 3   # last stage would be 4 by 4, so step back one


def test_task_metrics_and_gardner_helpers():
    truth = np.array([0, 1, 2, 2, IGNORE_INDEX])
    pred = np.array([0, 1, 1, 2, 0])
    metrics = task_metrics(truth, pred, 3)
    assert metrics["n"] == 4 and metrics["accuracy"] == 0.75 and metrics["within_one_grade"] == 1.0
    assert gardner_string(3, 0, 1) == "4AB" and gardner_string(0, 0, 0) == "1"
    good = is_good_quality(np.array([3, 3, 1]), np.array([0, 2, 0]), np.array([1, 0, 0]))
    assert good.tolist() == [True, False, False]
