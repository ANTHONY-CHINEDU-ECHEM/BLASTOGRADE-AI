"""Multi task grading network: one shared backbone, three classification heads."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from blastograde import TASKS

_RESNETS = {"resnet18": (models.resnet18, 512), "resnet34": (models.resnet34, 512), "resnet50": (models.resnet50, 2048)}


def _grayscale_stem(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Replace an RGB stem with a single channel stem, keeping pretrained filters by averaging."""
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
    return new


class BlastocystGrader(nn.Module):
    """Shared convolutional encoder with heads for expansion, ICM and TE.

    ``stages`` exposes the encoder as an ordered list of blocks so Grad CAM
    can attach to whichever stage gives a useful spatial resolution.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, dropout: float = 0.25):
        super().__init__()
        self.backbone_name = backbone
        if backbone in _RESNETS:
            factory, dim = _RESNETS[backbone]
            net = factory(weights="DEFAULT" if pretrained else None)
            net.conv1 = _grayscale_stem(net.conv1, pretrained)
            stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.stages = nn.ModuleList([stem, net.layer1, net.layer2, net.layer3, net.layer4])
        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            net.features[0][0] = _grayscale_stem(net.features[0][0], pretrained)
            blocks = list(net.features.children())
            self.stages = nn.ModuleList([nn.Sequential(*blocks[:3]), blocks[3], nn.Sequential(*blocks[4:6]),
                                         nn.Sequential(*blocks[6:8]), blocks[8]])
            dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(dim, len(classes)) for task, classes in TASKS.items()})

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.dropout(self.pool(self.features(x)).flatten(1))
        return {task: head(pooled) for task, head in self.heads.items()}

    def pick_cam_stage(self, image_size: int, minimum: int = 7) -> int:
        """Index of the deepest stage whose feature map is at least ``minimum`` pixels wide."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 1, image_size, image_size, device=next(self.parameters()).device)
            chosen = 0
            for index, stage in enumerate(self.stages):
                x = stage(x)
                if x.shape[-1] >= minimum:
                    chosen = index
        self.train(was_training)
        return chosen
