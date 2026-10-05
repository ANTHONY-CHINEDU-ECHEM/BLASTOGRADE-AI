"""Training loop for the multi task grader."""
from __future__ import annotations

import json
import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blastograde import TASKS, __version__
from blastograde.config import Config, resolve
from blastograde.data.dataset import IGNORE_INDEX, BlastocystDataset
from blastograde.engine.metrics import task_metrics
from blastograde.models.network import BlastocystGrader

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multitask_loss(outputs: dict, targets: dict, weights: dict, smoothing: float) -> tuple[torch.Tensor, dict]:
    """Weighted sum of cross entropies. Ungradable ICM and TE targets are masked out."""
    total, parts = 0.0, {}
    for task in TASKS:
        target = targets[task]
        if (target != IGNORE_INDEX).any():
            loss = F.cross_entropy(outputs[task], target, ignore_index=IGNORE_INDEX, label_smoothing=smoothing)
        else:
            loss = outputs[task].sum() * 0.0
        parts[task] = float(loss.detach())
        total = total + weights[task] * loss
    return total, parts


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[dict, dict, dict]:
    """Return predicted classes, class probabilities and targets for a loader."""
    model.eval()
    probs = {task: [] for task in TASKS}
    truth = {task: [] for task in TASKS}
    for images, targets in loader:
        outputs = model(images.to(device))
        for task in TASKS:
            probs[task].append(outputs[task].softmax(dim=1).cpu().numpy())
            truth[task].append(targets[task].numpy())
    probs = {task: np.concatenate(v) for task, v in probs.items()}
    truth = {task: np.concatenate(v) for task, v in truth.items()}
    return {task: p.argmax(axis=1) for task, p in probs.items()}, probs, truth


def summarise(preds: dict, truth: dict) -> dict:
    metrics = {task: task_metrics(truth[task], preds[task], len(classes)) for task, classes in TASKS.items()}
    metrics["mean_kappa"] = float(np.mean([metrics[t]["quadratic_kappa"] for t in TASKS]))
    return metrics


def run_training(cfg: Config) -> dict:
    """Train, keep the checkpoint with the best validation agreement, and save it."""
    set_seed(cfg.seed)
    device = pick_device()
    tr = cfg.training
    root = resolve(cfg.data.root)
    train_set = BlastocystDataset(root, "train", tr.image_size, train=True, seed=cfg.seed)
    val_set = BlastocystDataset(root, "val", tr.image_size)
    train_loader = DataLoader(train_set, batch_size=tr.batch_size, shuffle=True, num_workers=tr.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=tr.batch_size * 2, num_workers=tr.num_workers)
    logger.info("Training on %d images, validating on %d, device %s", len(train_set), len(val_set), device)

    model = BlastocystGrader(cfg.model.backbone, cfg.model.pretrained, cfg.model.dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    total_steps = max(tr.epochs * len(train_loader), 4)
    # Very short runs (unit tests, smoke checks) need a longer warm up share to keep the schedule valid.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=tr.learning_rate, total_steps=total_steps,
                                                    pct_start=min(0.5, max(0.2, 2.0 / total_steps)))
    report_dir = resolve(cfg.artifacts.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve(cfg.artifacts.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cam_stage = model.pick_cam_stage(tr.image_size) if cfg.model.cam_layer == "auto" else int(cfg.model.cam_layer)

    history, best, stale, started = [], -1.0, 0, time.time()
    for epoch in range(1, tr.epochs + 1):
        model.train()
        running, seen = {task: 0.0 for task in TASKS}, 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {task: t.to(device) for task, t in targets.items()}
            loss, parts = multitask_loss(model(images), targets, tr.task_weights, tr.label_smoothing)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()
            seen += 1
            for task in TASKS:
                running[task] += parts[task]
        preds, _, truth = predict_loader(model, val_loader, device)
        val = summarise(preds, truth)
        row = {"epoch": epoch, "seconds": round(time.time() - started, 1),
               **{f"train_loss_{t}": running[t] / seen for t in TASKS},
               **{f"val_accuracy_{t}": val[t]["accuracy"] for t in TASKS},
               **{f"val_kappa_{t}": val[t]["quadratic_kappa"] for t in TASKS}, "val_mean_kappa": val["mean_kappa"]}
        history.append(row)
        with open(report_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        logger.info("epoch %d | loss %s | val acc %s | mean kappa %.4f", epoch,
                    {t: round(row[f'train_loss_{t}'], 3) for t in TASKS},
                    {t: round(row[f'val_accuracy_{t}'], 3) for t in TASKS}, val["mean_kappa"])
        if val["mean_kappa"] > best:
            best, stale = val["mean_kappa"], 0
            torch.save({
                "state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                "backbone": cfg.model.backbone, "dropout": cfg.model.dropout, "image_size": tr.image_size,
                "cam_stage": cam_stage, "tasks": TASKS, "version": __version__, "epoch": epoch, "val_mean_kappa": best,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= tr.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break
    return {"best_val_mean_kappa": best, "epochs_run": len(history), "checkpoint": str(checkpoint_path)}


def load_checkpoint(path, device: torch.device | None = None) -> tuple[BlastocystGrader, dict]:
    """Rebuild the network from a checkpoint saved by :func:`run_training`."""
    device = device or pick_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BlastocystGrader(checkpoint["backbone"], pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in checkpoint["state_dict"].items()})
    return model.to(device).eval(), checkpoint
