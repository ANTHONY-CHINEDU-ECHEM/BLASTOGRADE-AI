"""Held out evaluation, Grad CAM localisation audit and figure generation."""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from blastograde import EXPANSION_CLASSES, QUALITY_CLASSES, TASKS
from blastograde.config import Config, resolve
from blastograde.data.dataset import IGNORE_INDEX, BlastocystDataset, encode_labels, load_labels
from blastograde.engine.metrics import gardner_string, is_good_quality
from blastograde.engine.train import load_checkpoint, pick_device, predict_loader, summarise
from blastograde.explain.gradcam import GradCAM, overlay, peak_location, tensor_to_gray

INK, PLUM, AMBER, MIST = "#1B1F3B", "#5B2A86", "#E0A100", "#9AA5B1"
TASK_TITLES = {"expansion": "Expansion", "icm": "Inner cell mass", "te": "Trophectoderm"}


def _style() -> None:
    plt.rcParams.update({"savefig.dpi": 160, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})


def cam_pointing_game(model, dataset, frame, preds, stage: int, device, limit: int = 400) -> dict:
    """Does the ICM heatmap peak on the inner cell mass?

    The renderer records where the inner cell mass was drawn, so the Grad CAM
    peak of the ICM head can be scored against ground truth. A hit means the
    peak lies within the cell cluster, allowing a tolerance of half a feature
    map cell. The chance rate is the share of feature map cells that would
    count as a hit, averaged over the same images.
    """
    eligible = np.where((dataset.targets["icm"] != IGNORE_INDEX) & (frame["icm_ref"].isin(["A", "B"]).to_numpy()))[0][:limit]
    cam = GradCAM(model, stage)
    hits, chance = [], []
    for start in range(0, len(eligible), 32):
        index = eligible[start:start + 32]
        batch = torch.stack([dataset[i][0] for i in index]).to(device)
        maps, _ = cam(batch, "icm", torch.as_tensor(preds["icm"][index], device=device))
        for heat, i in zip(maps, index):
            x, y = peak_location(heat)
            cx, cy, r = frame.loc[i, ["icm_x", "icm_y", "icm_r"]].astype(float)
            tolerance = r * 1.25 + 0.5 / heat.shape[0]
            hits.append(float(np.hypot(x - cx, y - cy) <= tolerance))
            grid = (np.arange(heat.shape[0]) + 0.5) / heat.shape[0]
            gx, gy = np.meshgrid(grid, grid)
            chance.append(float((np.hypot(gx - cx, gy - cy) <= tolerance).mean()))
    cam.close()
    return {"n": len(hits), "hit_rate": float(np.mean(hits)), "chance_rate": float(np.mean(chance)), "feature_map": int(maps.shape[-1])}


def run_evaluation(cfg: Config) -> dict:
    """Score the saved checkpoint on the test split and rebuild every figure."""
    _style()
    device = pick_device()
    root = resolve(cfg.data.root)
    model, checkpoint = load_checkpoint(resolve(cfg.artifacts.checkpoint), device)
    size = checkpoint["image_size"]
    dataset = BlastocystDataset(root, "test", size)
    frame = dataset.frame
    preds, probs, truth = predict_loader(model, DataLoader(dataset, batch_size=64), device)
    # Real datasets carry a single set of labels; the synthetic set also has the reference grades.
    has_reference = "expansion_ref" in frame.columns
    reference = encode_labels(frame, "_ref") if has_reference else truth

    report = {
        "n_test": len(dataset), "checkpoint_epoch": checkpoint["epoch"],
        "model_vs_annotator": summarise(preds, truth),
        "model_vs_reference": summarise(preds, reference),
        "annotator_vs_reference": summarise(truth, reference),
    }
    gradable = reference["icm"] != IGNORE_INDEX
    exact = (preds["expansion"] == reference["expansion"]) & (~gradable | ((preds["icm"] == reference["icm"]) & (preds["te"] == reference["te"])))
    annot_exact = (truth["expansion"] == reference["expansion"]) & (~gradable | ((truth["icm"] == reference["icm"]) & (truth["te"] == reference["te"])))
    report["full_gardner_exact_match"] = {"model": float(exact.mean()), "annotator": float(annot_exact.mean())}
    good_true = is_good_quality(reference["expansion"], reference["icm"], reference["te"])
    good_pred = is_good_quality(preds["expansion"], preds["icm"], preds["te"])
    tp, fp, fn = (good_pred & good_true).sum(), (good_pred & ~good_true).sum(), (~good_pred & good_true).sum()
    report["good_quality_3BB_or_better"] = {
        "prevalence": float(good_true.mean()), "accuracy": float((good_pred == good_true).mean()),
        "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
    }
    confidence = np.minimum.reduce([probs[t].max(axis=1) for t in TASKS])
    review = confidence < 0.6
    report["review_queue"] = {"share_flagged": float(review.mean()),
                              "exact_match_flagged": float(exact[review].mean()) if review.any() else None,
                              "exact_match_not_flagged": float(exact[~review].mean()) if (~review).any() else None}
    if has_reference and {"icm_x", "icm_y", "icm_r"} <= set(frame.columns):
        report["gradcam_icm_pointing_game"] = cam_pointing_game(model, dataset, frame, preds, checkpoint["cam_stage"], device)

    report_dir, figure_dir = resolve(cfg.artifacts.report_dir), resolve(cfg.artifacts.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    # Figure 1: confusion matrices against reference grades.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for ax, (task, classes) in zip(axes, TASKS.items()):
        matrix = np.array(report["model_vs_reference"][task]["confusion_matrix"], dtype=float)
        share = matrix / matrix.sum(axis=1, keepdims=True).clip(1)
        ax.imshow(share, cmap="Purples", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=9, color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
        m = report["model_vs_reference"][task]
        ax.set(xlabel="Predicted grade", ylabel="Reference grade",
               title=f"{TASK_TITLES[task]}\naccuracy {m['accuracy']:.1%}, kappa {m['quadratic_kappa']:.2f}")
    fig.tight_layout(); fig.savefig(figure_dir / "confusion_matrices.png"); plt.close(fig)

    # Figure 2: model against a single annotator, both scored on reference grades.
    fig, ax = plt.subplots(figsize=(7.2, 4))
    x = np.arange(3)
    annot = [report["annotator_vs_reference"][t]["accuracy"] * 100 for t in TASKS]
    ours = [report["model_vs_reference"][t]["accuracy"] * 100 for t in TASKS]
    ax.bar(x - 0.19, annot, 0.38, color=MIST, label="Single annotator")
    ax.bar(x + 0.19, ours, 0.38, color=PLUM, label="BlastoGrade AI")
    for xi, (a, o) in enumerate(zip(annot, ours)):
        ax.text(xi - 0.19, a + 0.6, f"{a:.1f}", ha="center", fontsize=9.5)
        ax.text(xi + 0.19, o + 0.6, f"{o:.1f}", ha="center", fontsize=9.5)
    ax.set_xticks(x, [TASK_TITLES[t] for t in TASKS])
    ax.set(ylabel="Agreement with reference grade (%)", ylim=(60, 103), title="Agreement with the reference standard")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figure_dir / "agreement_with_reference.png"); plt.close(fig)

    # Figure 3: training curves.
    history_path = report_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        for task, colour in zip(TASKS, [INK, PLUM, AMBER]):
            axes[0].plot(epochs, [h[f"train_loss_{task}"] for h in history], "o-", color=colour, label=TASK_TITLES[task])
            axes[1].plot(epochs, [h[f"val_kappa_{task}"] for h in history], "o-", color=colour, label=TASK_TITLES[task])
        axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Loss by task")
        axes[1].set(xlabel="Epoch", ylabel="Quadratic weighted kappa", title="Validation agreement by task")
        axes[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(figure_dir / "training_curves.png"); plt.close(fig)

    # Figure 4: Grad CAM gallery, one row per embryo, one heatmap per head.
    rng = np.random.default_rng(cfg.seed)
    candidates = np.where(gradable & exact)[0]
    if len(candidates) == 0:
        candidates = np.where(gradable)[0]
    chosen = []
    for grade in [2, 3, 4, 5]:
        pool = candidates[reference["expansion"][candidates] == grade]
        if len(pool):
            chosen.append(int(rng.choice(pool)))
    cam = GradCAM(model, checkpoint["cam_stage"])
    fig, axes = plt.subplots(len(chosen), 4, figsize=(10.4, 2.65 * len(chosen)), squeeze=False)
    for row, index in enumerate(chosen):
        tensor = dataset[index][0]
        gray = tensor_to_gray(tensor)
        label = gardner_string(preds["expansion"][index], preds["icm"][index], preds["te"][index])
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Predicted {label}", fontsize=10.5, loc="left")
        for col, task in enumerate(TASKS, start=1):
            heat, _ = cam(tensor.unsqueeze(0).to(device), task)
            axes[row, col].imshow(cv2.cvtColor(overlay(gray, heat[0]), cv2.COLOR_BGR2RGB))
            grade = TASKS[task][preds[task][index]]
            axes[row, col].set_title(f"{TASK_TITLES[task]}: {grade}", fontsize=10.5, loc="left")
        for ax in axes[row]:
            ax.axis("off")
    cam.close()
    fig.tight_layout(); fig.savefig(figure_dir / "gradcam_gallery.png"); plt.close(fig)

    # Figure 5: what each grade looks like.
    if not has_reference:
        return report
    full = load_labels(root)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.3))
    for col, grade in enumerate(EXPANSION_CLASSES):
        for row, quality in enumerate(QUALITY_CLASSES):
            ax = axes[row, col]
            ax.axis("off")
            pool = full[(full["expansion_ref"] == grade) & ((full["icm_ref"] == quality) & (full["te_ref"] == quality) | (int(grade) < 3))]
            if len(pool) > row:
                ax.imshow(cv2.imread(str(root / pool.iloc[row]["image"]), cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{grade}{quality}{quality}" if int(grade) >= 3 else f"Grade {grade}", fontsize=10.5)
    fig.suptitle("Rendered blastocysts across the Gardner scale", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(figure_dir / "dataset_gallery.png"); plt.close(fig)
    return report
