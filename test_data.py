import numpy as np
import torch

from blastograde.data.dataset import IGNORE_INDEX, BlastocystDataset, augment, encode_labels, preprocess
from blastograde.data.synth import render_blastocyst


def test_renderer_is_deterministic_and_well_formed():
    a = render_blastocyst(np.random.default_rng(5), 3, 0, 1, 128)
    b = render_blastocyst(np.random.default_rng(5), 3, 0, 1, 128)
    assert a.image.shape == (128, 128) and a.image.dtype == np.uint8
    assert np.array_equal(a.image, b.image)
    assert 0 < a.icm_x < 1 and 0 < a.icm_y < 1 and a.icm_r > 0


def test_expansion_changes_embryo_size():
    rng = np.random.default_rng(0)
    full = np.mean([render_blastocyst(rng, 2, 1, 1, 128).embryo_r for _ in range(20)])
    hatched = np.mean([render_blastocyst(rng, 5, 1, 1, 128).embryo_r for _ in range(20)])
    assert hatched > full * 1.15


def test_inner_cell_mass_grade_changes_cluster_size():
    rng = np.random.default_rng(0)
    grade_a = np.mean([render_blastocyst(rng, 3, 0, 1, 128).icm_r for _ in range(20)])
    grade_c = np.mean([render_blastocyst(rng, 3, 2, 1, 128).icm_r for _ in range(20)])
    assert grade_a > 2 * grade_c


def test_early_blastocysts_have_masked_quality_targets(tiny_dataset):
    _, labels = tiny_dataset
    encoded = encode_labels(labels)
    early = encoded["expansion"] < 2
    assert early.any()
    assert (encoded["icm"][early] == IGNORE_INDEX).all() and (encoded["te"][early] == IGNORE_INDEX).all()
    assert (encoded["icm"][~early] >= 0).all()


def test_dataset_items_have_expected_shape(tiny_dataset):
    root, _ = tiny_dataset
    dataset = BlastocystDataset(root, "train", 96, train=True)
    image, targets = dataset[0]
    assert image.shape == (1, 96, 96) and image.dtype == torch.float32
    assert set(targets) == {"expansion", "icm", "te"}


def test_augmentation_preserves_shape_and_range():
    gray = render_blastocyst(np.random.default_rng(2), 3, 1, 1, 96).image
    out = augment(gray, np.random.default_rng(3))
    assert out.shape == gray.shape and out.dtype == np.uint8


def test_preprocess_accepts_colour_and_resizes():
    colour = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert preprocess(colour, 96).shape == (1, 96, 96)
