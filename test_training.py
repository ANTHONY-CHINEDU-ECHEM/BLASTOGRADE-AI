from blastograde.config import Config, load_config
from blastograde.engine.train import load_checkpoint, run_training


def test_one_epoch_training_writes_a_loadable_checkpoint(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    cfg = dict(load_config())
    cfg["data"] = {**cfg["data"], "root": str(root)}
    cfg["training"] = {**cfg["training"], "epochs": 1, "batch_size": 8, "image_size": 64}
    cfg["artifacts"] = {"checkpoint": str(tmp_path / "m.pt"), "report_dir": str(tmp_path / "reports"), "figure_dir": str(tmp_path / "figs")}
    summary = run_training(Config(cfg))
    assert summary["epochs_run"] == 1
    model, checkpoint = load_checkpoint(tmp_path / "m.pt")
    assert checkpoint["image_size"] == 64 and not model.training
    assert (tmp_path / "reports" / "history.json").exists()
