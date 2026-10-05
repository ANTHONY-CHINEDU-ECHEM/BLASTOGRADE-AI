"""Command line entry point: ``python -m blastograde.cli <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from blastograde.config import load_config, resolve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="blastograde", description="BlastoGrade AI pipeline")
    parser.add_argument("command", choices=["generate", "train", "evaluate", "all", "grade"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", default=None, help="Image path for the grade command")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)

    if args.command in {"generate", "all"}:
        from blastograde.data.synth import generate_dataset

        labels = generate_dataset(resolve(cfg.data.root), cfg.data.n_images, cfg.data.render_size, cfg.data.stored_size,
                                  dict(cfg.data.split), dict(cfg.data.annotator_noise), cfg.seed)
        logging.info("Rendered %d images: %s", len(labels), labels["split"].value_counts().to_dict())
    if args.command in {"train", "all"}:
        from blastograde.engine.train import run_training

        print(json.dumps(run_training(cfg), indent=2))
    if args.command in {"evaluate", "all"}:
        from blastograde.engine.evaluate import run_evaluation

        report = run_evaluation(cfg)
        print(json.dumps({k: v for k, v in report.items() if k not in {"model_vs_annotator", "model_vs_reference", "annotator_vs_reference"}}, indent=2))
    if args.command == "grade":
        from blastograde.inference.grader import EmbryoGrader

        print(json.dumps(EmbryoGrader.load().grade_file(args.image), indent=2))


if __name__ == "__main__":
    main()
