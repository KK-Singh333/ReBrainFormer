"""Run a small validation-only regularization sweep; never evaluate the test set."""

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotypic-csv", type=Path,
                        default=PROJECT_ROOT / "test_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    parser.add_argument("--timeseries-dir", type=Path, default=PROJECT_ROOT / "test_data_npy")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "processed_features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tuning_runs")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--readout-dropouts", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--meso-bins", type=int, default=32)
    parser.add_argument("--correlation-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for name in ("lambda1", "lambda2", "lambda3"):
        parser.add_argument(f"--{name}", type=float, default=0.1,
                            help="Fixed geometry weight during the regularization sweep")
    args = parser.parse_args()
    # Fail before creating runs if the full training data is unavailable.
    if not args.phenotypic_csv.is_file() or not args.timeseries_dir.is_dir() or not args.feature_dir.is_dir():
        parser.error("Provide an existing phenotype CSV, time-series directory, and feature directory")

    sweep_dir = args.output_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    trials = list(itertools.product(
        args.hidden_dims, args.weight_decays, args.readout_dropouts, args.learning_rates
    ))
    ranked = []
    for index, (width, decay, dropout, learning_rate) in enumerate(trials, start=1):
        command = [sys.executable, str(PROJECT_ROOT / "train_abide.py")]
        options = {
            "phenotypic-csv": args.phenotypic_csv,
            "timeseries-dir": args.timeseries_dir,
            "feature-dir": args.feature_dir,
            "output-dir": sweep_dir / f"trial_{index:02d}",
            "hidden-dim": width, "weight-decay": decay,
            "readout-dropout": dropout, "learning-rate": learning_rate,
            "min-learning-rate": min(1e-6, learning_rate),
            "attention-dropout": args.attention_dropout,
            "label-smoothing": args.label_smoothing,
            "batch-size": args.batch_size, "epochs": args.epochs,
            "patience": args.patience, "seed": args.seed, "device": args.device,
            "lambda1": args.lambda1, "lambda2": args.lambda2, "lambda3": args.lambda3,
            "contrastive-weight": args.contrastive_weight, "temperature": args.temperature,
            "meso-bins": args.meso_bins,
        }
        for name, value in options.items():
            command.extend([f"--{name}", str(value)])
        if args.correlation_only:
            command.append("--correlation-only")
        command.append("--skip-test")
        print(f"Trial {index}/{len(trials)}: width={width}, wd={decay}, dropout={dropout}, lr={learning_rate}", flush=True)
        subprocess.run(command, check=True)
        metrics_path, = (sweep_dir / f"trial_{index:02d}").glob("*/metrics.json")
        metrics = json.loads(metrics_path.read_text())
        ranked.append({
            "run_dir": str(metrics_path.parent),
            "validation": metrics["validation"],
            "best_epoch": metrics["best_epoch"],
            "config": metrics["config"],
            # Re-run the chosen configuration once to evaluate the test split.
            "final_training_command": command[:-1],
        })
        ranked.sort(key=lambda result: result["validation"]["loss"])
        (sweep_dir / "ranking.json").write_text(json.dumps(ranked, indent=2))
    print(f"Validation ranking: {sweep_dir / 'ranking.json'}")
    print(f"Lowest validation loss: {ranked[0]['run_dir']}; test set remains unevaluated")


if __name__ == "__main__":
    main()
