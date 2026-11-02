"""Train the current classification model on ABIDE pipeline feature archives.

This uses model_input and roi_mask from processed_features/*.npz. The separate
local/global/meso arrays and contrastive loss are not connected to this model.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from datasets import AbideDataset, split_abide
from models.rebrainformer import ReBrainFormer


PROJECT_ROOT = Path(__file__).resolve().parent


class AbideFeatureDataset(Dataset):
    """Join labeled ABIDE time-series filenames to pipeline feature archives."""

    def __init__(self, phenotypic_csv, timeseries_dir, feature_dir):
        labeled = AbideDataset(phenotypic_csv, timeseries_dir)
        self.records = []
        missing = []
        for record in labeled.records:
            feature_path = Path(feature_dir) / f"{record['path'].stem}.npz"
            if feature_path.is_file():
                self.records.append({**record, "feature_path": feature_path})
            else:
                missing.append(record["subject_id"])
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} labeled subjects lack feature archives. "
                f"Finish preprocessing first. Examples: {missing[:5]}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["feature_path"], allow_pickle=False) as archive:
            model_input = archive["model_input"].astype(np.float32)
            roi_mask = archive["roi_mask"].astype(bool)
        n_rois = model_input.shape[0]
        if (
            model_input.shape != (n_rois, 4, n_rois)
            or roi_mask.shape != (n_rois,)
            or not roi_mask.any()
            or not np.isfinite(model_input).all()
        ):
            raise ValueError(f"Invalid model input or ROI mask: {record['feature_path']}")
        return (
            torch.from_numpy(model_input),
            torch.from_numpy(roi_mask),
            record["label"],
            record["subject_id"],
        )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, probabilities, subject_ids = [], [], []

    for model_input, roi_mask, targets, batch_ids in loader:
        model_input = model_input.to(device)
        roi_mask = roi_mask.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            # ReBrainFormer.forward currently takes one [N, 4, N] subject.
            logits = torch.stack([
                model(model_input[i], roi_mask[i])
                for i in range(len(targets))
            ])
            loss = F.cross_entropy(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(targets)
        labels.extend(targets.cpu().tolist())
        probabilities.extend(logits.detach().softmax(dim=1)[:, 1].cpu().tolist())
        subject_ids.extend(batch_ids)

    return (
        total_loss / len(loader.dataset),
        np.asarray(labels),
        np.asarray(probabilities),
        subject_ids,
    )


def classification_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_asd": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity_asd": float(recall_score(labels, predictions, zero_division=0)),
        "specificity_control": float(tn / (tn + fp)) if tn + fp else None,
        "f1_asd": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2 else None
        ),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypic-csv", type=Path,
        default=PROJECT_ROOT / "test_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
    )
    parser.add_argument("--timeseries-dir", type=Path, default=PROJECT_ROOT / "test_data_npy")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "processed_features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "training_runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot see a CUDA device")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    dataset = AbideFeatureDataset(
        args.phenotypic_csv, args.timeseries_dir, args.feature_dir
    )
    train_set, val_set, test_set = split_abide(dataset, seed=args.seed)
    loaders = [
        DataLoader(part, batch_size=args.batch_size, shuffle=(i == 0), num_workers=0)
        for i, part in enumerate((train_set, val_set, test_set))
    ]
    n_rois = dataset[0][0].shape[0]
    model_args = SimpleNamespace(
        Nodes=n_rois, classes=2,
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
    )
    model = ReBrainFormer(model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    run_dir = args.output_dir / f"abide_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = {
        "Nodes": n_rois,
        "classes": 2,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
    }
    (run_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    splits = {
        name: [dataset.records[index]["subject_id"] for index in subset.indices]
        for name, subset in zip(
            ("train", "validation", "test"), (train_set, val_set, test_set)
        )
    }
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    checkpoint_path = run_dir / "best_model.pt"
    print(
        f"Device: {device}; subjects: {len(dataset)}; "
        f"train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}",
        flush=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(model, loaders[0], device, optimizer)
        val_loss, val_labels, val_probabilities, _ = run_epoch(model, loaders[1], device)
        val_metrics = classification_metrics(val_labels, val_probabilities)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        auc = val_metrics["roc_auc"]
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_auc={auc_text}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
        if epoch - best_epoch >= args.patience:
            print(f"Stopping after {args.patience} epochs without validation improvement")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, labels, probabilities, subject_ids = run_epoch(model, loaders[2], device)
    test_metrics = {"loss": test_loss, **classification_metrics(labels, probabilities)}
    results = {
        "device": str(device), "subjects": len(dataset),
        "split_sizes": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "best_epoch": best_epoch, "validation": {"loss": best_val_loss, **best_val_metrics},
        "test": test_metrics,
        "config": {
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "lambda1": args.lambda1, "lambda2": args.lambda2, "lambda3": args.lambda3,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "label", "probability_asd", "predicted_label"])
        for subject_id, label, probability in zip(subject_ids, labels, probabilities):
            writer.writerow([subject_id, int(label), float(probability), int(probability >= 0.5)])
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Saved checkpoint, metrics, and predictions to {run_dir}")


if __name__ == "__main__":
    main()
