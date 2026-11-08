"""Train ABIDE classification with fused geometry and shared-encoder graph views."""

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
from modules.contrastive_loss import multiview_contrastive_loss
from preprocess.cache import validate_feature_archive, validate_timeseries


PROJECT_ROOT = Path(__file__).resolve().parent


class AbideFeatureDataset(Dataset):
    """Join labeled ABIDE time-series filenames to pipeline feature archives."""

    def __init__(self, phenotypic_csv, timeseries_dir, feature_dir, meso_bins=32):
        labeled = AbideDataset(phenotypic_csv, timeseries_dir)
        self.meso_bins = meso_bins
        self.records = []
        missing = []
        atlas_config, roi_count = None, None
        for record in labeled.records:
            feature_path = Path(feature_dir) / f"{record['path'].stem}.npz"
            if feature_path.is_file():
                metadata = validate_timeseries(record["path"])
                if atlas_config is None:
                    atlas_config, roi_count = metadata["config"], metadata["shape"][1]
                elif metadata["config"] != atlas_config or metadata["shape"][1] != roi_count:
                    raise ValueError("All subjects must share the extraction configuration, ROI count, and atlas order")
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
            validate_feature_archive(archive, record["path"])
            model_input = archive["model_input"].astype(np.float32)
            roi_mask = archive["roi_mask"].astype(bool)
            local = archive["local_features"].astype(np.float32)
            meso = archive["meso_features"].astype(np.float32)
            entanglement = archive["entanglement"].astype(np.float32)
        n_rois = model_input.shape[0]
        if (
            model_input.shape != (n_rois, 4, n_rois)
            or roi_mask.shape != (n_rois,)
            or not roi_mask.any()
            or not np.isfinite(model_input).all()
            or local.shape != (n_rois, 5) or entanglement.shape != (n_rois,)
            or meso.ndim != 2 or meso.shape[0] != n_rois or meso.shape[1] < 2
            or not all(np.isfinite(value).all() for value in (local, meso, entanglement))
        ):
            raise ValueError(f"Invalid model input or ROI mask: {record['feature_path']}")
        # Resample each ROI's meso representation over normalized scan time.
        # This avoids padding influencing the projector when scan lengths vary.
        meso = F.adaptive_avg_pool1d(torch.from_numpy(meso), self.meso_bins)
        return {
            "model_input": torch.from_numpy(model_input), "roi_mask": torch.from_numpy(roi_mask),
            "local_features": torch.from_numpy(local), "meso_features": meso,
            "entanglement": torch.from_numpy(entanglement),
            "label": record["label"], "subject_id": record["subject_id"],
        }


def run_epoch(model, loader, device, optimizer=None, label_smoothing=0.0, grad_clip=1.0,
              contrastive_weight=0.0, temperature=0.1, return_loss_details=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, probabilities, subject_ids = [], [], []
    classification_total, contrastive_total, contrastive_subjects = 0.0, 0.0, 0

    for batch in loader:
        if isinstance(batch, dict):
            model_input, roi_mask = batch["model_input"], batch["roi_mask"]
            targets, batch_ids = batch["label"], batch["subject_id"]
            features = {name: batch[name].to(device) for name in
                        ("local_features", "meso_features", "entanglement")}
        else:
            model_input, roi_mask, targets, batch_ids = batch
            features = None
        model_input = model_input.to(device)
        roi_mask = roi_mask.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            use_contrastive = training and contrastive_weight > 0 and len(targets) > 1
            if use_contrastive and len(set(batch_ids)) != len(batch_ids):
                raise ValueError("Contrastive batches must contain distinct subjects")
            output = model(model_input, roi_mask, features=features, return_views=use_contrastive)
            logits = output["logits"] if use_contrastive else output
            classification_loss = F.cross_entropy(
                logits, targets, label_smoothing=label_smoothing if training else 0.0
            )
            contrastive_loss = logits.new_zeros(())
            if use_contrastive:
                contrastive_loss = multiview_contrastive_loss(*output["views"], temperature=temperature)
                contrastive_subjects += len(targets)
            loss = classification_loss + contrastive_weight * contrastive_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += loss.item() * len(targets)
        classification_total += classification_loss.item() * len(targets)
        contrastive_total += contrastive_loss.item() * len(targets)
        labels.extend(targets.cpu().tolist())
        probabilities.extend(logits.detach().softmax(dim=1)[:, 1].cpu().tolist())
        subject_ids.extend(batch_ids)

    result = (
        total_loss / len(loader.dataset),
        np.asarray(labels),
        np.asarray(probabilities),
        subject_ids,
    )
    if return_loss_details:
        return (*result, {
            "classification_loss": classification_total / len(loader.dataset),
            "contrastive_loss": contrastive_total / len(loader.dataset),
            "contrastive_subjects": contrastive_subjects,
        })
    return result


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = float("inf"), 0

    def step(self, loss):
        if not np.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        improved = loss < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=32, help="Shared encoder width")
    parser.add_argument("--meso-bins", type=int, default=32)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--correlation-only", action="store_true",
                        help="Use the legacy correlation-only attention baseline without descriptors or biases")
    parser.add_argument("--attention-dropout", type=float, default=0.1,
                        help="Override --dropout for attention weights")
    parser.add_argument("--readout-dropout", type=float, default=0.5,
                        help="Override --dropout before the classifier")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm; 0 disables clipping")
    parser.add_argument("--skip-test", action="store_true",
                        help="Tune using validation only; leave test subjects unevaluated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lambda3", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay nonnegative")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error("--min-learning-rate must be between zero and --learning-rate")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    for name in ("attention_dropout", "readout_dropout"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.hidden_dim is not None and args.hidden_dim < 1:
        parser.error("--hidden-dim must be positive")
    if not 0.0 <= args.label_smoothing < 1.0 or args.grad_clip < 0:
        parser.error("--label-smoothing must be in [0, 1); --grad-clip must be nonnegative")
    if args.meso_bins < 2 or args.contrastive_weight < 0 or args.temperature <= 0 or args.min_delta < 0:
        parser.error("Invalid meso bins, contrastive weight, temperature, or minimum improvement")
    if args.correlation_only:
        args.contrastive_weight = 0.0
        args.lambda1 = args.lambda2 = args.lambda3 = 0.0
    elif args.contrastive_weight > 0 and args.batch_size < 2:
        parser.error("Contrastive learning requires --batch-size >= 2")
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
        args.phenotypic_csv, args.timeseries_dir, args.feature_dir, meso_bins=args.meso_bins
    )
    train_set, val_set, test_set = split_abide(dataset, seed=args.seed)
    loaders = [
        DataLoader(part, batch_size=args.batch_size, shuffle=(i == 0), num_workers=0)
        for i, part in enumerate((train_set, val_set, test_set))
    ]
    train_eval_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    n_rois = dataset[0]["model_input"].shape[0]
    model_args = SimpleNamespace(
        Nodes=n_rois, classes=2,
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        attention_dropout=args.attention_dropout,
        readout_dropout=args.readout_dropout,
        use_descriptors=not args.correlation_only, meso_bins=args.meso_bins,
    )
    model = ReBrainFormer(model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )

    run_dir = args.output_dir / f"abide_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = {
        "Nodes": n_rois,
        "classes": 2,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "dropout": args.dropout,
        "hidden_dim": model.hidden_dim,
        "attention_dropout": model.attention_dropout.p,
        "readout_dropout": model.readout_dropout.p,
        "use_descriptors": model.use_descriptors, "meso_bins": args.meso_bins,
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
    early_stopping = EarlyStopping(args.patience, args.min_delta)
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, _, _, _, loss_details = run_epoch(
            model, loaders[0], device, optimizer,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
            contrastive_weight=args.contrastive_weight, temperature=args.temperature,
            return_loss_details=True,
        )
        # Compare train and validation with dropout off and identical plain CE.
        train_eval_loss, train_labels, train_probabilities, _ = run_epoch(
            model, train_eval_loader, device
        )
        train_metrics = classification_metrics(train_labels, train_probabilities)
        val_loss, val_labels, val_probabilities, _ = run_epoch(model, loaders[1], device)
        val_metrics = classification_metrics(val_labels, val_probabilities)
        history.append({
            "epoch": epoch, "learning_rate": learning_rate,
            "train_loss": train_loss, "val_loss": val_loss, **val_metrics,
            "train_eval_loss": train_eval_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
            **loss_details,
        })
        auc = val_metrics["roc_auc"]
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        print(
            f"Epoch {epoch:02d}: lr={learning_rate:.2e} train_loss={train_loss:.4f} "
            f"train_eval_loss={train_eval_loss:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_auc={auc_text}",
            flush=True,
        )
        improved, should_stop = early_stopping.step(val_loss)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
        scheduler.step()
        if should_stop:
            print(f"Stopping after {args.patience} epochs without validation improvement")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = None
    if not args.skip_test:
        test_loss, labels, probabilities, subject_ids = run_epoch(model, loaders[2], device)
        test_metrics = {"loss": test_loss, **classification_metrics(labels, probabilities)}
        with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject_id", "label", "probability_asd", "predicted_label"])
            for subject_id, label, probability in zip(subject_ids, labels, probabilities):
                writer.writerow([subject_id, int(label), float(probability), int(probability >= 0.5)])
    results = {
        "device": str(device), "subjects": len(dataset),
        "split_sizes": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "best_epoch": best_epoch, "validation": {"loss": best_val_loss, **best_val_metrics},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_class_counts": {
            name: {
                "control": sum(dataset.records[i]["label"] == 0 for i in subset.indices),
                "asd": sum(dataset.records[i]["label"] == 1 for i in subset.indices),
            }
            for name, subset in zip(("train", "validation", "test"), (train_set, val_set, test_set))
        },
        "test": test_metrics,
        "config": {
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "scheduler": "CosineAnnealingLR", "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": model.hidden_dim,
            "attention_dropout": model.attention_dropout.p,
            "readout_dropout": model.readout_dropout.p,
            "label_smoothing": args.label_smoothing, "grad_clip": args.grad_clip,
            "skip_test": args.skip_test,
            "contrastive_weight": args.contrastive_weight, "temperature": args.temperature,
            "use_descriptors": model.use_descriptors, "meso_bins": args.meso_bins,
            "patience": args.patience, "min_delta": args.min_delta,
            "lambda1": args.lambda1, "lambda2": args.lambda2, "lambda3": args.lambda3,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Test evaluation skipped for tuning" if args.skip_test else
          f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
