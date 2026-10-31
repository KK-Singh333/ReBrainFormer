"""ABIDE subject loader for cached ROI time series.

Use the ABIDE phenotype CSV and the .npy files produced after extracting ROI
signals from func_preproc.nii.gz scans. Each .npy file must be [time, ROI],
with the same atlas and ROI order across subjects.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def _file_id(path):
    """Recover the ABIDE FILE_ID from supported cached time-series names."""
    name = path.name
    for suffix in (
        "_func_preproc.nii.npy",
        "_func_preproc.npy",
        "_rois_aal.npy",
        "_rois_cc200.npy",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


class AbideDataset(Dataset):
    """Return one ABIDE subject's ROI signals, correlation matrix and label.

    Labels are 0 for control and 1 for ASD. Only time-series files with a
    matching, labeled phenotype row are included. No network or extraction
    work occurs in ``__getitem__``.
    """

    def __init__(self, phenotypic_csv, timeseries_dir, expected_rois=None):
        self.phenotypic_csv = Path(phenotypic_csv)
        self.timeseries_dir = Path(timeseries_dir)
        self.expected_rois = expected_rois

        with self.phenotypic_csv.open(newline="") as stream:
            phenotypes = {
                row["FILE_ID"].strip(): row
                for row in csv.DictReader(stream)
                if row.get("FILE_ID", "").strip() not in ("", "no_filename")
                and row.get("DX_GROUP", "").strip() in ("1", "2")
            }

        if not self.timeseries_dir.is_dir():
            raise FileNotFoundError(f"Time-series directory not found: {self.timeseries_dir}")

        records = []
        seen = set()
        for path in sorted(self.timeseries_dir.glob("*.npy")):
            subject_id = _file_id(path)
            if subject_id is None or subject_id not in phenotypes:
                continue
            if subject_id in seen:
                raise ValueError(f"Multiple time-series files for subject {subject_id}")
            seen.add(subject_id)
            row = phenotypes[subject_id]
            records.append(
                {
                    "path": path,
                    "subject_id": subject_id,
                    "site": row["SITE_ID"].strip(),
                    "label": 1 if row["DX_GROUP"].strip() == "1" else 0,
                }
            )

        if not records:
            raise ValueError(
                "No labeled .npy time series matched the phenotype CSV. "
                "Extract ROI signals from the NIfTI files first."
            )
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        time_by_roi = np.load(record["path"], allow_pickle=False)
        if time_by_roi.ndim != 2:
            raise ValueError(f"Expected [time, ROI] array in {record['path']}; got {time_by_roi.shape}")
        time_points, roi_count = time_by_roi.shape
        if time_points < 2 or roi_count < 2:
            raise ValueError(f"Too few time points or ROIs in {record['path']}: {time_by_roi.shape}")
        if self.expected_rois is not None and roi_count != self.expected_rois:
            raise ValueError(
                f"Expected {self.expected_rois} ROIs in {record['path']}; got {roi_count}"
            )
        if not np.isfinite(time_by_roi).all():
            raise ValueError(f"Non-finite ROI signal in {record['path']}")

        roi_by_time = np.asarray(time_by_roi.T, dtype=np.float32)
        if np.any(np.std(roi_by_time, axis=1) == 0):
            raise ValueError(f"Constant ROI signal in {record['path']}; correlation is undefined")
        correlation = np.corrcoef(roi_by_time).astype(np.float32)
        if not np.isfinite(correlation).all():
            raise ValueError(f"Non-finite ROI correlation in {record['path']}")
        np.fill_diagonal(correlation, 0.0)

        return {
            "timeseries": torch.from_numpy(roi_by_time.copy()),  # [N, T]
            "correlation": torch.from_numpy(correlation),  # [N, N]
            "label": torch.tensor(record["label"], dtype=torch.long),
            "subject_id": record["subject_id"],
            "site": record["site"],
        }


def collate_abide(samples):
    """Pad time to the longest scan in a batch; preserve a validity mask."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    roi_counts = {sample["timeseries"].shape[0] for sample in samples}
    if len(roi_counts) != 1:
        raise ValueError("All subjects in a batch need the same atlas and ROI count")

    batch_size = len(samples)
    roi_count = samples[0]["timeseries"].shape[0]
    max_time = max(sample["timeseries"].shape[1] for sample in samples)
    timeseries = torch.zeros(batch_size, roi_count, max_time, dtype=torch.float32)
    time_mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    for i, sample in enumerate(samples):
        length = sample["timeseries"].shape[1]
        timeseries[i, :, :length] = sample["timeseries"]
        time_mask[i, :length] = True

    return {
        "timeseries": timeseries,  # [B, N, max(T)]
        "time_mask": time_mask,  # [B, max(T)]
        "lengths": torch.tensor([s["timeseries"].shape[1] for s in samples]),
        "correlation": torch.stack([s["correlation"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
        "subject_id": [s["subject_id"] for s in samples],
        "site": [s["site"] for s in samples],
    }


def split_abide(dataset, train_fraction=0.8, val_fraction=0.1, seed=42):
    """Make reproducible, subject-disjoint, diagnosis-stratified subsets."""
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Train and validation fractions must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test set")

    indices = np.arange(len(dataset))
    labels = np.array([record["label"] for record in dataset.records])
    if len(set(labels)) < 2:
        raise ValueError("Splitting requires both ASD and control subjects")
    try:
        train_idx, remainder_idx = train_test_split(
            indices,
            train_size=train_fraction,
            random_state=seed,
            stratify=labels,
        )
        val_ratio = val_fraction / (1 - train_fraction)
        val_idx, test_idx = train_test_split(
            remainder_idx,
            train_size=val_ratio,
            random_state=seed,
            stratify=labels[remainder_idx],
        )
    except ValueError as exc:
        raise ValueError(
            "Not enough subjects per class for a stratified train/validation/test split"
        ) from exc
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), Subset(dataset, test_idx.tolist())
