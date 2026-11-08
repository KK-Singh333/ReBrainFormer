"""Provenance for canonical [time, ROI] arrays and derived graph features."""

import hashlib
import json
from pathlib import Path

import numpy as np


TIMESERIES_VERSION = 1
FEATURE_VERSION = 2


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def metadata_path(path):
    return Path(str(path) + ".json")


def write_timeseries_metadata(path, source, config, frequency, timing=None):
    array = np.load(path, allow_pickle=False)
    metadata = {
        "version": TIMESERIES_VERSION, "axes": "time,roi", "shape": list(array.shape),
        "array_sha256": file_digest(path), "source": source_signature(source),
        "config": config, "frequency_hz": float(frequency),
    }
    if timing is not None:
        metadata["timing"] = timing
    metadata_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def validate_timeseries(path, source=None, config=None):
    path = Path(path)
    try:
        metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"))
        array = np.load(path, allow_pickle=False)
        valid = (
            metadata["version"] == TIMESERIES_VERSION and metadata["axes"] == "time,roi"
            and array.ndim == 2 and min(array.shape) >= 2
            and list(array.shape) == metadata["shape"] and np.isfinite(array).all()
            and metadata["array_sha256"] == file_digest(path)
        )
        if source is not None:
            valid = valid and metadata["source"] == source_signature(source)
        if config is not None:
            valid = valid and metadata["config"] == config
        if not valid:
            raise ValueError("stale or inconsistent metadata")
        return metadata
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Unverified or outdated time-series cache: {path}. Re-extract from the "
            "original NIfTI scan with pipeline.py (omit --skip-extraction). "
            "Transposing a legacy cache cannot restore truncated volumes."
        ) from error


def feature_metadata(timeseries_path, gamma, resolution):
    time_metadata = validate_timeseries(timeseries_path)
    return {
        "version": FEATURE_VERSION, "timeseries_sha256": time_metadata["array_sha256"],
        "timeseries_metadata_sha256": file_digest(metadata_path(timeseries_path)),
        "shape": time_metadata["shape"], "gamma": float(gamma),
        "resolution": float(resolution), "spectral_normalization": "mean_degree",
        "entanglement_bias": "valid_roi_zscore",
    }


def validate_feature_archive(archive, timeseries_path, gamma=None, resolution=None):
    try:
        saved = json.loads(str(archive["metadata_json"].item()))
        expected = feature_metadata(timeseries_path, saved["gamma"], saved["resolution"])
        if saved != expected or (gamma is not None and saved["gamma"] != gamma) or (
            resolution is not None and saved["resolution"] != resolution
        ):
            raise ValueError("feature provenance mismatch")
        required = {"model_input", "roi_mask", "local_features", "entanglement", "meso_features"}
        if not required.issubset(archive.files):
            raise ValueError("missing geometry descriptors")
        return saved
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Outdated graph features; rebuild them with pipeline.py") from error
