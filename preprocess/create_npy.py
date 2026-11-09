from pathlib import Path

import numpy as np
import nibabel as nib

from configs.fmri_extractor import fMRIConfig
from preprocess.load_graph import _fmri_extractor, _fmri_extractor_metadata
from preprocess.cache import validate_timeseries, write_timeseries_metadata
from configs.fmri_extractor import FMRI_EXTRACTOR_CONFIG
from preprocess.timing import TIMING_VERSION, resolve_scan_timing


def extraction_config(scan_config=None):
    settings = {key: value for key, value in FMRI_EXTRACTOR_CONFIG.items() if key != "infra"}
    return {
        "scan": dict(fMRIConfig if scan_config is None else scan_config),
        "timing_resolver_version": TIMING_VERSION,
        "extractor": {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for key, value in settings.items()
        },
    }


def audit_scan_timing(scans, scan_config):
    """Check every header before doing expensive extraction or writing arrays."""
    print(f"Checking timing for {len(scans)} scans before extraction...", flush=True)
    timings, errors = {}, []
    for index, scan in enumerate(scans, start=1):
        try:
            image = nib.load(scan)
            if len(image.shape) != 4 or image.shape[-1] < 2:
                raise ValueError(f"Expected 4D fMRI with at least two volumes; got {image.shape}")
            timings[scan] = resolve_scan_timing(scan, image.header, scan_config)
        except (OSError, ValueError) as error:
            errors.append(f"{scan.name}: {error}")
        if index % 100 == 0:
            print(f"Checked {index}/{len(scans)} scan headers", flush=True)
    if errors:
        raise ValueError(
            f"Timing preflight failed for {len(errors)}/{len(scans)} scans. "
            "No new arrays were extracted. First errors:\n" + "\n".join(errors[:10])
        )
    return timings


def _prepare_raw_data(target_dir, data_dir, limit=None, overwrite=False, unknown_time_unit=None):
    data_dir = Path(data_dir)
    target_dir = Path(target_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {data_dir}")

    scans = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )
    if not scans:
        raise FileNotFoundError(f"No .nii or .nii.gz scans in {data_dir}")
    if limit is not None:
        scans = scans[:limit]

    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    scan_config = dict(fMRIConfig)
    if unknown_time_unit is not None:
        scan_config["unknown_time_unit"] = unknown_time_unit
    config = extraction_config(scan_config)
    timings = audit_scan_timing(scans, scan_config)

    for index, scan in enumerate(scans, start=1):
        # Keep the existing .nii.npy naming expected by AbideDataset.
        output = target_dir / f"{scan.name.removesuffix('.gz')}.npy"

        if output.exists() and not overwrite:
            try:
                validate_timeseries(output, source=scan, config=config)
            except ValueError:
                print(f"[{index}/{len(scans)}] Re-extracting outdated cache {output}")
            else:
                print(f"[{index}/{len(scans)}] Reusing verified {output}")
                outputs.append(output)
                continue

        print(f"[{index}/{len(scans)}] Extracting {scan}")
        event = _fmri_extractor_metadata(str(scan), scan_config)
        if event is None:
            raise RuntimeError(f"Could not create fMRI event for {scan}")

        fmri = _fmri_extractor(event)
        if fmri is None:
            raise RuntimeError(f"Could not extract ROI signals from {scan}")

        # Neuralset's AtlasProjector returns [ROI, time]. The ABIDE loader
        # and graph pipeline consume [time, ROI].
        array = np.asarray(fmri.numpy() if hasattr(fmri, "numpy") else fmri)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Invalid [ROI, time] array from {scan}: {array.shape}")
        expected_volumes = round(event.duration * event.frequency)
        if array.shape[1] != expected_volumes:
            raise ValueError(
                f"Extraction lost volumes for {scan}: expected {expected_volumes}, got {array.shape[1]}"
            )

        np.save(output, array.T.copy())
        timing = timings[scan]
        write_timeseries_metadata(output, scan, config, event.frequency, timing=timing)
        outputs.append(output)

    return outputs
