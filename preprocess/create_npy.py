from pathlib import Path

import numpy as np

from configs.fmri_extractor import fMRIConfig
from preprocess.load_graph import _fmri_extractor, _fmri_extractor_metadata


def _prepare_raw_data(target_dir, data_dir, limit=None, overwrite=False):
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

    for index, scan in enumerate(scans, start=1):
        # Keep the existing .nii.npy naming expected by AbideDataset.
        output = target_dir / f"{scan.name.removesuffix('.gz')}.npy"

        if output.exists() and not overwrite:
            print(f"[{index}/{len(scans)}] Reusing {output}")
            outputs.append(output)
            continue

        print(f"[{index}/{len(scans)}] Extracting {scan}")
        event = _fmri_extractor_metadata(str(scan), fMRIConfig)
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

        np.save(output, array.T.copy())
        outputs.append(output)

    return outputs
