import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from neuralset import events, extractors

from configs.fmri_extractor import FMRI_EXTRACTOR_CONFIG, fMRIConfig


def _fmri_extractor_metadata(
    filepath: str,
    fMRIConfig: dict,
) -> events.etypes.Fmri | None:
    """
    Creates an Fmri event from a NIfTI file.
    """

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        subject = os.path.splitext(os.path.basename(filepath))[0]

        image = nib.load(filepath)
        if len(image.shape) != 4 or image.shape[-1] < 2:
            raise ValueError(f"Expected a 4D fMRI scan with at least two volumes: {image.shape}")
        frequency = fMRIConfig.get("freq")
        if frequency is None:
            tr = float(image.header.get_zooms()[3])
            time_unit = image.header.get_xyzt_units()[1]
            seconds_per_unit = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
            if time_unit not in seconds_per_unit:
                raise ValueError(
                    f"Unknown NIfTI time unit {time_unit!r}; verify TR and set "
                    "fMRIConfig['freq'] explicitly in Hz"
                )
            tr *= seconds_per_unit[time_unit]
            if not np.isfinite(tr) or tr <= 0:
                raise ValueError(f"Invalid NIfTI TR: {tr}")
            frequency = 1.0 / tr
        frequency = float(frequency)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"Invalid fMRI sampling frequency: {frequency}")

        fmri_event = events.etypes.Fmri(
           
            start=0.0,
            timeline="t1",
            duration=image.shape[-1] / frequency,
            filepath=filepath,
            subject=subject,
            frequency=frequency,
            space="T1w",
            preproc=fMRIConfig["preproc"],
            mask_filepath=None,
            spec={
                "resolution": fMRIConfig["resolution"],
                "registration": fMRIConfig["registration"],
            },
        )
        # print(fmri_event.read())
        return fmri_event

    except Exception as e:
        print(e)
        return None


def _fmri_extractor(event: events.etypes.Fmri) -> np.ndarray | None:
    """
    Extract ROI time series from an fMRI event.
    """

    extractor = extractors.neuro.FmriExtractor(
        **FMRI_EXTRACTOR_CONFIG
    )
    # print('hiiiii')

    try:
        fmri = extractor(
            event,
            start=event.start,
            duration=event.duration,
        )
        
        
        return fmri

    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ROI signals from one NIfTI scan.")
    parser.add_argument("nifti_path", type=Path, help="Path to a .nii or .nii.gz scan")
    args = parser.parse_args()

    event = _fmri_extractor_metadata(
        str(args.nifti_path),
        fMRIConfig,
    )

    if event is None:
        raise RuntimeError("Failed to create Fmri event.")

    fmri = _fmri_extractor(event)

    if fmri is not None:
        
        print(type(fmri))
        print(fmri.shape)
        print(fmri)
