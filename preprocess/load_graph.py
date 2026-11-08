import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from neuralset import events, extractors

from configs.fmri_extractor import FMRI_EXTRACTOR_CONFIG, fMRIConfig
from preprocess.timing import resolve_scan_timing


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
        timing = resolve_scan_timing(filepath, image.header, fMRIConfig)
        frequency = timing["frequency_hz"]
        if timing["source"] != "nifti_header":
            print(f"{os.path.basename(filepath)}: TR={timing['tr_seconds']:g}s ({timing['source']})")

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
