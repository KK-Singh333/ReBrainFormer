import os
import numpy as np
from neuralset import events, extractors

from configs.fmri_extractor import FMRI_EXTRACTOR_CONFIG


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

        fmri_event = events.etypes.Fmri(
           
            start=0.0,
            timeline="t1",
            duration=1000,
            filepath=filepath,
            subject=subject,
            frequency=float(fMRIConfig["freq"]),
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
   
    test_file_path = (
        r"E:\ReBrainFormer\ReBrainFormer\test_data"
        r"\ABIDE_51038_MRI_Resting_State_fMRI_br_raw_20120829200031740_S163361_I327304.nii"
    )

    fMRIConfig = {
        "preproc": "fmriprep",
        "resolution": "2.0mm",
        "registration": "msmall",
        "freq": 0.05,
    }

    event = _fmri_extractor_metadata(
        test_file_path,
        fMRIConfig,
    )

    if event is None:
        raise RuntimeError("Failed to create Fmri event.")

    fmri = _fmri_extractor(event)

    if fmri is not None:
        
        print(type(fmri))
        print(fmri.shape)
        print(fmri)