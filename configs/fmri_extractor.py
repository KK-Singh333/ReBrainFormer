from exca.map import MapInfra
from neuralset.extractors import neuro

FMRI_EXTRACTOR_CONFIG = {
    "offset": 0.0,
    "fwhm": None,

    "projection": neuro.AtlasProjector(
        atlas="aal",
    ),

    # "cleaning": neuro.FmriCleaner(
    #     standardize="zscore_sample",
    #     detrend=True,
    #     high_pass=None,
    #     low_pass=None,
    #     filter=None,
    #     ensure_finite=True,
    # ),
    "cleaning":None,

    "frequency": "native",

    "padding": None,

    "query": None,

    "infra": MapInfra(
        timeout_min=120,
        cpus_per_task=10,
        version="2",
    ),
}

fMRIConfig = {
        "preproc": "fmriprep",
        "resolution": "2.0mm",
        "registration": "msmall",
        "freq": 0.05,
    }
