"""Resolve missing ABIDE-I time units by matching published site protocols.

Only the unit is inferred. The numeric TR in the header is preserved, and
unrecognized filenames or TRs are rejected rather than assigned a default.
"""

import re
from pathlib import Path

import numpy as np


TIMING_VERSION = 1
SECONDS_PER_UNIT = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
PROTOCOL_ROOT = "https://fcon_1000.projects.nitrc.org/indi/abide/scan_params"
# Actual functional TRs (not minimum-TR fields) from the linked ABIDE-I protocols.
# CMU's protocol PDF documents both 1500 ms and 2000 ms acquisitions.
ABIDE_PROTOCOLS = {
    "caltech": ((2.0,), "Caltech"),
    "cmu": ((1.5, 2.0), "CMU"),
    "kki": ((2.5,), "KKI"),
    "maxmun": ((3.0,), "MaxMun"),
    "nyu": ((2.0,), "NYU"),
    "ohsu": ((2.5,), "OHSU"),
    "olin": ((1.5,), "Olin"),
    "sbl": ((2.2,), "SBL"),
    "sdsu": ((2.0,), "SDSU"),
    "stanford": ((2.0,), "Stanford"),
    "trinity": ((2.0,), "Trinity"),
    "ucla_1": ((3.0,), "UCLA1"),
    "ucla_2": ((3.0,), "UCLA2"),
    "leuven_1": ((1.667,), "Leuven-1"),
    "leuven_2": ((1.667,), "Leuven-2"),
    "um_1": ((2.0,), "UM1"),
    "um_2": ((2.0,), "UM2"),
    "pitt": ((1.5,), "Pitt"),
    "usm": ((2.0,), "USM"),
    "yale": ((2.0,), "Yale"),
}

# Leuven uses TR="shortest". Both site PDFs report minimum TR=1656 ms and
# actual TR=1667 ms, so scan-specific headers can fall between those values.
# These bounds identify the missing UNIT; they never replace the header TR.
ABIDE_TR_RANGES = {
    "leuven_1": (1.656, 1.667),
    "leuven_2": (1.656, 1.667),
}


def resolve_scan_timing(filepath, header, config):
    raw_tr = float(header.get_zooms()[3])
    header_unit = header.get_xyzt_units()[1]
    result = {"raw_tr": raw_tr, "header_time_unit": header_unit}
    if config.get("freq") is not None:
        frequency = float(config["freq"])
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"Invalid fMRI sampling frequency: {frequency}")
        return {**result, "frequency_hz": frequency, "tr_seconds": 1 / frequency,
                "source": "explicit_frequency"}
    if not np.isfinite(raw_tr) or raw_tr <= 0:
        raise ValueError(f"Invalid NIfTI TR: {raw_tr}")

    unit = header_unit
    source = "nifti_header"
    if header_unit == "unknown" and config.get("unknown_time_unit"):
        unit, source = config["unknown_time_unit"], "explicit_unknown_unit"
    elif header_unit == "unknown" and config.get("abide_protocol_fallback", True):
        match = re.fullmatch(r"(.+)_\d{7}_func_preproc\.nii(?:\.gz)?", Path(filepath).name, re.IGNORECASE)
        site = match[1].lower() if match else ""
        if site.startswith("cmu_"):
            site = "cmu"
        elif site.startswith("maxmun_"):
            site = "maxmun"
        if site in ABIDE_PROTOCOLS:
            allowed, protocol = ABIDE_PROTOCOLS[site]
            tr_range = ABIDE_TR_RANGES.get(site)
            matches = [name for name, scale in SECONDS_PER_UNIT.items()
                       if any(np.isclose(raw_tr * scale, tr, rtol=1e-3, atol=1e-6) for tr in allowed)
                       or (tr_range is not None and tr_range[0] - 1e-6 <= raw_tr * scale <= tr_range[1] + 1e-6)]
            if len(matches) == 1:
                unit, source = matches[0], "abide_i_protocol_match"
                result["protocol_url"] = f"{PROTOCOL_ROOT}/{protocol}/rest.pdf"
                if tr_range is not None:
                    result["protocol_tr_range_seconds"] = list(tr_range)
    if unit not in SECONDS_PER_UNIT:
        raise ValueError(
            f"Cannot resolve NIfTI time unit {header_unit!r} for {Path(filepath).name}; "
            f"raw header TR={raw_tr}. No unique published ABIDE-I protocol match. "
            "Verify the acquisition metadata, then pass --unknown-time-unit sec, msec, "
            "or usec, or set fMRIConfig['freq'] explicitly in Hz."
        )
    tr = raw_tr * SECONDS_PER_UNIT[unit]
    return {**result, "frequency_hz": 1 / tr, "tr_seconds": tr,
            "effective_time_unit": unit, "source": source}
