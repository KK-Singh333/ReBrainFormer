import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from preprocess.cache import validate_timeseries
from preprocess.create_npy import _prepare_raw_data
from preprocess.load_graph import _fmri_extractor_metadata
from preprocess.timing import resolve_scan_timing


def image_with_timing(raw_tr, unit="unknown"):
    image = nib.Nifti1Image(np.zeros((2, 2, 2, 196), dtype=np.float32), np.eye(4))
    image.header.set_zooms((3, 3, 3, raw_tr))
    image.header.set_xyzt_units("mm", unit)
    return image


class ScanTimingTests(unittest.TestCase):
    def test_caltech_missing_units_resolve_only_by_protocol_match(self):
        for value, unit in [(2, "sec"), (2000, "msec"), (2_000_000, "usec")]:
            with self.subTest(value=value):
                timing = resolve_scan_timing("Caltech_0051456_func_preproc.nii.gz", image_with_timing(value).header, {})
                self.assertEqual(timing["frequency_hz"], .5)
                self.assertEqual(timing["effective_time_unit"], unit)
                self.assertEqual(timing["source"], "abide_i_protocol_match")
                self.assertTrue(timing["protocol_url"].endswith("Caltech/rest.pdf"))

    def test_site_specific_and_cmu_multiple_protocols(self):
        for site, raw_tr, seconds in [
            ("Pitt", 1.5, 1.5), ("KKI", 2500, 2.5), ("Leuven_1", 1667, 1.667),
            ("CMU", 1.5, 1.5), ("CMU", 2, 2), ("CMU_a", 1500, 1.5),
            ("MaxMun_a", 3, 3), ("UCLA_2", 3000, 3), ("UM_1", 2, 2),
        ]:
            with self.subTest(site=site, raw_tr=raw_tr):
                result = resolve_scan_timing(f"{site}_0051456_func_preproc.nii.gz", image_with_timing(raw_tr).header, {})
                self.assertAlmostEqual(result["tr_seconds"], seconds)

    def test_unknown_subject_and_nonmatching_tr_are_rejected(self):
        for filename, tr in [("Other_0051456_func_preproc.nii.gz", 2),
                             ("Caltech_0051456_func_preproc.nii.gz", 1),
                             ("Caltech_rest.nii.gz", 2)]:
            with self.assertRaisesRegex(ValueError, "raw header TR="):
                resolve_scan_timing(filename, image_with_timing(tr).header, {})

    def test_header_unit_and_explicit_overrides_have_priority(self):
        name = "Caltech_0051456_func_preproc.nii.gz"
        timing = resolve_scan_timing(name, image_with_timing(3, "sec").header, {"unknown_time_unit": "msec"})
        self.assertEqual(timing["tr_seconds"], 3)
        self.assertEqual(timing["source"], "nifti_header")
        timing = resolve_scan_timing("other.nii.gz", image_with_timing(1700).header, {"unknown_time_unit": "msec"})
        self.assertAlmostEqual(timing["tr_seconds"], 1.7)
        timing = resolve_scan_timing(name, image_with_timing(2).header, {"freq": .25})
        self.assertEqual(timing["tr_seconds"], 4)
        with self.assertRaises(ValueError):
            resolve_scan_timing(name, image_with_timing(2).header, {"abide_protocol_fallback": False})

    def test_full_scan_survives_missing_units_and_provenance_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / "Caltech_0051456_func_preproc.nii.gz"
            nib.save(image_with_timing(2), scan)
            event = _fmri_extractor_metadata(str(scan), {"freq": None, "preproc": "custom", "resolution": "3mm", "registration": "custom"})
            self.assertIsNotNone(event)
            self.assertEqual(event.duration, 392)
            self.assertEqual(event.read().shape[-1], 196)
            with patch("preprocess.create_npy._fmri_extractor", side_effect=lambda event: event.read().get_fdata().reshape(8, -1)):
                path, = _prepare_raw_data(root / "arrays", root)
            self.assertEqual(np.load(path).shape, (196, 8))
            metadata = validate_timeseries(path)
            self.assertEqual(metadata["timing"]["source"], "abide_i_protocol_match")
            self.assertEqual(metadata["timing"]["tr_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
