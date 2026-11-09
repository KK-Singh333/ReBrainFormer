import tempfile
import unittest
import warnings
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker

from configs.fmri_extractor import FMRI_EXTRACTOR_CONFIG
from preprocess.atlas import FixedAtlasProjector
from preprocess.cache import metadata_path, validate_timeseries
from preprocess.correlation import roi_correlation
from preprocess.create_npy import _prepare_raw_data


def synthetic_atlas_and_scans():
    # Label 20 occupies a thin parcel at x=1. A grid sampling x=0,2,4 loses it.
    labels = np.zeros((5, 2, 2), dtype=np.int16)
    labels[0] = 10
    labels[1] = 20
    labels[2:] = 30
    atlas = nib.Nifti1Image(labels, np.eye(4))
    t = np.arange(10, dtype=np.float32)
    high_data = np.empty((5, 2, 2, len(t)), dtype=np.float32)
    for x, multiplier in enumerate([1, 2, 3, 3, 3]):
        high_data[x] = multiplier * t
    high = nib.Nifti1Image(high_data, np.eye(4))
    low = nib.Nifti1Image(high_data[::2].copy(), np.diag([2, 1, 1, 1]))
    for scan in (high, low):
        scan.header.set_zooms((*scan.header.get_zooms()[:3], 1.5))
        scan.header.set_xyzt_units("mm", "sec")
    return atlas, high, low, t


class FixedAtlasTests(unittest.TestCase):
    def test_parcels_stay_in_fixed_order_across_subject_grids(self):
        atlas, high, low, t = synthetic_atlas_and_scans()
        with patch.object(FixedAtlasProjector, "get_atlas", return_value=SimpleNamespace(maps=atlas)):
            projector = FixedAtlasProjector(atlas="aal")
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                full = projector.apply(high)
                missing = projector.apply(low)
                full_again = projector.apply(high)
            self.assertEqual(full.shape, missing.shape)
            np.testing.assert_allclose(full, np.stack([t, 2 * t, 3 * t]))
            np.testing.assert_allclose(missing, np.stack([t, np.zeros_like(t), 3 * t]))
            np.testing.assert_array_equal(full_again, full)
            self.assertTrue(any("zero-filled missing ROI labels [20]" in str(w.message) for w in captured))
            self.assertFalse(any("boolean values for 'standardize'" in str(w.message) for w in captured))
        correlation, roi_mask = roi_correlation(missing.T)
        np.testing.assert_array_equal(roi_mask, [True, False, True])
        np.testing.assert_array_equal(correlation[1], 0)

    def test_available_roi_signals_match_standard_masker(self):
        atlas, high, _, _ = synthetic_atlas_and_scans()
        reference = NiftiLabelsMasker(labels_img=atlas, standardize=None, reports=False).fit_transform(high).T
        with patch.object(FixedAtlasProjector, "get_atlas", return_value=SimpleNamespace(maps=atlas)):
            actual = FixedAtlasProjector(atlas="aal").apply(high)
        np.testing.assert_allclose(actual, reference)

    def test_real_neuralset_extraction_keeps_missing_parcels(self):
        atlas, high, low, _ = synthetic_atlas_and_scans()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nib.save(high, root / "Synthetic_0000001_func_preproc.nii.gz")
            nib.save(low, root / "Synthetic_0000002_func_preproc.nii.gz")
            projector = FixedAtlasProjector(atlas="aal")
            with patch.dict(FMRI_EXTRACTOR_CONFIG, {"projection": projector}), patch.object(
                FixedAtlasProjector, "get_atlas", return_value=SimpleNamespace(maps=atlas)
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                paths = _prepare_raw_data(root / "arrays", root)
            self.assertEqual([np.load(path).shape for path in paths], [(10, 3), (10, 3)])
            np.testing.assert_array_equal(np.load(paths[1])[:, 1], 0)
            metadata = validate_timeseries(paths[1])
            self.assertEqual(metadata["config"]["extractor"]["projection"]["column_order"], "sorted_original_atlas_ids")

    def test_old_cache_schema_is_rejected_and_automatically_rebuilt(self):
        atlas, _, low, _ = synthetic_atlas_and_scans()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nib.save(low, root / "Synthetic_0000001_func_preproc.nii.gz")
            projector = FixedAtlasProjector(atlas="aal")
            with patch.dict(FMRI_EXTRACTOR_CONFIG, {"projection": projector}), patch.object(
                FixedAtlasProjector, "get_atlas", return_value=SimpleNamespace(maps=atlas)
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                path, = _prepare_raw_data(root / "arrays", root)
                old = json.loads(metadata_path(path).read_text())
                old["version"] = 1
                metadata_path(path).write_text(json.dumps(old))
                with self.assertRaisesRegex(ValueError, "outdated"):
                    validate_timeseries(path)
                _prepare_raw_data(root / "arrays", root)
            self.assertEqual(validate_timeseries(path)["version"], 2)
            self.assertEqual(np.load(path).shape, (10, 3))


if __name__ == "__main__":
    unittest.main()
