import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from torch.utils.data import DataLoader

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from models.rebrainformer import ReBrainFormer
from modules.contrastive_loss import multiview_contrastive_loss
from pipeline import entanglement_attention_bias, node_entanglement, process_subject
from preprocess.cache import (
    feature_metadata, metadata_path, validate_feature_archive, validate_timeseries,
)
from preprocess.create_npy import _prepare_raw_data
from modules.local_enc import _serial_ricci_curvature
from train_abide import EarlyStopping, parse_args, run_epoch


def make_scan(path, volumes=24):
    data = np.random.default_rng(42).normal(size=(2, 2, 2, volumes)).astype(np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((3, 3, 3, 1.5))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)


def synthetic_projector(event):
    # No atlas download: treat the synthetic image's eight voxels as eight ROIs.
    return event.read().get_fdata().reshape(8, -1)


class GeometryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ReBrainFormer(SimpleNamespace(
            Nodes=8, classes=2, lambda1=0.1, lambda2=0.1, lambda3=0.1,
            hidden_dim=4, dropout=0.1, use_descriptors=True, meso_bins=6,
        ))
        self.x = torch.randn(4, 8, 4, 8)
        self.mask = torch.ones(4, 8, dtype=torch.bool)
        self.mask[:, -1] = False
        self.features = {
            "local_features": torch.randn(4, 8, 5),
            "meso_features": torch.randn(4, 8, 6),
            "entanglement": torch.rand(4, 8),
        }

    def test_every_descriptor_and_shared_encoder_receive_training_gradients(self):
        result = self.model(self.x, self.mask, features=self.features, return_views=True)
        self.assertEqual(result["logits"].shape, (4, 2))
        self.assertEqual([tuple(view.shape) for view in result["views"]], [(4, 4)] * 3)
        contrastive = multiview_contrastive_loss(*result["views"])
        loss = F.cross_entropy(result["logits"], torch.tensor([0, 1, 0, 1])) + 0.05 * contrastive
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (self.model.local_projector, self.model.global_projector,
                       self.model.meso_projector, self.model.contrastive_projector, self.model.qkv):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)

    def test_predictions_preserve_subject_alignment_and_single_batch_equivalence(self):
        self.model.eval()
        result = self.model(self.x, self.mask, features=self.features)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = self.model(self.x[permutation], self.mask[permutation], features={
            key: value[permutation] for key, value in self.features.items()
        })
        torch.testing.assert_close(result[permutation], permuted)
        single = self.model(self.x[0], self.mask[0], features={key: value[0] for key, value in self.features.items()})
        torch.testing.assert_close(result[0], single)

    def test_missing_roi_descriptors_do_not_change_predictions(self):
        self.model.eval()
        result = self.model(self.x, self.mask, features=self.features)
        changed = {key: value.clone() for key, value in self.features.items()}
        for value in changed.values():
            value[:, -1] = 1e6
        torch.testing.assert_close(result, self.model(self.x, self.mask, features=changed))

    def test_subject_positive_pair_order_matters(self):
        identity = torch.eye(4)
        aligned = multiview_contrastive_loss(identity, identity, identity)
        shuffled = multiview_contrastive_loss(identity, identity.roll(1, 0), identity.roll(2, 0))
        self.assertLess(float(aligned), float(shuffled))

    def test_training_loop_optimizes_contrastive_head_and_handles_singleton(self):
        samples = [
            {"model_input": self.x[i], "roi_mask": self.mask[i], "label": i % 2,
             "subject_id": f"subject_{i}", **{key: value[i] for key, value in self.features.items()}}
            for i in range(4)
        ]
        loader = DataLoader(samples, batch_size=3)
        original = self.model.contrastive_projector[0].weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        total, _, _, _, details = run_epoch(
            self.model, loader, "cpu", optimizer, contrastive_weight=.2, return_loss_details=True,
        )
        self.assertEqual(details["contrastive_subjects"], 3)
        self.assertGreater(details["contrastive_loss"], 0)
        self.assertAlmostEqual(total, details["classification_loss"] + .2 * details["contrastive_loss"], places=6)
        self.assertFalse(torch.equal(original, self.model.contrastive_projector[0].weight))

    def test_laplacian_scale_invariance_and_meaningful_bias(self):
        adjacency = np.array([[0, .2, .6], [.2, 0, .8], [.6, .8, 0]])
        first = node_entanglement(adjacency, gamma=1)
        second = node_entanglement(adjacency * 1000, gamma=1)
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-7)
        mask = np.array([True, True, True, False])
        bias = entanglement_attention_bias(np.r_[first, 0], mask)
        self.assertAlmostEqual(float(bias[0, mask].mean()), 0, places=5)
        self.assertAlmostEqual(float(bias[0, mask].std()), 1, places=5)
        np.testing.assert_array_equal(bias[:, -1], 0)
        np.testing.assert_array_equal(entanglement_attention_bias(np.ones(4), mask), 0)

    def test_early_stopping_ignores_negligible_improvements(self):
        stopping = EarlyStopping(patience=2, min_delta=1e-3)
        self.assertEqual(stopping.step(.6), (True, False))
        self.assertEqual(stopping.step(.59999), (False, False))
        self.assertEqual(stopping.step(.59998), (False, True))
        with patch("sys.argv", ["train_abide.py"]):
            args = parse_args()
        self.assertLess(args.patience, args.epochs)
        self.assertGreater(args.contrastive_weight, 0)


class CacheReliabilityTests(unittest.TestCase):
    def test_legacy_and_tampered_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / "SYNTH_0000001_func_preproc.nii.gz"
            make_scan(scan)
            cache = root / "arrays"
            cache.mkdir()
            legacy = cache / "SYNTH_0000001_func_preproc.nii.npy"
            np.save(legacy, np.zeros((8, 50)))
            with self.assertRaisesRegex(ValueError, "Re-extract"):
                validate_timeseries(legacy)
            with patch("preprocess.create_npy._fmri_extractor", side_effect=synthetic_projector) as extractor:
                paths = _prepare_raw_data(cache, root)
                self.assertEqual(extractor.call_count, 1)
                self.assertEqual(np.load(paths[0]).shape, (24, 8))
                _prepare_raw_data(cache, root)
                self.assertEqual(extractor.call_count, 1)
                np.save(paths[0], np.load(paths[0]).T)
                with self.assertRaises(ValueError):
                    validate_timeseries(paths[0])
                _prepare_raw_data(cache, root)
                self.assertEqual(extractor.call_count, 2)

    def test_extraction_rejects_lost_volumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_scan(root / "SYNTH_0000001_func_preproc.nii.gz", volumes=196)
            with patch("preprocess.create_npy._fmri_extractor", return_value=np.ones((8, 50))):
                with self.assertRaisesRegex(ValueError, "lost volumes"):
                    _prepare_raw_data(root / "arrays", root)

    def test_feature_cache_invalidates_on_parameters_and_timeseries_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_scan(root / "SYNTH_0000001_func_preproc.nii.gz")
            with patch("preprocess.create_npy._fmri_extractor", side_effect=synthetic_projector):
                path, = _prepare_raw_data(root / "arrays", root)
            with patch("pipeline.local_enc.computing_curvature", side_effect=_serial_ricci_curvature):
                process_subject(path, root / "graphs", root / "features", 1, 1)
            feature_path = root / "features" / f"{path.stem}.npz"
            with np.load(feature_path) as archive:
                metadata = validate_feature_archive(archive, path, 1, 1)
                self.assertEqual(metadata["spectral_normalization"], "mean_degree")
                self.assertGreater(float(archive["spectral_scale"]), 0)
                with self.assertRaisesRegex(ValueError, "rebuild"):
                    validate_feature_archive(archive, path, .1, 1)
                sidecar = json.loads(metadata_path(path).read_text())
                sidecar["frequency_hz"] *= 2
                metadata_path(path).write_text(json.dumps(sidecar))
                with self.assertRaisesRegex(ValueError, "rebuild"):
                    validate_feature_archive(archive, path, 1, 1)


if __name__ == "__main__":
    unittest.main()
