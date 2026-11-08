import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.rebrainformer import ReBrainFormer
from preprocess.load_graph import _fmri_extractor_metadata
from train_abide import run_epoch


class TrainingReliabilityTests(unittest.TestCase):
    def model(self, **overrides):
        config = dict(Nodes=12, classes=2, lambda1=1, lambda2=1, lambda3=1, dropout=0.2)
        config.update(overrides)
        return ReBrainFormer(SimpleNamespace(**config))

    def test_bottleneck_backward_and_missing_roi(self):
        torch.manual_seed(42)
        model = self.model(hidden_dim=4)
        x = torch.randn(12, 4, 12)
        mask = torch.ones(12, dtype=torch.bool)
        mask[-1] = False
        x[-1, 0] = 0
        x[:, 0, -1] = 0
        loss = model(x, mask).square().sum()
        loss.backward()
        self.assertEqual(model.classifier.in_features, 4)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_keys_and_queries_cannot_affect_prediction(self):
        model = self.model(hidden_dim=4).eval()
        x = torch.randn(12, 4, 12)
        mask = torch.ones(12, dtype=torch.bool)
        mask[-1] = False
        other = x.clone()
        # Invalid ROI feature rows are excluded by key masking and pooling.
        other[-1, 0] = 1e4
        other[:, 1:, -1] = 1e4
        torch.testing.assert_close(model(x, mask), model(other, mask))

    def test_dropout_overrides_and_legacy_dimensions(self):
        model = self.model(attention_dropout=0.1, readout_dropout=0.5)
        self.assertEqual(model.Wq.shape, (12, 12))
        self.assertEqual(model.attention_dropout.p, 0.1)
        self.assertEqual(model.readout_dropout.p, 0.5)
        self.assertEqual(self.model().readout_dropout.p, 0.2)

    def test_xavier_scale(self):
        model = self.model(hidden_dim=4)
        bound = math.sqrt(6 / (12 + 4))
        for weight in (model.Wq, model.Wk, model.Wv):
            self.assertLessEqual(float(weight.detach().abs().max()), bound)

    def test_training_regularizers_and_plain_evaluation_loss(self):
        torch.manual_seed(42)
        model = self.model(hidden_dim=4)
        dataset = TensorDataset(
            torch.randn(4, 12, 4, 12), torch.ones(4, 12, dtype=torch.bool),
            torch.tensor([0, 1, 0, 1]), torch.arange(4),
        )
        loader = DataLoader(dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train = run_epoch(model, loader, "cpu", optimizer, label_smoothing=0.05, grad_clip=1)
        result = run_epoch(model, loader, "cpu", label_smoothing=0.8)
        plain_result = run_epoch(model, loader, "cpu")
        self.assertTrue(np.isfinite(train[0]))
        self.assertAlmostEqual(result[0], plain_result[0])
        np.testing.assert_array_equal(result[1], [0, 1, 0, 1])

    def test_metadata_keeps_every_volume_and_converts_time_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.nii.gz"
            for unit, tr in [("sec", 1.5), ("msec", 1500), ("usec", 1_500_000)]:
                image = nib.Nifti1Image(np.zeros((2, 2, 2, 196), dtype=np.float32), np.eye(4))
                image.header.set_zooms((3, 3, 3, tr))
                image.header.set_xyzt_units("mm", unit)
                nib.save(image, path)
                event = _fmri_extractor_metadata(str(path), {"freq": None, "preproc": "custom", "resolution": "3mm", "registration": "custom"})
                self.assertIsNotNone(event)
                self.assertAlmostEqual(event.frequency, 1 / 1.5)
                self.assertAlmostEqual(event.duration, 294)
                self.assertEqual(event.read().shape[-1], 196)

    def test_explicit_frequency_for_unknown_header_time_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 196)), np.eye(4)), path)
            event = _fmri_extractor_metadata(str(path), {"freq": 0.5, "preproc": "custom", "resolution": "3mm", "registration": "custom"})
            self.assertEqual(event.duration, 392)
            self.assertEqual(event.read().shape[-1], 196)


if __name__ == "__main__":
    unittest.main()
