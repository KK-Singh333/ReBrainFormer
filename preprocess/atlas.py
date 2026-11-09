"""Deterministic atlas projection with a stable ROI order across subjects."""

import warnings
from os import PathLike

import nibabel as nib
import numpy as np
from neuralset.extractors import neuro
from nilearn.maskers import NiftiLabelsMasker
from pydantic import PrivateAttr


class FixedAtlasProjector(neuro.AtlasProjector):
    """Zero-fill parcels lost during resampling rather than dropping columns.

    Uses sorted non-background IDs from the ORIGINAL atlas, not from a
    particular subject's resampled atlas. Missing columns later produce a
    False ROI mask through the existing constant-signal handling.
    """

    column_order: str = "sorted_original_atlas_ids"
    projection_version: int = 1
    _roi_ids: list[int] = PrivateAttr(default_factory=list)

    def _get_masker(self):
        if self._masker is None:
            maps = self.get_atlas().maps
            maps = nib.load(maps) if isinstance(maps, (str, bytes, PathLike)) else maps
            labels = np.asarray(maps.dataobj)
            if labels.ndim != 3:
                raise ValueError("FixedAtlasProjector requires a deterministic 3D label atlas")
            if not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
                raise ValueError("Atlas label IDs must be finite integers")
            self._roi_ids = [int(value) for value in np.unique(labels) if value != 0]
            if not self._roi_ids:
                raise ValueError("Atlas contains no foreground parcels")
            self._masker = NiftiLabelsMasker(
                labels_img=maps, background_label=0, resampling_target="data",
                standardize=None, reports=False,
            )
        return self._masker

    def apply(self, rec):
        if len(rec.shape) != 4:
            raise ValueError(f"Atlas projection requires 4D data, got {rec.shape}")
        masker = self._get_masker()
        signals = np.asarray(masker.fit_transform(rec))  # [T, surviving ROI]
        ids = [int(masker.region_ids_[column]) for column in range(signals.shape[1])]
        canonical = {label: column for column, label in enumerate(self._roi_ids)}
        if len(set(ids)) != len(ids) or not set(ids).issubset(canonical):
            raise ValueError("Masker returned duplicate or unknown atlas labels")
        result = np.zeros((len(self._roi_ids), signals.shape[0]), dtype=np.float32)
        for column, label in enumerate(ids):
            result[canonical[label]] = signals[:, column]
        missing = sorted(set(self._roi_ids) - set(ids))
        if missing:
            warnings.warn(
                f"Fixed atlas order retained; zero-filled missing ROI labels {missing}",
                UserWarning, stacklevel=2,
            )
        return result
