"""ROI correlations that retain a fixed atlas when some signals are constant."""

import numpy as np


def roi_correlation(time_by_roi, min_std=1e-8):
    """Return [ROI, ROI] correlations and a valid-ROI mask.

    A constant signal has undefined Pearson correlations. Its row and column
    remain zero, so the atlas size and ROI ordering stay the same for every
    subject. Downstream models must use the returned mask when pooling and
    attending to nodes.
    """
    signals = np.asarray(time_by_roi, dtype=np.float64)
    if signals.ndim != 2 or min(signals.shape) < 2:
        raise ValueError(f"Expected [time, ROI] with at least 2 of each; got {signals.shape}")
    if not np.isfinite(signals).all():
        raise ValueError("ROI time series contains non-finite values")

    valid = signals.std(axis=0) > min_std
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        raise ValueError("Fewer than two ROIs have varying signals")

    correlation = np.zeros((signals.shape[1], signals.shape[1]), dtype=np.float32)
    block = np.corrcoef(signals[:, valid_indices], rowvar=False)
    if not np.isfinite(block).all():
        raise ValueError("Correlations remain non-finite after excluding constant ROIs")
    correlation[np.ix_(valid_indices, valid_indices)] = block.astype(np.float32)
    np.fill_diagonal(correlation, 0.0)
    return correlation, valid
