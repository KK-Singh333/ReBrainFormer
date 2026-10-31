"""Dataset adapters for brain imaging experiments."""

from .abide import AbideDataset, collate_abide, split_abide

__all__ = ["AbideDataset", "collate_abide", "split_abide"]
