"""Subject-level contrastive losses for aligned batches of graph views."""

import torch
import torch.nn.functional as F


def pairwise_contrastive_loss(z_a, z_b, temperature=0.1):
    """Compare two [B, D] embedding batches with matching subject order.

    Rows represent distinct subjects, not individual ROIs. Inputs must remain
    connected to their encoder to train it through this loss.
    """
    if z_a.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("Both inputs must have the same shape [B, D].")

    batch_size = z_a.size(0)
    if batch_size < 2:
        raise ValueError("Contrastive loss needs at least two subjects.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    logits = (z_a @ z_b.T) / temperature
    targets = torch.arange(batch_size, device=logits.device)

    loss_a_to_b = F.cross_entropy(logits, targets)
    loss_b_to_a = F.cross_entropy(logits.T, targets)
    return (loss_a_to_b + loss_b_to_a) / 2


def multiview_contrastive_loss(
    z_local, z_global, z_meso, temperature=0.1
):
    """Average symmetric losses across all three pairs of [B, D] views."""
    loss_local_global = pairwise_contrastive_loss(
        z_local, z_global, temperature
    )
    loss_local_meso = pairwise_contrastive_loss(
        z_local, z_meso, temperature
    )
    loss_global_meso = pairwise_contrastive_loss(
        z_global, z_meso, temperature
    )
    return (loss_local_global + loss_local_meso + loss_global_meso) / 3
