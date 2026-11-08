import math
import torch
import torch.nn as nn


class MesoProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # x: (N, T)
        return self.proj(x)  # (N, out_dim)


class CurvatureMLP(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=32, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReBrainFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        hidden_dim = getattr(args, "hidden_dim", None)
        self.hidden_dim = args.Nodes if hidden_dim is None else hidden_dim
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.use_descriptors = bool(getattr(args, "use_descriptors", False))
        self.classes = args.classes
        dropout = float(getattr(args, "dropout", 0.0))
        attention_dropout = getattr(args, "attention_dropout", None)
        readout_dropout = getattr(args, "readout_dropout", None)
        attention_dropout = dropout if attention_dropout is None else attention_dropout
        readout_dropout = dropout if readout_dropout is None else readout_dropout
        if any(not 0.0 <= value < 1.0 for value in (dropout, attention_dropout, readout_dropout)):
            raise ValueError("dropout rates must be in [0, 1)")
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.readout_dropout = nn.Dropout(readout_dropout)
        self.classifier = nn.Linear(self.hidden_dim, self.classes)
        if self.use_descriptors:
            self.input_projection = nn.Linear(args.Nodes, self.hidden_dim)
            self.local_projector = CurvatureMLP(out_dim=self.hidden_dim)
            self.global_projector = nn.Linear(1, self.hidden_dim)
            self.meso_projector = MesoProjector(getattr(args, "meso_bins", 32), self.hidden_dim)
            self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
            self.input_norm = nn.LayerNorm(self.hidden_dim)
            self.ff_norm = nn.LayerNorm(self.hidden_dim)
            self.output_norm = nn.LayerNorm(self.hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
            self.contrastive_projector = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            # Compatibility path for correlation-only runs and older checkpoints.
            self.Wq = nn.Parameter(torch.empty(args.Nodes, self.hidden_dim))
            self.Wv = nn.Parameter(torch.empty(args.Nodes, self.hidden_dim))
            self.Wk = nn.Parameter(torch.empty(args.Nodes, self.hidden_dim))
            for weight in (self.Wq, self.Wk, self.Wv):
                nn.init.xavier_uniform_(weight)

    def self_attention_map(self, H, K_bias, N_E, C, roi_mask=None):
        Q = H @ self.Wq
        K_proj = H @ self.Wk
        V = H @ self.Wv
        attention_scores = (
            (Q @ K_proj.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            + self.lambda1 * K_bias
            + self.lambda2 * N_E
            + self.lambda3 * C
        )
        if roi_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~roi_mask.unsqueeze(-2), torch.finfo(attention_scores.dtype).min
            )
        attention_weights = self.attention_dropout(
            torch.softmax(attention_scores, dim=-1)
        )
        return attention_weights @ V

    @staticmethod
    def pool(nodes, roi_mask):
        weights = roi_mask.unsqueeze(-1).to(nodes.dtype)
        return (nodes * weights).sum(dim=-2) / weights.sum(dim=-2)

    def encode(self, nodes, bias, roi_mask):
        """The same encoder is used for fused, local, global, and meso views."""
        q, k, v = self.qkv(self.input_norm(nodes)).chunk(3, dim=-1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.hidden_dim) + bias
        scores = scores.masked_fill(~roi_mask.unsqueeze(-2), torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        nodes = nodes + attention @ v
        nodes = nodes + self.feedforward(self.ff_norm(nodes))
        return self.pool(self.output_norm(nodes), roi_mask)

    def forward(self, x, roi_mask=None, features=None, return_views=False):
        if x.shape[-3:] != (self.args.Nodes, 4, self.args.Nodes):
            raise ValueError("Expected [N, 4, N] or [B, N, 4, N] model input")
        if roi_mask is None:
            roi_mask = torch.ones(x.shape[:-2], dtype=torch.bool, device=x.device)
        if roi_mask is not None:
            roi_mask = roi_mask.to(device=x.device, dtype=torch.bool)
            if roi_mask.shape != x.shape[:-2] or not roi_mask.any(dim=-1).all():
                raise ValueError("roi_mask must match nodes with at least one valid ROI per subject")
        H, K_bias, NE, C = x.unbind(dim=-2)
        # Mask both axes before any projection so missing features cannot leak in.
        H = H * roi_mask.unsqueeze(-1) * roi_mask.unsqueeze(-2)
        if self.use_descriptors:
            if features is None:
                raise ValueError("Geometry descriptors are required; rebuild features with pipeline.py")
            local = self.local_projector(features["local_features"].masked_fill(~roi_mask.unsqueeze(-1), 0))
            meso = self.meso_projector(features["meso_features"].masked_fill(~roi_mask.unsqueeze(-1), 0))
            # Recompute normalized NE from the raw descriptor on valid ROIs only.
            entanglement = features["entanglement"]
            count = roi_mask.sum(-1, keepdim=True)
            mean = (entanglement * roi_mask).sum(-1, keepdim=True) / count
            centered = (entanglement - mean) * roi_mask
            spread = (centered.square().sum(-1, keepdim=True) / count).sqrt()
            normalized = torch.where(spread > 1e-8, centered / spread.clamp_min(1e-8), 0)
            global_embedding = self.global_projector(normalized.unsqueeze(-1))
            NE = normalized.unsqueeze(-2).expand_as(NE)
            base = self.input_projection(H)
            local_bias, global_bias, meso_bias = self.lambda1 * K_bias, self.lambda2 * NE, self.lambda3 * C
            pooled = self.encode(
                base + (local + global_embedding + meso) / math.sqrt(3),
                local_bias + global_bias + meso_bias, roi_mask,
            )
            logits = self.classifier(self.readout_dropout(pooled))
            if return_views:
                views = [
                    self.contrastive_projector(self.encode(base + descriptor, bias, roi_mask))
                    for descriptor, bias in (
                        (local, local_bias), (global_embedding, global_bias), (meso, meso_bias)
                    )
                ]
                return {"logits": logits, "embedding": pooled, "views": views}
            return logits
        if return_views:
            raise ValueError("Contrastive views require the descriptor encoder")
        Emb = self.self_attention_map(H, K_bias, NE, C, roi_mask=roi_mask)
        pooled = self.pool(Emb, roi_mask)
        return self.classifier(self.readout_dropout(pooled))
