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
        self.Wq = nn.Parameter(torch.randn(args.Nodes, args.Nodes))
        self.Wv = nn.Parameter(torch.randn(args.Nodes, args.Nodes))
        self.Wk = nn.Parameter(torch.randn(args.Nodes, args.Nodes))
        self.classes = args.classes
        dropout = float(getattr(args, "dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.attention_dropout = nn.Dropout(dropout)
        self.readout_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(args.Nodes, self.classes)

    def self_attention_map(self, H, K_bias, N_E, C, roi_mask=None):
        Q = H @ self.Wq
        K_proj = H @ self.Wk
        V = H @ self.Wv
        attention_scores = (
            (Q @ K_proj.T) / math.sqrt(self.args.Nodes)
            + self.lambda1 * K_bias
            + self.lambda2 * N_E
            + self.lambda3 * C
        )
        if roi_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~roi_mask.unsqueeze(0), torch.finfo(attention_scores.dtype).min
            )
        attention_weights = self.attention_dropout(
            torch.softmax(attention_scores, dim=-1)
        )
        return attention_weights @ V

    def forward(self, x, roi_mask=None):
        if roi_mask is not None:
            roi_mask = roi_mask.to(device=x.device, dtype=torch.bool)
            if roi_mask.shape != (self.args.Nodes,) or not roi_mask.any():
                raise ValueError("roi_mask must be [Nodes] with at least one valid ROI")
        H = x[:, 0, :]
        K_bias = x[:, 1, :]
        NE = x[:, 2, :]
        C = x[:, 3, :]
        Emb = self.self_attention_map(H, K_bias, NE, C, roi_mask=roi_mask)
        pooled = Emb.mean(dim=0) if roi_mask is None else Emb[roi_mask].mean(dim=0)
        return self.classifier(self.readout_dropout(pooled))
