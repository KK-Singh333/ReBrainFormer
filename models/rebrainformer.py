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
        self.classifier = nn.Linear(args.Nodes, self.classes)

    def self_attention_map(self, H, K_bias, N_E, C):
        Q = H @ self.Wq
        K_proj = H @ self.Wk
        V = H @ self.Wv
        attention_scores = (
            (Q @ K_proj.T) / math.sqrt(self.args.Nodes)
            + self.lambda1 * K_bias
            + self.lambda2 * N_E
            + self.lambda3 * C
        )
        return torch.softmax(attention_scores, dim=-1) @ V

    def forward(self, x):
        H = x[:, 0, :]
        K_bias = x[:, 1, :]
        NE = x[:, 2, :]
        C = x[:, 3, :]
        Emb = self.self_attention_map(H, K_bias, NE, C)
        pooled = Emb.mean(dim=0)
        return self.classifier(pooled)