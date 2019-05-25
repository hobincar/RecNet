import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, feat_size, bottleneck_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.feat_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, hidden, feats):
        Wh = self.W(hidden)
        Uv = self.U(feats)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        weights = F.softmax(energies, dim=1)

        weighted_feats = feats * weights.expand_as(feats)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats, weights

