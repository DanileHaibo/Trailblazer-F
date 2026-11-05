"""Frozen DriveWAM structured MLP architecture."""
import torch
from torch import nn

class StructuredDraft(nn.Module):
    """Predict structured endpoint sequence with shifted-sigma interpolation."""
    def __init__(self, feature_dim=273, mode='flow'):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(nn.Linear(feature_dim + 24 + 10, 512), nn.SiLU(),
                                 nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 240))
        self.register_buffer('sigmas', torch.linspace(1, 0, 11))

    def forward(self, feature, x, index):
        if self.mode == 'no_feature':
            feature = torch.zeros_like(feature)
        endpoints = self.net(torch.cat([feature.float(), x.flatten(1).float(),
            torch.nn.functional.one_hot(index, 10).float()], 1)).reshape(-1, 10, 3, 8)
        if self.mode == 'residual':
            return x[:, None].float() + endpoints
        ids = (index[:, None]+torch.arange(10, device=x.device)[None]+1).clamp(max=10)
        ratio = self.sigmas[ids]/self.sigmas[index, None]
        return ratio[:, :, None, None]*x[:, None].float()+(1-ratio[:, :, None, None])*endpoints

