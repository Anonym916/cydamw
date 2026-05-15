import torch
import torch.nn as nn


class CoordAugmenter(nn.Module):
    """
    Coordinate encoding augmentation.
    Adds (x, y) coordinate channels and optionally jitters them for strong views.
    """

    def __init__(self, strength=0.0):
        super().__init__()
        self.strength = strength

    def forward(self, x, is_strong=False):
        """
        x: [B, C, H, W]
        """
        if x.dim() != 4:
            raise ValueError(f"CoordAugmenter expects 4D input, got {x.dim()}D")

        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

        if is_strong and self.strength > 0:
            coord = coord + torch.randn_like(coord) * self.strength

        return torch.cat([x, coord], dim=1)
