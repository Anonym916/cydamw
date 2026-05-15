import torch
import torch.nn as nn


class EncodingAugmenter(nn.Module):
    """
    Encoding-based strong augmentation for ViT.
    Perturbs positional encoding instead of pixel space.
    """

    def __init__(self, mode="rope", strength=0.0):
        super().__init__()
        self.mode = mode
        self.strength = strength

    def forward(self, pos_embed, is_strong=False):
        """
        pos_embed: [1, N, D]
        """
        if (not is_strong) or self.strength <= 0:
            return pos_embed

        if self.mode == "rope":
            return self._rope_jitter(pos_embed)

        raise NotImplementedError(f"Unknown encoding mode: {self.mode}")

    def _rope_jitter(self, pos_embed):
        noise = torch.randn_like(pos_embed) * self.strength
        return pos_embed * (1.0 + noise)
