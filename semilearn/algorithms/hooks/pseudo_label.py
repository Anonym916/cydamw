# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import smooth_targets

class PseudoLabelingHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def gen_ulb_targets(self,
                        algorithm,
                        logits,
                        use_hard_label=True,
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):

        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        noise_ratio = getattr(algorithm, 'pseudo_label_noise_ratio', 0.0)
        num_classes = algorithm.num_classes

        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)

            # inject hard label noise: with probability noise_ratio, replace with a random class
            if noise_ratio > 0.0:
                batch_size = pseudo_label.shape[0]
                # boolean mask indicating which samples get their label flipped
                flip_mask = torch.bernoulli(
                    torch.full((batch_size,), noise_ratio, device=pseudo_label.device)
                ).bool()
                # random class indices sampled uniformly from [0, num_classes)
                random_labels = torch.randint(
                    0, num_classes, (batch_size,), device=pseudo_label.device
                )
                pseudo_label = torch.where(flip_mask, random_labels, pseudo_label)

            return pseudo_label

        # return soft label
        if softmax:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = algorithm.compute_prob(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits

        # inject soft label noise: mix with uniform distribution
        # result = (1 - noise_ratio) * pseudo_label + noise_ratio * (1 / num_classes)
        if noise_ratio > 0.0:
            uniform = torch.full_like(pseudo_label, 1.0 / num_classes)
            pseudo_label = (1.0 - noise_ratio) * pseudo_label + noise_ratio * uniform

        return pseudo_label
        