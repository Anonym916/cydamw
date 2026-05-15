# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import (
    VisionTransformer,
    vit_tiny_patch2_32,
    vit_small_patch2_32,
    vit_small_patch16_224,
    vit_base_patch16_96,
    vit_base_patch16_224,
)
from .bert import bert_base_cased, bert_base_uncased, bert_base_cased_multihead, bert_base_uncased_multihead
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base

__all__ = [
    # WRN
    "wrn_28_2",
    "wrn_28_8",
    "wrn_var_37_2",

    # ResNet
    "resnet50",

    # VIT family
    "VisionTransformer",
    "vit_tiny_patch2_32",
    "vit_small_patch2_32",
    "vit_small_patch16_224",
    "vit_base_patch16_96",
    "vit_base_patch16_224",

    # NLP models
    "bert_base_cased",
    "bert_base_uncased",
    "bert_base_cased_multihead",
    "bert_base_uncased_multihead",

    # Audio models
    "wave2vecv2_base",
    "hubert_base",
]
