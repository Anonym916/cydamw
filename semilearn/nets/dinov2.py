import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
import os


class LoRALinear(nn.Module):
    def __init__(self, linear, r=4, alpha=1.0):
        super().__init__()

        self.linear = linear
        self.r = r
        self.alpha = alpha

        in_features = linear.in_features
        out_features = linear.out_features

        # 冻结原始权重
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scaling = alpha / r

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def apply_lora_to_dinov2(model, r=4):
    for name, module in model.named_modules():

        # Attention
        # if hasattr(module, "qkv"):
        #     module.qkv = LoRALinear(module.qkv, r=r)

        # if hasattr(module, "proj"):
        #     module.proj = LoRALinear(module.proj, r=r)

        # MLP
        if hasattr(module, "fc1"):
            module.fc1 = LoRALinear(module.fc1, r=r)

        if hasattr(module, "fc2"):
            module.fc2 = LoRALinear(module.fc2, r=r)

    return model


class CustomDINONormModel(nn.Module):
    def __init__(self, name, num_classes=8):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = Dinov2Model.from_pretrained(name)
        # set mlp with lora
        self.dino_model = apply_lora_to_dinov2(self.dino_model, r=4)
        for p in self.dino_model.parameters():
            p.requires_grad = False
        for m in self.dino_model.modules():
            if isinstance(m, LoRALinear):
                for p in m.parameters():
                    p.requires_grad = True
        for name, module in self.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                print(name, module.lora_A.shape, module.lora_B.shape)
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 256),  # (1024, 256)
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        # Extract features using DinoV2 model
        if return_embed:
            embed = self.dino_model(x)
            return embed

        out_dict = self.dino_model(x, output_hidden_states=True, return_dict=True)
        last_hidden_state = out_dict['last_hidden_state']
        pooled_output = torch.mean(last_hidden_state, 1)  # Perform mean pooling

        if only_fc:
            logits = self.classifier(pooled_output)
            return logits

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict
        
    
    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}dino_model.embeddings'.format(prefix), blocks=r'^{}dino_model.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


def dinov2_vitb14(pretrained=True, pretrained_path=None, num_classes=100, **kwargs):
    model = CustomDINONormModel(name='facebook/dinov2-base', num_classes=num_classes, **kwargs)  # facebookresearch/dinov2_vitb14
    return model


def dinov2_vitl14(pretrained=True, pretrained_path=None, **kwargs):
    model = CustomDINONormModel(name='facebookresearch/dinov2_vitl14', **kwargs)
    return model
