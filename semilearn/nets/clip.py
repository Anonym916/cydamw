import torch
import torch.nn as nn
from transformers import CLIPVisionModel
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

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scaling = alpha / r

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def apply_lora_to_clip(model, r=4):
    """
    遍历 CLIP 模型，将 Transformer block 中的 MLP 层替换为 LoRALinear
    """
    for name, module in model.named_modules():
        # Attention 投影层 (可选)
        # if hasattr(module, "q_proj"):
        #     module.q_proj = LoRALinear(module.q_proj, r=r)
        # if hasattr(module, "v_proj"):
        #     module.v_proj = LoRALinear(module.v_proj, r=r)

        # MLP 层
        # 在 HuggingFace 的 CLIP 实现中，MLP 的两个线性层名为 fc1 和 fc2
        if hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear):
            module.fc1 = LoRALinear(module.fc1, r=r)

        if hasattr(module, "fc2") and isinstance(module.fc2, nn.Linear):
            module.fc2 = LoRALinear(module.fc2, r=r)

    return model


class CustomCLIPNormModel(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", num_classes=8):
        super(CustomCLIPNormModel, self).__init__()
        
        # 仅加载 CLIP 的视觉编码器部分
        self.clip_model = CLIPVisionModel.from_pretrained(name)
        
        # 为 MLP 注入 LoRA
        self.clip_model = apply_lora_to_clip(self.clip_model, r=4)
        
        # 冻结 Backbone 所有参数
        for p in self.clip_model.parameters():
            p.requires_grad = False
            
        # 精确解冻 LoRA 参数，避免把原始 linear 层意外解冻
        for m in self.clip_model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True
                
        # 打印验证注入的 LoRA
        for name, module in self.clip_model.named_modules():
            if isinstance(module, LoRALinear):
                print(f"LoRA Injected: {name} | A: {module.lora_A.shape}, B: {module.lora_B.shape}")
                
        # CLIP ViT-Base 的 hidden size 也是 768
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 256),
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
            return_embed: return raw model output dict
        """
        if return_embed:
            embed = self.clip_model(x)
            return embed

        out_dict = self.clip_model(x, output_hidden_states=True, return_dict=True)
        last_hidden_state = out_dict['last_hidden_state']
        
        # 仿照您的代码使用 Mean Pooling 
        # (注：CLIP 原生通常使用 out_dict['pooler_output'] 即 CLS token，您可以根据实际需求切换)
        pooled_output = torch.mean(last_hidden_state, 1)

        if only_fc:
            logits = self.classifier(pooled_output)
            return logits

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict
        
    
    def group_matcher(self, coarse=False, prefix=''):
        # 适配 HuggingFace CLIP Vision Encoder 的层级命名结构
        matcher = dict(
            stem=r'^{}clip_model.vision_model.embeddings'.format(prefix), 
            blocks=r'^{}clip_model.vision_model.encoder.layers.(\d+)'.format(prefix)
        )
        return matcher

    def no_weight_decay(self):
        return []


def clip_vitb14(pretrained=True, pretrained_path=None, num_classes=100, **kwargs):
    model = CustomCLIPNormModel(name='openai/clip-vit-base-patch16', num_classes=num_classes, **kwargs)
    return model
