import math
from typing import Callable, Optional, Tuple, Union, List

import torch
from torch import Tensor
from torch.optim import Optimizer
# ============================================================
# ADOPT: Adaptive Directional OPTimizer (Local, Reproducible)
# ============================================================

def default_clip_lambda(step: int) -> float:
    return min(1.0, step ** 0.15)

class ADOPT(Optimizer):
    """
    ADOPT optimizer (local implementation, no external dependency).

    Core idea:
        - Normalize gradients by RMS (directional update)
        - Apply EMA on normalized gradients
        - Adaptive clipping in normalized-gradient space
        - Decoupled weight decay (AdamW-style)

    This implementation is SSL-safe and works with FixMatch / SemiLearn.
    """


    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        decouple: bool = True,
        clip_lambda: Optional[Callable[[int], float]] = None,
        warmup_steps: int = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        # Default: conservative SSL-friendly clipping
        if clip_lambda is None:
            clip_lambda = default_clip_lambda

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            clip_lambda=clip_lambda,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decouple = group["decouple"]
            clip_lambda = group["clip_lambda"]
            warmup_steps = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + 1

                # ------------------------------------------------------------
                # Decoupled weight decay (AdamW style)
                # ------------------------------------------------------------
                if weight_decay != 0.0 and decouple:
                    p.mul_(1.0 - lr * weight_decay)

                # ------------------------------------------------------------
                # RMS accumulator
                # ------------------------------------------------------------
                if step == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                    state["step"] = step
                    continue

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                normed_grad = grad / denom

                # ------------------------------------------------------------
                # Adaptive clipping in normalized-gradient space
                # ------------------------------------------------------------
                if clip_lambda is not None:
                    clip = clip_lambda(step)
                    normed_grad.clamp_(-clip, clip)

                # ------------------------------------------------------------
                # Warmup gate (important for SSL)
                # ------------------------------------------------------------
                if step <= warmup_steps:
                    update = grad
                else:
                    update = normed_grad

                exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)

                # ------------------------------------------------------------
                # Parameter update
                # ------------------------------------------------------------
                p.add_(exp_avg, alpha=-lr)

                state["step"] = step

        return loss


class CydamW(Optimizer):  # CydamW
    """
    Variant of AdamW with sinusoidally modulated momentum using previous gradient.
    """
    # weight_decay 0.1 0.001
    # gamma 0.1 0.3 0.4 0.6
    # omega 0.1 0.001
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
                 gamma=0.2, omega=0.01, eps=1e-8, clip_value=None):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        gamma=gamma, omega=omega, eps=eps, clip_value=clip_value)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            omega = group["omega"]
            eps = group["eps"]
            clip_value = group["clip_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = p.data.new_zeros(p.data.size())
                    state["v"] = p.data.new_zeros(p.data.size())
                    state["prev_grad"] = p.data.new_zeros(p.data.size())

                m, v, prev_grad = state["m"], state["v"], state["prev_grad"]
                t = state["step"] + 1

                v_new = beta2 * v + (1 - beta2) * (grad * grad)
                modulated = grad + gamma * math.sin(omega * t) * prev_grad
                denom = (v + eps).sqrt()
                normalized = modulated / denom

                if clip_value is not None:
                    normalized = torch.clamp(normalized, -clip_value, clip_value)

                m_new = beta1 * m + (1 - beta1) * normalized

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.data.add_(-lr * m_new)

                state["m"] = m_new
                state["v"] = v_new
                state["prev_grad"] = grad.clone()
                state["step"] = t

        return loss


class CySGD(Optimizer): # CySGD
    """
    Variant of SGD with sinusoidally modulated gradient using the previous gradient.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, gamma=0.2, omega=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        gamma=gamma, omega=omega)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            gamma = group['gamma']
            omega = group['omega']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]

                # 1. 标准的 L2 正则化 (Weight Decay)
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # 2. 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(p.data)
                    if momentum != 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']
                prev_grad = state['prev_grad']

                # 3. 核心创新点：正弦调制梯度 (Sinusoidal Modulation)
                # modulated_grad = g_t + γ * sin(ω * t) * g_{t-1}
                modulated_grad = d_p + gamma * math.sin(omega * t) * prev_grad

                # 4. SGD 动量逻辑 (使用调制后的梯度更新动量)
                if momentum != 0:
                    buf = state['momentum_buffer']
                    if t == 1:
                        buf.copy_(modulated_grad)
                    else:
                        buf.mul_(momentum).add_(modulated_grad, alpha=1 - dampening)

                    if nesterov:
                        modulated_grad = modulated_grad.add(buf, alpha=momentum)
                    else:
                        modulated_grad = buf

                # 5. 更新参数
                p.data.add_(modulated_grad, alpha=-lr)

                # 6. 保存当前的原始梯度(包含weight_decay，但不包含动量和调制)供下一步使用
                state['prev_grad'].copy_(d_p)

        return loss


class SGD(Optimizer):  # SGD
    """
    Standard Stochastic Gradient Descent (SGD) for baseline comparison.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]

                # 1. 标准的 L2 正则化 (Weight Decay)
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # 2. 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    if momentum != 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1

                # 3. 标准 SGD 动量逻辑 (直接使用原始梯度 d_p)
                if momentum != 0:
                    buf = state['momentum_buffer']
                    if state['step'] == 1:
                        buf.copy_(d_p)
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # 4. 更新参数
                p.data.add_(d_p, alpha=-lr)

        return loss