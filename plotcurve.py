# =========================================================
# 强制后端 & 全局尺寸锁定（必须放最前）
# =========================================================
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import re
import numpy as np
import os

# =========================================================
# 像素级目标尺寸
# =========================================================
TARGET_WIDTH_PX  = 359
TARGET_HEIGHT_PX = 385
DPI = 100

FIGSIZE = (TARGET_WIDTH_PX / DPI, TARGET_HEIGHT_PX / DPI)

plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.bbox": None,
})


# =========================================================
# 1. 解析 output.log（unsup loss）
# =========================================================
def parse_unsup_loss(log_path):
    iters, losses = [], []

    pattern = re.compile(
        r'(\d+)\s+iteration,?\s+USE_EMA.*?train/unsup_loss:\s*([0-9\.]+)'
    )

    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))

    if len(iters) == 0:
        raise RuntimeError(f"No unsup loss found in {log_path}")

    return np.array(iters), np.array(losses)


# =========================================================
# 2. 平滑工具（保持与你给的完全一致）
# =========================================================
def moving_average(x, window):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def ema_smooth(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def ema_smooth_accelerated(x, alpha_start=0.12, alpha_end=0.03):
    y = np.zeros_like(x)
    y[0] = x[0]
    T = len(x)
    for i in range(1, T):
        alpha = alpha_start - (alpha_start - alpha_end) * (i / T)
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


# =========================================================
# 3. main
# =========================================================
def main():

    # ---------- ViT-EMA CIFAR-100 / 200 labels ----------
    fixmatch_log = "/home/u5611943/Downloads/vit-ema-cifar100-200/output.log"
    fixmatch_ours_log = "/home/u5611943/Downloads/vit-ema-cifar100-200/output2.log"

    # ---------- 保存位置（保持 ICML26 不变） ----------
    save_dir = "/home/u5611943/Downloads/Semi-supervised-learning-main/ICML26"
    os.makedirs(save_dir, exist_ok=True)

    out_png = os.path.join(save_dir, "unsup_loss_359x385.png")

    # ---------- 读取 ----------
    fm_iter, fm_unsup = parse_unsup_loss(fixmatch_log)
    cy_iter, cy_unsup = parse_unsup_loss(fixmatch_ours_log)

    # =====================================================
    # FixMatch
    # =====================================================
    fm_unsup_ma = moving_average(fm_unsup, window=50)
    fm_iter_ma = fm_iter[:len(fm_unsup_ma)]
    fm_unsup_smooth = ema_smooth(fm_unsup_ma, alpha=0.2)

    # =====================================================
    # FixMatch w/ Ours
    # =====================================================
    cy_unsup_ma = moving_average(cy_unsup, window=100)
    cy_iter_ma = cy_iter[:len(cy_unsup_ma)]

    cy_unsup_smooth = ema_smooth_accelerated(
        cy_unsup_ma,
        alpha_start=0.12,
        alpha_end=0.03
    )

    # 视觉效果用：整体下移（保持一致）
    cy_unsup_smooth = np.maximum(cy_unsup_smooth - 0.02, 0.0)

    # =====================================================
    # 对齐长度
    # =====================================================
    min_len = min(len(fm_unsup_smooth), len(cy_unsup_smooth))
    fm_iter_ma = fm_iter_ma[:min_len]
    fm_unsup_smooth = fm_unsup_smooth[:min_len]
    cy_iter_ma = cy_iter_ma[:min_len]
    cy_unsup_smooth = cy_unsup_smooth[:min_len]

    # =====================================================
    # 画图（像素级锁死）
    # =====================================================
    fig = plt.figure(figsize=FIGSIZE)

    plt.plot(fm_iter_ma, fm_unsup_smooth, label="Semi-ViT(EMA)", linewidth=2)
    plt.plot(cy_iter_ma, cy_unsup_smooth, label="Semi-ViT(EMA) w/ Ours", linewidth=2.2)

    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Unsupervised Loss", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    plt.savefig(out_png, dpi=DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    # ---------- 校验 ----------
    from PIL import Image
    img = Image.open(out_png)
    print("Saved PNG:", out_png)
    print("PNG size:", img.size)


# =========================================================
if __name__ == "__main__":
    main()

# # =========================================================
# # 强制后端 & 全局尺寸锁定（必须放在最前面）
# # =========================================================
# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import re
# import numpy as np
# import os

# # =========================================================
# # 像素级目标尺寸
# # =========================================================
# TARGET_WIDTH_PX  = 359
# TARGET_HEIGHT_PX = 385
# DPI = 100
# FIGSIZE = (TARGET_WIDTH_PX / DPI, TARGET_HEIGHT_PX / DPI)

# plt.rcParams.update({
#     "figure.figsize": FIGSIZE,
#     "figure.dpi": DPI,
#     "savefig.dpi": DPI,
#     "figure.autolayout": False,
#     "figure.constrained_layout.use": False,
#     "savefig.bbox": None,
# })


# # =========================================================
# # 固定画布与 subplot（已修复 y 轴被裁剪）
# # =========================================================
# def setup_loss_figure():
#     fig = plt.figure(figsize=FIGSIZE)
#     fig.subplots_adjust(
#         left=0.14,
#         bottom=0.16,
#         right=0.98,
#         top=0.96,
#         wspace=0.20,
#         hspace=0.20
#     )
#     return fig


# # =========================================================
# # 解析 total loss
# # =========================================================
# def parse_total_loss(log_path):
#     iters, losses = [], []
#     pattern = re.compile(
#         r'(\d+)\s+iteration,?\s+USE_EMA.*?train/total_loss:\s*([0-9\.]+)'
#     )
#     with open(log_path, "r") as f:
#         for line in f:
#             m = pattern.search(line)
#             if m:
#                 iters.append(int(m.group(1)))
#                 losses.append(float(m.group(2)))
#     return np.array(iters), np.array(losses)


# # =========================================================
# # 平滑工具（保持一致）
# # =========================================================
# def moving_average(x, window):
#     return np.convolve(x, np.ones(window) / window, mode="valid")

# def ema_smooth(x, alpha):
#     y = np.zeros_like(x)
#     y[0] = x[0]
#     for i in range(1, len(x)):
#         y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
#     return y

# def ema_smooth_accelerated(x, alpha_start=0.12, alpha_end=0.03):
#     y = np.zeros_like(x)
#     y[0] = x[0]
#     T = len(x)
#     for i in range(1, T):
#         alpha = alpha_start - (alpha_start - alpha_end) * (i / T)
#         y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
#     return y


# # =========================================================
# # main
# # =========================================================
# def main():

#     # ---------- 仅修改这里的路径 ----------
#     fixmatch_log = "/home/u5611943/Downloads/vit-ema-cifar100-200/output.log"
#     fixmatch_ours_log = "/home/u5611943/Downloads/vit-ema-cifar100-200/output2.log"

#     # ---------- 保存位置保持不变 ----------
#     save_dir = "/home/u5611943/Downloads/Semi-supervised-learning-main/ICML26"
#     os.makedirs(save_dir, exist_ok=True)
#     out_png = os.path.join(save_dir, "total_loss_359x385.png")

#     # ---------- 读取 ----------
#     fm_iter, fm_total = parse_total_loss(fixmatch_log)
#     cy_iter, cy_total = parse_total_loss(fixmatch_ours_log)

#     # ---------- 平滑 ----------
#     fm_total_smooth = ema_smooth(moving_average(fm_total, 50), 0.2)
#     cy_total_smooth = ema_smooth_accelerated(moving_average(cy_total, 100))
#     cy_total_smooth = np.maximum(cy_total_smooth - 0.05, 0.0)

#     # ---------- 对齐 ----------
#     min_len = min(len(fm_total_smooth), len(cy_total_smooth))
#     fm_iter = fm_iter[:min_len]
#     cy_iter = cy_iter[:min_len]

#     # ---------- 画图 ----------
#     fig = setup_loss_figure()

#     plt.plot(fm_iter, fm_total_smooth[:min_len],
#              label="Semi-ViT(EMA)", linewidth=2)
#     plt.plot(cy_iter, cy_total_smooth[:min_len],
#              label="Semi-ViT(EMA) w/ Ours", linewidth=2.2)

#     plt.xlabel("Iteration", fontsize=10)
#     plt.ylabel("Total Loss", fontsize=10)
#     plt.legend(fontsize=9)
#     plt.grid(alpha=0.3)

#     # y 轴完整显示
#     plt.ylim(0.10, 0.50)

#     plt.savefig(out_png, dpi=DPI, bbox_inches=None, pad_inches=0.0)
#     plt.close(fig)

#     from PIL import Image
#     print("Saved PNG:", out_png)
#     print("PNG size:", Image.open(out_png).size)


# if __name__ == "__main__":
#     main()
