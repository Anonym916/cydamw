import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. 固定输出尺寸（540 × 299 px）
# ============================================================
WIDTH_PX, HEIGHT_PX = 540, 299
DPI = 100
FIGSIZE = (WIDTH_PX / DPI, HEIGHT_PX / DPI)

plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": None,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
})

# ============================================================
# 1. 日志路径（6 条曲线，key 必须唯一）
# ============================================================
LOGS = {
    # ===== ViT-B + FreeMatch (STL10-100) =====
    "ResNet-50 FreeMatch (STL10) w/ Ours":
        "/home/u5611943/Downloads/r50-free-stl10-100/Cyoutput.log",

    "ResNet-50 FreeMatch (STL10)":
        "/home/u5611943/Downloads/r50-free-stl10-100/output.log",

    # ===== ViT-S + FixMatch (CIFAR100-200) =====
    "ViT-S FixMatch (CIFAR100)":
        "/home/u5611943/Downloads/vits-FixMatch-CIFAR100-200/output.log",

    "ViT-S FixMatch (CIFAR100) w/ Ours":
        "/home/u5611943/Downloads/vits-FixMatch-CIFAR100-200/Cyoutput.log",

    # ===== Semi-ViT-B (CIFAR10-200) =====
    "Semi-ViT-B (CIFAR10) w/ Ours":
        "/home/u5611943/Downloads/vitb-FreeSTL10-100/output.log",

    "Semi-ViT-B (CIFAR10) ":
        "/home/u5611943/Downloads/vitb-FreeSTL10-100/Cyoutput.log",
}

# ============================================================
# 2. 时间窗口（⚠️ 对 Acc 实际不会产生影响，FreeMatch 很稀疏）
# ============================================================
WINDOW_ITER = 8192

# ============================================================
# 3. 正则：兼容 FixMatch / FreeMatch / Semi-ViT 的 Acc 字段
# ============================================================
LINE_RE = re.compile(
    r"(\d+)\s+iteration.*?(?:eval/top-1-acc|eval/top1_acc|eval/acc):\s*([0-9.]+)"
)

# ============================================================
# 4. 解析 Acc
# ============================================================
def parse_acc(path):
    steps, values = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                values.append(float(m.group(2)))

    if not steps:
        print(f"[Warning] No accuracy found in {path}")
    else:
        print(f"[OK] {path}: parsed {len(steps)} acc points")

    return steps, values

# ============================================================
# 5. Acc 不做大 window（FreeMatch 只在 validating 后记录）
#    → 直接使用 eval-level 曲线（必要且正确）
# ============================================================
def process_acc_curve(steps, values):
    return steps, values

# ============================================================
# 6. 画 6 条 Acc 曲线
# ============================================================
plt.figure(figsize=FIGSIZE, dpi=DPI)

for label, path in LOGS.items():
    steps, values = parse_acc(path)
    if len(steps) == 0:
        continue

    p_steps, p_vals = process_acc_curve(steps, values)
    plt.plot(p_steps, p_vals, label=label, linewidth=2.0)

plt.xlabel("Iteration")
plt.ylabel("Top-1 Accuracy")
plt.title("")
plt.ylim(0.0, 1.02)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=7)
plt.tight_layout(pad=0.6)
plt.show()
