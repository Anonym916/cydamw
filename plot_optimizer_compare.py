import os
import re
import matplotlib.pyplot as plt

# ============================================================
# 1. 日志所在目录（★ 改成你自己的路径）
# ============================================================
LOG_DIR = "/home/u5611943/Documents/optimizercompare"

LOG_FILES = {
    "AdamW": "AdamW.log",
    "ADOPT": "ADOPT.log",
    "CydamW": "CydamW.log",
    "SGD": "SGD.log",
}

# ============================================================
# 2. EMA 系数（wandb 风格）
# ============================================================
EMA_UTIL = 0.95    # util_ratio: 抑制抖动
EMA_LOSS = 0.85     # total_loss: 保留动态

# ============================================================
# 3. 样式：CydamW 深色，其它浅色
# ============================================================
STYLE = {
    "CydamW": dict(color="tab:red",   alpha=0.8,  linewidth=1.8),
    "AdamW":  dict(color="tab:blue",  alpha=0.35, linewidth=1.6),
    "ADOPT":  dict(color="tab:green", alpha=0.35, linewidth=1.6),
    "SGD":    dict(color="tab:gray",  alpha=0.35, linewidth=1.6),
}

# ============================================================
# 4. 日志行正则（严格匹配 SemiLearn INFO 行）
# ============================================================
LINE_PATTERN = re.compile(
    r"(?P<iter>\d+)\s+iteration.*?"
    r"train/total_loss:\s*(?P<total_loss>[0-9.eE+-]+).*?"
    r"train/util_ratio:\s*(?P<util_ratio>[0-9.eE+-]+)"
)

# ============================================================
# 5. wandb 风格 EMA
# ============================================================
def ema_smooth(values, alpha):
    if not values:
        return []
    smoothed = [values[0]]  # wandb: 首点不平滑
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed

# ============================================================
# 6. 解析单个日志
# ============================================================
def parse_log(path):
    steps, util, loss = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_PATTERN.search(line)
            if m:
                steps.append(int(m.group("iter")))
                util.append(float(m.group("util_ratio")))
                loss.append(float(m.group("total_loss")))
    return steps, util, loss

# ============================================================
# 7. 读取所有 optimizer 日志
# ============================================================
data = {}

for name, fname in LOG_FILES.items():
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}, skip.")
        continue

    steps, util, loss = parse_log(path)
    print(f"[OK] {name}: {len(steps)} points parsed")

    data[name] = {
        "steps": steps,
        "util": ema_smooth(util, EMA_UTIL),
        "loss": ema_smooth(loss, EMA_LOSS),
    }

if not data:
    raise RuntimeError("No logs parsed. Check LOG_DIR.")

# ============================================================
# 8. 画图（1×2 并列）
# ============================================================
plt.figure(figsize=(12, 4))

# -------- util_ratio --------
plt.subplot(1, 2, 1)
for name, d in data.items():
    plt.plot(
        d["steps"],
        d["util"],
        label=name,
        **STYLE.get(name, {})
    )
plt.xlabel("Iteration")
plt.ylabel("Label Util Ratio")
plt.title(f"")
plt.legend()
plt.grid(alpha=0.3)

# -------- total_loss --------
plt.subplot(1, 2, 2)
for name, d in data.items():
    plt.plot(
        d["steps"],
        d["loss"],
        label=name,
        **STYLE.get(name, {})
    )
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.title(f"")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_compare_ema.png", dpi=300)
plt.show()

print("[DONE] optimizer_compare_ema.png saved")
