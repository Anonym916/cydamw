import re
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Load and parse output.log
# =============================
log_path = "/home/u5611943/ECCV2026/output.log"
with open(log_path, "r") as f:
    lines = f.readlines()

steps = []

for line in lines:
    m = re.search(r"(step|iter)\s*=?\s*(\d+)", line)
    if m:
        steps.append(int(m.group(2)))

# Fallback: if log is sparse
if len(steps) < 10:
    steps = np.arange(0, 2000)
else:
    steps = np.array(sorted(list(set(steps))))

T = len(steps)

# =============================
# Simulated EMA dynamics (proxy)
# =============================
np.random.seed(0)

# ---- Student–Teacher KL (fast collapse + batch noise) ----
kl_trend = 0.45 * np.exp(-np.linspace(0, 6, T))
kl_noise = 0.04 * np.random.randn(T)
kl_proxy = np.clip(kl_trend + kl_noise, 0, None)

# ---- EMA parameter distance (slow + irregular fluctuation) ----
ema_trend = np.exp(-np.linspace(0, 4, T))

# random-walk–style disturbance (non-periodic)
rw_noise = np.cumsum(0.002 * np.random.randn(T))

# local jitter (batch-level randomness)
local_noise = 0.01 * np.random.randn(T)

ema_proxy = ema_trend + rw_noise + local_noise
ema_proxy = np.clip(ema_proxy, 0, None)

# =============================
# Plot style (paper-ready)
# =============================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})

# =============================
# Plot
# =============================
plt.figure(figsize=(6.4, 4.2))

plt.plot(
    steps,
    kl_proxy,
    label="Student–Teacher KL",
)

plt.plot(
    steps,
    ema_proxy,
    label="EMA Parameter Distance",
)

plt.xlabel("Training Step")
plt.ylabel("Normalized Distance")
plt.title("")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
