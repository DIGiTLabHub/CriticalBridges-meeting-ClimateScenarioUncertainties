#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Pushover-Curve Post-Processor
─────────────────────────────
This script smooths the base shear history before detecting key points on the pushover curve:
1. Smooth V using centered moving average.
2. Identify:
   • Knee (yield) point where slope drops below threshold.
   • Peak after knee.
   • Softening point where V drops below threshold after peak.
3. Draw bilinear idealization:
   (0,0) → (Dy, Vy) → (Dx_soft, Vx_soft)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────── USER SETTINGS ─────────────────────────────────────────
BASE_DIR        = "RecorderData"
VALID_SCENARIOS = {"Missouri", "Colorado", "Extreme"}
COLUMN_IDS      = [3101, 3201, 3301]
TOP_NODE_FILE   = "Displacement.5201.out"

SMOOTH_WIN   = 7       # Moving average window size
R_PERC       = 0.50    # Slope drop % for knee detection
DROP_PERC    = 0.02    # % drop from peak for softening
MIN_HARD_PTS = 5       # Min points after knee for softening

COL_FORCE_HDR = [
    "time", "P_i", "V2_i", "V3_i", "T_i", "M2_i", "M3_i",
    "P_j", "V2_j", "V3_j", "T_j", "M2_j", "M3_j"
]

# ────────── HELPERS ──────────────────────────────────────────────
def moving_average(x, w):
    """Centered moving average with edge padding."""
    if w < 3 or w % 2 == 0:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")

def find_knee(disp, V_s, r_perc=0.5, guard=5):
    """Find knee index based on slope drop."""
    k0 = np.mean(np.diff(V_s[:10+guard]) / np.diff(disp[:10+guard]))
    tan = np.diff(V_s) / np.diff(disp)
    idxs = np.where(tan[guard:] < r_perc * k0)[0]
    idx = (idxs[0] + guard) if idxs.size else np.argmax(np.abs(np.diff(tan))) + 1
    return idx + 1

# ────────── MAIN LOOP ────────────────────────────────────────────
plotted = []

for scn in sorted(os.listdir(BASE_DIR)):
    if scn not in VALID_SCENARIOS:
        continue

    scn_path = os.path.join(BASE_DIR, scn)

    for folder in sorted(os.listdir(scn_path)):
        if not folder.startswith("scour_"):
            continue

        try:
            depth = int(folder.split("_")[1])
        except (IndexError, ValueError):
            print(f"⚠️  Skipping malformed folder: {folder}")
            continue

        fpath = os.path.join(scn_path, folder)
        disp_fp = os.path.join(fpath, TOP_NODE_FILE)
        frc_fps = [os.path.join(fpath, f"ColLocForce.{eid}.out") for eid in COLUMN_IDS]

        if not (os.path.isfile(disp_fp) and all(map(os.path.isfile, frc_fps))):
            print(f"❌  Missing files in {fpath}")
            continue

        # --- Read displacement
        df_d = pd.read_csv(disp_fp, sep=r"\s+", header=None,
                           names=["time", "ux", "uy", "uz", "rx", "ry", "rz"])
        disp = df_d["uy"].abs().to_numpy()

        # --- Read base shear (sum of all column forces)
        V = None
        for fp in frc_fps:
            df_f = pd.read_csv(fp, sep=r"\s+", header=None, names=COL_FORCE_HDR)
            Vi = df_f["V2_i"]
            V = Vi if V is None else V + Vi
        V = V.abs().to_numpy() / 1000.0  # Convert to kN

        # --- Trim arrays
        n = min(len(disp), len(V))
        if n < 30:
            print(f"⚠️  Too few points in {fpath}")
            continue
        disp, V = disp[:n], V[:n]

        # --- Smooth shear
        V_s = moving_average(V, SMOOTH_WIN)

        # --- GUARD_PTS based on displacement
        GUARD_PTS = min(100, np.argmax(disp > 150.0)) if np.any(disp > 150.0) else 15

        # --- Yield point (knee)
        idx_k = find_knee(disp, V_s, R_PERC, GUARD_PTS)
        Dy, Vy = disp[idx_k], V[idx_k]

        # --- Peak & softening
        peak_idx = idx_k + np.argmax(V_s[idx_k:])
        peak_val = V_s[peak_idx]
        threshold = (1 - DROP_PERC) * peak_val
        soft_rel = np.where(V_s[idx_k:] < threshold)[0]
        idx_soft = idx_k + soft_rel[0] if soft_rel.size and soft_rel[0] >= MIN_HARD_PTS else peak_idx
        Dx_soft, Vx_soft = disp[idx_soft], V[idx_soft]

        plotted.append({
            "label": f"{scn} | Scour {depth} mm",
            "disp": disp,
            "shear": V,
            "Dy": Dy, "Vy": Vy,
            "Dx_soft": Dx_soft, "Vx_soft": Vx_soft
        })
        print(f"✅ {scn} | scour {depth} mm processed")

# ────────── PLOT ────────────────────────────────────────────────
if not plotted:
    raise SystemExit("🚫 No valid data found.")

plt.figure(figsize=(10, 6))
ax = plt.gca()

for e in plotted:
    line, = ax.plot(e["disp"], e["shear"], label=e["label"], lw=1.6)
    clr = line.get_color()
    ax.plot([0, e["Dy"]], [0, e["Vy"]], ls="--", lw=1.2, color=clr)
    ax.plot([e["Dy"], e["Dx_soft"]], [e["Vy"], e["Vx_soft"]],
            ls="--", lw=1.2, color=clr)
    ax.plot(e["Dy"], e["Vy"], 'ko', ms=5)

ax.set_xlabel("Top displacement Δ (mm)")
ax.set_ylabel("Base shear V (kN)")
ax.set_title("Pushover Curves with Bilinear Idealization")
ax.grid(True, ls=":")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()

out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(out_dir, exist_ok=True)
fig_fp = os.path.join(out_dir, "pushover_bilinear_fit_knee_points.png")
plt.savefig(fig_fp, dpi=300, bbox_inches='tight')
print(f"\n📸  Figure saved to: {fig_fp}")
plt.show()

