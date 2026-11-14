"""Comparison figures for the paper: (a) forest-style dot plot of the per-video AUC difference
CAM - method (filled = Holm-significant), (b) localisation vs faithfulness vs cost.
Reads RESULTADOS_estadistica.md (tables written by stats.py) and the timing JSON."""
import re, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS, TIMING, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})

# ---------------------------------------------------------------- parse the statistics tables
txt = open(STATS, encoding="utf-8").read()
blocks = re.split(r"\n## ", txt)
T = {}
for b in blocks:
    m = re.match(r"(\w+) — (\S+)", b)
    if not m:
        continue
    key = (m.group(1), m.group(2)); rows = {}
    for line in b.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 7 and c[0] not in ("Método", "---") and not c[0].startswith("-"):
            name = c[0].strip("*")
            auc = float(c[1].split("±")[0]) if "±" in c[1] else float("nan")
            aopc = float(c[3]) if c[3] not in ("—", "") else float("nan")
            d = float(c[4].replace("*", "").strip()) if c[4] not in ("—",) else 0.0
            sig = "*" in c[4]
            rows[name] = {"auc": auc, "aopc": aopc, "delta": d, "sig": sig}
    T[key] = rows
DS = [("xd_i3d", "XD-Violence\nI3D"), ("xd_clip", "XD-Violence\nCLIP"), ("ucf_i3d", "UCF-Crime\nI3D"),
      ("ucf_clip", "UCF-Crime\nCLIP"), ("thumos_i3d", "THUMOS14\nI3D")]
METH = ["Contextual Decomposition", "LRP-LSTM", "Murdoch-Szlam additive", "Murdoch-Szlam cell", "Integrated Gradients",
        "Sampled Shapley (s0)", "Occlusion", "Grad x Input", "Final memory", "Memory writes"]
SHORT = {"Contextual Decomposition": "Contextual Decomp.", "Murdoch-Szlam additive": "Murdoch-Szlam add.",
         "Murdoch-Szlam cell": "Murdoch-Szlam cell", "Sampled Shapley (s0)": "Sampled Shapley", "Grad x Input": "Grad × Input",
         "Final memory": "Final memory (ours)", "Memory writes": "Memory writes (ours)"}

# ---------------------------------------------------------------- (a) dot plot of differences
fig, axs = plt.subplots(1, 5, figsize=(7.2, 3.0), sharey=True)
y = np.arange(len(METH))[::-1]
for ax, (tag, ttl) in zip(axs, DS):
    rows = T[(tag, "lstm-const200")]
    ax.axvline(0, color=AXIS, lw=0.8)
    for yy, m in zip(y, METH):
        if m not in rows:
            continue
        r = rows[m]
        ax.plot([0, r["delta"]], [yy, yy], color=GRID, lw=1.0, zorder=1)
        ax.plot(r["delta"], yy, "o", ms=5, mew=1.2, color=BLUE, mfc=BLUE if r["sig"] else SURF, zorder=3)
    ax.set_title(ttl, fontsize=7.2, color=INK, loc="center")
    ax.set_xlim(-0.17, 0.17); ax.set_xticks([-0.1, 0, 0.1])
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
axs[0].set_yticks(y); axs[0].set_yticklabels([SHORT.get(m, m) for m in METH], color=INK2)
fig.supxlabel("Localisation AUC of the temporal CAM minus that of the method  (> 0: CAM better)", fontsize=7.2, color=INK2, y=0.02)
h1 = plt.Line2D([], [], marker="o", ls="", color=BLUE, mfc=BLUE, ms=5, label="significant (Wilcoxon, Holm p < 0.05)")
h2 = plt.Line2D([], [], marker="o", ls="", color=BLUE, mfc=SURF, ms=5, label="not significant")
fig.legend(handles=[h1, h2], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06), fontsize=6.8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/fig_cam_vs_methods.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_cam_vs_methods.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- (b) localisation vs faithfulness vs cost
tm = json.load(open(TIMING))
cost = {"Temporal CAM (ours)": 0.57, "Contextual Decomposition": tm["cd"]["mean_s"] * 1e3, "LRP-LSTM": tm["lrp"]["mean_s"] * 1e3,
        "Murdoch-Szlam additive": 0.57, "Integrated Gradients": tm["int_grad"]["mean_s"] * 1e3,
        "Sampled Shapley (s0)": tm["shapley"]["mean_s"] * 1e3, "Occlusion": tm["occlusion"]["mean_s"] * 1e3,
        "Grad x Input": tm["grad_x_input"]["mean_s"] * 1e3}
pts = {}
for m in cost:
    au, fa = [], []
    for tag, _ in DS:
        rows = T[(tag, "lstm-const200")]
        if m in rows:
            best = max(r["aopc"] for r in rows.values() if np.isfinite(r["aopc"]))
            au.append(rows[m]["auc"]); fa.append(rows[m]["aopc"] / best)
    pts[m] = (np.mean(au), np.mean(fa))
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for m, (a, f) in pts.items():
    ours = "ours" in m
    s = 18 + 55 * np.log10(1 + cost[m] / 0.5)
    ax.scatter(a, f, s=s, color=BLUE if ours else (ORANGE if m == "Contextual Decomposition" else MUTED),
               edgecolor=SURF, linewidth=1.2, zorder=3)
    off = {"Temporal CAM (ours)": (-7, -1), "Contextual Decomposition": (6, -14), "LRP-LSTM": (6, 7),
           "Murdoch-Szlam additive": (6, 0), "Integrated Gradients": (6, 3), "Sampled Shapley (s0)": (6, 6),
           "Occlusion": (6, -1), "Grad x Input": (6, -2)}[m]
    ax.annotate(f"{SHORT.get(m, m).replace(' (s0)', '')}  ({cost[m]:.1f} ms)" if not ours else f"Temporal CAM, ours ({cost[m]:.2f} ms)",
                (a, f), xytext=off, textcoords="offset points", fontsize=6.3, color=INK if ours else INK2,
                ha="right" if (off[0] < 0 or m == "Contextual Decomposition") else "left", va="center", fontweight="bold" if ours else "normal")
ax.set_xlabel("Localisation AUC (mean of 5 dataset–backbone pairs)")
ax.set_ylabel("Faithfulness (AOPC / best AOPC)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlim(0.60, 0.73); ax.set_ylim(0.55, 1.02)
ax.text(0.01, 0.02, "marker area grows with log(cost per video)", transform=ax.transAxes, fontsize=6, color=MUTED)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_tradeoff.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_tradeoff.png", dpi=300, bbox_inches="tight")
print("ok", {k: tuple(round(x, 3) for x in v) for k, v in pts.items()})
