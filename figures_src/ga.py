"""Graphical abstract (Elsevier: landscape, >= 1328 x 531 px). Uses the Shooting008 example data."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

RES, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, MUTED, AXIS, GT, WASH = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9", "#f0efec"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
SEQ = LinearSegmentedColormap.from_list("seq", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 8,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
key = "ucf_i3d|Shooting008_x264|0"
meta = {e["key"]: e for e in json.load(open(f"{RES}/examples.json"))}[key]
A = np.load(f"{RES}/examples.npz"); FR = json.load(open(f"{RES}/frames.json"))[key]["frames"]
T, st = meta["T"], meta["step_s"]; dur = T * st; t = (np.arange(T) + .5) * st; segs = meta["segs_s"]

fig = plt.figure(figsize=(8.4, 3.35))
W, H = fig.get_size_inches()
ax_ = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
canvas = ax_(0, 0, W, H); canvas.set_xlim(0, W); canvas.set_ylim(0, H); canvas.axis("off")


def title(x, s):
    canvas.text(x, 2.72, s, fontsize=8.6, fontweight="bold", color=INK, va="bottom")


canvas.text(0.15, 3.12, "What a recurrent memory keeps decides what a temporal explanation can show",
            fontsize=10.5, fontweight="bold", color=INK, va="center")
# ---------------------------------------------------------------- 1: video and model
title(0.15, "1  Untrimmed video, weak label")
ev = [f for f in FR if not f["peak"]][::2]
for k, f in enumerate(ev):
    a = ax_(0.15 + k * 0.52, 2.17, 0.5, 0.375)
    a.imshow(np.asarray(Image.open(f"{RES}/frames/{f['file']}").convert("RGB"))); a.axis("off")
a = ax_(0.15, 2.02, 2.58, 0.09); a.set_xlim(0, dur); a.axis("off")
for s0, s1 in segs:
    a.axvspan(s0, s1, color=GT, lw=0)
canvas.text(0.15, 1.86, "grey: annotated shooting (used only to evaluate)", fontsize=6.6, color=INK2, va="center")
for y, txt in ((1.3, "frozen encoder (I3D / CLIP), 16 frames per step"), (0.62, "1-layer (Bi-)LSTM  +  linear classifier")):
    canvas.add_patch(FancyBboxPatch((0.25, y - 0.2), 2.38, 0.4, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="white", ec=AXIS, lw=0.8))
    canvas.text(1.44, y, txt, ha="center", va="center", fontsize=7.2, color=INK)
canvas.add_patch(FancyArrowPatch((1.44, 1.1), (1.44, 0.83), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.8))
canvas.text(1.44, 0.24, "memory length set at initialisation:  $b_f=\\log(\\tau_0-1)$", ha="center", fontsize=6.8, color=INK2)
# ---------------------------------------------------------------- 2: memory maps
title(3.05, "2  Read the memory exactly")
for j, (init, lab) in enumerate((("const200", "long memory"), ("standard", "default (short)"))):
    M = A[f"{key}|{init}|map"].astype(float); M = M / (np.percentile(M[M > 0], 99) + 1e-12)
    a = ax_(3.15 + j * 1.27, 0.8, 1.08, 1.6)
    a.imshow(np.ma.masked_array(np.clip(M, 0, 1), np.tril(np.ones_like(M, bool), -1)), cmap=SEQ,
             norm=PowerNorm(0.5, 0, 1), origin="lower", extent=[0, dur, 0, dur], interpolation="nearest", aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(AXIS)
    a.set_title(lab, fontsize=7, color=INK2, pad=2)
    a.set_xlabel("read at τ", fontsize=6.5, color=MUTED, labelpad=1)
    if j == 0:
        a.set_ylabel("written at t", fontsize=6.5, color=MUTED, labelpad=1)
canvas.text(4.32, 0.36, "$c_T=\\sum_t\\, (\\prod_{k>t} f_k)\\odot(i_t\\odot g_t)$", ha="center", fontsize=7.6, color=INK)
canvas.text(4.32, 0.1, "what each step wrote, and how much is left", ha="center", fontsize=6.6, color=INK2)
# ---------------------------------------------------------------- 3: temporal CAM
title(5.95, "3  Temporal CAM (exact, 0.6 ms)")
for j, (init, lab) in enumerate((("const200", "long memory → points at the shooting"),
                                 ("standard", "default memory → points at the last seconds"))):
    a = ax_(6.0, 1.62 - j * 0.98, 2.25, 0.62)
    for s0, s1 in segs:
        a.axvspan(s0, s1, color=GT, lw=0)
    c = A[f"{key}|{init}|cam"]
    a.bar(t, c, width=st * 0.85, color=np.where(c >= 0, RED, BLUE), lw=0)
    a.axhline(0, color=AXIS, lw=0.6); a.set_xlim(0, dur); a.axis("off")
    a.set_title(lab, fontsize=7, color=INK2, pad=1, loc="left")
canvas.text(6.0, 0.36, "$\\sum_t s_t = \\mathrm{logit}-b$;  red: for the class, blue: against", fontsize=6.8, color=INK2)
canvas.text(6.0, 0.1, "also: retention maps, causal write-blocking tests", fontsize=6.6, color=INK2)
fig.savefig(f"{OUT}/graphical_abstract.png", dpi=300)
fig.savefig(f"{OUT}/graphical_abstract.pdf")
print("ok", fig.get_size_inches() * 300)
