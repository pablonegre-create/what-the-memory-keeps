"""Paper figures (English labels). Reads the figure data written by
src/figdata.py (out/fig) and writes PDF + PNG into the output folder."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

RES = sys.argv[1]            # figure data folder (out/fig)
OUT = sys.argv[2]            # output folder
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette (reference instance, light)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
POS, NEG, GTFILL = "#e34948", "#2a78d6", "#e1e0d9"          # diverging poles; annotation wash
SEQ = LinearSegmentedColormap.from_list("seq", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.4, "lines.solid_capstyle": "round"})

meta = {e["key"]: e for e in json.load(open(f"{RES}/examples.json"))}
A = np.load(f"{RES}/examples.npz")
FR = json.load(open(f"{RES}/frames.json"))
S = json.load(open(f"{RES}/summary.json"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def ax_in(fig, x, y, w, h):
    W, Hh = fig.get_size_inches()
    return fig.add_axes([x / W, y / Hh, w / W, h / Hh])


def gt_spans(ax, segs, dur, alpha=1.0):
    for a, b in segs:
        ax.axvspan(max(0, a), min(b, dur), color=GTFILL, alpha=alpha, lw=0, zorder=0)


def clean_time_axis(ax, dur, show=True):
    ax.set_xlim(0, dur)
    ax.spines["left"].set_visible(False); ax.set_yticks([])
    if not show:
        ax.spines["bottom"].set_visible(False); ax.set_xticks([])


def cam_row(ax, t, cam, st, segs, dur, label, peak_letter):
    gt_spans(ax, segs, dur)
    ax.bar(t, cam, width=st * 0.82, color=np.where(cam >= 0, POS, NEG), lw=0, zorder=2)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    m = np.max(np.abs(cam)) * 1.25
    ax.set_ylim(-m * 0.55, m)
    k = int(np.argmax(cam))
    ax.annotate(peak_letter, (t[k], cam[k]), xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=INK2)
    clean_time_axis(ax, dur, show=False)
    return k


# ================================================================= Figure 1: hero example
def hero(key, name):
    e = meta[key]; T, st = e["T"], e["step_s"]; dur = T * st; t = (np.arange(T) + .5) * st
    segs = e["segs_s"]; fr = FR[key]["frames"]
    fig = plt.figure(figsize=(7.2, 3.15))
    LW = 5.45; X0 = 0.95
    # filmstrip (10 evenly spaced frames, at their time positions)
    strip = ax_in(fig, X0, 2.25, LW, 0.62)
    strip.set_xlim(0, dur); strip.set_ylim(0, 1); strip.axis("off")
    ev = [f for f in fr if not f["peak"]]
    slot = dur / len(ev); tw = slot * 0.94
    th = (LW * tw / dur) * 0.75 / 0.62                 # 4:3 thumbnail height in axis units
    for k, f in enumerate(ev):                          # evenly spaced slots, connector to the true time
        c = (k + .5) * slot
        img = np.asarray(Image.open(f"{RES}/frames/{f['file']}").convert("RGB"))
        strip.imshow(img, extent=[c - tw / 2, c + tw / 2, 1 - th, 1], aspect="auto", zorder=2)
        strip.plot([c, f["t"]], [1 - th, 0], color=AXIS, lw=0.6, zorder=1, clip_on=False)
    strip.set_ylim(0, 1)
    strip.text(-0.01, 0.5, "Video", transform=strip.transAxes, ha="right", va="center", color=INK2)
    # annotation row
    gt = ax_in(fig, X0, 2.02, LW, 0.16)
    gt_spans(gt, segs, dur); clean_time_axis(gt, dur, show=False); gt.set_ylim(0, 1)
    import re as _re
    what = _re.match(r"[A-Za-z]+", e["name"]).group(0).lower() if e["tag"].startswith("ucf") else (
        "violence" if e["tag"].startswith("xd") else e["class"])
    gt.text(-0.01, 0.5, f"Annotated {what}",
            transform=gt.transAxes, ha="right", va="center", color=INK2)
    # CAM rows
    tauL = S["video"].get(f"{e['tag']}|const200", {})
    a1 = ax_in(fig, X0, 1.18, LW, 0.74)
    kL = cam_row(a1, t, A[f"{key}|const200|cam"], st, segs, dur, "Temporal CAM\nlong memory", "A")
    a2 = ax_in(fig, X0, 0.36, LW, 0.74)
    kS = cam_row(a2, t, A[f"{key}|standard|cam"], st, segs, dur, "Temporal CAM\nshort memory (default)", "B")
    a2.spines["bottom"].set_visible(True); a2.set_xticks(np.arange(0, dur + 1e-6, 10 if dur < 120 else 30))
    a2.set_xlabel("Time (s)")
    # peak frames on the right
    peaks = [f for f in fr if f["peak"]]
    for j, (f, y, lab, kk) in enumerate([(peaks[0], 1.18, "A", kL), (peaks[1], 0.36, "B", kS)]):
        ax = ax_in(fig, X0 + LW + 0.12, y, 0.99, 0.6)
        ax.imshow(np.asarray(Image.open(f"{RES}/frames/{f['file']}").convert("RGB"))); ax.axis("off")
        ax.set_title(f"{lab}: maximum at {f['t']:.0f} s", fontsize=7, color=INK2, pad=2)
    # legend for the bars
    lx = ax_in(fig, X0 + LW + 0.12, 2.02, 0.99, 0.75); lx.axis("off")
    for yy, col, txt in ((0.85, POS, "for the class"), (0.62, NEG, "against the class")):
        lx.add_patch(Rectangle((0.0, yy - 0.06), 0.09, 0.12, color=col, transform=lx.transAxes, lw=0))
        lx.text(0.14, yy, txt, transform=lx.transAxes, va="center", fontsize=6.8, color=INK2)
    lx.add_patch(Rectangle((0.0, 0.33), 0.09, 0.12, color=GTFILL, transform=lx.transAxes, lw=0))
    lx.text(0.14, 0.39, "annotated", transform=lx.transAxes, va="center", fontsize=6.8, color=INK2)
    save(fig, name)


# ================================================================= Figure 2: memory maps
def memory_maps(key, name):
    e = meta[key]; T, st = e["T"], e["step_s"]; dur = T * st; segs = e["segs_s"]
    lab = A[f"{key}|lab"].astype(bool)
    fig = plt.figure(figsize=(7.2, 3.7))
    share = {}
    for j, (init, title) in enumerate((("const200", "Long memory"), ("standard", "Short memory (default)"))):
        M = A[f"{key}|{init}|map"].astype(np.float64)
        ax = ax_in(fig, 0.55 + j * 2.75, 1.32, 2.2, 2.2)
        Mn = M / (np.percentile(M[M > 0], 99) + 1e-12)
        mask = np.tril(np.ones_like(M, bool), -1)
        ax.imshow(np.ma.masked_array(np.clip(Mn, 0, 1), mask), cmap=SEQ, norm=PowerNorm(0.5, 0, 1), origin="lower",
                  extent=[0, dur, 0, dur], interpolation="nearest")
        for a, b in segs:      # annotated writing times: grey bar outside the right edge
            ax.add_patch(Rectangle((dur * 1.02, a), dur * 0.03, b - a, color=AXIS, clip_on=False, lw=0))
        ax.set_xlim(0, dur); ax.set_ylim(0, dur)
        ax.set_title(title, color=INK, pad=4)
        ax.set_xlabel("Reading time τ (s)"); ax.set_ylabel("Writing time t (s)" if j == 0 else "")
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        tot = M.sum(0); share[init] = M[lab].sum(0) / np.maximum(tot, 1e-12)
    # colour key
    ck = ax_in(fig, 6.05, 1.32, 0.12, 2.2)
    ck.imshow(np.linspace(0, 1, 256)[:, None], cmap=SEQ, norm=PowerNorm(0.5, 0, 1), aspect="auto", origin="lower")
    ck.set_xticks([]); ck.set_yticks([0, 255]); ck.set_yticklabels(["0", "max"]); ck.yaxis.tick_right()
    ck.set_title("Memory from t\nleft at τ", fontsize=6.5, color=INK2, pad=3)
    # share of the memory that comes from annotated steps
    ax = ax_in(fig, 0.55, 0.3, 4.95, 0.62)
    gt_spans(ax, segs, dur)
    t = (np.arange(T) + .5) * st
    for init, col, lab_ in (("const200", BLUE, "Long memory"), ("standard", ORANGE, "Short memory")):
        ax.plot(t, share[init], color=col, lw=1.6)
        ax.text(dur * 1.01, share[init][-1], lab_, color=INK2, fontsize=6.8, va="center")
    ax.set_xlim(0, dur); ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Share of memory\nfrom annotated steps")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, name)


# ================================================================= Figure 3: explanation barcodes
def barcodes(keys, name):
    rows = [("cam", "Temporal CAM (ours)"), ("mem", "Final memory"), ("write", "Memory writes"),
            ("occlusion", "Occlusion"), ("int_grad", "Integrated Gradients"), ("shapley", "Sampled Shapley")]
    fig = plt.figure(figsize=(7.2, 0.6 + 1.78 * len(keys)))
    H = fig.get_size_inches()[1]
    import re as _re
    for p, key in enumerate(keys):
        e = meta[key]; T, st = e["T"], e["step_s"]; dur = T * st
        y0 = H - 0.4 - (p + 1) * 1.78
        ttl = {"ucf_i3d": "UCF-Crime", "xd_i3d": "XD-Violence", "thumos_i3d": "THUMOS14"}[e["tag"]]
        if e["tag"].startswith("ucf"):
            e = dict(e, **{"class": _re.match(r"[A-Za-z]+", e["name"]).group(0)})
        ax = ax_in(fig, 1.45, y0 + 1.2, 5.6, 0.12)
        gt_spans(ax, e["segs_s"], dur); clean_time_axis(ax, dur, show=False)
        ax.text(-0.01, 0.5, "Annotation", transform=ax.transAxes, ha="right", va="center", color=INK2, fontsize=7)
        ax.set_title(f"{ttl} · {e['class']}", loc="left", fontsize=7.5, color=INK, pad=3)
        img = []
        for r, _ in rows:
            kk = f"{key}|const200|{r}"
            v = A[kk] if kk in A.files else None
            if v is None:
                img.append(np.zeros(T)); continue
            v = np.maximum(np.asarray(v, float).reshape(-1)[:T], 0)
            img.append(v / (v.max() + 1e-12))
        ax = ax_in(fig, 1.45, y0 + 0.08, 5.6, 1.05)
        ax.imshow(np.array(img), cmap=SEQ, aspect="auto", interpolation="nearest", extent=[0, dur, len(rows), 0])
        for a, b in e["segs_s"]:
            ax.axvline(a, color=INK2, lw=0.5, alpha=0.3); ax.axvline(b, color=INK2, lw=0.5, alpha=0.3)
        ax.set_yticks(np.arange(len(rows)) + .5); ax.set_yticklabels([r[1] for r in rows], color=INK2)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, dur)
        if p == len(keys) - 1:
            ax.set_xlabel("Time (s)")
    save(fig, name)


# ================================================================= Figure 4: memory length dose-response
def dose(name):
    D = S["dose"]
    const = ["standard", "const10", "const50", "const200", "const500", "const1000"]
    other = [("chrono", "chrono"), ("loguniform", "log-uniform"), ("uniform_gate", "uniform gate")]
    panels = [("cam_point", "CAM peak inside the action"), ("auc", "Video-level AUC"), ("det30", "Detection, >30 s after the action")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.15))
    for ax, (m, ttl) in zip(axs, panels):
        x = [D[k]["tau"][0] for k in const]; y = [D[k][m][0] for k in const]; s = [D[k][m][1] for k in const]
        ax.plot(x, y, color=BLUE, lw=1.6, zorder=2)
        ax.errorbar(x, y, yerr=s, fmt="o", ms=4.5, color=BLUE, mec=SURF, mew=1.2, elinewidth=0.8, capsize=0, zorder=3)
        for k, lab in other:
            ax.errorbar(D[k]["tau"][0], D[k][m][0], yerr=D[k][m][1], fmt="D", ms=4, color=ORANGE, mec=SURF,
                        mew=1.0, elinewidth=0.8, zorder=3)
        ax.set_xscale("log"); ax.set_title(ttl, loc="left", color=INK)
        ax.set_xlabel("Learned memory τ (steps, log)")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.annotate("default", (D["standard"]["tau"][0], D["standard"][m][0]), xytext=(6, -7 if m == "cam_point" else -2),
                    textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        ax.annotate("200", (D["const200"]["tau"][0], D["const200"][m][0]), xytext=(-5, 7),
                    textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
        if m == "cam_point":
            for k, lab in other:
                off = (0, 9) if k == "uniform_gate" else (6, -1)
                ax.annotate(lab, (D[k]["tau"][0], D[k][m][0]), xytext=off, textcoords="offset points",
                            fontsize=6.2, color=INK2, va="center", ha="left" if k != "uniform_gate" else "center")
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", ms=4.5, label="all cells with the same initial memory (default, 10, 50, 200, 500, 1000 steps)"),
                        plt.Line2D([], [], color=ORANGE, marker="D", ls="", ms=4, label="cells with different initial memories")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=6.6)
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


# ================================================================= Figure 5: tail / duration effects
def tails(name):
    tags = [("xd_i3d", "XD-Violence I3D"), ("xd_clip", "XD-Violence CLIP"), ("ucf_i3d", "UCF-Crime I3D"), ("thumos_i3d", "THUMOS14")]
    series = [("const200", BLUE, "LSTM, long memory"), ("standard", ORANGE, "LSTM, short memory"), ("poolmean", AQUA, "No memory (mean pool)")]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey="row")
    for c, (tag, ttl) in enumerate(tags):
        for r, (var, met, ylab) in enumerate((("tail_s", "det", "Recall of positive videos"), ("act_s", "loc", "Localisation AUC"))):
            ax = axs[r, c]
            for mdl, col, lab in series:
                b = S["bins"].get(f"{tag}|{mdl}", {}).get(var)
                if not b:
                    continue
                y = np.array([x[met][0] if x[met][0] is not None else np.nan for x in b])
                s = np.array([x[met][1] if x[met][1] is not None else 0 for x in b])
                xx = np.arange(len(b))
                ax.plot(xx, y, color=col, lw=1.6, marker="o", ms=3.8, mec=SURF, mew=1.0, label=lab)
                ax.fill_between(xx, y - s, y + s, color=col, alpha=0.10, lw=0)
            ax.set_xticks(np.arange(len(b))); ax.set_xticklabels([x["bin"] for x in b], rotation=0, fontsize=6.3)
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(ttl, loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(ylab)
        axs[0, c].set_xlabel("Time after the action", fontsize=6.8)
        axs[1, c].set_xlabel("Action duration", fontsize=6.8)
    axs[0, 0].legend(loc="lower left", fontsize=6.3)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, name)


# ================================================================= Figure 6: interventions
def interventions(name):
    tags = [("xd_i3d", "XD I3D"), ("xd_clip", "XD CLIP"), ("ucf_i3d", "UCF I3D"), ("ucf_clip", "UCF CLIP"), ("thumos_i3d", "THUMOS14")]
    bars = [("gt_write", BLUE, "Block memory writes during the action"),
            ("nongt_write", ORANGE, "Block as many writes outside it"),
            ("gt_input", AQUA, "Delete the action from the input")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    for ax, (mdl, ttl) in zip(axs, (("const200", "Long memory"), ("standard", "Short memory (default)"))):
        w = 0.24
        for j, (k, col, lab) in enumerate(bars):
            for i, (tag, _) in enumerate(tags):
                v = S["interv"].get(f"{tag}|{mdl}", {}).get(k)
                if not v:
                    continue
                x = i + (j - 1) * (w + 0.03)
                m, s = v
                ax.bar(x, m, width=w, color=col, lw=0)
                ax.plot([x, x], [m - s, m + s], color=INK2, lw=0.8)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(range(len(tags))); ax.set_xticklabels([t[1] for t in tags])
        ax.set_xlim(-0.6, len(tags) - 0.4); ax.set_title(ttl, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("Drop of the class logit")
    axs[0].set_ylim(-0.6, 3.0)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for _, c, _ in bars], labels=[l for _, _, l in bars],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=6.8)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    hero("ucf_i3d|Shooting008_x264|0", "fig_example_ucf_shooting008")
    hero("ucf_i3d|Shooting004_x264|0", "fig_example_ucf_shooting004")
    hero("ucf_i3d|Fighting033_x264|0", "fig_example_ucf_fighting033")
    memory_maps("ucf_i3d|Shooting008_x264|0", "fig_memory_map_shooting008")
    barcodes(["ucf_i3d|Shooting008_x264|0", "xd_i3d|Haywire.2011__#00-16-07_00-16-49_label_B1-0-0|0",
              "thumos_i3d|video_test_0000814|18"], "fig_methods_barcodes")
    dose("fig_memory_dose_response")
    tails("fig_tail_duration")
    interventions("fig_interventions")
