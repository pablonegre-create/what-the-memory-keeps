"""Check that features match annotations and compute exact duration/coverage stats."""
import os, json, numpy as np

F, A = "data/feat", "data/annot"


def T(path):
    return np.load(path, mmap_mode="r").shape[0]


def stats(rows, label):
    """rows: list of (video_len_s, [(start_s, end_s), ...]) for positive videos."""
    vl, act, cov, tail, nseg, onset = [], [], [], [], [], []
    for L, segs in rows:
        segs = sorted((max(0, s), min(e, L)) for s, e in segs if s < L and e > s)
        if not segs or L <= 0:
            continue
        # union of segments
        u, cur = [], list(segs[0])
        for s, e in segs[1:]:
            if s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                u.append(cur); cur = [s, e]
        u.append(cur)
        a = sum(e - s for s, e in u)
        vl.append(L); act.append(a); cov.append(a / L); tail.append(L - u[-1][1])
        nseg.append(len(u)); onset.append(u[0][0] / L)
    vl, act, cov, tail, nseg, onset = map(np.array, (vl, act, cov, tail, nseg, onset))
    q = lambda x: "  ".join(f"{v:7.1f}" for v in np.percentile(x, [5, 25, 50, 75, 95]))
    print(f"\n--- {label}: {len(vl)} positive videos ---")
    print("percentiles               p5      p25      p50      p75      p95")
    print("video length (s)      ", q(vl))
    print("action duration (s)   ", q(act))
    print("coverage (%)          ", q(100 * cov))
    print("tail after action (s) ", q(tail))
    print("onset position (%)    ", q(100 * onset))
    print(f"videos with >1 occurrence: {np.mean(nseg > 1) * 100:.0f}%  (max {nseg.max()})")
    print("coverage bins <10|10-25|25-50|50-75|75-90|>90 %:",
          np.histogram(cov, [0, .1, .25, .5, .75, .9, 1.01])[0].tolist())
    print("action dur bins <2|2-5|5-10|10-30|30-60|>60 s:",
          np.histogram(act, [0, 2, 5, 10, 30, 60, 1e9])[0].tolist())
    print("corr(log action, log coverage) = %.2f" % np.corrcoef(np.log(act), np.log(cov))[0, 1])


# ---------------- XD-Violence (I3D, 16 frames per step, 24 fps) ----------------
print("=" * 70, "\nXD-Violence I3D")
d = f"{F}/xd_i3d/i3d-features"
for sub in ["RGB", "Flow", "RGBTest", "FlowTest"]:
    names = {f.rsplit("__", 1)[0] for f in os.listdir(f"{d}/{sub}")}
    crops = {f.rsplit("__", 1)[1] for f in os.listdir(f"{d}/{sub}")}
    print(f"{sub}: {len(names)} videos, crops {sorted(crops)}")
ann = {}
for line in open(f"{A}/vadclip_annotations.txt"):
    t = line.split()
    if t:
        v = list(map(int, t[1:]))
        ann[t[0]] = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
test = sorted({f.rsplit("__", 1)[0] for f in os.listdir(f"{d}/RGBTest")})
rows, bad, sumT, mism = [], 0, 0, 0
for n in test:
    t_rgb, t_flow = T(f"{d}/RGBTest/{n}__0.npy"), T(f"{d}/FlowTest/{n}__0.npy")
    mism += t_rgb != t_flow
    sumT += t_rgb
    nframes = t_rgb * 16
    if "label_A" in n:
        continue
    segs = ann.get(n)
    if segs is None:
        bad += 1
        continue
    if max(e for _, e in segs) > nframes + 32:
        bad += 1
    rows.append((nframes / 24, [(s / 24, e / 24) for s, e in segs]))
gt = np.load(f"{A}/vadclip_gt.npy")
print(f"test videos {len(test)}, violent with annotation {len(rows)}, "
      f"annotation missing or beyond video end: {bad}, RGB/flow length mismatch: {mism}")
print(f"sum of steps x16 = {sumT * 16} frames vs CLIP-based GT {len(gt)} frames")
stats(rows, "XD-Violence test (exact lengths)")
tr = [f for f in os.listdir(f"{d}/RGB") if f.endswith("__0.npy")]
lens = np.array([T(f"{d}/RGB/{f}") for f in tr]) * 16 / 24
print(f"\nXD train videos {len(tr)} (violent {sum('label_A' not in f for f in tr)}), "
      f"length s p5/p50/p95/max = {np.percentile(lens, [5, 50, 95]).round(1).tolist()} / {lens.max():.0f}")

# ---------------- UCF-Crime (CLIP, 16 frames per step, 30 fps) ----------------
print("\n" + "=" * 70, "\nUCF-Crime CLIP")
d = f"{F}/ucf_clip/UCFClipFeatures"
where = {}
for c in os.listdir(d):
    for f in os.listdir(f"{d}/{c}"):
        if f.endswith("__0.npy"):
            where[f[:-7]] = f"{d}/{c}/{f}"
print(f"videos with features: {len(where)}")
ua = {}
for line in open(f"{A}/vadclip_Temporal_Anomaly_Annotation.txt"):
    t = line.split()
    if t:
        v = list(map(int, t[2:6]))
        ua[t[0]] = (t[1], [(v[i], v[i + 1]) for i in (0, 2) if v[i] != -1])
testlist = [l.split(",")[0].split("/")[-1].rsplit("__", 1)[0]
            for l in open(f"{A}/vadclip_ucf_CLIP_rgbtest.csv").read().splitlines()[1:]]
missing = [n for n in testlist if n not in where]
sumT = sum(T(where[n]) for n in testlist if n in where)
gtu = np.load(f"{A}/vadclip_gt_ucf.npy")
print(f"test list {len(testlist)}, missing features {len(missing)}; "
      f"sum steps x16 = {sumT * 16} vs GT {len(gtu)} frames")
rows, rows_v = [], []
violent = {"Abuse", "Assault", "Fighting", "Shooting", "Robbery", "Arrest"}
for n, (cls, segs) in ua.items():
    if n not in where or not segs:
        continue
    L = T(where[n]) * 16 / 30
    r = (L, [(s / 30, e / 30) for s, e in segs])
    rows.append(r)
    if cls in violent:
        rows_v.append(r)
stats(rows, "UCF-Crime test, all anomalies (exact lengths)")
stats(rows_v, "UCF-Crime test, violent subset")

# ---------------- THUMOS14 (I3D, stride 4 frames) ----------------
print("\n" + "=" * 70, "\nTHUMOS14 I3D")
db = json.load(open(f"{F}/thumos/thumos/annotations/thumos14.json"))["database"]
subs = {}
for k, v in db.items():
    subs.setdefault(v["subset"], []).append(k)
print({k: len(v) for k, v in subs.items()})
ex = next(iter(db.values()))
print("example keys:", list(ex.keys()), "fps:", ex.get("fps"), "ann keys:", list(ex["annotations"][0].keys()))
rows, ratio, ninst, ncls = [], [], [], []
for k, v in db.items():
    p = f"{F}/thumos/thumos/i3d_features/{k}.npy"
    if not os.path.exists(p):
        continue
    ratio.append(T(p) * 4 / v["fps"] / v["duration"])
    if v["subset"].lower() != "test":
        continue
    segs = [tuple(a["segment"]) for a in v["annotations"]]
    rows.append((v["duration"], segs))
    ninst.append(len(segs)); ncls.append(len({a["label"] for a in v["annotations"]}))
print(f"(steps x 4 / fps) / duration: p5/p50/p95 = {np.percentile(ratio, [5, 50, 95]).round(3).tolist()}")
print(f"test: instances per video median {np.median(ninst):.0f} (p95 {np.percentile(ninst, 95):.0f}), "
      f"videos with >1 class: {np.mean(np.array(ncls) > 1) * 100:.0f}%")
stats(rows, "THUMOS14 test")
