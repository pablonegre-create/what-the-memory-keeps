"""Dataset adapters. Every dataset returns the same structure:
  tr/te: dict(names, X list of (T,d) float32 normalised, Y (N,) or (N,C) float32)
  gt[name][cls] -> bool array (T,) of annotated steps (test positives only)
  seg_s[name][cls] -> list of (start_s, end_s) annotated segments in seconds
  step_s[name] -> duration of one step in seconds;  n_out, classes."""
import os, glob, json
import numpy as np
from common import step_labels

F = "data/feat"


def _norm(tr, te):
    allx = np.concatenate([x[::4] for x in tr["X"]])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    for d in (tr, te):
        d["X"] = [((x - mu) / sd).astype(np.float32) for x in d["X"]]
    return mu, sd


def _gt_from_frames(segs_frames, T, fps, clip=16):
    lab = step_labels(segs_frames, T, clip) > 0.5
    return lab, [(s / fps, e / fps) for s, e in segs_frames]


def load_xd(backbone):
    if backbone == "i3d":
        d = f"{F}/xd_i3d/i3d-features"
        path = lambda split, n: f"{d}/{'RGB' if split == 'train' else 'RGBTest'}/{n}__0.npy"
        names = {s: sorted(f[:-7] for f in os.listdir(f"{d}/{'RGB' if s == 'train' else 'RGBTest'}"))
                 for s in ("train", "test")}
    else:
        d = f"{F}/xd_clip"
        sub = {"train": "XDTrainClipFeatures", "test": "XDTestClipFeatures"}
        path = lambda split, n: f"{d}/{sub[split]}/{n}__0.npy"
        names = {s: sorted(f[:-7] for f in os.listdir(f"{d}/{sub[s]}")) for s in ("train", "test")}
    out = {}
    for s in ("train", "test"):
        X = [np.load(path(s, n)).astype(np.float32) for n in names[s]]
        Y = np.array([0.0 if "label_A" in n else 1.0 for n in names[s]], np.float32)
        out[s] = {"names": names[s], "X": X, "Y": Y}
    ann = {}
    for line in open("data/annot/vadclip_annotations.txt"):
        t = line.split()
        if t:
            v = list(map(int, t[1:]))
            ann[t[0]] = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
    gt, seg_s, step_s = {}, {}, {}
    for n, x in zip(out["test"]["names"], out["test"]["X"]):
        step_s[n] = 16 / 24
        if n in ann:
            lab, ss = _gt_from_frames(ann[n], len(x), 24)
            gt[n], seg_s[n] = {0: lab}, {0: ss}
    mu, sd = _norm(out["train"], out["test"])
    return {"tr": out["train"], "te": out["test"], "gt": gt, "seg_s": seg_s, "step_s": step_s,
            "n_out": 1, "classes": ["violence"], "epochs": 30}


def load_ucf(backbone):
    root = f"{F}/ucf_clip/UCFClipFeatures"
    where, folder = {}, {}
    for c in os.listdir(root):
        for f in os.listdir(f"{root}/{c}"):
            if f.endswith("__0.npy"):
                where[f[:-7]] = f"{root}/{c}/{f}"; folder[f[:-7]] = c
    test = [l.split(",")[0].split("/")[-1].rsplit("__", 1)[0]
            for l in open("data/annot/vadclip_ucf_CLIP_rgbtest.csv").read().splitlines()[1:]]
    tset = set(test)
    names = {"test": sorted(tset), "train": sorted(set(where) - tset)}
    path = (lambda n: where[n]) if backbone == "clip" else (lambda n: f"{F}/ucf_i3d/{n}.npy")
    out = {}
    for s in ("train", "test"):
        ns = [n for n in names[s] if os.path.exists(path(n))]
        X = [np.load(path(n)).astype(np.float32) for n in ns]
        Y = np.array([0.0 if "Normal" in folder[n] else 1.0 for n in ns], np.float32)
        out[s] = {"names": ns, "X": X, "Y": Y}
    ann = {}
    for line in open("data/annot/vadclip_Temporal_Anomaly_Annotation.txt"):
        t = line.split()
        if t:
            v = list(map(int, t[2:6]))
            ann[t[0]] = [(v[i], v[i + 1]) for i in (0, 2) if v[i] != -1]
    gt, seg_s, step_s = {}, {}, {}
    for n, x in zip(out["test"]["names"], out["test"]["X"]):
        step_s[n] = 16 / 30
        if ann.get(n):
            lab, ss = _gt_from_frames(ann[n], len(x), 30)
            gt[n], seg_s[n] = {0: lab}, {0: ss}
    _norm(out["train"], out["test"])
    return {"tr": out["train"], "te": out["test"], "gt": gt, "seg_s": seg_s, "step_s": step_s,
            "n_out": 1, "classes": ["anomaly"], "epochs": 30}


def load_thumos(backbone="i3d"):
    assert backbone == "i3d"
    db = json.load(open(f"{F}/thumos/thumos/annotations/thumos14.json"))["database"]
    classes = sorted({a["label"] for v in db.values() for a in v["annotations"]})
    cid = {c: i for i, c in enumerate(classes)}
    out, gt, seg_s, step_s = {}, {}, {}, {}
    for s, sub in (("train", "validation"), ("test", "test")):
        names, X, Y = [], [], []
        for k in sorted(db):
            v = db[k]
            p = f"{F}/thumos/thumos/i3d_features/{k}.npy"
            if v["subset"].lower() != sub or not os.path.exists(p):
                continue
            x4 = np.load(p).astype(np.float32)
            T = len(x4) // 4
            if T < 2:
                continue
            x = x4[:T * 4].reshape(T, 4, -1).mean(1)          # 4 features (stride 4) -> 16 frames
            y = np.zeros(len(classes), np.float32)
            for a in v["annotations"]:
                y[cid[a["label"]]] = 1
            names.append(k); X.append(x); Y.append(y)
            if s == "test":
                fps = v["fps"]; step_s[k] = 16 / fps
                gt[k], seg_s[k] = {}, {}
                for c in {cid[a["label"]] for a in v["annotations"]}:
                    sf = [(a["segment"][0] * fps, a["segment"][1] * fps) for a in v["annotations"]
                          if cid[a["label"]] == c]
                    lab, ss = _gt_from_frames(sf, T, fps)
                    gt[k][c], seg_s[k][c] = lab, ss
        out[s] = {"names": names, "X": X, "Y": np.array(Y, np.float32)}
    _norm(out["train"], out["test"])
    return {"tr": out["train"], "te": out["test"], "gt": gt, "seg_s": seg_s, "step_s": step_s,
            "n_out": len(classes), "classes": classes, "epochs": 60}


LOADERS = {"xd": load_xd, "ucf": load_ucf, "thumos": load_thumos}
