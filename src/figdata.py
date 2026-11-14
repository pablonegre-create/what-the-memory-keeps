"""Pick example videos with a fixed rule and compute everything the qualitative figures need:
per-step explanations of both LSTMs (long / short memory) and the memory maps
M[t, tau] = || prod_{k=t+1..tau} f_k * (i_t g_t) ||  (how much of step t is still in memory at tau).
Writes out/fig/examples.npz and out/fig/examples.json."""
import json, os
import numpy as np
import torch
from common import Model, replay, decompose
from data import LOADERS

OUT, FIG, H = "out/full", "out/fig", 128
os.umask(0o077); os.makedirs(FIG, exist_ok=True)
VIOLENT = ("Fighting", "Assault", "Abuse", "Robbery", "Shooting", "Arrest")


def pv(tag, init):
    return {(p["name"], p["cls"]): p for p in json.load(open(f"{OUT}/{tag}_{init}_s0.json"))["per_video"]}


def pick(tag, n_good, need_violent=False, min_seg=1):
    L, S = pv(tag, "const200"), pv(tag, "standard")
    cand = []
    for k, p in L.items():
        q = S.get(k, {})
        if "auc_cam" not in p or "auc_cam" not in q:
            continue
        if need_violent and not p["name"].startswith(VIOLENT):
            continue
        if not (60 <= p["T"] <= 450 and 0.08 <= p["cov"] <= 0.6 and p["n_seg"] >= min_seg):
            continue
        cand.append((k, p, q))
    good = [c for c in cand if c[1]["point_cam"] == 1 and c[1]["auc_cam"] >= .8 and c[2]["point_cam"] == 0]
    good.sort(key=lambda c: -(c[1]["auc_cam"] - c[2]["auc_cam"]))
    allc = sorted(cand, key=lambda c: c[1]["auc_cam"])
    typical = allc[len(allc) // 2] if allc else None
    fail = [c for c in allc if c[1]["point_cam"] == 0 and c[1]["auc_cam"] < .45]
    chosen = [("good", c) for c in good[:n_good]]
    if typical:
        chosen.append(("typical", typical))
    if fail:
        chosen.append(("failure", fail[len(fail) // 2]))
    return chosen


def mem_map(rec):
    f, ig = rec["f"], rec["i"] * rec["g"]
    Lc = torch.cumsum(torch.log(f.clamp_min(1e-300)), 0)                  # (T,H)
    keep = torch.exp((Lc[None, :, :] - Lc[:, None, :]).clamp(max=0))       # [t,tau,H] for tau>=t
    T = len(f); tri = torch.triu(torch.ones(T, T, dtype=torch.bool))
    M = (keep * ig[:, None, :]).norm(dim=2) * tri
    return M.float().numpy()


meta, arrays = [], {}
for tag, ds, bb, kw in [("ucf_i3d", "ucf", "i3d", dict(n_good=4, need_violent=True)),
                        ("xd_i3d", "xd", "i3d", dict(n_good=3)),
                        ("thumos_i3d", "thumos", "i3d", dict(n_good=2, min_seg=3))]:
    D = LOADERS[ds](bb)
    models = {}
    for init in ("const200", "standard"):
        m = Model(D["tr"]["X"][0].shape[1], H, 0.0, D["n_out"])
        m.load_state_dict(torch.load(f"{OUT}/{tag}_{init}_s0.pt", weights_only=True, map_location="cpu")); m.eval()
        models[init] = m
    sc = {init: np.load(f"{OUT}/scores_{tag}_{init}_s0.npz") for init in ("const200", "standard")}
    for role, ((name, cls), p, q) in pick(tag, **kw):
        i = D["te"]["names"].index(name); x = torch.from_numpy(D["te"]["X"][i])
        key = f"{tag}|{name}|{cls}"
        e = {"key": key, "tag": tag, "name": name, "cls": int(cls), "class": D["classes"][cls], "role": role,
             "T": len(x), "step_s": D["step_s"][name], "segs_s": D["seg_s"][name][cls],
             "auc": {}, "point": {}}
        for init, m in models.items():
            rec = replay(m, x)
            d = decompose(rec, m, cls)
            arrays[f"{key}|{init}|cam"] = d["cam"].numpy().astype(np.float32)
            arrays[f"{key}|{init}|mem"] = d["mem"].numpy().astype(np.float32)
            arrays[f"{key}|{init}|write"] = d["write"].numpy().astype(np.float32)
            arrays[f"{key}|{init}|map"] = mem_map(rec).astype(np.float16)
            e[f"logit_{init}"] = float(rec["logits"][cls])
            for meth in ("occlusion", "int_grad", "grad_x_input", "shapley"):
                k2 = f"{name}|{cls}|{meth}"
                if k2 in sc[init].files:
                    arrays[f"{key}|{init}|{meth}"] = sc[init][k2].astype(np.float32)
        arrays[f"{key}|lab"] = D["gt"][name][cls].astype(np.uint8)
        for src, init in ((p, "const200"), (q, "standard")):
            e["auc"][init] = {k[4:]: v for k, v in src.items() if k.startswith("auc_")}
            e["point"][init] = {k[6:]: v for k, v in src.items() if k.startswith("point_")}
        meta.append(e)
        print(f"{role:8s} {key}: T {e['T']} AUC CAM long {p['auc_cam']:.2f} short {q['auc_cam']:.2f}", flush=True)
    del D
np.savez_compressed(f"{FIG}/examples.npz", **arrays)
json.dump(meta, open(f"{FIG}/examples.json", "w"), indent=1)
print("saved", len(meta), "examples")
