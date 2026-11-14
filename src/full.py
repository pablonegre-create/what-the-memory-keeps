"""Full experiment. For each (dataset, backbone, init, seed): train the LSTM, then evaluate
detection, the memory explanations (write / memory / temporal CAM) against the temporal
annotation, perturbation and gradient baselines, faithfulness (deletion / keep curves),
interventions on the memory (blocking writes), and a randomisation sanity check.
Pooling classifiers without memory (mean / max) are trained as references.
Configuration is read from cfg.json (no command-line arguments)."""
import json, os, time, math
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from common import Model, replay_any as replay_many, decompose_any as decompose
from data import LOADERS
from exp import init_bias

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = json.load(open(os.environ.get("CFG", "cfg.json")))
CELL = CFG.get("cell", "lstm")
BIDIR = CFG.get("bidir", False)
SMOKE = CFG.get("smoke", False)
OUT = CFG.get("out", "out/full")
os.makedirs(OUT, exist_ok=True)
TMAX, BS, LR, H, BLOCK = 1200, 32, 1e-3, 128, 5
QS = [0.1, 0.2, 0.3, 0.5]
t0 = time.time()


def log(m):
    print(f"[{(time.time() - t0) / 60:7.1f} min] {m}", flush=True)


# ------------------------------------------------------------------ helpers
def cap(x, tmax):
    return x if len(x) <= tmax else x[np.linspace(0, len(x) - 1, tmax).round().astype(int)]


def to_batch(seqs):
    seqs = [s if torch.is_tensor(s) else torch.from_numpy(s) for s in seqs]
    return (nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(DEV),
            torch.tensor([len(s) for s in seqs]))


@torch.no_grad()
def logits_of(model, seqs, budget=300_000):
    """Logits (n, n_out) of many sequences, batched by length."""
    model.eval()
    order = np.argsort([len(s) for s in seqs]); out = np.zeros((len(seqs), model.n_out))
    k = 0
    while k < len(order):
        j = k
        while j < len(order) and (j - k + 1) * len(seqs[order[j]]) <= budget:
            j += 1
        j = max(j, k + 1)
        xb, lb = to_batch([seqs[i] for i in order[k:j]])
        o = model(xb, lb).reshape(j - k, -1).double().cpu().numpy()
        out[order[k:j]] = o
        k = j
    return out


def video_metrics(Y, P):
    if Y.ndim == 1:
        return {"auc": roc_auc_score(Y, P), "ap": average_precision_score(Y, P),
                "acc": float(((P > 0) == (Y > .5)).mean())}
    ok = Y.sum(0) > 0
    return {"map": float(np.mean([average_precision_score(Y[:, c], P[:, c]) for c in np.where(ok)[0]]))}


def loc_metrics(lab, s, rng):
    s = np.asarray(s, np.float64) + 1e-12 * rng.rand(len(s))
    k = int(lab.sum())
    pos = np.maximum(s, 0)
    nrm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return {"auc": roc_auc_score(lab, s), "point": float(lab[np.argmax(s)]),
            "pk": float(lab[np.argsort(-s)[:k]].mean()),
            "mass": float(pos[lab].sum() / (pos.sum() + 1e-12)),
            "tpr": float((nrm[lab] > .5).mean()), "fpr": float((nrm[~lab] > .5).mean()),
            "last10": float(np.argmax(s) / max(1, len(s) - 1) > .9)}


def describe(name, cls, D, T):
    st = D["step_s"][name]; vid = T * st
    segs = sorted((max(0, a), min(b, vid)) for a, b in D["seg_s"][name][cls] if a < vid)
    u = []
    for a, b in segs:
        if u and a <= u[-1][1]:
            u[-1][1] = max(u[-1][1], b)
        else:
            u.append([a, b])
    act = sum(b - a for a, b in u)
    return {"T": T, "video_s": vid, "act_s": act, "cov": act / vid if vid else 0,
            "tail_s": vid - u[-1][1] if u else vid, "n_seg": len(u),
            "onset": u[0][0] / vid if u else 0, "seg_mean_s": act / max(1, len(u))}


# ------------------------------------------------------------------ training
def train_lstm(D, init, seed):
    X, Y = D["tr"]["X"], D["tr"]["Y"]
    idx = np.random.RandomState(0).permutation(len(X)); nv = max(2, len(X) // 10)
    val_i, tr_i = idx[:nv], idx[nv:]
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model = Model(X[0].shape[1], H, 0.3, D["n_out"], cell=CELL, bidir=BIDIR).to(DEV)
    init_bias(model, init, gen=torch.Generator().manual_seed(seed))
    bias = [p for n, p in model.named_parameters() if "bias" in n]
    other = [p for n, p in model.named_parameters() if "bias" not in n]
    opt = torch.optim.Adam([{"params": other, "weight_decay": 1e-4},
                            {"params": bias, "weight_decay": 0.0}], lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Yt = torch.from_numpy(Y)
    best, state, epochs = 1e9, None, (1 if SMOKE else D["epochs"])
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr_i)
        for k in range(0, len(perm), BS):
            ids = perm[k:k + BS]
            xb, lb = to_batch([cap(X[i], TMAX) for i in ids])
            loss = lossf(model(xb, lb), Yt[ids].to(DEV))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pv = logits_of(model, [cap(X[i], TMAX) for i in val_i])
        pv = pv[:, 0] if D["n_out"] == 1 else pv
        vl = float(lossf(torch.from_numpy(pv), Yt[val_i].double()))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); model.drop.p = 0.0
    return model, best


def train_pool(D, kind, seed):
    """Linear classifier on mean- or max-pooled features (no memory)."""
    X, Y = D["tr"]["X"], D["tr"]["Y"]
    pool = (lambda x: x.mean(0)) if kind == "mean" else (lambda x: x.max(0))
    P = torch.from_numpy(np.stack([pool(x) for x in X])).to(DEV)
    Yt = torch.from_numpy(Y).to(DEV)
    idx = np.random.RandomState(0).permutation(len(X)); nv = max(2, len(X) // 10)
    val_i, tr_i = torch.tensor(idx[:nv]), torch.tensor(idx[nv:])
    torch.manual_seed(seed)
    fc = nn.Linear(P.shape[1], D["n_out"]).to(DEV)
    opt = torch.optim.Adam(fc.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); best, state = 1e9, None
    for ep in range(30 if SMOKE else 400):
        fc.train(); out = fc(P[tr_i]).reshape(Yt[tr_i].shape)
        loss = lossf(out, Yt[tr_i]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = lossf(fc(P[val_i]).reshape(Yt[val_i].shape), Yt[val_i]).item()
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in fc.state_dict().items()}
    fc.load_state_dict(state)
    return fc


# ------------------------------------------------------------------ explanation baselines
def occlusion(model, x, full):
    T = len(x); starts = list(range(0, T, BLOCK))
    seqs = [torch.cat([x[:s], x[s + BLOCK:]]) if T > BLOCK else x for s in starts]
    out = logits_of(model, seqs)
    sc = np.zeros((T, model.n_out))
    for s, v in zip(starts, out):
        sc[s:s + BLOCK] = full - v
    return sc


def grads(model, x, cls, steps=32):
    model.train()                       # cuDNN backward needs train mode (dropout is 0)
    xs = x[None].to(DEV)
    xi = xs.clone().requires_grad_(True)
    o = model(xi, torch.tensor([len(x)])).reshape(1, -1)[:, cls].sum()
    o.backward()
    gxi = (xi.grad * xi).sum(-1)[0].detach().cpu().numpy()
    igs = torch.zeros_like(xs)
    for a0 in range(0, steps, 8):
        al = torch.arange(a0 + 1, min(steps, a0 + 8) + 1, device=DEV, dtype=xs.dtype)[:, None, None] / steps
        xb = (al * xs).requires_grad_(True)
        model(xb, torch.tensor([len(x)] * len(al))).reshape(len(al), -1)[:, cls].sum().backward()
        igs += xb.grad.sum(0, keepdim=True)
    ig = (igs / steps * xs).sum(-1)[0].detach().cpu().numpy()
    model.eval()
    return gxi, ig


def shapley(model, x, M=16, nb=40):
    """Permutation-sampling Shapley values of contiguous blocks (removal = deletion)."""
    T = len(x); nb = min(nb, T); edges = np.linspace(0, T, nb + 1).round().astype(int)
    blocks = [np.arange(edges[b], edges[b + 1]) for b in range(nb)]
    rng = np.random.RandomState(0); perms = [rng.permutation(nb) for _ in range(M)]
    zero = torch.zeros(1, x.shape[1])
    seqs = []
    for p in perms:
        inc = np.zeros(nb, bool)
        seqs.append(zero)
        for b in p:
            inc[b] = True
            ids = np.concatenate([blocks[j] for j in np.where(inc)[0]])
            seqs.append(x[ids])
    v = logits_of(model, seqs).reshape(M, nb + 1, -1)
    phi = np.zeros((nb, model.n_out))
    for m, p in enumerate(perms):
        for r, b in enumerate(p):
            phi[b] += v[m, r + 1] - v[m, r]
    phi /= M
    sc = np.zeros((T, model.n_out))
    for b in range(nb):
        sc[blocks[b]] = phi[b] / max(1, len(blocks[b]))
    return sc


# ------------------------------------------------------------------ evaluation of one LSTM
def evaluate_lstm(model, D, init, seed, do_extra):
    rng = np.random.RandomState(seed)
    te = D["te"]; Xte = [torch.from_numpy(x) for x in te["X"]]
    P = logits_of(model, Xte)
    res = {"video": video_metrics(te["Y"], P.squeeze(-1) if model.n_out == 1 else P)}
    pos = [(te["names"].index(n), n, c) for n in D["gt"] for c in D["gt"][n]]
    if SMOKE:
        pos = pos[:12]
    vids = sorted({i for i, _, _ in pos})
    recs = dict(zip(vids, replay_many(model, [Xte[i] for i in vids], device=DEV)))
    fm = np.mean([recs[i]["f"].mean(0).numpy() for i in vids], 0)
    tau = 1 / (1 - np.clip(fm, 1e-6, 1 - 1e-9))
    res["tau"] = {"median": float(np.median(tau)), "p90": float(np.percentile(tau, 90)),
                  "gt50": int((tau > 50).sum()), "all": tau.tolist()}
    errs = []
    methods = ["cam", "mem", "write", "erase", "occlusion", "grad_x_input", "int_grad", "random"]
    if do_extra:
        methods.append("shapley")
    loc = {m: [] for m in methods}
    faith = {m: {"del": [], "keep": []} for m in methods if m != "erase"}
    interv = {k: [] for k in ["topcam", "randk", "gt_write", "nongt_write", "gt_input"]}
    perv, store, keep_curve = [], {}, {d: [] for d in [5, 20, 50, 100, 200]}
    masks, mask_meta = [], []
    occ_cache, shap_cache = {}, {}
    for i, n, c in pos:
        x = Xte[i]; T = len(x); lab = D["gt"][n][c]
        rec = recs[i]; d = decompose(rec, model, c)
        errs.append((d["recon_err"], d["cam_err"]))
        kp = d["keep_mean"].numpy()
        for dd in keep_curve:
            if T - 1 - dd >= 0:
                keep_curve[dd].append(kp[T - 1 - dd])
        full = P[i]
        if i not in occ_cache:
            occ_cache[i] = occlusion(model, x, full)
            if do_extra:
                shap_cache[i] = shapley(model, x, M=2 if SMOKE else 16)
        gxi, ig = grads(model, x, c, 4 if SMOKE else 32)
        sc = {"cam": d["cam"].numpy(), "mem": d["mem"].numpy(), "write": d["write"].numpy(),
              "erase": d["erase"].numpy(), "occlusion": occ_cache[i][:, c], "grad_x_input": gxi,
              "int_grad": ig, "random": rng.rand(T)}
        if do_extra:
            sc["shapley"] = shap_cache[i][:, c]
        pv = {"name": n, "cls": int(c), "logit": float(full[c])} | describe(n, c, D, T)
        if lab.any() and not lab.all():
            for m in methods:
                r = loc_metrics(lab, sc[m], rng); loc[m].append(r)
                pv[f"auc_{m}"] = r["auc"]; pv[f"point_{m}"] = r["point"]
            # faithfulness: delete top-q steps / keep only top-q steps
            seqs, meta = [], []
            for m in faith:
                order = np.argsort(-(sc[m] + 1e-12 * rng.rand(T)))
                for q in QS:
                    k = max(1, int(round(q * T))); top = np.sort(order[:k])
                    rest = np.setdiff1d(np.arange(T), top)
                    seqs += [x[rest] if len(rest) else x[:1] * 0, x[top]]; meta += [(m, q)]
            out = logits_of(model, seqs)[:, c]
            for j, (m, q) in enumerate(meta):
                faith[m]["del"].append(full[c] - out[2 * j]); faith[m]["keep"].append(out[2 * j + 1])
            # interventions on the memory: block writing (i = 0) at chosen steps
            k = max(1, int(round(0.1 * T)))
            top = np.argsort(-sc["cam"])[:k]; rnd = rng.choice(T, k, replace=False)
            gtw = np.where(lab)[0]; non = np.where(~lab)[0]
            nong = rng.choice(non, min(len(gtw), len(non)), replace=False)
            for key, ids in (("topcam", top), ("randk", rnd), ("gt_write", gtw), ("nongt_write", nong)):
                m_ = torch.zeros(T, dtype=torch.float64); m_[ids] = 1
                masks.append((x, m_)); mask_meta.append((key, len(perv), c))
            rest = np.setdiff1d(np.arange(T), gtw)
            v = logits_of(model, [x[rest] if len(rest) else x[:1] * 0])[0, c]
            interv["gt_input"].append(full[c] - v)
        perv.append(pv)
        if seed == 0 and len(store) < 400:
            store[f"{n}|{c}"] = {m: sc[m].astype(np.float32) for m in sc} | {"lab": lab}
    if masks:
        rr = replay_many(model, [m[0] for m in masks], [m[1] for m in masks], keep=False, device=DEV)
        for (key, pi, c), r in zip(mask_meta, rr):
            interv[key].append(perv[pi]["logit"] - float(r["logits"][c]))
    res["decomp_err"] = {"c_T": max(e[0] for e in errs), "cam_sum": max(e[1] for e in errs)}
    res["loc"] = {m: {k: float(np.mean([r[k] for r in loc[m]])) for k in loc[m][0]} for m in loc if loc[m]}
    res["faith"] = {m: {"aopc_del": float(np.mean(v["del"])), "keep": float(np.mean(v["keep"]))}
                    for m, v in faith.items() if v["del"]}
    res["interv"] = {k: float(np.mean(v)) for k, v in interv.items() if v}
    res["keep_curve"] = {str(k): float(np.median(v)) for k, v in keep_curve.items() if v}
    res["n_loc"] = len(loc["cam"])
    if do_extra:   # randomisation sanity check: same init, untrained weights
        torch.manual_seed(1000 + seed)
        mr = Model(model.lstm.input_size, H, 0.0, model.n_out, cell=CELL, bidir=BIDIR).to(DEV)
        init_bias(mr, init, gen=torch.Generator().manual_seed(1000 + seed)); mr.eval()
        rrec = dict(zip(vids, replay_many(mr, [Xte[i] for i in vids], device=DEV)))
        rho, aucr = [], []
        for i, n, c in pos:
            lab = D["gt"][n][c]
            cr = decompose(rrec[i], mr, c)["cam"].numpy()
            ct = decompose(recs[i], model, c)["cam"].numpy()
            rho.append(spearmanr(cr, ct).correlation)
            if lab.any() and not lab.all():
                aucr.append(roc_auc_score(lab, cr + 1e-12 * rng.rand(len(cr))))
        res["random_model"] = {"spearman_cam": float(np.nanmean(rho)), "cam_auc": float(np.mean(aucr))}
    res["per_video"] = perv
    return res, store


def evaluate_pool(fc, kind, D, seed):
    rng = np.random.RandomState(seed)
    te = D["te"]; W = fc.weight.detach().double().cpu().numpy(); b = fc.bias.detach().double().cpu().numpy()
    pool = (lambda x: x.mean(0)) if kind == "mean" else (lambda x: x.max(0))
    P = np.stack([W @ pool(x).astype(np.float64) + b for x in te["X"]])
    res = {"video": video_metrics(te["Y"], P.squeeze(-1) if P.shape[1] == 1 else P)}
    loc, perv = [], []
    for n in D["gt"]:
        i = te["names"].index(n); x = te["X"][i].astype(np.float64)
        for c, lab in D["gt"][n].items():
            if kind == "mean":
                s = x @ W[c]
            else:
                am = x.argmax(0); s = np.zeros(len(x))
                np.add.at(s, am, W[c] * x[am, np.arange(x.shape[1])])
            pv = {"name": n, "cls": int(c), "logit": float(P[i, c])} | describe(n, c, D, len(x))
            if lab.any() and not lab.all():
                r = loc_metrics(lab, s, rng); loc.append(r); pv["auc_contrib"] = r["auc"]
            perv.append(pv)
    res["loc"] = {"contrib": {k: float(np.mean([r[k] for r in loc])) for k in loc[0]}} if loc else {}
    res["per_video"] = perv
    return res


# ------------------------------------------------------------------ main loop
for ds, bb in CFG["runs"]:
    tag0 = f"{ds}_{bb}"
    D = LOADERS[ds](bb)
    if SMOKE:
        for s, m in (("tr", 80), ("te", 60)):
            keep = list(range(min(m, len(D[s]["names"]))))
            if s == "te":
                pos = [k for k, n in enumerate(D["te"]["names"]) if n in D["gt"]][:20]
                keep = sorted(set(keep[:40]) | set(pos))
            D[s] = {"names": [D[s]["names"][k] for k in keep], "X": [D[s]["X"][k] for k in keep],
                    "Y": D[s]["Y"][keep]}
        D["gt"] = {n: D["gt"][n] for n in D["te"]["names"] if n in D["gt"]}
    log(f"{tag0}: train {len(D['tr']['X'])}, test {len(D['te']['X'])}, annotated test {len(D['gt'])}, "
        f"dim {D['tr']['X'][0].shape[1]}, classes {D['n_out']}")
    for kind in ("mean", "max"):
        for seed in CFG["seeds"]:
            f = f"{OUT}/{tag0}_pool{kind}_s{seed}.json"
            if os.path.exists(f):
                continue
            r = evaluate_pool(train_pool(D, kind, seed), kind, D, seed)
            json.dump(r, open(f, "w"))
            log(f"  pool {kind} s{seed}: {r['video']} loc {r['loc'].get('contrib', {}).get('auc', float('nan')):.3f}")
    for init in CFG["inits"]:
        for seed in CFG["seeds"]:
            f = f"{OUT}/{tag0}_{init}_s{seed}.json"
            if os.path.exists(f):
                continue
            model, vl = train_lstm(D, init, seed)
            extra = seed == CFG.get("extra_seed", 0)
            r, store = evaluate_lstm(model, D, init, seed, extra)
            r["val_loss"] = vl
            json.dump(r, open(f, "w"))
            torch.save(model.state_dict(), f"{OUT}/{tag0}_{init}_s{seed}.pt")
            if store:
                np.savez_compressed(f"{OUT}/scores_{tag0}_{init}_s{seed}.npz",
                                    **{f"{k}|{m}": v for k, d in store.items() for m, v in d.items()})
            lc = r["loc"]
            log(f"  {init} s{seed}: {r['video']} tau {r['tau']['median']:.1f} | loc AUC cam {lc['cam']['auc']:.3f} "
                f"mem {lc['mem']['auc']:.3f} write {lc['write']['auc']:.3f} occ {lc['occlusion']['auc']:.3f} "
                f"ig {lc['int_grad']['auc']:.3f} | point cam {lc['cam']['point']:.2f} | "
                f"err {r['decomp_err']['cam_sum']:.1e} | interv {r['interv']}")
    del D
log("all done")
