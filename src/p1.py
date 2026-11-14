"""Pilot: effect of the forget-gate bias initialisation on memory length, detection and
localisation. Variants: standard (b_f=0), keras-like (b_f=1), chrono (Tallec & Ollivier 2018)."""
import json, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from common import Model, load_xd, xd_annotations, step_labels, replay, decompose

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev, TMAX, EPOCHS, BS, LR, H = "cuda", 1200, 30, 32, 1e-3, 128
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

ntr, Xtr, ytr = load_xd("train")
nte, Xte, yte = load_xd("test")
allx = np.concatenate([x[::4] for x in Xtr])
mu, sd = allx.mean(0), allx.std(0) + 1e-6
Xtr = [(x - mu) / sd for x in Xtr]; Xte = [(x - mu) / sd for x in Xte]
ann = xd_annotations()
rng = np.random.RandomState(0)
idx = rng.permutation(len(Xtr)); val_i, tr_i = idx[:len(idx) // 10], idx[len(idx) // 10:]


def cap(x):
    return x if len(x) <= TMAX else x[np.linspace(0, len(x) - 1, TMAX).round().astype(int)]


def batch(X, ids):
    seqs = [torch.from_numpy(cap(X[i])) for i in ids]
    return nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(dev), torch.tensor([len(s) for s in seqs])


@torch.no_grad()
def predict(model, X, ids, bs=16, full=False):
    model.eval(); out = []
    for k in range(0, len(ids), bs):
        if full:
            seqs = [torch.from_numpy(X[i]) for i in ids[k:k + bs]]
            xb = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(dev)
            lb = torch.tensor([len(s) for s in seqs])
        else:
            xb, lb = batch(X, ids[k:k + bs])
        out.append(model(xb, lb).cpu())
    return torch.cat(out).numpy()


def init_bias(model, kind):
    L = model.lstm
    with torch.no_grad():
        if kind == "standard":
            return
        L.bias_ih_l0.zero_(); L.bias_hh_l0.zero_()
        if kind == "keras":
            L.bias_ih_l0[H:2 * H] = 1.0
        elif kind == "chrono":
            u = torch.empty(H).uniform_(1, 500 - 1)          # memory lengths 1..500 steps
            L.bias_ih_l0[H:2 * H] = torch.log(u)              # forget gate
            L.bias_ih_l0[:H] = -torch.log(u)                  # input gate


def tau_of(model, X):
    """Mean forget gate per cell over test steps -> time constant 1/(1-f)."""
    fs = []
    for x in X[:200]:
        rec = replay(model, torch.from_numpy(x), torch.float64, "cpu")
        fs.append(rec["f"].mean(0).numpy())
    f = np.mean(fs, 0)
    return 1 / (1 - np.clip(f, 1e-6, 1 - 1e-9))


results = {}
for kind in ["standard", "keras", "chrono"]:
    torch.manual_seed(0); np.random.seed(0)
    model = Model(Xtr[0].shape[1], H).to(dev)
    init_bias(model, kind)
    tau0 = tau_of(model, Xte)
    bias = [p for n, p in model.named_parameters() if "bias" in n]
    other = [p for n, p in model.named_parameters() if "bias" not in n]
    opt = torch.optim.Adam([{"params": other, "weight_decay": 1e-4},
                            {"params": bias, "weight_decay": 0.0}], lr=LR)
    lossf, yt = nn.BCEWithLogitsLoss(), torch.from_numpy(ytr)
    best, state = -1, None
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(tr_i)
        for k in range(0, len(perm), BS):
            ids = perm[k:k + BS]; xb, lb = batch(Xtr, ids)
            loss = lossf(model(xb, lb), yt[ids].to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        auc = roc_auc_score(ytr[val_i], predict(model, Xtr, val_i))
        if auc > best:
            best, state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); model.drop.p = 0.0
    torch.save({"state": state, "mu": mu, "sd": sd, "kind": kind}, f"out/p1_{kind}.pt")
    p = predict(model, Xte, list(range(len(Xte))), full=True)
    tau = tau_of(model, Xte)
    r = {"val_auc": best, "test_auc": roc_auc_score(yte, p), "test_ap": average_precision_score(yte, p),
         "test_acc": float(((p > 0) == (yte > .5)).mean()),
         "tau_init_median": float(np.median(tau0)), "tau_median": float(np.median(tau)),
         "tau_p90": float(np.percentile(tau, 90)), "cells_tau_gt_50": int((tau > 50).sum())}
    loc = {m: {"auc": [], "point": [], "last10": []} for m in ["cam", "mem", "write"]}
    keep_at = {d: [] for d in [5, 20, 50, 100]}
    det_tail = {"<2s": [], "2-30s": [], ">30s": []}
    for i in np.where(yte > .5)[0]:
        n = nte[i]
        if n not in ann:
            continue
        x = torch.from_numpy(Xte[i]); T = len(x)
        lab = step_labels(ann[n], T) > .5
        d = decompose(replay(model, x, torch.float64, "cpu"), model)
        keep = d["keep_mean"].numpy()
        for dd in keep_at:
            if T - 1 - dd >= 0:
                keep_at[dd].append(keep[T - 1 - dd])
        tail = (T * 16 - min(max(e for _, e in ann[n]), T * 16)) / 24
        det_tail["<2s" if tail < 2 else "2-30s" if tail < 30 else ">30s"].append(p[i] > 0)
        if lab.all() or not lab.any():
            continue
        for m in loc:
            s = d[m].numpy() + 1e-12 * np.random.rand(T)
            loc[m]["auc"].append(roc_auc_score(lab, s)); loc[m]["point"].append(float(lab[np.argmax(s)]))
            loc[m]["last10"].append(float(np.argmax(s) / max(1, T - 1) > .9))
    r["loc"] = {m: {k: float(np.mean(v)) for k, v in loc[m].items()} for m in loc}
    r["keep_at"] = {str(k): float(np.median(v)) for k, v in keep_at.items()}
    r["det_by_tail"] = {k: (float(np.mean(v)), len(v)) for k, v in det_tail.items()}
    results[kind] = r
    log(f"{kind}: {json.dumps(r)}")

print("\n=========== summary ===========")
print(f"{'':10s} {'vidAUC':>7s} {'acc':>6s} {'tau0':>6s} {'tau':>6s} {'tau>50':>6s} "
      f"{'CAM AUC/pt/last10':>20s} {'mem AUC':>8s} {'write AUC':>9s}  keep@5/20/50/100   det tail <2/2-30/>30")
for k, r in results.items():
    c = r["loc"]["cam"]
    print(f"{k:10s} {r['test_auc']:7.3f} {r['test_acc']:6.3f} {r['tau_init_median']:6.1f} {r['tau_median']:6.1f} "
          f"{r['cells_tau_gt_50']:6d} {c['auc']:7.3f}/{c['point']:.2f}/{c['last10']:.2f} {r['loc']['mem']['auc']:8.3f} "
          f"{r['loc']['write']['auc']:9.3f}  " + "/".join(f"{v:.2f}" for v in r["keep_at"].values()) + "   "
          + "/".join(f"{v[0]:.2f}" for v in r["det_by_tail"].values()))
json.dump(results, open("out/p1.json", "w"), indent=1)
log("done")
