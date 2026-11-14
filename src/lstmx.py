"""LSTM-specific explanation baselines on the already trained LSTMs (out/full):
  ms_cell : Murdoch & Szlam 2017, cell decomposition   w.(o_T (tanh c_t - tanh c_{t-1}))
  ms_add  : Murdoch & Szlam 2017, additive decomposition (telescoping over retained contributions)
  lrp     : LRP for LSTM (Arras et al. 2017), epsilon rule, gates as constants (signal-take-all)
  cd      : Contextual Decomposition (Murdoch, Liu & Yu 2018), per block of 5 steps
Same localisation and faithfulness metrics as full.py, plus a per-video timing benchmark."""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from common import Model, replay_many, replay_batch, decompose
from data import LOADERS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV, F64, H, BLOCK, QS = "cuda", torch.float64, 128, 5, [0.1, 0.2, 0.3, 0.5]
CFG = json.load(open(os.environ.get("CFG", "cfg.json")))
OUT = CFG.get("out", "out/full")
t0 = time.time()
log = lambda m: print(f"[{(time.time() - t0) / 60:7.1f} min] {m}", flush=True)


# ------------------------------------------------------------------ helpers (as in full.py)
def to_batch(seqs):
    seqs = [s if torch.is_tensor(s) else torch.from_numpy(s) for s in seqs]
    return (nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(DEV), torch.tensor([len(s) for s in seqs]))


@torch.no_grad()
def logits_of(model, seqs, budget=300_000):
    model.eval()
    order = np.argsort([len(s) for s in seqs]); out = np.zeros((len(seqs), model.n_out)); k = 0
    while k < len(order):
        j = k
        while j < len(order) and (j - k + 1) * len(seqs[order[j]]) <= budget:
            j += 1
        j = max(j, k + 1)
        xb, lb = to_batch([seqs[i] for i in order[k:j]])
        out[order[k:j]] = model(xb, lb).reshape(j - k, -1).double().cpu().numpy(); k = j
    return out


def loc_metrics(lab, s, rng):
    s = np.asarray(s, np.float64) + 1e-12 * rng.rand(len(s)); k = int(lab.sum())
    pos = np.maximum(s, 0); nrm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return {"auc": roc_auc_score(lab, s), "point": float(lab[np.argmax(s)]),
            "pk": float(lab[np.argsort(-s)[:k]].mean()), "mass": float(pos[lab].sum() / (pos.sum() + 1e-12)),
            "tpr": float((nrm[lab] > .5).mean()), "fpr": float((nrm[~lab] > .5).mean()),
            "last10": float(np.argmax(s) / max(1, len(s) - 1) > .9)}


def W(model):
    L = model.lstm
    return (L.weight_ih_l0.detach().to(DEV, F64), L.weight_hh_l0.detach().to(DEV, F64),
            (L.bias_ih_l0 + L.bias_hh_l0).detach().to(DEV, F64), model.fc.weight.detach().to(DEV, F64),
            model.fc.bias.detach().to(DEV, F64))


# ------------------------------------------------------------------ LSTM-specific methods
def ms_cell(rec, model, cls):
    w = model.fc.weight[cls].detach().double().cpu(); c = rec["c"]; oT = rec["o"][-1]
    tc = torch.tanh(c); prev = torch.cat([torch.zeros_like(tc[:1]), tc[:-1]])
    return ((tc - prev) * oT) @ w


def ms_add(rec, model, cls):
    w = model.fc.weight[cls].detach().double().cpu(); R = decompose(rec, model, cls)["R"]
    S = torch.cumsum(R, 0); tS = torch.tanh(S)
    prev = torch.cat([torch.zeros_like(tS[:1]), tS[:-1]])
    return ((tS - prev) * rec["o"][-1]) @ w


@torch.no_grad()
def lrp(rec, x, model, cls, eps=1e-3):
    W_ih, W_hh, b, Wfc, bfc = W(model)
    i, f, g, o, c = (rec[k].to(DEV) for k in "ifgoc")
    hs = o * torch.tanh(c); T = len(c); x = x.to(DEV, F64)
    stab = lambda z: z + eps * torch.where(z >= 0, 1.0, -1.0)
    w = Wfc[cls]; zout = w @ hs[-1] + bfc[cls]
    Rh = (w * hs[-1]) / stab(zout) * zout
    Wg, Ug, bg = W_ih[2 * H:3 * H], W_hh[2 * H:3 * H], b[2 * H:3 * H]
    Rc_next = torch.zeros(H, dtype=F64, device=DEV); Rx = torch.zeros(T, dtype=F64, device=DEV)
    for t in range(T - 1, -1, -1):
        Rc = Rc_next + Rh                                       # h_t -> c_t (signal-take-all)
        cprev = c[t - 1] if t > 0 else torch.zeros(H, dtype=F64, device=DEV)
        hprev = hs[t - 1] if t > 0 else torch.zeros(H, dtype=F64, device=DEV)
        den = stab(c[t])
        Rc_next = f[t] * cprev / den * Rc
        Rg = i[t] * g[t] / den * Rc
        gx, gh = Wg @ x[t], Ug @ hprev
        rho = Rg / stab(gx + gh + bg)
        Rx[t] = (gx * rho).sum()
        Rh = hprev * (Ug.T @ rho)
    return Rx.cpu()


def _three(a, b_, c, act):
    abc, ac, bc, cc = act(a + b_ + c), act(a + c), act(b_ + c), act(c)
    return 0.5 * (ac - cc + abc - bc), 0.5 * (bc - cc + abc - ac), cc


def _tanh_two(a, b_):
    tab = torch.tanh(a + b_)
    return 0.5 * (torch.tanh(a) + tab - torch.tanh(b_)), 0.5 * (torch.tanh(b_) + tab - torch.tanh(a))


@torch.no_grad()
def cd(x, model, block=BLOCK):
    """Contextual Decomposition, one relevant block of `block` steps per batch row.
    Returns (T, n_out) per-step scores (block score / block length)."""
    W_ih, W_hh, b, Wfc, bfc = W(model)
    T = len(x); nb = (T + block - 1) // block
    xp = x.to(DEV, F64) @ W_ih.T
    rel_h = torch.zeros(nb, H, dtype=F64, device=DEV); irr_h = torch.zeros_like(rel_h)
    rel_c = torch.zeros_like(rel_h); irr_c = torch.zeros_like(rel_h)
    rows = torch.arange(nb, device=DEV)
    bi, bf, bg_, bo = b[:H], b[H:2 * H], b[2 * H:3 * H], b[3 * H:]
    for t in range(T):
        inS = (rows == t // block).unsqueeze(1)
        zx = xp[t].unsqueeze(0)
        rel = rel_h @ W_hh.T + torch.where(inS, zx, torch.zeros_like(zx))
        irr = irr_h @ W_hh.T + torch.where(inS, torch.zeros_like(zx), zx)
        ri, ii, bci = _three(rel[:, :H], irr[:, :H], bi, torch.sigmoid)
        rg, ig, bcg = _three(rel[:, 2 * H:3 * H], irr[:, 2 * H:3 * H], bg_, torch.tanh)
        nrel = ri * (rg + bcg) + bci * rg
        nirr = ii * (rg + ig + bcg) + (ri + bci) * ig
        bb = bci * bcg
        nrel = nrel + torch.where(inS, bb, torch.zeros_like(bb))
        nirr = nirr + torch.where(inS, torch.zeros_like(bb), bb)
        if t > 0:
            rf, iff, bcf = _three(rel[:, H:2 * H], irr[:, H:2 * H], bf, torch.sigmoid)
            nrel = nrel + (rf + bcf) * rel_c
            nirr = nirr + (rf + iff + bcf) * irr_c + iff * rel_c
        o = torch.sigmoid(rel[:, 3 * H:] + irr[:, 3 * H:] + bo)
        rel_c, irr_c = nrel, nirr
        tr, ti = _tanh_two(rel_c, irr_c)
        rel_h, irr_h = o * tr, o * ti
    sb = rel_h @ Wfc.T                                         # (nb, n_out)
    sc = sb[torch.arange(T, device=DEV) // block] / block
    return sc.cpu().numpy()


# ------------------------------------------------------------------ evaluation
SMOKE = CFG.get("smoke_x", False)


def run(D, tag0, init, seed, rng):
    f = f"{OUT}/{tag0}_{init}_s{seed}_x.json" if not SMOKE else f"/tmp/{tag0}_x_smoke.json"
    if os.path.exists(f) and not SMOKE:
        return
    model = Model(D["tr"]["X"][0].shape[1], H, 0.0, D["n_out"]).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/{tag0}_{init}_s{seed}.pt", weights_only=True)); model.eval()
    te = D["te"]; Xte = [torch.from_numpy(x) for x in te["X"]]
    pos = [(te["names"].index(n), n, c) for n in D["gt"] for c in D["gt"][n]]
    if SMOKE:
        pos = pos[:6]
    vids = sorted({i for i, _, _ in pos})
    recs = dict(zip(vids, replay_many(model, [Xte[i] for i in vids], device=DEV)))
    P = logits_of(model, Xte)
    if SMOKE:   # sanity: the additive decompositions sum to the logit
        i, n, c = pos[0]
        rec = recs[i]; bias = model.fc.bias[c].item()
        log(f"  check {tag0}: ms_add sum+b-logit {float(ms_add(rec, model, c).sum()) + bias - float(rec['logits'][c]):.1e}, "
            f"ms_cell sum+b-logit {float(ms_cell(rec, model, c).sum()) + bias - float(rec['logits'][c]):.1e}, "
            f"lrp sum {float(lrp(rec, Xte[i], model, c).sum()):.3f} vs logit {float(rec['logits'][c]):.3f}")
    meth = ["ms_cell", "ms_add", "lrp", "cd"]
    loc = {m: [] for m in meth}; faith = {m: {"del": [], "keep": []} for m in meth}; perv = []
    cd_cache = {}
    for i, n, c in pos:
        x = Xte[i]; T = len(x); lab = D["gt"][n][c]; rec = recs[i]
        if i not in cd_cache:
            cd_cache[i] = cd(x, model)
        sc = {"ms_cell": ms_cell(rec, model, c).numpy(), "ms_add": ms_add(rec, model, c).numpy(),
              "lrp": lrp(rec, x, model, c).numpy(), "cd": cd_cache[i][:, c]}
        pv = {"name": n, "cls": int(c)}
        if lab.any() and not lab.all():
            seqs, meta = [], []
            for m in meth:
                r = loc_metrics(lab, sc[m], rng); loc[m].append(r)
                pv[f"auc_{m}"] = r["auc"]; pv[f"point_{m}"] = r["point"]
                order = np.argsort(-(sc[m] + 1e-12 * rng.rand(T)))
                for q in QS:
                    k = max(1, int(round(q * T))); top = np.sort(order[:k]); rest = np.setdiff1d(np.arange(T), top)
                    seqs += [x[rest] if len(rest) else x[:1] * 0, x[top]]; meta.append(m)
            out = logits_of(model, seqs)[:, c]
            for j, m in enumerate(meta):
                faith[m]["del"].append(P[i, c] - out[2 * j]); faith[m]["keep"].append(out[2 * j + 1])
        perv.append(pv)
    res = {"loc": {m: {k: float(np.mean([r[k] for r in loc[m]])) for k in loc[m][0]} for m in meth if loc[m]},
           "faith": {m: {"aopc_del": float(np.mean(v["del"])), "keep": float(np.mean(v["keep"]))} for m, v in faith.items()},
           "per_video": perv}
    if not SMOKE:
        json.dump(res, open(f, "w"))
    log(f"  {tag0} {init} s{seed}: " + " ".join(f"{m} {res['loc'][m]['auc']:.3f}/{res['faith'][m]['aopc_del']:.2f}" for m in meth))


def timing(D, tag0):
    """Seconds per video (batch of one) for every explanation method, on up to 100 positives."""
    from full_methods import occlusion, grads, shapley        # copies of the full.py baselines
    model = Model(D["tr"]["X"][0].shape[1], H, 0.0, D["n_out"]).to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/{tag0}_const200_s0.pt", weights_only=True)); model.eval()
    te = D["te"]; names = [n for n in D["gt"]][:(5 if SMOKE else 100)]
    tm = {m: [] for m in ["cam", "ms_add", "lrp", "cd", "occlusion", "grad_x_input", "int_grad", "shapley"]}
    lens = []
    for n in names:
        i = te["names"].index(n); x = torch.from_numpy(te["X"][i]); lens.append(len(x))
        full = logits_of(model, [x])[0]

        def tick(m, fn):
            torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize()
            tm[m].append(time.perf_counter() - s)
        tick("cam", lambda: decompose(replay_batch(model, [x], device=DEV)[0], model, 0))
        rec = replay_batch(model, [x], device=DEV)[0]
        tick("ms_add", lambda: ms_add(rec, model, 0))
        tick("lrp", lambda: lrp(rec, x, model, 0))
        tick("cd", lambda: cd(x, model))
        tick("occlusion", lambda: occlusion(model, x, full, logits_of))
        tick("grad_x_input", lambda: grads(model, x, 0, steps=1))
        tick("int_grad", lambda: grads(model, x, 0, steps=32))
        tick("shapley", lambda: shapley(model, x, logits_of))
    r = {m: {"mean_s": float(np.mean(v)), "median_s": float(np.median(v))} for m, v in tm.items()}
    r["n"] = len(names); r["median_T"] = float(np.median(lens))
    if not SMOKE:
        json.dump(r, open(f"{OUT}/timing_{tag0}.json", "w"), indent=1)
    log(f"  timing {tag0}: " + " ".join(f"{m} {v['mean_s'] * 1000:.0f} ms" for m, v in r.items() if isinstance(v, dict)))


for ds, bb in CFG["runs"]:
    tag0 = f"{ds}_{bb}"
    D = LOADERS[ds](bb)
    log(f"{tag0}: {len(D['gt'])} annotated test videos")
    if tag0 == "xd_i3d" and (SMOKE or not os.path.exists(f"{OUT}/timing_{tag0}.json")):
        timing(D, tag0)
    for init in CFG["inits"]:
        for seed in (CFG["seeds"][:1] if SMOKE else CFG["seeds"]):
            run(D, tag0, init, seed, np.random.RandomState(seed))
    del D
log("all done")
