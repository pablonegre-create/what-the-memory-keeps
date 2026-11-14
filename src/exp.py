"""Reusable training / evaluation for the memory-explanation experiments."""
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from common import Model, step_labels, replay, decompose

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"


def cap(x, tmax):
    return x if len(x) <= tmax else x[np.linspace(0, len(x) - 1, tmax).round().astype(int)]


def batch(X, ids, tmax=None):
    seqs = [torch.from_numpy(cap(X[i], tmax) if tmax else X[i]) for i in ids]
    return (nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(DEV),
            torch.tensor([len(s) for s in seqs]))


@torch.no_grad()
def predict(model, X, ids, tmax=None, bs=16):
    model.eval(); out = []
    for k in range(0, len(ids), bs):
        xb, lb = batch(X, ids[k:k + bs], tmax)
        out.append(model(xb, lb).cpu())
    return torch.cat(out).numpy()


def init_bias(model, kind, tmax_mem=500, gen=None):
    """Forget/input gate bias initialisation. All non-default kinds follow
    b_f = log(u), b_i = -log(u) with u = tau - 1 (initial time constant tau)."""
    L, H = model.lstm, model.lstm.hidden_size
    if kind == "standard":
        return
    if getattr(model, "cell", "lstm") == "gru":
        # GRU (gates r, z, n): z is the retention gate; tau = 1 + u  ->  b_z = log(u)
        assert kind.startswith("const"), kind
        u = torch.full((H,), float(kind[5:]) - 1)
        with torch.no_grad():
            L.bias_ih_l0.zero_(); L.bias_hh_l0.zero_()
            L.bias_ih_l0[H:2 * H] = torch.log(u)
        return
    if kind == "keras":
        with torch.no_grad():
            L.bias_ih_l0.zero_(); L.bias_hh_l0.zero_(); L.bias_ih_l0[H:2 * H] = 1.0
        return
    if kind.startswith("const"):
        u = torch.full((H,), float(kind[5:]) - 1)
    elif kind == "chrono":
        u = torch.empty(H).uniform_(1, tmax_mem - 1, generator=gen)
    elif kind == "loguniform":
        u = torch.exp(torch.empty(H).uniform_(0, float(np.log(tmax_mem - 1)), generator=gen))
    elif kind == "uniform_gate":  # Gu et al. 2020: sigmoid(b_f) ~ U(1/r, 1-1/r)
        f = torch.empty(H).uniform_(1 / tmax_mem, 1 - 1 / tmax_mem, generator=gen)
        u = f / (1 - f)
    else:
        raise ValueError(kind)
    with torch.no_grad():
        for s in ("", "_reverse") if getattr(L, "bidirectional", False) else ("",):
            bih, bhh = getattr(L, "bias_ih_l0" + s), getattr(L, "bias_hh_l0" + s)
            bih.zero_(); bhh.zero_()
            bih[H:2 * H] = torch.log(u)
            bih[:H] = -torch.log(u)


def train_model(Xtr, ytr, tr_i, val_i, init="chrono", seed=0, hidden=128, epochs=30,
                bs=32, lr=1e-3, tmax=1200, p_drop=0.3):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model = Model(Xtr[0].shape[1], hidden, p_drop).to(DEV)
    init_bias(model, init, gen=torch.Generator().manual_seed(seed))
    bias = [p for n, p in model.named_parameters() if "bias" in n]
    other = [p for n, p in model.named_parameters() if "bias" not in n]
    opt = torch.optim.Adam([{"params": other, "weight_decay": 1e-4},
                            {"params": bias, "weight_decay": 0.0}], lr=lr)
    lossf, yt = nn.BCEWithLogitsLoss(), torch.from_numpy(ytr)
    best, state = -1, None
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr_i)
        for k in range(0, len(perm), bs):
            ids = perm[k:k + bs]; xb, lb = batch(Xtr, ids, tmax)
            loss = lossf(model(xb, lb), yt[ids].to(DEV))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        auc = roc_auc_score(ytr[val_i], predict(model, Xtr, val_i, tmax))
        if auc > best:
            best, state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); model.drop.p = 0.0
    return model, best


def taus(model, X, n=200):
    fs = [replay(model, torch.from_numpy(x), torch.float64, "cpu")["f"].mean(0).numpy() for x in X[:n]]
    return 1 / (1 - np.clip(np.mean(fs, 0), 1e-6, 1 - 1e-9))


def evaluate(model, names, X, y, ann, fps=24, clip=16, seed=0):
    """Video-level metrics, localisation of decomposition scores, retention, detection by tail."""
    rng = np.random.RandomState(seed)
    p = predict(model, X, list(range(len(X))))
    r = {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
         "acc": float(((p > 0) == (y > .5)).mean())}
    loc = {m: {"auc": [], "point": [], "last10": []} for m in ["cam", "mem", "write", "random"]}
    keep_at = {d: [] for d in [5, 20, 50, 100]}
    det = {"<2s": [], "2-30s": [], ">30s": []}
    per_video = []
    for i in np.where(y > .5)[0]:
        n = names[i]
        if n not in ann:
            continue
        x = torch.from_numpy(X[i]); T = len(x)
        lab = step_labels(ann[n], T, clip) > .5
        d = decompose(replay(model, x, torch.float64, "cpu"), model)
        keep = d["keep_mean"].numpy()
        for dd in keep_at:
            if T - 1 - dd >= 0:
                keep_at[dd].append(keep[T - 1 - dd])
        tail = (T * clip - min(max(e for _, e in ann[n]), T * clip)) / fps
        det["<2s" if tail < 2 else "2-30s" if tail < 30 else ">30s"].append(p[i] > 0)
        pv = {"name": n, "T": T, "tail_s": tail, "cov": float(lab.mean()), "logit": float(p[i])}
        if not (lab.all() or not lab.any()):
            sc = {"cam": d["cam"].numpy(), "mem": d["mem"].numpy(), "write": d["write"].numpy(),
                  "random": rng.rand(T)}
            for m, s in sc.items():
                s = s + 1e-12 * rng.rand(T)
                a = roc_auc_score(lab, s)
                loc[m]["auc"].append(a); loc[m]["point"].append(float(lab[np.argmax(s)]))
                loc[m]["last10"].append(float(np.argmax(s) / max(1, T - 1) > .9))
                pv[f"auc_{m}"] = a
        per_video.append(pv)
    r["loc"] = {m: {k: float(np.mean(v)) for k, v in loc[m].items()} for m in loc}
    r["keep_at"] = {str(k): float(np.median(v)) for k, v in keep_at.items()}
    r["det_by_tail"] = {k: float(np.mean(v)) for k, v in det.items()}
    r["per_video"] = per_video
    return r
