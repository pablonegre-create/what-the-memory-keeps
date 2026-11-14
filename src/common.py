"""Shared code: model (LSTM or GRU), step-by-step replay (single on CPU, batched on GPU) and the
exact decomposition of the final memory into per-step contributions. Also legacy XD helpers."""
import os
import numpy as np
import torch
import torch.nn as nn

XD = "data/feat/xd_i3d/i3d-features"
XD_ANN = "data/annot/vadclip_annotations.txt"


# ----------------------------------------------------------------- legacy XD helpers
def xd_names(split):
    sub = "RGB" if split == "train" else "RGBTest"
    return sorted({f.rsplit("__", 1)[0] for f in os.listdir(f"{XD}/{sub}")})


def load_xd(split, stream="RGB", crop=0):
    sub = stream + ("" if split == "train" else "Test")
    names = xd_names(split)
    X = [np.load(f"{XD}/{sub}/{n}__{crop}.npy").astype(np.float32) for n in names]
    y = np.array([0.0 if "label_A" in n else 1.0 for n in names], np.float32)
    return names, X, y


def xd_annotations():
    ann = {}
    for line in open(XD_ANN):
        t = line.split()
        if t:
            v = list(map(int, t[1:]))
            ann[t[0]] = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
    return ann


def step_labels(segs, T, clip=16):
    """Fraction of annotated frames in each step of `clip` frames. segs in frames."""
    fr = np.zeros(T * clip, np.float32)
    for s, e in segs:
        fr[max(0, int(round(s))):min(int(round(e)), T * clip)] = 1.0
    return fr.reshape(T, clip).mean(1)


# ----------------------------------------------------------------- model
class Model(nn.Module):
    """Single-layer unidirectional LSTM or GRU read out from h_T by one linear layer.
    (The recurrent module is stored as `lstm` in both cases for checkpoint compatibility.)"""

    def __init__(self, d_in, hidden=128, p_drop=0.3, n_out=1, cell="lstm", bidir=False):
        super().__init__()
        self.n_out, self.cell, self.bidir = n_out, cell, bidir
        self.drop = nn.Dropout(p_drop)
        self.lstm = (nn.LSTM if cell == "lstm" else nn.GRU)(d_in, hidden, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), n_out)

    def forward(self, x, lengths):
        x = self.drop(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.lstm(packed)
        h = hn[0] if self.cell == "lstm" else hn
        out = self.fc(torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1])
        return out.squeeze(-1) if self.n_out == 1 else out


def _weights(model, device, dtype):
    L = model.lstm
    return (L.weight_ih_l0.detach().to(device, dtype), L.weight_hh_l0.detach().to(device, dtype),
            (L.bias_ih_l0 + L.bias_hh_l0).detach().to(device, dtype),
            model.fc.weight.detach().to(device, dtype), model.fc.bias.detach().to(device, dtype))


@torch.no_grad()
def replay(model, x, dtype=torch.float64, device="cpu", wmask=None):
    """Re-run the LSTM step by step on one sequence and record every gate (LSTM only)."""
    W_ih, W_hh, b, Wfc, bfc = _weights(model, device, dtype)
    H = W_hh.shape[1]
    xp = x.to(device, dtype) @ W_ih.T + b
    h = torch.zeros(H, dtype=dtype, device=device)
    c = torch.zeros(H, dtype=dtype, device=device)
    rec = {k: torch.empty(x.shape[0], H, dtype=dtype, device=device) for k in "ifgoch"}
    for t in range(x.shape[0]):
        z = xp[t] + W_hh @ h
        i = torch.sigmoid(z[:H]); f = torch.sigmoid(z[H:2 * H])
        g = torch.tanh(z[2 * H:3 * H]); o = torch.sigmoid(z[3 * H:])
        if wmask is not None:
            i = i * (1 - wmask[t])
        c = f * c + i * g
        h = o * torch.tanh(c)
        for k, v in zip("ifgoch", (i, f, g, o, c, h)):
            rec[k][t] = v
    rec["logits"] = Wfc @ h + bfc
    rec["logit"] = rec["logits"][0].item()
    return rec


@torch.no_grad()
def replay_batch(model, xs, wmasks=None, dtype=torch.float64, device="cuda", keep=True):
    """Batched exact replay of many sequences (float64). wmasks: optional (T_b,) 0/1 tensors that
    block writing at those steps (LSTM: i = 0; GRU: z = 1). Returns per-sequence dicts with the
    gates (CPU) and logits. For GRU, 'f' holds the retention gate z so retention code is shared."""
    gru = getattr(model, "cell", "lstm") == "gru"
    L = model.lstm
    W_ih = L.weight_ih_l0.detach().to(device, dtype); W_hh = L.weight_hh_l0.detach().to(device, dtype)
    b_ih = L.bias_ih_l0.detach().to(device, dtype); b_hh = L.bias_hh_l0.detach().to(device, dtype)
    Wfc = model.fc.weight.detach().to(device, dtype); bfc = model.fc.bias.detach().to(device, dtype)
    H, B = W_hh.shape[1], len(xs)
    lens = torch.tensor([len(x) for x in xs], device=device)
    Tm = int(lens.max())
    X = torch.zeros(B, Tm, xs[0].shape[1], dtype=dtype, device=device)
    for k, x in enumerate(xs):
        X[k, :len(x)] = x.to(device, dtype)
    XP = X @ W_ih.T + (b_ih if gru else b_ih + b_hh)
    del X
    M = None
    if wmasks is not None:
        M = torch.zeros(B, Tm, dtype=dtype, device=device)
        for k, m in enumerate(wmasks):
            M[k, :len(m)] = m.to(device, dtype)
    h = torch.zeros(B, H, dtype=dtype, device=device)
    c = torch.zeros(B, H, dtype=dtype, device=device)
    keys = "rznh" if gru else "ifgoc"
    if keep:
        S = {k: torch.empty(B, Tm, H, dtype=dtype, device=device) for k in keys}
    for t in range(Tm):
        act = (t < lens).unsqueeze(1)
        if gru:
            hh = h @ W_hh.T + b_hh
            r = torch.sigmoid(XP[:, t, :H] + hh[:, :H])
            z = torch.sigmoid(XP[:, t, H:2 * H] + hh[:, H:2 * H])
            n = torch.tanh(XP[:, t, 2 * H:] + r * hh[:, 2 * H:])
            if M is not None:
                z = z + (1 - z) * M[:, t:t + 1]
            hn = (1 - z) * n + z * h
            h = torch.where(act, hn, h)
            vals = (r, z, n, hn)
        else:
            zz = XP[:, t] + h @ W_hh.T
            i = torch.sigmoid(zz[:, :H]); f = torch.sigmoid(zz[:, H:2 * H])
            g = torch.tanh(zz[:, 2 * H:3 * H]); o = torch.sigmoid(zz[:, 3 * H:])
            if M is not None:
                i = i * (1 - M[:, t:t + 1])
            cn = f * c + i * g
            hn = o * torch.tanh(cn)
            c = torch.where(act, cn, c); h = torch.where(act, hn, h)
            vals = (i, f, g, o, cn)
        if keep:
            for k, v in zip(keys, vals):
                S[k][:, t] = v
    logits = h @ Wfc.T + bfc
    out = []
    for k in range(B):
        r_ = {"logits": logits[k].cpu()}
        if keep:
            T = int(lens[k])
            r_ |= {key: S[key][k, :T].cpu() for key in keys}
            if gru:
                r_["f"] = r_["z"]
        out.append(r_)
    return out


def replay_many(model, xs, wmasks=None, budget=1_500_000, keep=True, device="cuda"):
    """Group sequences by length so that batch*length stays under `budget` (~14 GB of GPU
    memory in float64 with H=128), and replay them on the GPU."""
    order = np.argsort([len(x) for x in xs])
    res = [None] * len(xs)
    k = 0
    while k < len(order):
        j = k
        while j < len(order) and (j - k + 1) * len(xs[order[j]]) <= budget:
            j += 1
        j = max(j, k + 1)
        ids = order[k:j]
        rr = replay_batch(model, [xs[i] for i in ids],
                          None if wmasks is None else [wmasks[i] for i in ids], keep=keep, device=device)
        for i, r in zip(ids, rr):
            res[i] = r
        k = j
    return res


def _retention(f):
    logf = torch.log(f.clamp_min(1e-300))
    return torch.exp(torch.cumsum(logf.flip(0), 0).flip(0) - logf)    # prod_{k>t} f_k


@torch.no_grad()
def decompose(rec, model, cls=0):
    """Exact split of the final memory into per-step contributions (option 2) and their
    projection onto the classifier for class `cls` (option 3, temporal CAM).
    LSTM: c_T = sum_t (prod_{k>t} f_k) i_t g_t, logit = w.(o_T tanh c_T) + b (proportional tanh share).
    GRU:  h_T = sum_t (prod_{k>t} z_k) (1-z_t) n_t,  logit = w.h_T + b (linear, exact)."""
    gru = getattr(model, "cell", "lstm") == "gru"
    w = model.fc.weight[cls].detach().to(rec["f"].device, rec["f"].dtype)
    logits = rec["logits"] if "logits" in rec else torch.tensor([rec["logit"]])
    if gru:
        z, wr = rec["z"], (1 - rec["z"]) * rec["n"]
        keep = _retention(z); R = keep * wr
        hT = rec["h"][-1]
        cam = R @ w
        recon = (R.sum(0) - hT).abs().max().item()
        erase = (1 - z).mean(1)
    else:
        wr = rec["i"] * rec["g"]
        keep = _retention(rec["f"]); R = keep * wr
        cT, oT = rec["c"][-1], rec["o"][-1]
        ratio = torch.where(cT.abs() > 1e-12, torch.tanh(cT) / cT, torch.ones_like(cT))
        cam = R @ (w * oT * ratio)
        recon = (R.sum(0) - cT).abs().max().item()
        erase = (1 - rec["f"]).mean(1)
    return {"cam": cam, "mem": R.norm(dim=1), "write": wr.norm(dim=1), "erase": erase,
            "keep_mean": keep.mean(1), "R": R, "recon_err": recon,
            "cam_err": abs(cam.sum().item() + model.fc.bias[cls].item() - float(logits[cls]))}


# ----------------------------------------------------------------- bidirectional support
class DirView:
    """One direction of a bidirectional model, seen as a unidirectional model whose classifier
    is the matching half of the linear layer (the bias is kept only in the forward view)."""

    def __init__(self, model, reverse):
        import types
        L = model.lstm; H = L.hidden_size; s = "_reverse" if reverse else ""
        self.cell, self.n_out = model.cell, model.n_out
        self.lstm = types.SimpleNamespace(**{k: getattr(L, k + s) for k in
                                            ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")}, hidden_size=H)
        sl = slice(H, 2 * H) if reverse else slice(0, H)
        self.fc = types.SimpleNamespace(weight=model.fc.weight[:, sl],
                                        bias=torch.zeros_like(model.fc.bias) if reverse else model.fc.bias)


def views(model):
    return [DirView(model, False), DirView(model, True)] if getattr(model, "bidir", False) else [model]


def replay_any(model, xs, wmasks=None, keep=True, device="cuda"):
    """replay_many for uni- or bidirectional models. The backward direction is replayed on the
    time-reversed sequence; its records stay in reversed order (decompose_any flips them back)."""
    vs = views(model)
    fw = replay_many(vs[0], xs, wmasks, keep=keep, device=device)
    if len(vs) == 1:
        return fw
    bw = replay_many(vs[1], [x.flip(0) for x in xs], None if wmasks is None else [m.flip(0) for m in wmasks],
                     keep=keep, device=device)
    out = []
    for a, b in zip(fw, bw):
        r = {"fw": a, "bw": b, "logits": a["logits"] + b["logits"]}
        if keep:
            r["f"] = torch.cat([a["f"], b["f"].flip(0)], 1)
        out.append(r)
    return out


@torch.no_grad()
def decompose_any(rec, model, cls=0):
    """decompose() for uni- or bidirectional models: the per-step scores of both directions add up,
    so the temporal CAM still sums exactly to the logit minus the bias."""
    if "fw" not in rec:
        return decompose(rec, model, cls)
    vf, vb = views(model)
    a, b = decompose(rec["fw"], vf, cls), decompose(rec["bw"], vb, cls)
    out = {k: a[k] + b[k].flip(0) for k in ("cam", "mem", "write")}
    out["erase"] = (a["erase"] + b["erase"].flip(0)) / 2
    out["keep_mean"] = a["keep_mean"]            # forward retention towards the end of the video
    out["keep_mean_bw"] = b["keep_mean"].flip(0)  # backward retention towards its start
    out["R"] = None
    out["recon_err"] = max(a["recon_err"], b["recon_err"])
    out["cam_err"] = abs(out["cam"].sum().item() + model.fc.bias[cls].item() - float(rec["logits"][cls]))
    return out
