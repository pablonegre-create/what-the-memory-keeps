"""Model-agnostic baselines (same definitions as in full.py), importable for timing."""
import numpy as np
import torch

DEV, BLOCK = "cuda", 5


def occlusion(model, x, full, logits_of):
    T = len(x); starts = list(range(0, T, BLOCK))
    seqs = [torch.cat([x[:s], x[s + BLOCK:]]) if T > BLOCK else x for s in starts]
    out = logits_of(model, seqs)
    sc = np.zeros((T, model.n_out))
    for s, v in zip(starts, out):
        sc[s:s + BLOCK] = full - v
    return sc


def grads(model, x, cls, steps=32):
    """steps=1 -> gradient x input; steps>1 -> integrated gradients (zero baseline)."""
    model.train()
    xs = x[None].to(DEV)
    igs = torch.zeros_like(xs)
    for a0 in range(0, steps, 8):
        al = torch.arange(a0 + 1, min(steps, a0 + 8) + 1, device=DEV, dtype=xs.dtype)[:, None, None] / steps
        xb = (al * xs).requires_grad_(True)
        model(xb, torch.tensor([len(x)] * len(al))).reshape(len(al), -1)[:, cls].sum().backward()
        igs += xb.grad.sum(0, keepdim=True)
    model.eval()
    return (igs / steps * xs).sum(-1)[0].detach().cpu().numpy()


def shapley(model, x, logits_of, M=16, nb=40):
    T = len(x); nb = min(nb, T); edges = np.linspace(0, T, nb + 1).round().astype(int)
    blocks = [np.arange(edges[b], edges[b + 1]) for b in range(nb)]
    rng = np.random.RandomState(0); perms = [rng.permutation(nb) for _ in range(M)]
    zero = torch.zeros(1, x.shape[1]); seqs = []
    for p in perms:
        inc = np.zeros(nb, bool); seqs.append(zero)
        for b in p:
            inc[b] = True
            seqs.append(x[np.concatenate([blocks[j] for j in np.where(inc)[0]])])
    v = logits_of(model, seqs).reshape(M, nb + 1, -1)
    phi = np.zeros((nb, model.n_out))
    for m, p in enumerate(perms):
        for r, b in enumerate(p):
            phi[b] += v[m, r + 1] - v[m, r]
    return phi / M
