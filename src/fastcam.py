"""Vectorised temporal CAM: one cuDNN forward pass gives every hidden state h_t; the gates are then
recomputed for all steps at once (h_{t-1} is known), so no step-by-step loop is needed.
Checks equality with the float64 replay and times it against the baselines (XD-I3D, 100 videos)."""
import json, time
import numpy as np
import torch
from scipy.stats import spearmanr
from common import Model, replay_batch, decompose
from data import LOADERS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"


@torch.no_grad()
def fast_cam(model, x, cls=0, dtype=torch.float32):
    L = model.lstm; H = L.hidden_size
    xb = x.to(DEV)[None]
    hs, _ = L(xb)                                   # (1,T,H) cuDNN
    hs = hs[0].to(dtype)
    hprev = torch.cat([torch.zeros_like(hs[:1]), hs[:-1]])
    z = xb[0].to(dtype) @ L.weight_ih_l0.T.to(dtype) + hprev @ L.weight_hh_l0.T.to(dtype) \
        + (L.bias_ih_l0 + L.bias_hh_l0).to(dtype)
    i = torch.sigmoid(z[:, :H]); f = torch.sigmoid(z[:, H:2 * H])
    g = torch.tanh(z[:, 2 * H:3 * H]); o = torch.sigmoid(z[:, 3 * H:])
    logf = torch.log(f.clamp_min(1e-30))
    keep = torch.exp(torch.cumsum(logf.flip(0), 0).flip(0) - logf)
    R = keep * i * g
    cT = R.sum(0)
    w = model.fc.weight[cls].to(dtype)
    ratio = torch.where(cT.abs() > 1e-12, torch.tanh(cT) / cT, torch.ones_like(cT))
    return R @ (w * o[-1] * ratio)


D = LOADERS["xd"]("i3d")
model = Model(D["tr"]["X"][0].shape[1], 128, 0.0, 1).to(DEV)
model.load_state_dict(torch.load("out/full/xd_i3d_const200_s0.pt", weights_only=True)); model.eval()
names = [n for n in D["gt"]]
res = {"maxabs": [], "rho": [], "t_fast32": [], "t_fast64": [], "t_replay": []}
for k, n in enumerate(names):
    x = torch.from_numpy(D["te"]["X"][D["te"]["names"].index(n)])
    ref = decompose(replay_batch(model, [x], device=DEV)[0], model, 0)["cam"].numpy()
    f32 = fast_cam(model, x).cpu().numpy()
    res["maxabs"].append(float(np.abs(f32 - ref).max() / (np.abs(ref).max() + 1e-12)))
    res["rho"].append(float(spearmanr(f32, ref).correlation))
    if k < 100:
        for key, fn in (("t_fast32", lambda: fast_cam(model, x)), ("t_fast64", lambda: fast_cam(model, x, dtype=torch.float64)),
                        ("t_replay", lambda: decompose(replay_batch(model, [x], device=DEV)[0], model, 0))):
            fn(); torch.cuda.synchronize(); s = time.perf_counter()
            for _ in range(3):
                fn()
            torch.cuda.synchronize(); res[key].append((time.perf_counter() - s) / 3)
out = {"n": len(names), "rel_maxabs_err_max": max(res["maxabs"]), "rel_maxabs_err_median": float(np.median(res["maxabs"])),
       "spearman_min": min(res["rho"]), "spearman_median": float(np.median(res["rho"])),
       **{k: {"mean_ms": 1000 * float(np.mean(v)), "median_ms": 1000 * float(np.median(v))} for k, v in res.items() if k.startswith("t_")}}
json.dump(out, open("out/full/timing_fastcam_xd_i3d.json", "w"), indent=1)
print(json.dumps(out, indent=1))
