"""Extension of the initialisation ablation to longer memories (XD-Violence I3D-RGB, 3 seeds)."""
import json, time
import numpy as np
import torch
from common import Model, load_xd, xd_annotations
from exp import train_model, evaluate, taus, init_bias, DEV

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
ntr, Xtr, ytr = load_xd("train"); nte, Xte, yte = load_xd("test")
allx = np.concatenate([x[::4] for x in Xtr]); mu, sd = allx.mean(0), allx.std(0) + 1e-6
Xtr = [(x - mu) / sd for x in Xtr]; Xte = [(x - mu) / sd for x in Xte]
ann = xd_annotations()
idx = np.random.RandomState(0).permutation(len(Xtr))
val_i, tr_i = idx[:len(idx) // 10], idx[len(idx) // 10:]

KINDS = ["const500", "const1000"]
SEEDS = [0, 1, 2]
res = {}
for kind in KINDS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        m0 = Model(Xtr[0].shape[1]).to(DEV)
        init_bias(m0, kind, gen=torch.Generator().manual_seed(seed)); m0.eval(); m0.drop.p = 0.0
        tau0 = taus(m0, Xte, 100)
        model, val = train_model(Xtr, ytr, tr_i, val_i, init=kind, seed=seed)
        tau = taus(model, Xte, 100)
        r = evaluate(model, nte, Xte, yte, ann, seed=seed)
        r |= {"val_auc": val, "tau0_med": float(np.median(tau0)), "tau_med": float(np.median(tau)),
              "tau_gt50": int((tau > 50).sum()), "tau0": tau0.tolist(), "tau": tau.tolist()}
        res[f"{kind}|{seed}"] = r
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, f"out/p3_{kind}_{seed}.pt")
        log(f"{kind} seed {seed}: AUC {r['auc']:.3f} acc {r['acc']:.3f} "
            f"tau {r['tau0_med']:.1f}->{r['tau_med']:.1f} "
            f"CAM {r['loc']['cam']['auc']:.3f}/{r['loc']['cam']['point']:.2f}")
        json.dump(res, open("out/p3.json", "w"))

print("\n================ summary (mean +- sd over seeds) ================")
cols = [("vid AUC", lambda r: r["auc"]), ("acc", lambda r: r["acc"]),
        ("tau0", lambda r: r["tau0_med"]), ("tau", lambda r: r["tau_med"]),
        ("tau>50", lambda r: r["tau_gt50"]),
        ("CAM AUC", lambda r: r["loc"]["cam"]["auc"]), ("CAM pt", lambda r: r["loc"]["cam"]["point"]),
        ("CAM end", lambda r: r["loc"]["cam"]["last10"]), ("mem AUC", lambda r: r["loc"]["mem"]["auc"]),
        ("wr AUC", lambda r: r["loc"]["write"]["auc"]), ("keep50", lambda r: r["keep_at"]["50"]),
        ("det>30s", lambda r: r["det_by_tail"][">30s"])]
print("init         " + "".join(f"{c:>15s}" for c, _ in cols))
for kind in KINDS:
    rs = [res[f"{kind}|{s}"] for s in SEEDS]
    row = "".join(f"{np.mean([f(r) for r in rs]):8.3f}+-{np.std([f(r) for r in rs]):<5.3f}" for _, f in cols)
    print(f"{kind:13s}{row}")
log("done")
