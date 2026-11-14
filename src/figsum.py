"""Compact numbers for the quantitative figures (mean and sd over seeds).
Output: out/fig/summary.json"""
import json, glob, re, collections
import numpy as np

OUT, FIG = "out/full", "out/fig"
R = collections.defaultdict(dict)
for f in glob.glob(f"{OUT}/*_s[0-9].json"):
    m = re.match(r".*/(\w+?_\w+?)_(\w+)_s(\d)\.json", f)
    R[(m.group(1), m.group(2))][int(m.group(3))] = json.load(open(f))
TAIL = [(0, 2, "<2 s"), (2, 10, "2-10 s"), (10, 30, "10-30 s"), (30, 1e9, ">30 s")]
ACT = [(0, 5, "<5 s"), (5, 15, "5-15 s"), (15, 60, "15-60 s"), (60, 1e9, ">60 s")]
COV = [(0, .1, "<10 %"), (.1, .25, "10-25 %"), (.25, .5, "25-50 %"), (.5, .9, "50-90 %"), (.9, 1.01, ">90 %")]
S = {"bins": {}, "interv": {}, "loc": {}, "video": {}, "faith": {}}


def msd(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.mean(v)), float(np.std(v))] if v else [None, None]


for (tag, mdl), seeds in R.items():
    key = f"{tag}|{mdl}"
    rs = list(seeds.values())
    S["video"][key] = {k: msd([r["video"].get(k) for r in rs]) for k in rs[0]["video"]}
    S["loc"][key] = {m: {k: msd([r["loc"][m][k] for r in rs if m in r["loc"]]) for k in ("auc", "point", "last10")}
                     for m in rs[0]["loc"]}
    if "interv" in rs[0]:
        S["interv"][key] = {k: msd([r["interv"].get(k) for r in rs]) for k in rs[0]["interv"]}
        S["faith"][key] = {m: msd([r["faith"][m]["aopc_del"] for r in rs if m in r["faith"]]) for m in rs[0]["faith"]}
    aucs_key = "auc_cam" if mdl in ("const200", "standard") else "auc_contrib"
    b = {}
    for var, bins in (("tail_s", TAIL), ("act_s", ACT), ("cov", COV)):
        b[var] = []
        for lo, hi, lab in bins:
            det, loc, n = [], [], 0
            for r in rs:
                sel = [p for p in r["per_video"] if lo <= p[var] < hi]
                n = len(sel)
                if sel:
                    det.append(np.mean([p["logit"] > 0 for p in sel]))
                    a = [p[aucs_key] for p in sel if aucs_key in p]
                    if a:
                        loc.append(np.mean(a))
            b[var].append({"bin": lab, "n": n, "det": msd(det), "loc": msd(loc)})
    S["bins"][key] = b
# dose-response from the initialisation sweeps (p2 + p3, XD-I3D, 3 seeds)
dose = {}
for f in ("out/p2.json", "out/p3.json"):
    for k, r in json.load(open(f)).items():
        kind, seed = k.split("|")
        dose.setdefault(kind, []).append({"tau": r["tau_med"], "auc": r["auc"], "cam_auc": r["loc"]["cam"]["auc"],
                                          "cam_point": r["loc"]["cam"]["point"], "cam_end": r["loc"]["cam"]["last10"],
                                          "det30": r["det_by_tail"][">30s"]})
S["dose"] = {k: {m: msd([d[m] for d in v]) for m in v[0]} for k, v in dose.items()}
for f in glob.glob(f"{OUT}/timing_*.json"):
    S.setdefault("timing", {})[f.split("timing_")[1][:-5]] = json.load(open(f))
json.dump(S, open(f"{FIG}/summary.json", "w"), indent=1)
print("ok", len(R), "model groups;", len(S["dose"]), "inits in dose sweep")
