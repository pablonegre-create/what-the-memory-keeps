"""Aggregate all localisation results (generic + LSTM-specific baselines, LSTM and GRU) and run
paired Wilcoxon signed-rank tests per video (seed-averaged), Holm-corrected within each
dataset x model. Output: out/stats.md, out/stats.json"""
import json, glob, re, collections
import numpy as np
from scipy.stats import wilcoxon

ROWS = collections.defaultdict(lambda: collections.defaultdict(dict))   # (tag, model)[seed] -> {(name,cls): {meth: auc}}
SUM = collections.defaultdict(lambda: collections.defaultdict(list))    # (tag, model)[meth] -> per-seed (auc, point, aopc)
VID = {}
for out, arch in (("out/full", "lstm"), ("out/gru", "gru"), ("out/bilstm", "bilstm")):
    for f in glob.glob(f"{out}/*_s[0-9].json") + glob.glob(f"{out}/*_s[0-9]_x.json"):
        m = re.match(r".*/(\w+?_\w+?)_(\w+?)_s(\d)(_x)?\.json", f)
        tag, mdl, seed = m.group(1), m.group(2), int(m.group(3))
        if mdl.startswith("pool"):
            continue
        key = (tag, f"{arch}-{mdl}")
        r = json.load(open(f))
        for pv in r["per_video"]:
            d = ROWS[key][seed].setdefault((pv["name"], pv["cls"]), {})
            d.update({k[4:]: v for k, v in pv.items() if k.startswith("auc_")})
        for meth, v in r["loc"].items():
            fa = r.get("faith", {}).get(meth, {}).get("aopc_del")
            SUM[key][meth].append((v["auc"], v["point"], fa, seed))
        if "video" in r:
            VID.setdefault(key, []).append(r["video"])

methods = ["cam", "mem", "write", "ms_add", "ms_cell", "lrp", "cd", "occlusion", "int_grad", "grad_x_input", "random"]
names = {"cam": "Temporal CAM (ours)", "mem": "Final memory", "write": "Memory writes", "ms_add": "Murdoch-Szlam additive",
         "ms_cell": "Murdoch-Szlam cell", "lrp": "LRP-LSTM", "cd": "Contextual Decomposition", "occlusion": "Occlusion",
         "int_grad": "Integrated Gradients", "grad_x_input": "Grad x Input", "shapley": "Sampled Shapley (s0)", "random": "Random"}
L, J = ["# Comparación estadística de métodos de explicación\n",
        "AUC de localización por vídeo promediado sobre semillas; test de Wilcoxon pareado (CAM frente a cada método), corrección de Holm dentro de cada dataset × modelo. "
        "Δ = media(CAM − método); 'gana' = % de vídeos en que el CAM tiene mayor AUC. AOPC = fidelidad (caída del logit al borrar los pasos más importantes).\n"], {}
for key in sorted(ROWS):
    tag, model = key
    seeds = ROWS[key]
    vids = sorted(set.intersection(*[set(s) for s in seeds.values()]))
    avg = {m: np.array([np.mean([seeds[s][v][m] for s in seeds if m in seeds[s].get(v, {})]) if any(m in seeds[s].get(v, {}) for s in seeds) else np.nan
                        for v in vids]) for m in methods}
    if 0 in seeds:
        avg["shapley"] = np.array([seeds[0].get(v, {}).get("shapley", np.nan) for v in vids])
        cam0 = np.array([seeds[0].get(v, {}).get("cam", np.nan) for v in vids])
    vd = VID.get(key, [])
    vm = (f"vídeo AUC {np.mean([v['auc'] for v in vd]):.3f}" if vd and "auc" in vd[0] else
          f"vídeo mAP {np.mean([v['map'] for v in vd]):.3f}" if vd else "")
    L.append(f"\n## {tag} — {model}  (n = {len(vids)} vídeo-clase; {vm})\n")
    L.append("| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |")
    L.append("|---|---|---|---|---|---|---|")
    tests = []
    for m in methods + ["shapley"]:
        if m not in avg or np.all(np.isnan(avg[m])):
            continue
        s = SUM[key].get(m, [])
        auc = f"{np.mean([x[0] for x in s]):.3f} ± {np.std([x[0] for x in s]):.3f}" if s else "—"
        pt = f"{np.mean([x[1] for x in s]):.2f}" if s else "—"
        fa = [x[2] for x in s if x[2] is not None]
        aopc = f"{np.mean(fa):.2f}" if fa else "—"
        if m == "cam":
            L.append(f"| **{names[m]}** | {auc} | {pt} | {aopc} | — | — | — |"); continue
        a = (cam0 if m == "shapley" else avg["cam"]); b = avg[m]
        ok = ~np.isnan(a) & ~np.isnan(b)
        d = a[ok] - b[ok]
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        tests.append([m, auc, pt, aopc, d.mean(), (d > 0).mean(), p])
    ps = np.array([t[-1] for t in tests]); order = np.argsort(ps); holm = np.empty_like(ps); run = 0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i])); holm[i] = run
    for t, ph in zip(tests, holm):
        m, auc, pt, aopc, dm, win, p = t
        star = "" if ph >= .05 else " *"
        L.append(f"| {names[m]} | {auc} | {pt} | {aopc} | {dm:+.3f}{star} | {100 * win:.0f} % | {ph:.1e} |")
        J[f"{tag}|{model}|{m}"] = {"delta": dm, "win": win, "p_holm": ph}
open("out/stats.md", "w").write("\n".join(L))
json.dump(J, open("out/stats.json", "w"), indent=1)
print("\n".join(L))
