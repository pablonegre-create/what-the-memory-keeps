"""Aggregate the full experiment (mean ± sd over seeds) and analyse the effect of action
duration, coverage, tail and repetitions on detection and on the temporal explanation.
Writes out/full/report.md and out/full/report.json."""
import json, glob, os, re, collections
import numpy as np
from sklearn.linear_model import LogisticRegression

OUT = json.load(open(os.environ.get("CFG", "cfg.json"))).get("out", "out/full")
files = sorted(glob.glob(f"{OUT}/*_s[0-9].json"))
R = collections.defaultdict(dict)                     # R[(dsbb, model)][seed] = result
for f in files:
    m = re.match(r".*/(\w+?_\w+?)_(\w+)_s(\d)\.json", f)
    R[(m.group(1), m.group(2))][int(m.group(3))] = json.load(open(f))
dsbbs = sorted({k[0] for k in R})
lines, rep = [], {}


def ms(vals, f="{:.3f}"):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "—"
    return (f + "±" + f).format(np.mean(v), np.std(v)) if len(v) > 1 else f.format(v[0])


def P(s=""):
    lines.append(s)


P("# Informe de la experimentación completa\n")
P("Media ± desviación sobre semillas. `const200` = memoria larga (modelo principal); `standard` = "
  "inicialización por defecto (memoria corta); `poolmean`/`poolmax` = clasificador lineal sin memoria.\n")

# ---------------------------------------------------------------- detection
P("## 1. Detección a nivel de vídeo\n")
P("| Dataset_backbone | Modelo | AUC / mAP | AP | Exactitud | τ mediana (pasos) |")
P("|---|---|---|---|---|---|")
for d in dsbbs:
    for mdl in ["const200", "standard", "poolmean", "poolmax"]:
        rs = list(R.get((d, mdl), {}).values())
        if not rs:
            continue
        v = rs[0]["video"]
        if "map" in v:
            P(f"| {d} | {mdl} | {ms([r['video']['map'] for r in rs])} | — | — | "
              f"{ms([r.get('tau', {}).get('median') for r in rs], '{:.1f}')} |")
        else:
            P(f"| {d} | {mdl} | {ms([r['video']['auc'] for r in rs])} | {ms([r['video']['ap'] for r in rs])} | "
              f"{ms([r['video']['acc'] for r in rs])} | {ms([r.get('tau', {}).get('median') for r in rs], '{:.1f}')} |")
P()

# ---------------------------------------------------------------- localisation
METH = ["cam", "mem", "write", "occlusion", "int_grad", "grad_x_input", "shapley", "random"]
NAMES = {"cam": "CAM temporal (opción 3)", "mem": "Memoria final (opción 2)", "write": "Escritura (opción 1)",
         "occlusion": "Oclusión por bloques", "int_grad": "Integrated Gradients",
         "grad_x_input": "Gradiente × entrada", "shapley": "Shapley muestreado (semilla 0)", "random": "Azar"}
P("## 2. Localización frente a la anotación temporal (vídeos positivos con ambas clases)\n")
for d in dsbbs:
    for mdl in ["const200", "standard"]:
        rs = list(R.get((d, mdl), {}).values())
        if not rs:
            continue
        P(f"### {d} — {mdl} (n = {rs[0].get('n_loc')} vídeo-clase)\n")
        P("| Método | AUC | Pointing | P@k | Masa en la acción | % pasos acción marcados (TPR) | % pasos fuera marcados (FPR) | Máx. en último 10 % |")
        P("|---|---|---|---|---|---|---|---|")
        for m in METH:
            vals = [r["loc"][m] for r in rs if m in r["loc"]]
            if not vals:
                continue
            P(f"| {NAMES[m]} | " + " | ".join(ms([v[k] for v in vals]) for k in
                                              ["auc", "point", "pk", "mass", "tpr", "fpr", "last10"]) + " |")
        for pm in ["poolmean", "poolmax"]:
            prs = list(R.get((d, pm), {}).values())
            if prs and "contrib" in prs[0]["loc"]:
                vals = [r["loc"]["contrib"] for r in prs]
                P(f"| (referencia sin memoria) {pm}: contribución exacta | " + " | ".join(
                    ms([v[k] for v in vals]) for k in ["auc", "point", "pk", "mass", "tpr", "fpr", "last10"]) + " |")
        P()

# ---------------------------------------------------------------- faithfulness and interventions
P("## 3. Fidelidad: borrar los pasos más importantes (caída del logit, AOPC) y quedarse solo con ellos\n")
P("| Dataset_backbone | Modelo | " + " | ".join(f"{m} borrar / quedarse" for m in METH if m != "erase") + " |")
P("|---|---|" + "---|" * len(METH))
for d in dsbbs:
    for mdl in ["const200", "standard"]:
        rs = list(R.get((d, mdl), {}).values())
        if not rs:
            continue
        cells = []
        for m in METH:
            v = [r["faith"][m] for r in rs if m in r.get("faith", {})]
            cells.append(f"{ms([x['aopc_del'] for x in v], '{:.2f}')} / {ms([x['keep'] for x in v], '{:.2f}')}" if v else "—")
        P(f"| {d} | {mdl} | " + " | ".join(cells) + " |")
P()
P("## 4. Intervenciones en la memoria (caída del logit al impedir escribir, i = 0)\n")
P("| Dataset_backbone | Modelo | Top-10 % CAM | 10 % al azar | Pasos de la acción | Mismo nº fuera de la acción | Borrar la acción de la entrada |")
P("|---|---|---|---|---|---|---|")
for d in dsbbs:
    for mdl in ["const200", "standard"]:
        rs = list(R.get((d, mdl), {}).values())
        if not rs:
            continue
        P(f"| {d} | {mdl} | " + " | ".join(ms([r["interv"].get(k) for r in rs], "{:.2f}") for k in
                                           ["topcam", "randk", "gt_write", "nongt_write", "gt_input"]) + " |")
P()
P("## 5. Retención y prueba de aleatorización\n")
P("| Dataset_backbone | Modelo | Retención a 5 / 20 / 50 / 100 / 200 pasos | Spearman CAM entrenado vs aleatorio | AUC CAM modelo aleatorio | Error descomposición |")
P("|---|---|---|---|---|---|")
for d in dsbbs:
    for mdl in ["const200", "standard"]:
        rs = R.get((d, mdl), {})
        if not rs:
            continue
        kc = " / ".join(ms([r["keep_curve"].get(k) for r in rs.values()], "{:.2f}") for k in ["5", "20", "50", "100", "200"])
        r0 = rs.get(0, {}).get("random_model", {})
        err = max(r["decomp_err"]["cam_sum"] for r in rs.values())
        P(f"| {d} | {mdl} | {kc} | {r0.get('spearman_cam', float('nan')):.3f} | {r0.get('cam_auc', float('nan')):.3f} | {err:.1e} |")
P()

# ---------------------------------------------------------------- duration / coverage analysis
P("## 6. Duración, cobertura y cola: efecto en la detección y en la explicación\n")
BINS = {"cov": ([0, .1, .25, .5, .9, 1.01], ["<10 %", "10-25 %", "25-50 %", "50-90 %", ">90 %"]),
        "act_s": ([0, 5, 15, 60, 1e9], ["<5 s", "5-15 s", "15-60 s", ">60 s"]),
        "tail_s": ([0, 2, 10, 30, 1e9], ["<2 s", "2-10 s", "10-30 s", ">30 s"]),
        "n_seg": ([1, 2, 4, 1e9], ["1", "2-3", "≥4"])}
for d in dsbbs:
    P(f"### {d}\n")
    for var, (edges, labels) in BINS.items():
        P(f"**Por {var}** — detección (logit > 0) y AUC de localización\n")
        P("| Modelo | " + " | ".join(labels) + " |")
        P("|---|" + "---|" * len(labels))
        for mdl in ["const200", "standard", "poolmean", "poolmax"]:
            rs = R.get((d, mdl), {})
            if not rs:
                continue
            key = "auc_cam" if mdl in ("const200", "standard") else "auc_contrib"
            rows = [pv for r in rs.values() for pv in r["per_video"]]
            cells = []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = [pv for pv in rows if a <= pv[var] < b]
                if not sel:
                    cells.append("—"); continue
                det = np.mean([pv["logit"] > 0 for pv in sel])
                au = [pv[key] for pv in sel if key in pv]
                n = len({(pv["name"], pv["cls"]) for pv in sel})
                cells.append(f"{det:.2f} / {np.mean(au):.2f} (n={n})" if au else f"{det:.2f} / — (n={n})")
            P(f"| {mdl} | " + " | ".join(cells) + " |")
        P()
    # regression with bootstrap over videos (seeds pooled)
    P("**Regresión** (variables estandarizadas; IC 95 % por bootstrap sobre vídeos). "
      "Detección: logística. Localización: lineal sobre el AUC del CAM.\n")
    P("| Modelo | Respuesta | log duración acción | log duración vídeo | log(1+cola) | >1 aparición |")
    P("|---|---|---|---|---|---|")
    for mdl in ["const200", "standard", "poolmean"]:
        rs = R.get((d, mdl), {})
        if not rs:
            continue
        key = "auc_cam" if mdl != "poolmean" else "auc_contrib"
        rows = [pv for r in rs.values() for pv in r["per_video"]]
        vids = sorted({(pv["name"], pv["cls"]) for pv in rows})
        by = collections.defaultdict(list)
        for pv in rows:
            by[(pv["name"], pv["cls"])].append(pv)

        def design(sel):
            Xd = np.array([[np.log(max(p["act_s"], .1)), np.log(p["video_s"]), np.log1p(p["tail_s"]), float(p["n_seg"] > 1)]
                           for p in sel])
            return Xd

        Xall = design(rows); mu, sd = Xall.mean(0), Xall.std(0) + 1e-9
        rng = np.random.RandomState(0)
        for resp in ("det", "loc"):
            coefs = []
            for bsi in range(300):
                pick = [vids[j] for j in rng.randint(0, len(vids), len(vids))] if bsi else vids
                sel = [p for v in pick for p in by[v] if resp == "det" or key in p]
                if len(sel) < 20:
                    continue
                Xb = (design(sel) - mu) / sd
                if resp == "det":
                    yb = np.array([p["logit"] > 0 for p in sel])
                    if yb.all() or not yb.any():
                        continue
                    coefs.append(LogisticRegression(C=1e6, max_iter=2000).fit(Xb, yb).coef_[0])
                else:
                    yb = np.array([p[key] for p in sel])
                    A = np.c_[np.ones(len(Xb)), Xb]
                    coefs.append(np.linalg.lstsq(A, yb, rcond=None)[0][1:])
            if not coefs:
                continue
            c0, cb = coefs[0], np.array(coefs[1:])
            lo, hi = np.percentile(cb, 2.5, 0), np.percentile(cb, 97.5, 0)
            cells = [f"{c0[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]" for j in range(4)]
            P(f"| {mdl} | {'detección' if resp == 'det' else 'AUC CAM'} | " + " | ".join(cells) + " |")
            rep[f"{d}|{mdl}|{resp}"] = {"coef": c0.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    P()

open(f"{OUT}/report.md", "w").write("\n".join(lines))
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
print("\n".join(lines))
