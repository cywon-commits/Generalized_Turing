"""results_v6/knob.csv 와 figures/knob_lambda_f.png"""
import json, glob, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
for fn in sorted(glob.glob("results_v6/knob/*_t0.?.json")):
    try: d = json.load(open(fn))
    except Exception: continue
    des = d["design"]
    for e in d["etas"]:
        for s in e["states"]:
            base = dict(case=d["case"], target_abs_alpha0=d["target"], gamma=d["gamma"], eta=e["eta"], stripe_alpha0_over_s0=e["stripe"]["alpha"],
                        stripe_beta0_over_s0=e["stripe"]["beta"], stripe_kappa0_max_nonneutral=e["stripe_kappa0_max_nonneutral"],
                        w_peak_over_s0=des.get("peak_over_s0"), w_sector_max=json.dumps({k: round(v, 1) for k, v in des.get("sector_max_over_s0", {}).items()}))
            if s.get("exists"):
                fa = s["fillin"]
                rows.append(dict(base, state=s["type"], exists=True, s=s["s"], lam_f_max_over_s0=fa["lam_full_max"], alpha=fa["alpha"], beta=fa["beta"], qprod=fa["qprod"], Lam2=fa["Lam2"]))
            else:
                rows.append(dict(base, state=s.get("seed"), exists=False, note=s.get("note", "")))
keys = sorted({k for r in rows for k in r})
w = csv.DictWriter(open("results_v6/knob.csv", "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(rows)
os.makedirs("results_v6/figures", exist_ok=True)
COL = {"oct": "#1f77b4", "ctrlL": "#d62728", "ctrlH": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, st in zip(ax, ("standing", "spiral")):
    for case in COL:
        for tgt, ls in ((0.1, "-"), (0.3, "--"), (0.6, ":")):
            S = sorted([r for r in rows if r["case"] == case and r["target_abs_alpha0"] == tgt and r.get("exists") and r["state"] == st], key=lambda r: r["eta"])
            if S: a.plot([r["eta"] for r in S], [r["lam_f_max_over_s0"] for r in S], "o" + ls, color=COL[case], label=f"{case} |α0|={tgt}", ms=3)
    a.axhline(0, color="k", lw=.7); a.set_xlabel("η"); a.set_ylabel("max eigenvalue (fill-in sector) / σ0"); a.set_title(st); a.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig("results_v6/figures/knob_lambda_f.png", dpi=110)
print(len(rows), "rows;", "max lam_f over all =", max(r["lam_f_max_over_s0"] for r in rows if r.get("exists")))
