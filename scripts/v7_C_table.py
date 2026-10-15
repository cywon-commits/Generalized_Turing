import sys, glob, re, os, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ring_sim.specdiag import diagnose
rows = []
B = {(int(r["Cx"]), float(r["eta"])): r for r in csv.DictReader(open("results_v5/B_branches.csv")) if r.get("exists") == "True"}
def add(kind, Cx, N, eta, diag, maxeig, note=""):
    rows.append(dict(kind=kind, Cx=Cx, N=N, eta=eta, outer_W16=diag["outer_W16"], inner_W16=diag["inner_W16"], outer_width_deg=diag["outer_width_deg"],
                     inner_width_deg=diag["inner_width_deg"], outer_diag_over_axis=diag["outer_diag_over_axis"], max_nonneutral_over_s0=maxeig, note=note))
for r in csv.DictReader(open("results_v7/C_baseline_v4.csv")):
    eta = float(r["eta"]); b = B.get((10, eta))
    add("v4 branch (stable for eta>=0.1364)", 10, 96, eta, {k: float(r[k]) for k in ("outer_W16", "inner_W16", "outer_width_deg", "inner_width_deg", "outer_diag_over_axis")},
        float(b["max_nonneutral"]) if b else "")
for f in sorted(glob.glob("runs_v4/n8s_cx24*_eta*.npy")):
    if "relaxed" in f: continue
    eta = float(re.findall(r"eta([0-9.]+)\.npy", f)[0])
    d = diagnose(np.load(f).reshape(2, 4, 240, 240)[0], 24); b = B.get((24, eta))
    add("v5 stable Cx=24 branch (different amplitude)", 24, 240, eta, d, float(b["max_nonneutral"]) if b else "")
for f in sorted(glob.glob("results_v7/C/embed_*.json")):
    d = json.load(open(f))
    if d.get("converged"):
        add("v7 embedded from Cx=10 modes (top %d)" % d["topk"], 24, 240, d["eta"], d["diag"], d["max_nonneutral"], "converged but different/unstable state")
    else:
        rows.append(dict(kind="v7 embedded from Cx=10 modes (top %d)" % d["topk"], Cx=24, N=240, eta=d["eta"], note="Newton-Krylov " + d.get("error", "failed")))
keys = sorted({k for r in rows for k in r}, key=lambda k: ["kind", "Cx", "N", "eta"].index(k) if k in ("kind", "Cx", "N", "eta") else 9)
w = csv.DictWriter(open("results_v7/C_approximants.csv", "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(rows)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for kind, c, mk in (("v4 branch (stable for eta>=0.1364)", "#1f77b4", "o"), ("v5 stable Cx=24 branch (different amplitude)", "#d62728", "s"), ("v7 embedded from Cx=10 modes (top 24)", "#7f7f7f", "^")):
    S = sorted([r for r in rows if r["kind"] == kind and "outer_W16" in r], key=lambda r: r["eta"])
    if not S: continue
    ax[0].plot([r["eta"] for r in S], [r["outer_W16"] for r in S], mk + "-", color=c, label=kind[:34], ms=4)
    ax[1].plot([r["eta"] for r in S], [r["inner_W16"] for r in S], mk + "-", color=c, ms=4)
    ax[2].plot([r["eta"] for r in S], [r["outer_diag_over_axis"] for r in S], mk + "-", color=c, ms=4)
ax[0].set_ylabel("outer W16"); ax[1].set_ylabel("inner W16"); ax[2].set_ylabel("A_diag/A_axis (outer)")
for a in ax: a.set_xlabel("η")
ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig("results_v7/figures/C_W16_vs_eta.png", dpi=110)
