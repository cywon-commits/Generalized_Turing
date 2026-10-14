import csv, json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("results_v6/figures", exist_ok=True)
R = list(csv.DictReader(open("results_v6/coeffs.csv")))
f = lambda r, k: float(r[k])
COL = {"oct": "#1f77b4", "ctrlL": "#d62728", "ctrlH": "#2ca02c"}
# (i) λ_f(η): 전체 vs 판정식
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for a, case in zip(ax, COL):
    for typ, mk in (("standing", "o"), ("spiral", "s")):
        S = sorted([r for r in R if r["case"] == case and r["type"] == typ and r["incl12"] == "False" and r["kind"] in ("standing", "p1") and r["branch"] in ("1", "2")], key=lambda r: f(r, "eta"))
        S = [r for r in S if r["branch"] == min(x["branch"] for x in S if x["type"] == typ)]
        a.plot([f(r, "eta") for r in S], [f(r, "lam_full") for r in S], mk + "-", color=COL[case], label=f"{typ} full")
        a.plot([f(r, "eta") for r in S], [f(r, "Lam_formula_at_s") for r in S], mk + "--", color="k", mfc="none", label=f"{typ} formula")
    a.set_title(case); a.set_xlabel("η"); a.set_ylabel("λ_f / σ0"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig("results_v6/figures/lambda_f_full_vs_formula.png", dpi=110); plt.close(fig)
# (ii) α(s), β(s), Q(s) 맞춤 (8회 η=0.10 정상파형)
d = json.load(open("results_v6/task1/oct_eta0.1.json")); s0 = d["sigma0"]
e = [x for x in d["states"] if x["type"] == "standing"][0]
S = np.array([x["s"] for x in e["scan"]]); fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a, key, ttl in zip(ax, ("alpha", "beta", "qprod"), ("α(s)/σ0", "β(s)/σ0", "Q_c'Q_b'(s)/σ0²")):
    y = np.array([x[key] for x in e["scan"]]) / (s0 if key != "qprod" else s0 ** 2)
    a.plot(S, y, "o", color="#1f77b4")
    if key != "qprod": a.plot(S, np.polyval(np.polyfit(S, y, 1), S), "k--", label="linear fit")
    else: a.plot(S, (y @ S) / (S @ S) * S, "k--", label="∝ s")
    a.set_title(ttl); a.set_xlabel("s"); a.legend()
fig.tight_layout(); fig.savefig("results_v6/figures/coefficient_fits_oct_eta0.10.png", dpi=110); plt.close(fig)
# (iii) R vs η
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for case in COL:
    for typ, mk in (("standing", "o"), ("spiral", "s")):
        S = sorted([r for r in R if r["case"] == case and r["type"] == typ and r["incl12"] == "False" and r["kind"] in ("standing", "p1") and r["branch"] == "1"], key=lambda r: f(r, "eta"))
        ax.plot([f(r, "eta") for r in S], [f(r, "R") for r in S], mk + "-", color=COL[case], label=f"{case} {typ}")
ax.axhline(1, color="k", lw=.7); ax.set_xlabel("η"); ax.set_ylabel("R"); ax.set_yscale("log"); ax.legend(fontsize=6); ax.set_title("judgement ratio R (fill-in opens iff R > 1)")
fig.tight_layout(); fig.savefig("results_v6/figures/R_vs_eta.png", dpi=110)
# 1:2 포함/제외 비교
rows = []
key = lambda r: (r["case"], r["eta"], r["type"], r["branch"], r["kind"])
base = {key(r): r for r in R if r["incl12"] == "False"}
for r in R:
    if r["incl12"] == "True":
        b = base[key(r)]
        rows.append((r["case"], f(r, "eta"), r["type"], r["kind"], f(r, "alpha_s") - f(b, "alpha_s"), f(r, "beta_s") - f(b, "beta_s"), f(r, "qprod_s") / max(abs(f(b, "qprod_s")), 1e-30) - 1, f(r, "lam_eff_self") - f(b, "lam_eff_self")))
print("1:2 모드 포함 - 제외: Δα, Δβ (σ0), Δq/q, Δλ_eff (σ0)")
import statistics
for case in COL:
    for typ in ("standing", "spiral"):
        S = [x for x in rows if x[0] == case and x[2] == typ]
        if S: print(case, typ, "max|Δα|=%.4f max|Δβ|=%.4f max|Δq/q|=%.4f max|Δλ|=%.1e" % tuple(max(abs(x[i]) for x in S) for i in (4, 5, 6, 7)))
