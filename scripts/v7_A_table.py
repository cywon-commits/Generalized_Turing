"""results_v7/A_scaling.csv + 요약: σ0 스케일링 (η_r 지수, 계수의 σ0 의존, R(σ0) 상대 변화, λ_f/σ0)"""
import csv, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
eta_r = {}
for f in glob.glob("results_v7/A/eta_r_*.json"):
    d = json.load(open(f)); eta_r[(d["case"], d["f"])] = d["eta_r"]
for fn in sorted(glob.glob("results_v7/A/*_f*_k*.json")):
    try: d = json.load(open(fn))
    except Exception: continue
    if not d.get("scan"): continue
    s0 = d["sigma0"]; eta = d["eta"]
    S = np.array([x["s"] for x in d["scan"]]); A = np.array([x["alpha"] for x in d["scan"]]); B = np.array([x["beta"] for x in d["scan"]]); Q = np.array([x["qprod"] for x in d["scan"]])
    a0 = -d["stripe"]["alpha"]; b0 = -d["stripe"]["beta"]
    pa = np.polyfit(S, A, 1); pb = np.polyfit(S, B, 1)
    X, Y = -pa[0], -pb[0]                         # 원 단위: 감쇠율 / s
    Qc = float((Q @ S) / (S @ S))                 # 원 단위: (rate²)/s
    R = Qc / (np.sqrt(a0 * Y) + np.sqrt(b0 * X)) ** 2
    ss = np.linspace(0, 3 * d["s"], 3001)
    lam = lambda s: 0.5 * (-a0 - X * s - b0 - Y * s) + np.sqrt((0.5 * ((-a0 - X * s) - (-b0 - Y * s))) ** 2 + max(Qc, 0) * s)
    Lam = np.array([lam(s) for s in ss])
    rows.append(dict(case=d["case"], f=d["f"], kf=d["kf"], sigma0=s0, eta_r=d["eta_r"], eta=eta, s=d["s"], s_over_sigma0=d["s"] / s0,
                     alpha0_over_s0=a0 / s0, beta0_over_s0=b0 / s0, X=X, Y=Y, X_s_over_s0=X * d["s"] / s0, Y_s_over_s0=Y * d["s"] / s0,
                     Qc=Qc, Qc_over_eta2=Qc / eta ** 2, R=R, s_star_over_s=float(ss[np.argmax(Lam)]) / d["s"], Lam_formula_over_s0=lam(d["s"]) / s0,
                     lam_f_full_over_s0=d["rhombus"]["lam_full"] / s0, lam2_over_s0=d["rhombus"]["Lam2"] / s0, kappa0_stripe_alpha_check=d["stripe"]["lam_full"] / s0))
keys = list(rows[0])
w = csv.DictWriter(open("results_v7/A_scaling.csv", "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(rows)
# 요약
print("η_r(f): ", {k: round(v, 6) for k, v in sorted(eta_r.items())})
for case in ("oct", "ctrlL"):
    fs = sorted(f for (c, f) in eta_r if c == case); er = [eta_r[(case, f)] for f in fs]
    p = np.polyfit(np.log(fs), np.log(er), 1)
    print(case, "η_r ∝ σ0^%.4f  (prefactor η_r(σ0=ref)=%.5f)" % (p[0], np.exp(p[1])))
print("\ncase kf  f   |a0|/s0 |b0|/s0   X      Y     Qc/eta^2   R      lam_f/s0")
for r in sorted(rows, key=lambda r: (r["case"], r["kf"], r["f"])):
    print(r["case"], r["kf"], r["f"], "%.3f %.3f %.2f %.2f %.3f %.4f %.3f" % (r["alpha0_over_s0"], r["beta0_over_s0"], r["X"], r["Y"], r["Qc_over_eta2"], r["R"], r["lam_f_full_over_s0"]))
print("\nR 의 σ0 에 대한 상대 변화 (max-min)/mean, 같은 κ_f:")
for case in ("oct", "ctrlL"):
    for kf in sorted({r["kf"] for r in rows}):
        Rv = [r["R"] for r in rows if r["case"] == case and r["kf"] == kf]
        if len(Rv) == 3: print(case, kf, "R=", np.round(Rv, 4), "rel.var = %.1f%%" % (100 * (max(Rv) - min(Rv)) / np.mean(Rv)))
os.makedirs("results_v7/figures", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for case, c in (("oct", "#1f77b4"), ("ctrlL", "#d62728")):
    for kf, mk in zip((1.05, 1.15, 1.3, 1.5), "osv^"):
        S = sorted([r for r in rows if r["case"] == case and r["kf"] == kf], key=lambda r: r["f"])
        ax[0].plot([r["sigma0"] for r in S], [r["R"] for r in S], mk + "-", color=c, label=f"{case} κ={kf}", ms=4)
        ax[1].plot([r["sigma0"] for r in S], [r["lam_f_full_over_s0"] for r in S], mk + "-", color=c, ms=4)
        ax[2].plot([r["sigma0"] for r in S], [r["alpha0_over_s0"] for r in S], mk + "-", color=c, ms=4)
ax[0].set_ylabel("R"); ax[1].set_ylabel("λ_f / σ0"); ax[2].set_ylabel("|α0| / σ0")
for a in ax: a.set_xscale("log"); a.set_xlabel("σ0")
ax[0].legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig("results_v7/figures/A_R_vs_sigma0.png", dpi=110)

# 16 배 창 (연속 3 개 σ0) 별 R 의 상대 변화 (max-min)/mean
FL = sorted({r["f"] for r in rows})
print("\n16배 창별 R 상대 변화 (max-min)/mean [%]  (창 = 연속한 σ0 세 개: f 가 4배씩)")
win = {}
for case in ("oct", "ctrlL"):
    for kf in sorted({r["kf"] for r in rows}):
        for i in range(len(FL) - 2):
            fs = FL[i:i + 3]
            Rv = [next((r["R"] for r in rows if r["case"] == case and r["kf"] == kf and r["f"] == f_), np.nan) for f_ in fs]
            if np.all(np.isfinite(Rv)) and np.mean(Rv) > 0:
                win[(case, kf, fs[0])] = 100 * (max(Rv) - min(Rv)) / np.mean(Rv)
for case in ("oct", "ctrlL"):
    print(case, "창 시작 f:", [f for f in FL[:3]])
    for kf in sorted({r["kf"] for r in rows}):
        print("  κ_f=%.2f" % kf, ["%.1f%%" % win.get((case, kf, f), float("nan")) for f in FL[:3]])
