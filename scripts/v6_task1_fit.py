"""v6 과제 1 후처리: results_v6/task1/*.json -> coeffs.csv (판정식 계수, R, 재현 오차)."""
import json, glob, csv, os
import numpy as np
rows = []
for fn in sorted(glob.glob("results_v6/task1/*.json")):
    d = json.load(open(fn)); s0 = d["sigma0"]; case, eta = d["case"], d["eta"]
    stripe = [s for s in d["states"] if s["type"] == "stripe"][0]["runs"]
    st = [r for r in stripe if not r["incl12"]][0]
    a0, b0 = -st["alpha"], -st["beta"]  # |α0|, |β0| (줄무늬, s=0)
    br = 0
    for ent in d["states"]:
        if ent["type"] == "stripe": continue
        br += 1
        scan = ent.get("scan", [])
        S = np.array([x["s"] for x in scan]); A = np.array([x["alpha"] for x in scan]); B = np.array([x["beta"] for x in scan]); Q = np.array([x["qprod"] for x in scan])
        # 선형 맞춤: α(s) = -|α0| - X s  (절편 자유), β 동일, Q = Qc s (원점 통과)
        pa = np.polyfit(S, A, 1); pb = np.polyfit(S, B, 1)
        Qc = float((Q @ S) / (S @ S))
        pa2 = np.polyfit(S, A, 2); pb2 = np.polyfit(S, B, 2)
        X, Y = -pa[0], -pb[0]
        ra = float(np.abs(np.polyval(pa, S) - A).max()); rb = float(np.abs(np.polyval(pb, S) - B).max())
        R = Qc / (np.sqrt(max(a0 * Y, 0)) + np.sqrt(max(b0 * X, 0))) ** 2 if X > 0 and Y > 0 else float("nan")
        ss = np.linspace(0, 3 * ent["s"], 3001)
        lam = lambda s: 0.5 * (-a0 - X * s - b0 - Y * s) + np.sqrt((0.5 * ((-a0 - X * s) - (-b0 - Y * s))) ** 2 + max(Qc, 0) * s)
        Lam = lam(ss); sstar = float(ss[np.argmax(Lam)])
        for run in ent["runs"]:
            rows.append(dict(case=case, eta=eta, type=ent["type"], branch=br, kind=run["kind"], incl12=run["incl12"], s=ent["s"],
                             alpha0_over_s0=a0 / s0, beta0_over_s0=b0 / s0, X_over_s0=X / s0, Y_over_s0=Y / s0, Qc_over_s0sq=Qc / s0 ** 2,
                             alpha_s=run["alpha"] / s0, beta_s=run["beta"] / s0, qprod_s=run["qprod"] / s0 ** 2, degenerate=run["degenerate"],
                             R=R, s_star=sstar, Lam_formula_at_s=float(lam(ent["s"])) / s0, Lam2_actual=run["Lam2"] / s0,
                             lam_eff_self=run["lam_self"] / s0, lam_full=run["lam_full"] / s0, lam_full_weight=run["lam_full_weight"],
                             fit_resid_alpha=ra / s0, fit_resid_beta=rb / s0, quad_alpha=pa2[0] * ent["s"] ** 2 / s0, quad_beta=pb2[0] * ent["s"] ** 2 / s0))
os.makedirs("results_v6", exist_ok=True)
w = csv.DictWriter(open("results_v6/coeffs.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    if not r["incl12"]:
        print(r["case"], r["eta"], r["type"][:4], r["kind"], "s=%.2e" % r["s"], "a0=%.2f b0=%.2f X=%.0f Y=%.0f R=%.3f" % (r["alpha0_over_s0"], r["beta0_over_s0"], r["X_over_s0"], r["Y_over_s0"], r["R"]),
              "Lam_formula=%.3f Lam2=%.3f eff=%.3f full=%.3f" % (r["Lam_formula_at_s"], r["Lam2_actual"], r["lam_eff_self"], r["lam_full"]))
