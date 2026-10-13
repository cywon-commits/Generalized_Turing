"""results_v3/stageA/*.json -> results_v3/stageA.csv, results_v3/figures/lambda_f_eta.png, results_v3/tables.md"""
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO = {"ctrlL": 1.82, "oct": 2 * np.cos(np.radians(22.5)), "ctrlH": 1.87, "dodec": 2 * np.cos(np.radians(15))}
MARG = {"ctrlL": -6.82, "oct": -4.44, "ctrlH": -2.88, "dodec": 0.0}
rows = []
for f in sorted(glob.glob("results_v3/stageA/*.json")):
    d = json.load(open(f))
    for st in d["states"]:
        i = st["info"]
        k0 = st["kappa0"]; kb = st["kappa_bp"]
        nn = [e["lam_over_s0"][0] for e in k0 if abs(e["lam_over_s0"][0]) > 1e-3]
        rows.append(dict(case=d["case"], rho=RHO[d["case"]], margin_1to2=MARG[d["case"]], eta=d["eta"], state=i["type"], seed=st["seed"],
                         Ac=i["Ac"], Aa=i["Aa"], a_p1=i["a_p1"], a_p3=i["a_p3"], b_p1=i["b_p1"], b_p3=i["b_p3"], A2a_p0=i["A2a"], A2b_p0=i["A2b"],
                         A2a_over_Aa2=(i["A2a"] / i["Aa"] ** 2 if i["Aa"] > 1e-6 else float("nan")),
                         kappa0_lam1=k0[0]["lam_over_s0"][0], kappa0_max_nonneutral=st["kappa0_max_nonneutral"],
                         kappa_bp_lam_max=st["lambda_max_bp_over_s0"], lambda_f_core=st["lambda_f_over_s0"],
                         W_core_top=kb[0]["W_core"], coset_max_nonzero_kappa=st.get("coset_max_nonzero_kappa_over_s0"),
                         newton_res=st["newton_res"], Cx=d["lattice"]["Cx"], N_exact=None))
with open("results_v3/stageA.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# 그림
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
col = {"ctrlL": "C0", "oct": "C3", "ctrlH": "C2", "dodec": "k"}
lab = {"ctrlL": "ctrl L (1.82)", "oct": "8-fold (1.848)", "ctrlH": "ctrl H (1.87)", "dodec": "12-fold ref (1.932)"}
for ai, typ in enumerate(("standing", "spiral")):
    for case in ("ctrlL", "oct", "ctrlH", "dodec"):
        pts = sorted((r["eta"], r["kappa_bp_lam_max"]) for r in rows if r["case"] == case and r["state"] == typ)
        if not pts: continue
        e = sorted(set(p[0] for p in pts))
        ax[ai].plot([p[0] for p in pts], [p[1] for p in pts], "o", ms=3, color=col[case], alpha=0.4)
        ax[ai].plot(e, [max(p[1] for p in pts if p[0] == x) for x in e], "-" if case != "dodec" else "s", color=col[case], label=lab[case])
    ax[ai].axhline(0, color="k", lw=0.8); ax[ai].set_xlabel("η"); ax[ai].set_ylabel("λ_f / σ0  (max eigenvalue, fill-in sector κ=b')")
    ax[ai].set_title(f"fill-in eigenvalue of the {typ} rhombus"); ax[ai].legend(fontsize=7)
for case in ("ctrlL", "oct", "ctrlH", "dodec"):
    for typ, ls in (("standing", "-"), ("spiral", "--")):
        pts = sorted((r["eta"], r["kappa0_max_nonneutral"]) for r in rows if r["case"] == case and r["state"] == typ and r["kappa0_max_nonneutral"] is not None)
        e = sorted(set(p[0] for p in pts))
        if e: ax[2].plot(e, [max(p[1] for p in pts if p[0] == x) for x in e], ls, marker="o" if case == "dodec" else None, color=col[case], label=f"{lab[case]} {typ}" if True else None)
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xlabel("η"); ax[2].set_ylabel("max non-neutral λ / σ0 (κ=0 sector)")
ax[2].set_title("stability of the rhombus itself (κ=0): solid standing, dashed spiral"); ax[2].legend(fontsize=5, ncol=2)
fig.tight_layout(); os.makedirs("results_v3/figures", exist_ok=True); fig.savefig("results_v3/figures/lambda_f_eta.png", dpi=95)

# 표
out = ["| η | 사례 | 상태 | A_c | A_a(p1,p3) | A_2a/A_a² | κ=0 최대(비중립) λ/σ0 | κ=b′ 최대 λ_f/σ0 |", "|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["eta"], -r["rho"] if False else r["rho"], r["state"])):
    if r["state"] == "stripe" or r["eta"] not in (0.082, 0.09, 0.10, 0.12, 0.13): continue
    out.append(f"| {r['eta']} | {r['case']} | {r['state']} | {r['Ac']:.2e} | ({r['a_p1']:.1e}, {r['a_p3']:.1e}) | {r['A2a_over_Aa2']:.1f} | {r['kappa0_max_nonneutral']:+.2f} | {r['kappa_bp_lam_max']:+.2f} |")
open("results_v3/tables.md", "w").write("\n".join(out) + "\n")
print(len(rows), "rows")
# 요약: 최대 fill-in 고유값, 1:2 여유 대비 (η=0.10 정상파형/나선형)
for typ in ("standing", "spiral"):
    print(typ, "η=0.10 λ_f(max in κ=b') by ratio:", {c: round(max(r["kappa_bp_lam_max"] for r in rows if r["case"] == c and r["eta"] == 0.10 and r["state"] == typ), 2) for c in ("ctrlL", "oct", "ctrlH", "dodec")})
print("global max λ_f over all states, η:", max((r["kappa_bp_lam_max"], r["case"], r["eta"], r["state"]) for r in rows if r["state"] != "stripe"))
