import csv, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TOL = 0.005  # σ0 단위: 최대 고유값 < TOL 이면 안정
def load(pat): 
    out = []
    for f in sorted(glob.glob(f"results_v7/B/{pat}.json")):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out
def best(states, t):
    S = [s for s in states if s["type"] == t]
    return min(S, key=lambda s: s["maxeig_over_s0"]) if S else None
rows = []
for d in load("exp*"):
    st = d.get("states", [])
    for t in ("standing", "spiral"):
        b = best(st, t)
        rows.append(dict(exp=d["kind"], name=d["name"], rho=d.get("rho", "sqrt3"), sigma1_over_sigma0=d.get("sigma1_over_sigma0", ""), etap_ratio=d.get("etap_ratio", ""),
                         kf=d.get("kf", ""), eta_r=d.get("eta_r", ""), eta=d.get("eta", 0.4 if d["kind"] == "exp1" else ""), margin_1to2=d.get("margin_1to2", ""),
                         stacking=t, n_states=len([s for s in st if s["type"] == t]), exists=b is not None,
                         maxeig_over_s0=b["maxeig_over_s0"] if b else "", stable=(b["maxeig_over_s0"] < TOL) if b else False, s=b["s"] if b else ""))
w = csv.DictWriter(open("results_v7/B_stacking.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("== exp1 (fig 3 조건) ==")
for r in rows:
    if r["exp"] == "exp1": print(r["stacking"], "n=", r["n_states"], "maxeig/σ0 = %s" % (round(r["maxeig_over_s0"], 4) if r["exists"] else None), "stable" if r["stable"] else "")
for d in load("exp1*"):
    for s in d["states"]: print("   ", s["type"], "Ac=%.4f Aa=%.4f maxeig=%.4f" % (s["Ac"], s["Aa"], s["maxeig_over_s0"]))
print("\n== exp2 (σ1/σ0 × η') ==  sig1/s0 etap kf eta_r | standing(maxeig) spiral(maxeig)")
E2 = [r for r in rows if r["exp"] == "exp2"]
for key in sorted({(r["sigma1_over_sigma0"], r["etap_ratio"], r["kf"]) for r in E2}):
    sub = {r["stacking"]: r for r in E2 if (r["sigma1_over_sigma0"], r["etap_ratio"], r["kf"]) == key}
    fm = lambda r: ("%.3f%s" % (r["maxeig_over_s0"], "*" if r["stable"] else "")) if r["exists"] else "none"
    er = sub["standing"]["eta_r"]
    print("%5.1f %5.1f %4.1f  eta_r=%s | S: %s  P: %s" % (key[0], key[1], key[2], ("%.4f" % er) if er != "" and er == er else "nan", fm(sub["standing"]), fm(sub["spiral"])))
print("\n== exp3 (비율) == rho eta | standing, spiral (maxeig/σ0, * = 안정)")
E3 = [r for r in rows if r["exp"] == "exp3"]
cross = {}
for rho in ("sqrt3", "L1.82", "oct", "H1.87", "dodec"):
    line = []
    cr = None
    for eta in sorted({r["eta"] for r in E3}):
        sub = {r["stacking"]: r for r in E3 if r["rho"] == rho and r["eta"] == eta}
        if not sub: continue
        fm = lambda r: ("%.2f%s" % (r["maxeig_over_s0"], "*" if r["stable"] else "")) if r["exists"] else "-"
        line.append("%.3f:S%s/P%s" % (eta, fm(sub["standing"]), fm(sub["spiral"])))
        if cr is None and sub["standing"]["stable"] and sub["spiral"]["exists"] and not sub["spiral"]["stable"]: cr = eta
    cross[rho] = cr
    m = [r["margin_1to2"] for r in E3 if r["rho"] == rho][0]
    print(rho, "margin_1:2 = %.2f" % m, "first eta with (standing stable & spiral unstable) =", cr); print("   ", "  ".join(line))
os.makedirs("results_v7/figures", exist_ok=True)
col = lambda sS, sP: "#2ca02c" if (sS and sP) else "#1f77b4" if sS else "#d62728" if sP else "#aaaaaa"
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for r in [x for x in E2]:
    pass
pts = {}
for r in E2:
    k = (r["sigma1_over_sigma0"], r["etap_ratio"], r["kf"], r["eta"])
    pts.setdefault(k, {})[r["stacking"]] = r["stable"]
for (s1, er, kf, eta), v in pts.items():
    if eta == "": continue
    ax[0].scatter(s1 + (0.04 if er == 0 else -0.04), eta, c=col(v.get("standing"), v.get("spiral")), marker="s" if er == 0 else "o", s=60, edgecolors="k")
ax[0].set_xlabel("σ1/σ0 (circle η'=-η/2, square η'=0)"); ax[0].set_ylabel("η"); ax[0].set_title("stable stacking: blue=standing only, red=spiral only, green=both, grey=none")
pts3 = {}
for r in E3: pts3.setdefault((r["margin_1to2"], r["eta"], r["rho"]), {})[r["stacking"]] = r["stable"]
for (m, eta, rho), v in pts3.items():
    ax[1].scatter(m, eta, c=col(v.get("standing"), v.get("spiral")), s=40, edgecolors="k")
ax[1].set_xlabel("1:2 margin σ0(K*)/σ0"); ax[1].set_ylabel("η"); ax[1].set_title("ratio scan (σ1=-σ0, η'=-η/2)")
fig.tight_layout(); fig.savefig("results_v7/figures/B_stacking_maps.png", dpi=110)
