"""A0.csv: 마름모 (정상파형·나선형) 안정성. 불안정 판정 = κ=0 비중립 최대 고유값 또는 전 잉여류 최대(κ≠0) 가 > 0.01σ0."""
import json, glob, csv, os
rows = []
for case in ("oct", "ctrlL", "ctrlH"):
    for eta in (0.12, 0.13, 0.15, 0.16, 0.18):
        fn = f"results_v3/stageA/{case}_eta{eta}.json"
        if not os.path.exists(fn): continue
        d = json.load(open(fn))
        for s in d["states"]:
            t = s["info"]["type"]
            if t == "stripe": continue
            k0 = s.get("kappa0_max_nonneutral"); cs = s.get("coset_max_nonzero_kappa_over_s0")
            worst = max([x for x in (k0, cs) if x is not None], default=None)
            rows.append(dict(case=case, eta=eta, type=t, Ac=s["info"]["Ac"], Aa=s["info"]["Aa"], kappa0_max=k0, coset_max=cs,
                             lambda_f=s.get("lambda_f_over_s0"), stable=(worst is not None and worst < 0.01)))
w = csv.DictWriter(open("results_v5/A0.csv", "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
for r in rows: print(r["case"], r["eta"], r["type"], "k0max=%.3f" % r["kappa0_max"] if r["kappa0_max"] is not None else "-", "coset=%.3f" % r["coset_max"] if r["coset_max"] is not None else "-", "STABLE" if r["stable"] else "unstable")
