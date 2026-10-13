"""R (기준) 마름모: v3 축소 셀 Galerkin 으로 η = 0.14, 0.15, 0.18 의 정상상태/안정성 (정상파형·나선형)"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import setup, A_SAT, CASES
from ring_sim.bloch import find_states, eigen_sector, Cell, coset_reps
out = []
for eta in (0.12, 0.14, 0.15, 0.18):
    par, s0, k0, (c0, ck), lat = setup("oct", eta)
    for f in find_states(c0, A_SAT, T=3e5):
        i = f["info"]
        if i["type"] == "stripe": continue
        e0 = eigen_sector(c0, c0, f["X"], nev=6, core=(), k0=k0)
        nz = [e["lam"].real / s0 for e in e0 if abs(e["lam"].real / s0) > 1e-3]
        best = (-1e9, None)
        for kap in coset_reps(lat["a"], lat["b"]):
            cell = Cell(par, lat["a"], lat["b"], k0 / lat["Cx"], 3 * k0, kappa=kap, P=c0.P); cell._k1 = k0 / CASES["oct"]
            ev = eigen_sector(cell, c0, f["X"], nev=1, core=(), k0=k0)
            lam = ev[0]["lam"].real / s0
            if kap != (0.0, 0.0) and lam > best[0]: best = (lam, kap)
        out.append(dict(eta=eta, type=i["type"], Ac=i["Ac"], Aa=i["Aa"], a_p1=i["a_p1"], a_p3=i["a_p3"], A2a=i["A2a"],
                        kappa0_max=max(nz) if nz else None, coset_max=best[0], coset_arg=best[1]))
        print(out[-1], flush=True)
json.dump(out, open("results_v4/rhombus_ref.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
