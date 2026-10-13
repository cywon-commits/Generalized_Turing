"""단계 A 교차 검증: 축소 셀(조밀) 고유값 vs 전체 영역 행렬 없는 Arnoldi (선형화 흐름 사상).
사용: python scripts/v3_arnoldi_check.py case eta branch(spiral|standing) [dt] [k]"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import setup, A_SAT, CASES
from ring_sim.bloch import find_states, Cell, eigen_sector, coset_reps
from ring_sim.arnoldi import TangentFlow, embed
from ring_sim.exactlattice import search

case, eta, branch = sys.argv[1], float(sys.argv[2]), sys.argv[3]
dt = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
kev = int(sys.argv[5]) if len(sys.argv) > 5 else 8
par, s0, k0, (c0, ck), lat = setup(case, eta)
seed = {"spiral": "spiral_p1p3", "standing": "sym_standing"}[branch]
fs = [f for f in find_states(c0, A_SAT, seeds=(seed,), nrep=1) if f["info"]["type"] == branch]
X = fs[0]["X"]
# 조밀 (전 잉여류) 고유값 모음
dense = []
for kap in coset_reps(lat["a"], lat["b"]):
    cell = Cell(par, lat["a"], lat["b"], k0 / lat["Cx"], 3 * k0, kappa=kap, P=c0.P)
    cell._k1 = k0 / CASES[case]
    for e in eigen_sector(cell, c0, X, nev=6, core=(), k0=k0):
        dense.append((e["lam"].real / s0, kap))
dense.sort(key=lambda x: -x[0])
print("dense top 14 (λ/σ0, κ):", [(round(l, 4), k) for l, k in dense[:14]], flush=True)
N = LAT_N = int(json.load(open("results_v3/exact_lattices.json"))[case]["N"])
Cx = lat["Cx"]
L = 2 * np.pi * Cx / k0
u, v = embed(c0, X, lat["a"], lat["b"], N)
tau = 1.0 / s0
TF = TangentFlow(par, u, N, L, dt=dt, tau=tau)
t = time.time()
w, V = TF.eigenvalues(k=kev, ncv=3 * kev + 10, tol=1e-5, maxiter=200)
lam = np.log(np.abs(w)) / tau / s0
order = np.argsort(-lam)
print("Arnoldi (N=%d, dt=%g, τ=1/σ0) top λ/σ0:" % (N, dt), [round(float(lam[i]), 4) for i in order], "time %.0fs" % (time.time() - t), flush=True)
json.dump(dict(case=case, eta=eta, branch=branch, dt=dt, dense=[(l, list(k)) for l, k in dense[:14]], arnoldi=[float(lam[i]) for i in order],
               arnoldi_mu=[[float(np.abs(w[i])), float(np.angle(w[i]))] for i in order]), open(f"results_v3/arnoldi_{case}_{eta}_{branch}.json", "w"))
