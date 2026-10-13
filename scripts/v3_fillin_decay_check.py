"""fill-in 섹터 (κ=b') 의 선도 고유값을 전체 영역 접선 흐름으로 직접 확인: 섭동을 b', c'=a+b' 및 그 켤레에만 심고
시간 t 동안 적분하여 ln||δ|| 의 기울기를 측정 (섹터가 정확히 분리되므로 기울기 → 선도 고유값)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import setup, A_SAT, CASES
from ring_sim.bloch import find_states, Cell, eigen_sector
from ring_sim.arnoldi import TangentFlow, embed

case, eta, branch = sys.argv[1], float(sys.argv[2]), sys.argv[3]
dt = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
par, s0, k0, (c0, ck), lat = setup(case, eta)
seed = {"spiral": "spiral_p1p3", "standing": "sym_standing"}[branch]
X = [f for f in find_states(c0, A_SAT, seeds=(seed,), nrep=1) if f["info"]["type"] == branch][0]["X"]
ev = eigen_sector(ck, c0, X, nev=4, core=((0, 0), (1, 0)), k0=k0)
print("dense κ=b' top λ/σ0:", [round(e["lam"].real / s0, 4) for e in ev], flush=True)
N = int(json.load(open("results_v3/exact_lattices.json"))[case]["N"])
L = 2 * np.pi * lat["Cx"] / k0
u, v = embed(c0, X, lat["a"], lat["b"], N)
TF = TangentFlow(par, u, N, L, dt=dt, tau=1.0 / s0)
S = TF.S
# 섭동: b' 와 c' 모드 (및 켤레), 모든 층/성분에 무작위 진폭
rng = np.random.default_rng(0)
bp = np.array(lat["bprime"]); cp = np.array(lat["a"]) + bp
arr = np.zeros((2, 4, N, N), complex)
for vec in (bp, cp):
    for f in (0, 1):
        for n in range(4):
            c = rng.standard_normal() + 1j * rng.standard_normal()
            arr[f, n, vec[0] % N, vec[1] % N] = 1e-6 * c
            arr[f, n, (-vec[0]) % N, (-vec[1]) % N] = 1e-6 * np.conj(c)
d = (np.fft.ifft2(arr, axes=(2, 3)) * N * N).real
V = S.state_from_real(d[0], d[1])
tau = 1.0 / s0
ts, ns = [], []
nsteps = int(round(1.5 * tau / dt))
for i in range(1, nsteps + 1):
    V = S.step(V, nl=TF._nl)
    if i % 20 == 0:
        ts.append(i * dt); ns.append(np.sqrt((np.abs(V[0]) ** 2).sum()))
ts, ns = np.array(ts), np.array(ns)
m = ts > 0.6 * tau
slope = np.polyfit(ts[m], np.log(ns[m]), 1)[0] / s0
print("tangent-flow decay slope (t in [0.6τ,1.5τ]) λ/σ0 = %.4f" % slope, flush=True)
json.dump(dict(case=case, eta=eta, branch=branch, dense=[e["lam"].real / s0 for e in ev], measured=float(slope)),
          open(f"results_v3/fillin_check_{case}_{eta}_{branch}.json", "w"))
