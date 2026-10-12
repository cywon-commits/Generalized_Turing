"""v2 필수 검증 3: η'=-η/2 에서 p=0 줄무늬의 포화 진폭이 η 와 무관하고 √σ0 에 비례함을 두 D_v 로 확인.
작은 영역 (32², 바깥 파장 4개) 에 한 방향 줄무늬만 심고 포화까지 적분."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.model import Params, peak, solve_kappa, RATIOS
from ring_sim.integrator import Solver


def stripe_amp(Dv, eta, T=40000, N=32, R0=4, A0=0.02, record=False):
    base = Params(Dv=Dv)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    ku, kv = solve_kappa(RATIOS["dodec"], 1, base)
    par = Params(Dv=Dv, eta=eta, etap=-eta / 2, ku=ku, kv=kv)
    L = 2 * np.pi * R0 / k0
    S = Solver(par, N=N, L=L, dt=1.0)
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = A0 * np.cos(2 * np.pi * R0 * X / L)[None] * np.ones((4, 1, 1))
    V = S.state_from_real(u, np.zeros_like(u))
    # v 성분은 정상 관계로
    q = S.k2.ravel()[S.idx]
    V[1, 0] = V[0, 0] / (par.d + par.Dv * q)
    ts, amps = [], []
    for i in range(1, T + 1):
        V = S.step(V)
        if i % 500 == 0:
            F = np.abs(V[0, 0]) / (4 * N * N)
            ts.append(i); amps.append(2 * F.max())
    return s0, amps[-1], ts, amps


def job(a):
    Dv, eta = a
    s0, A, ts, amps = stripe_amp(Dv, eta)
    # 수렴 확인: 마지막 20% 구간의 상대 변화
    k = len(amps) // 5
    drift = abs(amps[-1] - amps[-k]) / amps[-1]
    return dict(Dv=Dv, eta=eta, sigma0=s0, A_sat=A, A_over_sqrt_sigma0=A / np.sqrt(s0), drift_last20pct=drift)


if __name__ == "__main__":
    # σ0 를 대략 4배 차이나게 하는 두 D_v
    Dvs = [2.50309, 2.51236]
    jobs = [(D, e) for D in Dvs for e in (0.05, 0.1, 0.2)]
    with Pool(4) as p:
        res = p.map(job, jobs)
    for r in res:
        print(json.dumps(r))
    json.dump(res, open("results_v2/validation3.json", "w"), indent=1)
