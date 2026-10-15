"""v7 D1: 역위상 섹터만 불안정 (동위상 -σ0 탈조) 대 동위상 섹터만 불안정 (역위상 -σ0 탈조). 시드 3, 영역 12 파장, 잡음 시작."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from ring_sim.model import Params
from ring_sim.bilayer import *

SIG = 6.58e-3   # σ0 (Dv = 1.02 Dv_c 의 값)
ETA = 0.4


def case_params(case):
    if case.startswith("antiphase_m"):   # 동위상 봉우리 -m σ, 역위상 +σ
        m = float(case[len("antiphase_m"):])
        base = Params(Dv=Dv_for_peak(-m * SIG, lo=1.5, hi=2.9), eta=ETA, etap=0.0)
        ku, kv = solve_kappa_nl(1.0, 1.0 / m, base)
    elif case == "antiphase_only":   # 동위상 봉우리 -σ, 역위상 +σ (같은 k0)
        base = Params(Dv=Dv_for_peak(-SIG), eta=ETA, etap=0.0)
        ku, kv = solve_kappa_nl(1.0, 1.0, base)   # σ1 = -r σ0(base) = +SIG
    else:                          # 동위상 +σ, 역위상 -σ
        base = Params(Dv=Dv_for_peak(+SIG), eta=ETA, etap=0.0)
        ku, kv = solve_kappa_nl(1.0, 1.0, base)   # σ1 = -SIG
    return Params(**{**base.__dict__, "ku": ku, "kv": kv})


def run(job):
    case, seed = job
    fn = f"results_v7/D/D1_{case}_s{seed}.json"
    if os.path.exists(fn): return job, "cached"
    par = case_params(case)
    s_in, q0 = peak_nl(par, 0); s_an, q1 = peak_nl(par, 1); k0 = np.sqrt(q0)
    L = 2 * np.pi * 12 / k0; N = 128
    S = make_solver(par, L, N, dt=2.0)
    V = noise_state(S, seed)
    T = float(os.environ.get("D_T", 6e4)); nsteps = int(T / S.dt)
    hist = []
    for i in range(nsteps):
        V = S.step(V)
        if (i + 1) % (nsteps // 12) == 0:
            u, P0, P1 = sector_spectra(S, V); hist.append(dict(t=(i + 1) * S.dt, P_in=float(P0.sum()), P_anti=float(P1.sum()), umax=float(np.abs(u).max())))
    u, P0, P1 = sector_spectra(S, V)
    dirs_in, tot_in = peak_directions(P0, N, L, k0); dirs_an, tot_an = peak_directions(P1, N, L, k0)
    res = dict(case=case, seed=seed, sigma_in=s_in, sigma_anti=s_an, kappa=[par.ku, par.kv], Dv=par.Dv, k0=k0, L=L, N=N, T=T, hist=hist,
               dirs_in=dirs_in, dirs_anti=dirs_an, P_in=tot_in, P_anti=tot_an)
    os.makedirs("results_v7/D", exist_ok=True)
    np.save(f"results_v7/D/D1_{case}_s{seed}_u.npy", u.astype(np.float32))
    json.dump(res, open(fn, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return job, f"P_in={tot_in:.2e} P_anti={tot_an:.2e} dirs_in={dirs_in} dirs_anti={dirs_an}"


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["antiphase_only", "inphase_only"]
    jobs = [(c, s) for c in cases for s in (0, 1, 2)]
    with Pool(3, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run, jobs): print(r, flush=True)
