"""v7 D2: 혼합 삼파 PDE 잡음 시작 (시드 3, 영역 30 파장, N=256). 두 층 상관·이동, 링별 세기 기록."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from ring_sim.bilayer import *
from v7_D2 import params, RHO
from v7_D_analysis import orient


def run(seed):
    fn = f"results_v7/D/D2_pde_s{seed}.json"
    if os.path.exists(fn): return seed, "cached"
    par = params(); s0, q0 = peak_nl(par, 0); k0 = np.sqrt(q0)
    L = 2 * np.pi * 30 / k0; N = 256
    S = make_solver(par, L, N, dt=2.0)
    V = noise_state(S, seed)
    T = float(os.environ.get("D_T", 6e4)); n = int(T / S.dt)
    for i in range(n): V = S.step(V)
    u, P0, P1 = sector_spectra(S, V)
    o_out = orient(u, k0, L); o_in = orient(u, k0 / RHO, L)
    C = np.fft.ifft2(np.fft.fft2(u[0] - u[0].mean()) * np.conj(np.fft.fft2(u[1] - u[1].mean()))).real
    corr_u = float(np.corrcoef(u[0].ravel(), u[1].ravel())[0, 1])
    res = dict(seed=seed, k0=k0, L=L, N=N, T=T, outer_in_phase_power=o_out["in"]["P"], outer_anti_power=o_out["anti"]["P"], inner_in_phase_power=o_in["in"]["P"], inner_anti_power=o_in["anti"]["P"],
               outer_psi6=o_out["in"]["psi6"], inner_psi6=o_in["anti"]["psi6"], corr_u0_u1=corr_u, umax=float(u.max()), umin=float(u.min()))
    np.save(f"results_v7/D/D2_pde_s{seed}_u.npy", u.astype(np.float32))
    json.dump(res, open(fn, "w"))
    return seed, res


if __name__ == "__main__":
    with Pool(3, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run, (0, 1, 2)): print(r, flush=True)
