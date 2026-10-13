"""v3 단계 A: 다섯 파수 c(0°,k0), a(θ*,k1), b(-θ*,k1), c'(2θ*,k0), b'(3θ*,k1) 가 모두 정수 격자 벡터가 되는
가장 작은 주기 영역. 조건: 각도 오차 < tol_deg, 성장률 오차 |σ(k_격자) - σ(k_이상)| < tol_sigma * σ0.
c = (Cx, 0), a = (Cx/2, ay), b = (Cx/2, -ay) (a+b=c 가 정확히 닫힘), c' = a + b' (정확히 닫힘)."""
import numpy as np
from scipy.fft import next_fast_len
from .model import Params, peak, growth_rate, solve_kappa


def _ang(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def search(rho, Dv=2.50309, r=1.0, tol_deg=1.0, tol_sigma=0.1, cx_max=96, ppw=9.14, n_keep=5):
    base = Params(Dv=Dv)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    ku, kv = solve_kappa(rho, r, base)
    par = Params(Dv=Dv, ku=ku, kv=kv)
    s1pk, q1 = peak(par, 1)
    th = np.degrees(np.arccos(rho / 2))
    sols = []
    for Cx in range(4, cx_max + 1, 2):
        dk = k0 / Cx
        sg0 = lambda L: abs(float(growth_rate(par, 0, (L * dk) ** 2)) - s0) < tol_sigma * s0
        sg1 = lambda L: abs(float(growth_rate(par, 1, (L * dk) ** 2)) - s1pk) < tol_sigma * abs(s0)
        ax = Cx // 2
        for ay in range(1, Cx):
            a = np.array([ax, ay])
            if abs(_ang(a) - th) > tol_deg or not sg1(np.hypot(*a)):
                continue
            m = Cx + 2
            for bx in range(-m, m + 1):
                for by in range(0, m + 1):
                    bp = np.array([bx, by])
                    if abs(_ang(bp) - 3 * th) > tol_deg or not sg1(np.hypot(*bp)):
                        continue
                    cp = a + bp
                    if abs(_ang(cp) - 2 * th) > tol_deg or not sg0(np.hypot(*cp)):
                        continue
                    sols.append(dict(Cx=Cx, a=(int(a[0]), int(a[1])), bprime=(int(bp[0]), int(bp[1])), cprime=(int(cp[0]), int(cp[1]))))
        if sols and len({s["Cx"] for s in sols}) >= 1 and Cx >= sols[0]["Cx"] + 0:
            break
    if not sols:
        return None
    Cx = sols[0]["Cx"]
    best = []
    for s in sols:
        dk = k0 / Cx
        a, bp, cp = np.array(s["a"]), np.array(s["bprime"]), np.array(s["cprime"])
        err = max(abs(_ang(a) - th), abs(_ang(bp) - 3 * th), abs(_ang(cp) - 2 * th))
        s["max_ang_err_deg"] = float(err)
        s["rel_k_err"] = [float(np.hypot(*a) / (Cx / rho) - 1), float(np.hypot(*bp) / (Cx / rho) - 1), float(np.hypot(*cp) / Cx - 1)]
        best.append(s)
    best.sort(key=lambda s: s["max_ang_err_deg"] + 100 * max(abs(x) for x in s["rel_k_err"]))
    N = next_fast_len(int(np.ceil(ppw * Cx)))
    return dict(rho=rho, theta_star=float(th), Cx=Cx, N=int(N), L=float(2 * np.pi * Cx / k0), solutions=best[:n_keep], n_solutions=len(sols))
