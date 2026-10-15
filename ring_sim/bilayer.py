"""v7 과제 D: ℤ2 이중층 (nl=2). 동위상 섹터 p=0, 역위상 섹터 p=1 (층간 결합 고유값 s=4)."""
import numpy as np
from scipy.optimize import fsolve, brentq, minimize_scalar
from .model import Params, growth_rate, peak
from .integrator import Solver

NL = 2


def peak_nl(par, p, qmax=1.0, n=4001):
    qs = np.linspace(0, qmax, n)
    g = growth_rate(par, p, qs, NL)
    i = int(np.argmax(g))
    r = minimize_scalar(lambda q: -float(growth_rate(par, p, q, NL)), bounds=(qs[max(i - 1, 0)], qs[min(i + 1, n - 1)]), method="bounded", options={"xatol": 1e-13})
    return float(-r.fun), float(r.x)


def solve_kappa_nl(ratio, r, base):
    """역위상 섹터 (p=1) 의 봉우리가 k0/ratio 에서 값 -r σ0(base) 가 되는 (κu, κv). σ0(base) = 동위상 봉우리 (음수 가능)."""
    s0, q0 = peak_nl(base, 0)
    k0 = np.sqrt(q0)
    tgt = (k0 / ratio) ** 2

    def f(x):
        par = Params(**{**base.__dict__, "ku": x[0], "kv": x[1]})
        s1, q1 = peak_nl(par, 1)
        return [q1 - tgt, s1 - (-r * s0)]
    sol = fsolve(f, [0.035, 0.085], xtol=1e-12)
    return float(sol[0]), float(sol[1])


def Dv_for_peak(sigma, lo=2.2, hi=2.9):
    return brentq(lambda D: peak_nl(Params(Dv=D), 0)[0] - sigma, lo, hi, xtol=1e-12)


def make_solver(par, L, N=128, dt=2.0):
    return Solver(par, N=N, L=L, dt=dt, nl=NL)


def noise_state(S, seed, amp=1e-3):
    rng = np.random.default_rng(seed)
    u = amp * rng.standard_normal((NL, S.N, S.N))
    uh = S.to_hat(u)
    V = np.stack([uh, np.zeros_like(uh)])
    S.hermitize(V)
    return V


def sector_spectra(S, V):
    u = S.to_real(V[0])
    N = S.N
    F = np.fft.fft2(u, axes=(1, 2)) / (N * N)
    f0 = 0.5 * (F[0] + F[1])
    f1 = 0.5 * (F[0] - F[1])
    return u, np.abs(f0) ** 2, np.abs(f1) ** 2


def peak_directions(P, N, L, k0, w=0.25, frac=0.15):
    """링 대역 안의 스펙트럼 피크 방향 (도, 0-180) 목록: 최대 세기의 frac 이상인 국소 극대"""
    k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / L
    KX, KY = np.meshgrid(k, k, indexing="ij")
    kk = np.hypot(KX, KY)
    band = np.abs(kk - k0) <= w * k0
    Pm = np.where(band, P, 0)
    m = Pm.max()
    if m <= 0: return [], 0.0
    loc = np.ones_like(Pm, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj: loc &= Pm >= np.roll(np.roll(Pm, di, 0), dj, 1)
    ok = loc & (Pm >= frac * m)
    ang = np.degrees(np.arctan2(KY[ok], KX[ok])) % 180.0
    return sorted(set(np.round(ang, 0))), float(Pm.sum())
