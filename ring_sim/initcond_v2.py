"""v2 초기 조건: S (줄무늬 + 잡음 1e-4), R (마름모 + 잡음 1e-5), N (잡음 1e-3)."""
import numpy as np
from .model import NLAYER, s_p


def rhombus_vectors(R0, ratio):
    """c=(R0,0), a=(R0/2, y), b=(R0/2,-y): 격자에서 a+b=c 가 정확히 닫히고 k0/|a| 가 ratio 에 가장 가까운 정수 y."""
    assert R0 % 2 == 0
    best = min(range(1, R0), key=lambda y: abs(R0 / np.hypot(R0 / 2, y) - ratio))
    return (R0, 0), (R0 // 2, best), (R0 // 2, -best)


def make_state_v2(S, ratio, ic, seed, R0, A_sat, ab_frac=0.3, noise=None):
    rng = np.random.default_rng(seed)
    N, L = S.N, S.L
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    n = np.arange(NLAYER)[:, None, None]
    u = np.zeros((NLAYER, N, N))
    info = dict(ic=ic, A_sat=A_sat)
    phase = lambda: rng.uniform(0, 2 * np.pi)
    if ic in ("S", "R"):
        c, a, b = rhombus_vectors(R0, ratio)
        u += A_sat * np.cos(2 * np.pi * (c[0] * X + c[1] * Y)[None] / L + phase())
        info.update(c=c, a=a, b=b)
        if ic == "R":
            # a: e^{+iπn/2}, b: e^{-iπn/2} (층 Fourier 섹터 p=1, p=3 → 삼각형 (1,1,0) 이 살아있는 배치)
            u += ab_frac * A_sat * np.cos(2 * np.pi * (a[0] * X + a[1] * Y)[None] / L + phase() + np.pi * n / 2)
            u += ab_frac * A_sat * np.cos(2 * np.pi * (b[0] * X + b[1] * Y)[None] / L + phase() - np.pi * n / 2)
    amp = noise if noise is not None else {"S": 1e-4, "R": 1e-5, "N": 1e-3}[ic]
    u += amp * rng.standard_normal(u.shape)
    uh = S.to_hat(u)
    V = np.stack([uh, np.zeros_like(uh)])
    S.hermitize(V)
    par = S.par
    q = S.k2.ravel()[S.idx]
    for p in range(NLAYER):
        V[1, p] = V[0, p] / (par.d + s_p(p) * par.kv + par.Dv * q)
    return V, info
