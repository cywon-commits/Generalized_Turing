"""모형 매개변수, 선형 분산 관계, 결합률(κ_u, κ_v) 해법."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, fsolve

NLAYER = 4


@dataclass
class Params:
    a: float = 1.0
    b: float = 1.6
    d: float = 1.5
    Du: float = 1.0
    Dv: float = 2.55
    c: float = 1.0
    eta: float = 0.4
    ku: float = 0.0
    kv: float = 0.0


def s_p(p):
    return 4.0 * np.sin(np.pi * np.asarray(p) / NLAYER) ** 2


def lin_matrix(par, p, q):
    """섹터 p, q=k^2 에서의 2x2 선형 연산자 (…,2,2) 실수."""
    s = s_p(p)
    q = np.asarray(q, dtype=float)
    L = np.zeros(np.broadcast(s, q).shape + (2, 2))
    L[..., 0, 0] = par.a - s * par.ku - par.Du * q
    L[..., 0, 1] = -par.b
    L[..., 1, 0] = 1.0
    L[..., 1, 1] = -par.d - s * par.kv - par.Dv * q
    return L


def growth_rate(par, p, q):
    """최대 고유값 실수부 σ_p(q)."""
    L = lin_matrix(par, p, q)
    return np.linalg.eigvals(L).real.max(axis=-1)


def peak(par, p, qmax=1.0, n=4001):
    """σ_p(q) 의 최대값과 그 때의 q (미세 격자 + 정밀화)."""
    from scipy.optimize import minimize_scalar
    qs = np.linspace(0, qmax, n)
    i = int(np.argmax(growth_rate(par, p, qs)))
    lo, hi = qs[max(i - 1, 0)], qs[min(i + 1, n - 1)]
    r = minimize_scalar(lambda q: -float(growth_rate(par, p, q)), bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-13})
    return float(-r.fun), float(r.x)


def base_params(**kw):
    return Params(**kw)


def solve_kappa(ratio, r, base=None):
    """k0/k1 = ratio, σ1 = -r σ0 를 만족하는 (κu, κv)."""
    base = base or Params()
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    q1_target = (k0 / ratio) ** 2

    def f(x):
        par = Params(**{**base.__dict__, "ku": x[0], "kv": x[1]})
        s1, q1 = peak(par, 1)
        return [q1 - q1_target, s1 - (-r * s0)]

    sol = fsolve(f, [0.07, 0.17], xtol=1e-12)
    return float(sol[0]), float(sol[1])


# 표 3 (의뢰서) -------------------------------------------------------------
RATIOS = {
    "square": 2 ** 0.5,
    "hex": 3 ** 0.5,
    "ctrlA": 1.6,
    "oct": 2 * np.cos(np.deg2rad(22.5)),
    "ctrlB": 1.8,
    "dec": 2 * np.cos(np.deg2rad(18.0)),
    "dodec": 2 * np.cos(np.deg2rad(15.0)),
}
