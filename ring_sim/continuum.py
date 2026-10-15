"""v7: 임계원 위에 정확히 놓인 연속체 Bloch 셀 (정수 격자 이탈 없음).
단위 dk = k0.  a, b = 안쪽 파 (k1 = k0/ρ, 각 ±φ, cosφ = ρ/2) -> c = a + b = (1,0) 정확히 바깥 원 위.
b′ = 각 3φ 의 안쪽 파, c′ = a + b′ 도 정확히 바깥 원 위 (모든 비율 ρ < 2 에서 가능).
Λ = span{a, b} 의 Bloch 오프셋 κ = b′ 셀이 fill-in 부분공간."""
import numpy as np
from scipy.fft import next_fast_len
from .model import Params, peak, solve_kappa
from .bloch import Cell


def cont_vectors(rho):
    phi = np.arccos(rho / 2)
    k1 = 1.0 / rho
    a = (k1 * np.cos(phi), k1 * np.sin(phi))
    b = (k1 * np.cos(phi), -k1 * np.sin(phi))
    bp = (k1 * np.cos(3 * phi), k1 * np.sin(3 * phi))
    return a, b, bp


def cont_setup(rho, Dv, eta, r=1.0, kcut_factor=3.0, etap_ratio=-0.5):
    """반환: par, s0, k0, (c0, ck), (a, b, bp). 모든 길이는 k0 단위 (dk = k0)."""
    base = Params(Dv=Dv)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    ku, kv = solve_kappa(rho, r, base)
    par = Params(Dv=Dv, ku=ku, kv=kv, eta=eta, etap=etap_ratio * eta)
    a, b, bp = cont_vectors(rho)
    cells = [Cell(par, a, b, k0, kcut_factor * k0, kappa=kap) for kap in [(0.0, 0.0), bp]]
    P = next_fast_len(4 * max(c.Mx for c in cells) + 2)
    cells = [Cell(par, a, b, k0, kcut_factor * k0, kappa=c.kappa, P=P) for c in cells]
    for c in cells:
        c._k1 = k0 / rho
    return par, s0, k0, cells, (a, b, bp)
