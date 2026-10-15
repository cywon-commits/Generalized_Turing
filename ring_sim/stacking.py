"""v7 과제 B: 정상파형/나선형 마름모 적층의 존재·안정성 (연속체 Bloch 셀, 여러 Bloch 오프셋)."""
import numpy as np
from scipy.fft import next_fast_len
from .model import Params, peak, solve_kappa
from .bloch import Cell, find_states, relax, newton, classify_state
from .continuum import cont_vectors


def build(rho, Dv, eta, r=1.0, etap_ratio=-0.5, nb=3, kcut_factor=3.0):
    """κ=0 셀 + nb x nb Bloch 오프셋 (κ = i a/nb + j b/nb) 셀들. 모든 길이 k0 단위 (dk = k0)."""
    base = Params(Dv=Dv)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    ku, kv = solve_kappa(rho, r, base)
    par = Params(Dv=Dv, ku=ku, kv=kv, eta=eta, etap=etap_ratio * eta)
    a, b, bp = cont_vectors(rho)
    a_, b_ = np.array(a), np.array(b)
    kaps = [(0.0, 0.0)] + [tuple(i / nb * a_ + j / nb * b_) for i in range(nb) for j in range(nb) if (i, j) != (0, 0)]
    cells = [Cell(par, a, b, k0, kcut_factor * k0, kappa=k) for k in kaps]
    P = next_fast_len(4 * max(c.Mx for c in cells) + 2)
    cells = [Cell(par, a, b, k0, kcut_factor * k0, kappa=c.kappa, P=P) for c in cells]
    return par, s0, k0, cells


def max_eig(cells, X0, s0):
    """모든 Bloch 오프셋에서의 최대 비중립 고유값 / σ0 와 그 오프셋 인덱스"""
    best, arg = -1e9, None
    c0 = cells[0]
    for i, c in enumerate(cells):
        w = np.linalg.eigvals(c.jacobian(X0[:4], c0))
        wr = w.real[np.abs(w) > 1e-3 * s0] if i == 0 else w.real
        if len(wr) and wr.max() / s0 > best:
            best, arg = wr.max() / s0, i
    return best, arg
