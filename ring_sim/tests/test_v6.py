import numpy as np
import scipy.linalg as sla
from ring_sim.model import Params, nonlinear_u
from ring_sim import fillin as F


def _N2(x, par):
    return nonlinear_u(x, par, axis=0)


def test_sector_quadratic_ratio_is_two():
    """u 만의 D4 2차항 (η'=-η/2): Q_{1x3->0} / Q_{0x1->1} = 2"""
    par = Params(Dv=2.50309, eta=0.1, etap=-0.05, c=0.0)  # 3차항 제거 -> 순수 2차
    n = np.arange(4)
    e = [np.exp(1j * np.pi * p * n / 2) for p in range(4)]
    B = lambda x, y: _N2(x + y, par) - _N2(x, par) - _N2(y, par)
    proj = lambda f, p: np.vdot(e[p], f) / 4
    q130 = proj(B(e[1], e[3]), 0)
    q011 = proj(B(e[0], e[1]), 1)
    q000 = proj(B(e[0], e[0]), 0)
    assert abs(q000) < 1e-12
    assert abs(q130 / q011 - 2.0) < 1e-12


def test_schur_selfconsistent_hits_exact_eigenvalue():
    rng = np.random.default_rng(0)
    n, d = 14, 2
    J = rng.standard_normal((n, n)) * 0.3
    J = J - 3 * np.eye(n)
    J[0, 0] = 0.5; J[1, 1] = 0.2
    w = np.linalg.eigvals(J)
    V = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Vi = np.linalg.inv(V)
    R, Lf = V[:, :d], Vi[:d].conj().T  # l^H r = identity (쌍직교)
    lam, M, parts, M0 = F.self_consistent(J.astype(complex), R, Lf)
    assert np.min(np.abs(w - lam)) < 1e-9


def test_full_P_gives_J():
    rng = np.random.default_rng(1)
    n = 6
    J = rng.standard_normal((n, n)).astype(complex)
    V = rng.standard_normal((n, n)) + 1j * np.eye(n)
    Vi = np.linalg.inv(V)
    M, _ = F.schur(J, V, Vi.conj().T, 0.0)
    assert np.allclose(M, Vi @ J @ V)
