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


def test_cell12_jacobian_matches_finite_difference():
    from ring_sim.bloch12 import Cell12, Knob
    from ring_sim.model import Params
    par = Params(Dv=2.50309, eta=0.1, etap=-0.05, ku=0.07, kv=0.17)
    kn = Knob(Dy=4.0, kw=0.5, ky=0.5, gamma=-0.6, betaw=1.0)
    c = Cell12(par, kn, (2, 1), (2, -1), 0.12, 0.34)
    rng = np.random.default_rng(0)
    # 실공간 실수장이 되도록 X(-k) = conj X(k) 를 만족하는 임의 상태
    X = np.zeros((16, c.M), complex)
    for j in range(c.M):
        jm = c.lookup[(-c.m[j], -c.n[j])]
        if jm >= j:
            z = (rng.standard_normal(16) + 1j * rng.standard_normal(16)) * 0.02
            X[:, j] += z
            X[:, jm] += np.conj(z) if jm != j else 0
            if jm == j:
                X[:, j] = X[:, j].real
    J = c.jacobian(X, c)
    dx = rng.standard_normal((16, c.M)) * 1e-6
    Lb = c.Lblock()
    fd = (c.residual(X + dx, Lb) - c.residual(X - dx, Lb)) / 2
    lin = (J @ dx.T.reshape(-1)).reshape(c.M, 16).T
    assert np.abs(fd - lin).max() < 1e-9 * max(1, np.abs(lin).max()) + 1e-12


def test_stripe_cross_saturation_ratio_is_two():
    """줄무늬 상태에서 바깥 교차계수 비 g_cc'/g = (σ_lin(c') - α0)/σ_lin(c) ≈ 2 (Q000 = 0 설계)"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from v3_stageA import setup, A_SAT, CASES
    from ring_sim.bloch import seed_state, newton
    from ring_sim import fillin as F
    par, s0, k0, (c0, ck), lat = setup("oct", 0.10)
    X = seed_state(c0, A_SAT, ab_frac=0.0, rng=np.random.default_rng(3))
    X, r, _ = newton(c0, X)
    assert r < 1e-10
    J = ck.jacobian(X[:4], c0)
    U = F.sector_transform(ck.M)
    Jt = U @ J @ U.conj().T
    jc = ck.lookup[(1, 0)]
    R, Lf = F.basis_vectors(ck, par, [(jc, 0)])
    lam, M, parts, M0 = F.self_consistent(Jt, R, Lf)
    s_c = F.top_pair(par, 0, c0.q[c0.lookup[(1, 1)]])[0]
    s_cp = F.top_pair(par, 0, ck.q[jc])[0]
    ratio = (s_cp - lam) / s_c
    assert abs(ratio - 2.0) < 0.05
