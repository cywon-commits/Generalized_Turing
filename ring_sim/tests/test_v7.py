import numpy as np
from ring_sim.continuum import cont_setup, cont_vectors
from ring_sim.specdiag import diagnose
from ring_sim.model import Params, growth_rate
from ring_sim.bilayer import solve_kappa_nl, peak_nl, Cell2


def test_continuum_vectors_on_rings():
    for rho in (np.sqrt(3), 1.82, 2 * np.cos(np.pi / 8), 1.87):
        a, b, bp = cont_vectors(rho)
        a, b, bp = map(np.array, (a, b, bp))
        assert abs(np.hypot(*(a + b)) - 1) < 1e-12 and abs(np.hypot(*(a + bp)) - 1) < 1e-12
        assert abs(np.hypot(*a) * rho - 1) < 1e-12 and abs(np.hypot(*bp) * rho - 1) < 1e-12


def test_w16_of_sixteen_mode_state_is_one():
    N, Cx = 96, 10
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 2 * np.cos(np.pi / 8)
    u = np.zeros((4, N, N))
    for j in range(4):  # 바깥 4 쌍 (p=0), 안쪽 4 쌍 (p=1 만, 층 위상 e^{iπ n/2})
        th = np.pi * j / 4
        kx, ky = int(round(Cx * np.cos(th))), int(round(Cx * np.sin(th)))
        u += np.cos(2 * np.pi * (kx * X + ky * Y))[None]
        th2 = np.pi / 8 + np.pi * j / 4
        kx2, ky2 = int(round(Cx / rho * np.cos(th2))), int(round(Cx / rho * np.sin(th2)))
        n = np.arange(4)[:, None, None]
        u += np.cos(2 * np.pi * (kx2 * X + ky2 * Y)[None] + np.pi * n / 2)
    d = diagnose(u, Cx, top=16)
    assert d["outer_W16"] > 0.999 and d["inner_W16"] > 0.999


def test_bilayer_kappa_places_antiphase_peak():
    base = Params(Dv=2.55)
    s0, q0 = peak_nl(base, 0)
    ku, kv = solve_kappa_nl(np.sqrt(3), -1.0, base)
    par = Params(**{**base.__dict__, "ku": ku, "kv": kv})
    s1, q1 = peak_nl(par, 1)
    assert abs(s1 - s0) < 1e-9 * max(1, abs(s0)) + 1e-10
    assert abs(np.sqrt(q0 / q1) - np.sqrt(3)) < 1e-6


def test_cell2_jacobian_matches_finite_difference():
    par = Params(Dv=2.55, eta=0.4, etap=0.0, ku=0.01, kv=0.02)
    c = Cell2(par, (0.33, 0.19), (0.33, -0.19), 0.45, 1.2)
    rng = np.random.default_rng(0)
    X = np.zeros((4, c.M), complex)
    for j in range(c.M):
        jm = c.lookup[(-c.m[j], -c.n[j])]
        if jm >= j:
            z = (rng.standard_normal(4) + 1j * rng.standard_normal(4)) * 0.02
            X[:, j] += z
            if jm != j: X[:, jm] += np.conj(z)
            else: X[:, j] = X[:, j].real
    J = c.jacobian(X[:2], c)
    dx = rng.standard_normal((4, c.M)) * 1e-6
    Lb = c.Lblock()
    fd = (c.residual(X + dx, Lb) - c.residual(X - dx, Lb)) / 2
    lin = (J @ dx.T.reshape(-1)).reshape(c.M, 4).T
    assert np.abs(fd - lin).max() < 1e-9 * max(1, np.abs(lin).max()) + 1e-12
