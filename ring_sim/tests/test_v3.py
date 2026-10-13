"""v3 단위 시험: 정확 격자 탐색, 축소 셀 선형 연산자, 정상상태 Newton, Arnoldi(행렬 없는 흐름 사상) 일치."""
import numpy as np
import pytest
from ring_sim.model import Params, peak, solve_kappa, growth_rate, lin_matrix, NLAYER
from ring_sim.bloch import Cell, make_cells, seed_state, relax, newton, sector_amps, eigen_sector, classify_state
from ring_sim.exactlattice import search

DV = 2.50309
RHO8 = 2 * np.cos(np.radians(22.5))


def _setup(eta, rho=RHO8, Cx=10, a=(5, 2)):
    base = Params(Dv=DV)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    ku, kv = solve_kappa(rho, 1, base)
    par = Params(Dv=DV, ku=ku, kv=kv, eta=eta, etap=-eta / 2)
    return par, s0, k0, k0 / Cx, a, (a[0], -a[1])


def test_exact_lattice_oct():
    r = search(RHO8, cx_max=40)
    assert r["Cx"] == 10 and r["N"] == 96
    s = r["solutions"][0]
    a, bp, cp = np.array(s["a"]), np.array(s["bprime"]), np.array(s["cprime"])
    assert np.array_equal(cp, a + bp)  # c' = a + b' 가 정확히 닫힘
    assert s["max_ang_err_deg"] < 1.0


def test_block_operator_matches_sector_dispersion():
    """u*=0 이면 셀의 선형 연산자 고유값 = 모드별 섹터 2x2 연산자 (model.lin_matrix) 고유값의 합집합."""
    par, s0, k0, dk, a, b = _setup(0.1)
    c0, = make_cells(par, a, b, dk, 3 * k0, [])[:1]
    J = c0.jacobian(np.zeros((4, c0.M), complex), c0)
    w = np.sort_complex(np.linalg.eigvals(J))
    exp = []
    for j in range(c0.M):
        for p in range(NLAYER):
            exp += list(np.linalg.eigvals(lin_matrix(par, p, c0.q[j])))
    exp = np.sort_complex(np.array(exp))
    assert np.abs(w - exp).max() < 1e-9
    top = max(float(growth_rate(par, p, c0.q[j])) for j in range(c0.M) for p in range(NLAYER))
    assert w.real.max() == pytest.approx(top, abs=1e-12)
    assert top == pytest.approx(s0, rel=0.02)  # 격자에 k0 가 정확히 들어 있다


@pytest.mark.parametrize("eta", [0.05, 0.10, 0.2])
def test_stripe_amplitude_independent_of_eta(eta):
    """줄무늬 정상상태 (a,b=0) 의 진폭은 η 와 무관 (v2 검증 3, 격자 정확 해): 2|F0[c]| = A_sat."""
    par, s0, k0, dk, a, b = _setup(eta)
    c0 = make_cells(par, a, b, dk, 3 * k0, [])[0]
    X = np.zeros((8, c0.M), complex)
    j, jm = c0.lookup[(1, 1)], c0.lookup[(-1, -1)]
    X[:4, j] = 0.009 / 2; X[:4, jm] = 0.009 / 2
    X[4:] = X[:4] / (par.d + par.Dv * c0.q[None, :])
    X, r, it = newton(c0, X)
    assert r < 1e-12
    A = 2 * sector_amps(c0, X)[0, j]
    assert A == pytest.approx(0.018153639577850524, rel=1e-4)


def test_rhombus_threshold_and_newton():
    """줄무늬의 a,b 성장 고유값: η<0.081 이면 음, η>0.081 이면 양 (v2 와 같은 문턱). 마름모 정상상태 Newton 잔차 < 1e-10."""
    lam = {}
    for eta in (0.07, 0.10):
        par, s0, k0, dk, a, b = _setup(eta)
        c0 = make_cells(par, a, b, dk, 3 * k0, [])[0]
        X = np.zeros((8, c0.M), complex)
        j, jm = c0.lookup[(1, 1)], c0.lookup[(-1, -1)]
        X[:4, j] = 0.018153639577850524 / 2; X[:4, jm] = X[:4, j]
        X[4:] = X[:4] / (par.d + par.Dv * c0.q[None, :])
        X, r, it = newton(c0, X)
        lam[eta] = max(e["lam"].real for e in eigen_sector(c0, c0, X, nev=8, k0=k0)) / s0
    assert lam[0.07] < 0.0 < lam[0.10]
    assert lam[0.10] == pytest.approx(0.685, abs=0.05)  # v2 의 전체 영역 시뮬레이션 성장률 0.64 와 근접
    par, s0, k0, dk, a, b = _setup(0.10)
    c0 = make_cells(par, a, b, dk, 3 * k0, [])[0]
    X = seed_state(c0, 0.018153639577850524, rng=np.random.default_rng(1), chir=("p1", "p3"))
    X, _ = relax(c0, X, T=200000, h=50)
    X, r, it = newton(c0, X)
    assert r < 1e-10
    info = classify_state(c0, X)
    assert info["type"] in ("spiral", "standing", "mixed") and info["Aa"] > 1e-3


def test_arnoldi_uniform_state_matches_dispersion():
    """u*=0 에서 행렬 없는 흐름 사상 Arnoldi 의 최대 고유값이 선형 분산 σ0 (격자 최대 모드 성장률) 과 일치."""
    from ring_sim.arnoldi import TangentFlow
    from ring_sim.integrator import Solver
    par, s0, k0, dk, a, b = _setup(0.1)
    N, R0 = 48, 7
    L = 2 * np.pi * R0 / k0
    tau = 300.0
    TF = TangentFlow(par, np.zeros((4, N, N)), N, L, dt=2.0, tau=tau)
    w, V = TF.eigenvalues(k=4, ncv=14, tol=1e-8, maxiter=100)
    lam = np.log(np.abs(w)) / tau
    S = TF.S
    q = S.k2.ravel()[S.idx]
    top = max(float(growth_rate(par, p, q).max()) for p in range(NLAYER))
    assert lam.max() == pytest.approx(top, abs=2e-6)
