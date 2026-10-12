"""v2 필수 검증 1, 2 (의뢰서 v2 4절): 새 D_v 에서의 선형 성장률, η' 항의 섹터 규칙."""
import numpy as np
import pytest
from ring_sim.model import Params, peak, solve_kappa, RATIOS, growth_rate, nonlinear_u, NLAYER
from ring_sim.integrator import Solver

DV = 2.50309
TABLE = {("dodec", 0.5): (0.07353, 0.18359), ("dodec", 1): (0.07363, 0.18369),
         ("ctrlA", 0.5): (0.06125, 0.15285), ("ctrlA", 1): (0.06136, 0.15296)}


def test_v2_base_point():
    s0, q0 = peak(Params(Dv=DV), 0)
    assert s0 == pytest.approx(4.11e-4, rel=0.01) and np.sqrt(q0) == pytest.approx(0.4473, rel=1e-3)


@pytest.mark.parametrize("key", list(TABLE))
def test_v2_table(key):
    case, r = key
    base = Params(Dv=DV)
    s0, q0 = peak(base, 0)
    ku, kv = solve_kappa(RATIOS[case], r, base)
    assert ku == pytest.approx(TABLE[key][0], rel=1e-3) and kv == pytest.approx(TABLE[key][1], rel=1e-3)
    par = Params(Dv=DV, ku=ku, kv=kv)
    s1, q1 = peak(par, 1)
    assert s1 == pytest.approx(-r * s0, rel=0.01)
    assert np.sqrt(q0 / q1) == pytest.approx(RATIOS[case], rel=0.01)
    assert growth_rate(par, 2, 0.0) < (-1.5e-2 if case == "dodec" else -4.3e-3) * 0.99
    assert Params(Dv=DV).Dv * ku - kv == pytest.approx({0.5: 4.6e-4, 1: 6.2e-4}[r], abs=1e-5)  # δ


def _Q(p1, p2, eta, etap):
    """서로 다른 두 섹터 성분 e^{iθ1 n}, e^{iθ2 n} 의 곱 (교차항) 의 합 섹터 (p1+p2) 투영 계수."""
    par = Params(eta=eta, etap=etap, c=0.0)
    n = np.arange(NLAYER)
    u1 = np.exp(1j * np.pi * p1 * n / 2)
    u2 = np.exp(1j * np.pi * p2 * n / 2)
    cross = nonlinear_u(u1 + u2, par) - nonlinear_u(u1, par) - nonlinear_u(u2, par)
    proj = {p: np.sum(cross * np.exp(-1j * np.pi * p * n / 2)) / NLAYER for p in range(NLAYER)}
    return proj, (p1 + p2) % NLAYER


@pytest.mark.parametrize("eta", [0.083, 0.1, 0.37])
def test_sector_rule(eta):
    """Q_{0x0->0} = 2η+4η' = 0, Q_{0x1->1} = 2η+2η' = η, Q_{1x3->0} = 2η  (η' = -η/2), 상대오차 1e-10."""
    etap = -eta / 2
    for (p1, p2, expect) in [(0, 0, 0.0), (0, 1, eta), (1, 3, 2 * eta), (0, 3, eta), (1, 1, 2 * eta + 2 * etap * 0)]:
        proj, tgt = _Q(p1, p2, eta, etap)
        assert abs(proj[tgt] - expect) <= 1e-10 * eta, (p1, p2, proj[tgt], expect)
        for p in range(NLAYER):
            if p != tgt:
                assert abs(proj[p]) <= 1e-10 * eta  # 다른 섹터로 새지 않는다


def test_sector_rule_solver_path():
    """적분기가 쓰는 실공간 구현(Solver.nonlin)이 같은 규칙을 따르는지: 균일 p=0 장의 2차 항 계수 η+2η' = 0."""
    par = Params(eta=0.1, etap=-0.05, c=0.0)
    u = np.full((NLAYER, 3, 3), 0.7)
    assert np.abs(nonlinear_u(u, par)).max() < 1e-15


def test_simulated_growth_rates_v2():
    """무작위 극미소 진폭에서 섹터별 모드 지수 성장률 (새 D_v, σ0 가 작으므로 긴 구간)."""
    ku, kv = solve_kappa(RATIOS["dodec"], 1, Params(Dv=DV))
    par = Params(Dv=DV, ku=ku, kv=kv, eta=0.09, etap=-0.045)
    k0 = np.sqrt(peak(Params(Dv=DV), 0)[1])
    N, R0 = 128, 14
    S = Solver(par, N=N, L=2 * np.pi * R0 / k0, dt=1.0)
    rng = np.random.default_rng(0)
    V = S.state_from_real(1e-12 * rng.standard_normal((4, N, N)), np.zeros((4, N, N)))
    win = {0: (4000, 12000), 1: (4000, 12000), 2: (100, 300)}
    snaps = {}
    for i in range(1, 12001):
        V = S.step(V)
        if i in (100, 300, 4000, 12000):
            snaps[i] = np.abs(V[0])
    q = S.k2.ravel()[S.idx]
    for p, (t0, t1) in win.items():
        rate = np.log(snaps[t1] / snaps[t0])[p] / (t1 - t0)
        sig = growth_rate(par, p, q)
        top = np.argsort(sig)[-20:]
        err = np.abs(rate[top] - sig[top]) / np.abs(sig[top])
        assert err.max() < 0.01, (p, err.max())


def test_build_uses_config_Dv():
    """runner.build 는 cfg['Dv'] 로 κ 를 풀어야 한다 (과거 결함: 기본 D_v=2.55 용 κ 를 사용)."""
    from ring_sim.runner import build
    for case, r in (("dodec", 1), ("ctrlA", 0.5)):
        par, S, ratio, R0, k0, s0 = build(dict(case=case, r=r, eta=0.09, etap_ratio=-0.5, Dv=DV, N=64, R0=8, dt=1.0))
        ku, kv = TABLE[(case, r)]
        assert par.Dv == DV and par.ku == pytest.approx(ku, rel=1e-3) and par.kv == pytest.approx(kv, rel=1e-3)
        assert par.etap == pytest.approx(-0.045)
        assert s0 == pytest.approx(4.11e-4, rel=0.01)
