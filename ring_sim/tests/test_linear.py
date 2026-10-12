"""선형 성장률 검증 (의뢰서 4절 '구현 검증'): 표 3 의 σ0, σ1, σ2, 최대 성장 파수를 1% 이내로 재현."""
import numpy as np
import pytest
from ring_sim.model import Params, peak, solve_kappa, RATIOS, growth_rate, s_p
from ring_sim.integrator import Solver

TABLE = {  # (case, r): (ku, kv)  의뢰서 표 3
    ("dodec", 0.5): (0.07861, 0.19280), ("dodec", 1): (0.08025, 0.19444), ("dodec", 2): (0.08356, 0.19777),
    ("ctrlA", 0.5): (0.06626, 0.16132), ("ctrlA", 1): (0.06792, 0.16299), ("ctrlA", 2): (0.07121, 0.16629),
    ("hex", 1): (0.07369, 0.17771), ("oct", 1): (0.07777, 0.18811), ("dec", 1): (0.07941, 0.19230),
    ("ctrlB", 1): (0.07617, 0.18403), ("square", 1): (0.05690, 0.13489),
}


@pytest.mark.parametrize("key", list(TABLE))
def test_kappa_table(key):
    ku, kv = solve_kappa(RATIOS[key[0]], key[1])
    assert ku == pytest.approx(TABLE[key][0], rel=1e-3)
    assert kv == pytest.approx(TABLE[key][1], rel=1e-3)


@pytest.mark.parametrize("key", list(TABLE))
def test_peaks(key):
    case, r = key
    s0, q0 = peak(Params(), 0)
    assert s0 == pytest.approx(0.00658, rel=1e-3) and np.sqrt(q0) == pytest.approx(0.4487, rel=1e-3)
    par = Params(ku=TABLE[key][0], kv=TABLE[key][1])
    s1, q1 = peak(par, 1)
    assert s1 == pytest.approx(-r * s0, rel=0.01)
    assert np.sqrt(q0) / np.sqrt(q1) == pytest.approx(RATIOS[case], rel=0.01)
    assert growth_rate(par, 2, 0.0) < -0.013  # σ2 (q=0 근방이 최대)
    assert peak(par, 0)[0] == pytest.approx(s0)


def test_simulated_growth_rates():
    """무작위 미소 진폭에서 섹터별 모드 지수 성장률을 측정해 σ_p(q) 와 비교."""
    par = Params(ku=0.08025, kv=0.19444, eta=0.4)
    s0, q0 = peak(Params(), 0)
    k0 = np.sqrt(q0)
    N, R0 = 128, 14
    L = 2 * np.pi * R0 / k0
    S = Solver(par, N=N, L=L, dt=1.0)
    rng = np.random.default_rng(0)
    V = S.state_from_real(1e-12 * rng.standard_normal((4, N, N)), np.zeros((4, N, N)))
    # p=0 은 늦은 구간, 감쇠 섹터(p=1,2) 는 비선형 구동(p=0 성장의 제곱)에 오염되기 전 초기 구간에서 측정
    win = {0: (800, 1200), 1: (100, 300), 2: (100, 300)}
    snaps = {}
    for i in range(1, 1201):
        V = S.step(V)
        if i in (100, 300, 800, 1200):
            snaps[i] = np.abs(V[0])
    q = S.k2.ravel()[S.idx]
    rate = {p: np.log(snaps[w[1]] / snaps[w[0]])[p] / (w[1] - w[0]) for p, w in win.items()}
    for p in (0, 1, 2):
        sig = growth_rate(par, p, q)
        top = np.argsort(sig)[-20:]
        err = np.abs(rate[p][top] - sig[top]) / np.abs(sig[top])
        assert err.max() < 0.01, (p, err.max())
    # 최대 성장 파수
    unst = growth_rate(par, 0, q) > 0
    i0 = np.flatnonzero(unst)[np.argmax(rate[0][unst])]
    assert np.sqrt(q[i0]) == pytest.approx(k0, rel=0.01)


def test_hermitian_projection():
    """ky=0 열의 반-에르미트 성분은 실공간에 안 보이지만 반올림 오차로 σ0 로 자라 진단을 오염시킨다 -> 매 스텝 제거."""
    par = Params(ku=0.08025, kv=0.19444, eta=0.4)
    k0 = np.sqrt(peak(Params(), 0)[1])
    N, R0 = 64, 7
    S = Solver(par, N=N, L=2 * np.pi * R0 / k0, dt=1.0)
    rng = np.random.default_rng(1)
    V = S.state_from_real(1e-3 * rng.standard_normal((4, N, N)), np.zeros((4, N, N)))
    V[0, :, S._h_pos[:4]] += 1e-3j  # 인위적 반-에르미트 섭동
    V = S.step(V)
    a = V[:, :, S._h_pos]
    b = np.conj(V[:, S._h_pp][:, :, S._h_partner])
    assert np.abs(a - b).max() < 1e-12 * max(1.0, np.abs(a).max())
