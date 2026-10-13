import numpy as np
from ring_sim.invariant import Rings, closure, closure_exact
from ring_sim.network import build_from_params, outer_from_inner, errors

RHO8 = 2 * np.cos(np.pi / 8)


def test_oct_cx10_closes_at_16():
    R = Rings(RHO8); dk = R.k0 / 10
    c = closure([(5, 2), (2, 5), (-2, 5), (-5, 2)], R, dk)
    assert c["closed"] and len(c["K"]) == 16 and (c["n_out"], c["n_in"]) == (8, 8)


def test_rhombus_closes_at_6_for_any_ratio():
    for rho in (RHO8, 1.82, 1.87):
        R = Rings(rho); k1 = 1 / rho; th = 2 * np.arccos(rho / 2)
        rh = [k1 * np.exp(1j * th / 2), k1 * np.exp(-1j * th / 2)]
        assert closure_exact(rh, 1.0, k1)["sizes"][-1] == 6


def test_exact_cyclic_network_only_for_resonant_ratio():
    sizes = {}
    for rho in (RHO8, 1.82):
        k1 = 1 / rho; th = 2 * np.arccos(rho / 2)
        sizes[rho] = closure_exact([k1 * np.exp(1j * (th / 2 + j * th)) for j in range(4)], 1.0, k1)["sizes"][-1]
    assert sizes[RHO8] == 16 and sizes[1.82] == 14


def test_pell_lattices_converge():
    errs = []
    for cx in (10, 24, 58):
        p = cx // 2; q = int(round(p * np.tan(np.pi / 8)))
        I = build_from_params(8, (p, q)); c = outer_from_inner(I)
        a, ro, ri = errors(8, I, c)
        # 정확 격자: c_j = I_{j-1} + I_j 가 항등적으로 성립
        assert all((c[j] == I[j - 1] + I[j]).all() for j in range(8))
        errs.append(a)
    assert errs[0] > errs[1] > errs[2]
