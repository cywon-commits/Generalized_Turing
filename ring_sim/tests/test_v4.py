import numpy as np
import pytest
from ring_sim.model import Params
from ring_sim.netstate import Net
from ring_sim.network import search_network, build_from_params, outer_from_inner


@pytest.mark.parametrize("n,Cx", [(8, 10), (6, 14), (10, 26)])
def test_network_lattice_closes_exactly(n, Cx):
    s = search_network(n, cx_max=60)[0]
    assert s["Cx"] == Cx
    I = [np.array(v) for v in s["inner"]]
    c = [np.array(v) for v in s["outer"]]
    for j in range(n):  # c_j = I_{j-1} + I_j 정확히 닫힘
        assert np.array_equal(c[j], I[j - 1] + I[j])
    assert s["max_ang_err"] < 1.0


@pytest.mark.parametrize("name,size", [("D4", 8), ("D4L", 16), ("Mx+My+L", 8), ("Mx+My*L", 4)])
def test_symmetry_projection_idempotent(name, size):
    par = Params(Dv=2.50309, ku=0.07, kv=0.17, eta=0.1, etap=-0.05)
    net = Net(par, 32, 4, 0.4473, 4.1e-4, dt=8.0)
    net.set_sym(name)
    assert len(net._group) == size
    u = np.random.default_rng(0).standard_normal((4, 32, 32))
    p = net._project(u)
    assert np.abs(net._project(p) - p).max() < 1e-12
