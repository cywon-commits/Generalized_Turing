"""저장된 정상상태에서 상위 고유값과 불안정 고유벡터 구성 재계산: python scripts/v4_eigcomp.py STATE eta"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_branch import lattice, state_spec
from v4_probe import make
from ring_sim.netstate import composition
name, eta = sys.argv[1], float(sys.argv[2])
n, outs, ins, sym, sector = state_spec(name)
inner, outer, Cx, N = lattice(n)
net, s0 = make(2 * np.cos(np.pi / n), Cx, N, eta)
x = np.load(f"runs_v4/{name}_eta{eta}.npy")
lam, w, V = net.eigenvalues(x, k=8, tol=1e-6)
print("λ/σ0:", [round(float(l), 4) for l in lam])
for i, l in enumerate(lam[:6]):
    if abs(l) < 2e-3: continue
    c = composition(net, np.real(V[:, i]), inner, outer)
    print(f"mode {i} λ={l:+.3f}:", {k: round(v, 3) if not isinstance(v, list) else [round(a, 3) for a in v] for k, v in c.items()})
