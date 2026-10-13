"""쌍안정 시험: 닫힌 8회 연결망(N8-S)과 정상파형 마름모(R)가 같은 η 에서 둘 다 안정한지, 둘 사이 직선 보간 초기 조건이 어느 끌개로 가는지.
사용: python scripts/v4_bistable.py eta"""
import sys, json, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import setup, A_SAT
from v4_probe import make
from v4_branch import lattice
from ring_sim.bloch import find_states
from ring_sim.arnoldi import embed

eta = float(sys.argv[1])
inner, outer, Cx, N = lattice(8)
net, s0 = make(2 * np.cos(np.pi / 8), Cx, N, eta)
xN = np.load(f"runs_v4/n8s_eta{eta}.npy")
# 마름모 (v3 Galerkin, 정상파형) 를 96² 격자에 심고 Newton 으로 다듬는다
par, s0b, k0, (c0, ck), lat = setup("oct", eta)
st = [f for f in find_states(c0, A_SAT, seeds=("sym_standing",), nrep=1, T=3e5) if f["info"]["type"] == "standing"][0]
u, v = embed(c0, st["X"], lat["a"], lat["b"], N)
xR = net.to_vec(u, v)
net.set_sym("My+L")
xR, res = net.newton(xR, f_tol=1e-10, maxiter=15, inner_maxiter=30)
xR = net.project_vec(xR)  # 진짜 정상상태는 사영된 상태
net.set_sym(None)
print('rhombus unconstrained residual after projection: %.1e' % np.abs(net.residual(xR)).max(), flush=True)
print(f"rhombus (standing) NK residual {res:.1e}", flush=True)
lamR, _, _ = net.eigenvalues(xR, k=8, tol=1e-5)
print("rhombus full-domain Arnoldi λ/σ0:", [round(float(l), 4) for l in lamR], flush=True)
np.save(f"runs_v4/r8s_eta{eta}.npy", xR)
# 병진 정렬: 두 장의 상호상관 최대화
uR, uN = xR.reshape(net.shape)[0], xN.reshape(net.shape)[0]
cc = np.fft.ifft2(np.fft.fft2(uN, axes=(1, 2)) * np.conj(np.fft.fft2(uR, axes=(1, 2))), axes=(1, 2)).real.sum(0)
si, sj = np.unravel_index(np.argmax(cc), cc.shape)
xR_al = np.roll(xR.reshape(net.shape), (si, sj), axis=(2, 3)).reshape(-1)
print("alignment shift", (si, sj), flush=True)

def classify(x):
    mo = net.modes(x, [tuple(v) for v in outer[:4]])[:, 0]
    mi = net.modes(x, [tuple(v) for v in inner[:4]])
    return mo, mi

out = dict(eta=eta, rhombus_eig=[float(l) for l in lamR], results=[])
for s in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
    x0 = (1 - s) * xR_al + s * xN
    t = time.time()
    x = net.relax(x0, 6e4)
    mo, mi = classify(x)
    nnet = int((mo > 1e-3).sum())
    label = "network(4 outer)" if nnet == 4 else ("rhombus/chain (%d outer)" % nnet if nnet >= 1 else "no outer")
    print(f"s={s}: final outer|F0|={['%.2e' % a for a in mo]} inner p1={['%.1e' % a[1] for a in mi]} -> {label} [{time.time()-t:.0f}s]", flush=True)
    out["results"].append(dict(s=s, outer=[float(a) for a in mo], inner_p1=[float(a[1]) for a in mi], label=label))
    json.dump(out, open(f"results_v4/bistable_eta{eta}.json", "w"))
