"""쌍안정 시험의 시간 경로: 보간 초기 조건 x(s) = (1-s) x_R + s x_N (병진 정렬) 의 바깥/안쪽 진폭 vs 시간. 사용: python scripts/v4_bistable_traj.py eta"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_probe import make
from v4_branch import lattice
eta = float(sys.argv[1])
inner, outer, Cx, N = lattice(8)
net, s0 = make(2 * np.cos(np.pi / 8), Cx, N, eta)
xN = np.load(f"runs_v4/n8s_proj_eta{eta}.npy") if os.path.exists(f"runs_v4/n8s_proj_eta{eta}.npy") else np.load(f"runs_v4/n8s_eta{eta}.npy")
xR = np.load(f"runs_v4/r8s_eta{eta}.npy")
uR, uN = xR.reshape(net.shape)[0], xN.reshape(net.shape)[0]
cc = np.fft.ifft2(np.fft.fft2(uN, axes=(1, 2)) * np.conj(np.fft.fft2(uR, axes=(1, 2))), axes=(1, 2)).real.sum(0)
si, sj = np.unravel_index(np.argmax(cc), cc.shape)
xR_al = np.roll(xR.reshape(net.shape), (si, sj), axis=(2, 3)).reshape(-1)
out = dict(eta=eta, paths=[])
for s in (0.0, 0.3, 0.4, 0.45, 0.5, 0.6, 1.0):
    x = (1 - s) * xR_al + s * xN
    ts, so, si_ = [], [], []
    t = 0.0
    for blk in range(21):
        mo = net.modes(x, [tuple(v) for v in outer[:4]])[:, 0]; mi = net.modes(x, [tuple(v) for v in inner[:4]])
        ts.append(t); so.append([float(a) for a in mo]); si_.append([float(a[1]) for a in mi])
        x = net.relax(x, 3000.0); t += 3000.0
    final = int((np.array(so[-1]) > 1e-3).sum())
    out["paths"].append(dict(s=s, t=ts, outer=so, inner_p1=si_, final=("network" if final == 4 else "rhombus")))
    print(f"s={s}: final outer {['%.2e' % a for a in so[-1]]} -> {out['paths'][-1]['final']}", flush=True)
    json.dump(out, open(f"results_v4/bistable_traj_eta{eta}.json", "w"))
