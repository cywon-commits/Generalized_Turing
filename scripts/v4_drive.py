"""닫힘 효과 지표 (의뢰서 3절 5): 안쪽 모드 방정식에서 2차(삼각형) 항의 크기 / |σ1|.
정상상태 u* 에서 Nq = η u² + η' u (u_{n+1}+u_{n-1}) 의 안쪽 파 I 의 섹터 p 푸리에 성분 / (|û_I,p| |σ1|)."""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_probe import make
from v4_branch import lattice

def drive(net, x, inner):
    u = x.reshape(net.shape)[0]
    p = net.par
    Nq = p.eta * u * u + p.etap * u * (np.roll(u, 1, 0) + np.roll(u, -1, 0))
    N = net.N
    Fu = np.fft.fft2(u, axes=(1, 2)) / (N * N)
    Fq = np.fft.fft2(Nq, axes=(1, 2)) / (N * N)
    lay = lambda F, p_: np.sum(F * np.exp(-1j * np.pi * p_ * np.arange(4) / 2)[:, None, None], 0) / 4
    r = []
    for v in inner[:4]:
        for sec in (1, 3):
            a = abs(lay(Fu, sec)[v[0] % N, v[1] % N]); q = abs(lay(Fq, sec)[v[0] % N, v[1] % N])
            if a > 1e-8:
                r.append(q / a / abs(-net.s0))   # σ1 = -σ0
    return float(np.mean(r)) if r else float("nan")

if __name__ == "__main__":
    inner, outer, Cx, N = lattice(8)
    out = []
    for name, etas in (("n8s", (0.14, 0.15, 0.16, 0.18)), ("r8s", (0.14,))):
        for eta in etas:
            f = f"runs_v4/{name}_proj_eta{eta}.npy" if os.path.exists(f"runs_v4/{name}_proj_eta{eta}.npy") else f"runs_v4/{name}_eta{eta}.npy"
            if not os.path.exists(f): continue
            net, s0 = make(2 * np.cos(np.pi / 8), Cx, N, eta)
            x = np.load(f)
            mi = net.modes(x, [tuple(v) for v in inner[:4]])
            if name == "r8s":
                # 마름모의 안쪽 모드는 (5,2), (5,-2)  -> inner[0], inner[7]
                inner_r = [inner[0], inner[7]]
                d = drive(net, x, inner_r)
                Ain = float(np.mean([net.modes(x, [tuple(v)])[0][1] for v in inner_r]))
            else:
                d = drive(net, x, inner); Ain = float(np.mean(mi[:, 1]))
            mo = net.modes(x, [tuple(v) for v in outer[:4]])[:, 0]
            out.append(dict(state=name, eta=eta, drive_over_sigma1=d, inner_p1=Ain, outer_mean=float(mo.mean()), ratio_in_out=Ain / float(mo.max())))
            print(out[-1], flush=True)
    json.dump(out, open("results_v4/drive_indicator.json", "w"))
