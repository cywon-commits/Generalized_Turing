"""v7 과제 C: C_x=10 N8-S 의 상위 모드를 C_x=24 (N=240) 격자에 옮겨 심고 대칭 부분공간 Newton-Krylov + 전체 공간 Arnoldi.
사용: python scripts/v7_C_embed.py ETA [TOPK]   -> results_v7/C/embed_cx24_eta<η>.json, runs_v7/cx24_eta<η>.npy"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_probe import make
from ring_sim.specdiag import diagnose

eta = float(sys.argv[1]); topk = int(sys.argv[2]) if len(sys.argv) > 2 else 24
CX, N = 24, 240
os.makedirs("results_v7/C", exist_ok=True); os.makedirs("runs_v7", exist_ok=True)
fn = f"results_v7/C/embed_cx24_eta{eta}_k{topk}.json"
rho = 2 * np.cos(np.pi / 8)
x10 = np.load(f"runs_v4/n8s_proj_eta{eta}.npy").reshape(2, 4, 96, 96)
F = np.fft.fft2(x10, axes=(2, 3)) / (96 * 96)           # (2,4,96,96)
P = (np.abs(F[0]) ** 2).sum(0); P[0, 0] = 0
idx = np.argsort(P.ravel())[::-1][:topk]
Fn = np.zeros((2, 4, N, N), complex)
used = []
for f in idx:
    i, j = divmod(int(f), 96)
    kx = i if i <= 48 else i - 96; ky = j if j <= 48 else j - 96
    kx2, ky2 = int(round(kx * CX / 10)), int(round(ky * CX / 10))
    Fn[:, :, kx2 % N, ky2 % N] = F[:, :, i, j]
    used.append(((kx, ky), (kx2, ky2)))
xseed = np.fft.ifft2(Fn, axes=(2, 3)).real * (N * N)
net, s0 = make(rho, CX, N, eta)
net.set_sym("D4L")
t0 = time.time()
x = net.to_vec(xseed[0], xseed[1])
res = dict(Cx=CX, N=N, eta=eta, topk=topk, modes=used[:8])
try:
    x = net.relax(x, float(os.environ.get("RELAX_T", 4e4)))
    x, r = net.newton(x, f_tol=1e-10, maxiter=15, inner_maxiter=30)
    x = net.project_vec(x)
    res.update(converged=bool(r < 1e-9), residual=float(r))
except Exception as ex:
    res.update(converged=False, error=type(ex).__name__)
if res.get("converged"):
    np.save(f"runs_v7/cx24_eta{eta}_k{topk}.npy", x)
    u = x.reshape(2, 4, N, N)[0]
    res["diag"] = diagnose(u, CX)
    net.set_sym(None)
    lam, w, V = net.eigenvalues(x, k=10, tol=1e-5)
    nn = [l for l in lam if abs(l) > 2e-3]
    res["eig"] = [float(l) for l in lam]; res["max_nonneutral"] = float(max(nn)) if nn else None
    res["amp_inner_p1"] = [float(a[1]) for a in net.modes(x, [tuple(v) for v in net_inner[:4]])] if False else None
res["time_s"] = time.time() - t0
json.dump(res, open(fn, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(json.dumps({k: v for k, v in res.items() if k not in ("modes", "eig")}, default=str))
