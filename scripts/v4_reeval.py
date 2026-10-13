"""저장된 N8-S 상태를 사영한 뒤 (진짜 정상상태) 전체 공간 Arnoldi + 불안정 고유벡터 구성 + 진폭. 사용: python scripts/v4_reeval.py eta1 eta2 ..."""
import sys, json, os, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_branch import lattice
from v4_probe import make
from ring_sim.netstate import composition
inner, outer, Cx, N = lattice(8)
for eta in [float(a) for a in sys.argv[1:]]:
    cands = [f for f in (f"runs_v4/n8s_eta{eta}.npy", f"runs_v4/n8s_fine_eta{eta}.npy", f"runs_v4/n8s_fold_eta{eta}.npy", f"runs_v4/n8s_fineB_eta{eta}.npy") if os.path.exists(f)]
    if not cands: print("missing", eta); continue
    x = np.load(cands[0])
    net, s0 = make(2 * np.cos(np.pi / 8), Cx, N, eta)
    net.set_sym("D4L"); xp = net.project_vec(x); res = float(np.abs(net.residual(xp)).max())
    net.set_sym(None); res_full = float(np.abs(net.residual(xp)).max())
    t = time.time()
    lam, w, V = net.eigenvalues(xp, k=10, tol=1e-6, maxiter=500)
    rec = dict(eta=eta, residual_proj=res, residual_full=res_full, eig=[float(l) for l in lam])
    nn = [l for l in lam if abs(l) > 2e-3]
    rec["max_nonneutral"] = float(max(nn))
    if rec["max_nonneutral"] > 1e-3:
        i0 = int(np.argmax([l if abs(l) > 2e-3 else -1e9 for l in lam]))
        rec["unstable_comp"] = composition(net, np.real(V[:, i0]), inner, outer)
    mo = net.modes(xp, [tuple(v) for v in outer[:4]]); mi = net.modes(xp, [tuple(v) for v in inner[:4]])
    rec["outer_p0"] = [float(a[0]) for a in mo]; rec["inner_p1"] = [float(a[1]) for a in mi]; rec["inner_p3"] = [float(a[3]) for a in mi]
    rec["two_k_p0"] = [float(a[0]) for a in net.modes(xp, [tuple(2 * v) for v in inner[:4]])]
    np.save(f"runs_v4/n8s_proj_eta{eta}.npy", xp)
    print(eta, "res(full)=%.1e" % res_full, "max nonneutral λ/σ0=%.4f" % rec["max_nonneutral"], "eig", [round(l, 3) for l in lam[:6]], "[%.0fs]" % (time.time() - t), flush=True)
    json.dump(rec, open(f"results_v4/stages/n8s_reeval_{eta}.json", "w"))
