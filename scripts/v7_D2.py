"""v7 D2: 혼합 삼파 (바깥 동위상 k0, 안쪽 역위상 k1 = k0/√3, δ=0: 두 섹터 모두 +σ0). 이중층 Bloch 셀 Newton + PDE 잡음 시작.
출력: results_v7/D/D2_cell.json, D2_pde_s*.json"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.model import Params
from ring_sim.bilayer import *

DV, ETA, RHO = 2.55, 0.4, np.sqrt(3)


def params():
    base = Params(Dv=DV, eta=ETA, etap=0.0)
    s0, _ = peak_nl(base, 0)
    ku, kv = solve_kappa_nl(RHO, -1.0, base)     # δ=0: σ1 = +σ0 at k1 = k0/√3
    return Params(**{**base.__dict__, "ku": ku, "kv": kv})


def seed(c, kind, rng, A=0.06):
    X = np.zeros((4, c.M), complex)
    def put(mn, v):
        j, jm = c.lookup[mn], c.lookup[(-mn[0], -mn[1])]
        X[:2, j] += v / 2; X[:2, jm] += np.conj(v) / 2
    ph = lambda: np.exp(1j * rng.uniform(0, 2 * np.pi))
    inph, anti = np.array([1.0, 1.0]), np.array([1.0, -1.0])
    if kind == "rhombus":
        put((1, 1), A * ph() * inph); put((1, 0), 0.7 * A * ph() * anti); put((0, 1), 0.7 * A * ph() * anti)
    elif kind == "hex_triads":   # 바깥 동위상 삼각 {(-1,-1),(2,-1),(-1,2)} + 안쪽 역위상 삼각 {(1,0),(-1,1),(0,-1)}
        for mn in ((-1, -1), (2, -1), (-1, 2)): put(mn, A * ph() * inph)
        for mn in ((1, 0), (-1, 1), (0, -1)): put(mn, 0.7 * A * ph() * anti)
    elif kind == "hex_inner_both":   # 안쪽 삼각 두 개 (30°, 90° 회전) 모두
        for mn in ((1, 0), (-1, 1), (0, -1), (-1, 0), (1, -1), (0, 1)): put(mn, 0.5 * A * ph() * anti)
        put((1, 1), A * ph() * inph)
    else:
        for j in range(c.M):
            X[:2, j] = 1e-3 * (rng.standard_normal(2) + 1j * rng.standard_normal(2))
        X[:2, c.lookup[(0, 0)]] = 0
        for j in range(c.M):
            jm = c.lookup[(-c.m[j], -c.n[j])]; X[:2, jm] = np.conj(X[:2, j])
    for j in range(c.M):
        X[2:, j] = X[:2, j] / (c.par.d + c.par.Dv * c.q[j])
    return X


def cell_search():
    par = params()
    cells, s0, k0 = cell2_setup(par, RHO, nb=3)
    c0 = cells[0]
    out = dict(sigma0=s0, sigma_in=peak_nl(par, 0)[0], sigma_anti=peak_nl(par, 1)[0], q_in=peak_nl(par, 0)[1], q_anti=peak_nl(par, 1)[1], kappa=[par.ku, par.kv], states=[])
    seen = []
    for kind in ("rhombus", "hex_triads", "hex_inner_both", "noise"):
        for rep in range(4):
            rng = np.random.default_rng(10 * rep + 3)
            X = seed(c0, kind, rng)
            X = relax2(c0, X, T=4e4 * 4.1e-4 / s0 * 4, h=10.0)
            if not np.all(np.isfinite(X)): continue
            X, r = newton2(c0, X)
            if r > 1e-9: continue
            u = c0.to_real(X[:2]).real
            if np.abs(u).max() < 1e-6: continue
            P = np.abs(X[:2]) ** 2
            key = (round(float(np.sum(P)), 8),)
            if key in seen: continue
            seen.append(key)
            # 지배 모드와 층 구조
            order = np.argsort(-(np.abs(X[0] + X[1]) ** 2 + np.abs(X[0] - X[1]) ** 2))[:12]
            modes = [dict(mn=[int(c0.m[j]), int(c0.n[j])], k=float(np.sqrt(c0.q[j]) / k0), inphase=float(abs(X[0, j] + X[1, j]) / 2), antiphase=float(abs(X[0, j] - X[1, j]) / 2)) for j in order]
            sh, corr = layer_shift(c0, X)
            lam_max = -1e9
            for i, c in enumerate(cells):
                w = np.linalg.eigvals(c.jacobian(X[:2], c0))
                wr = w.real[np.abs(w) > 1e-3 * s0] if i == 0 else w.real
                lam_max = max(lam_max, wr.max() / s0)
            out["states"].append(dict(seed=kind, rep=rep, resid=float(r), shift=sh, shift_corr=corr, maxeig_over_s0=float(lam_max), umax=float(u.max()), umin=float(u.min()), modes=modes))
            print(kind, rep, "shift", np.round(sh, 3), "corr %.3f" % corr, "maxeig/σ0 %.3f" % lam_max, "top modes", [(m["mn"], round(m["inphase"], 4), round(m["antiphase"], 4)) for m in modes[:5]], flush=True)
    os.makedirs("results_v7/D", exist_ok=True)
    json.dump(out, open("results_v7/D/D2_cell.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    cell_search()
