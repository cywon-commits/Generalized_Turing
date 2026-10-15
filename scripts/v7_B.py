"""v7 과제 B: 적층 유형 지도. 사용: python scripts/v7_B.py [--np 4]  -> results_v7/B/*.json, B_stacking.csv (v7_B_table.py)
exp1: 원고 그림 3 조건 (ρ=√3, δ=0, η=0.4, η'=0, Dv=1.02 Dv_c).  exp2: σ1/σ0 × η' 주사.  exp3: 비율 의존 (σ1=-σ0, η'=-η/2)."""
import json, os, sys, time, warnings
import numpy as np
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from ring_sim.model import Params, peak, growth_rate
from ring_sim.stacking import build, max_eig
from ring_sim.bloch import find_states, seed_state, newton, classify_state
OUT = "results_v7/B"
DV_REF = 2.50309
S0_REF = peak(Params(Dv=DV_REF), 0)[0]
RHO = {"sqrt3": np.sqrt(3), "L1.82": 1.82, "oct": 2 * np.cos(np.pi / 8), "H1.87": 1.87, "dodec": 2 * np.cos(np.pi / 12)}


def dump(res, fn):
    with open(fn + ".tmp", "w") as fh:
        json.dump(res, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)


def states_at(rho, Dv, eta, r, etap_ratio, nb):
    par, s0, k0, cells = build(rho, Dv, eta, r=r, etap_ratio=etap_ratio, nb=nb)
    c0 = cells[0]
    f = s0 / S0_REF
    fs = find_states(c0, 0.9 * np.sqrt(s0), T=3e5 / f, h=50.0 / f)
    out = []
    for x in fs:
        i = x["info"]
        if i["type"] == "stripe" or i["Aa"] < 1e-6:
            continue
        lam, arg = max_eig(cells, x["X"], s0)
        out.append(dict(type=i["type"], Ac=i["Ac"], Aa=i["Aa"], s=i["Aa"] ** 2, a_p1=i["a_p1"], a_p3=i["a_p3"], A2a=i["A2a"],
                        maxeig_over_s0=lam, sector=arg, newton_res=x["newton_res"], seed=x["seed"]))
    margin = float(growth_rate(Params(Dv=Dv), 0, (2 / rho) ** 2 * k0 ** 2)) / s0
    return dict(sigma0=s0, n_sectors=len(cells), states=out, margin_1to2=margin)


def eta_threshold(rho, Dv, r, etap_ratio, f_s0):
    from ring_sim.continuum import cont_setup
    def lam(eta):
        par, s0, k0, (c0, ck), _ = cont_setup(rho, Dv, eta, r=r, etap_ratio=etap_ratio)
        X = seed_state(c0, 0.9 * np.sqrt(s0), ab_frac=0.0, rng=np.random.default_rng(3)); X, res, _ = newton(c0, X)
        if res > 1e-8: return np.nan
        w = np.linalg.eigvals(c0.jacobian(X[:4], c0)); nz = w.real[np.abs(w) > 1e-3 * s0]
        return nz.max() / s0
    grid = np.arange(0.03, 0.30, 0.02)
    vals = [lam(e) for e in grid]
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            return brentq(lam, grid[i], grid[i + 1], xtol=1e-6), vals
    return float("nan"), vals


def job(spec):
    kind = spec["kind"]; name = spec["name"]; fn = f"{OUT}/{name}.json"
    if os.path.exists(fn): return name, "cached"
    t0 = time.time()
    if kind == "exp1":
        Dvc = 2.5
        res = states_at(RHO["sqrt3"], 1.02 * Dvc, 0.4, -1.0, 0.0, 6)
        res.update(spec)
    elif kind == "exp2":
        r, er, kf = spec["r"], spec["etap_ratio"], spec["kf"]
        etr, vals = eta_threshold(RHO["sqrt3"], DV_REF, r, er, S0_REF)
        res = dict(spec, eta_r=etr, thr_scan=vals)
        if np.isfinite(etr):
            res.update(states_at(RHO["sqrt3"], DV_REF, kf * etr, r, er, 3)); res["eta"] = kf * etr
    else:
        res = states_at(RHO[spec["rho"]], DV_REF, spec["eta"], 1.0, -0.5, 3); res.update(spec)
    res["time_s"] = time.time() - t0
    dump(res, fn)
    return name, round(res["time_s"])


def specs():
    S = [dict(kind="exp1", name="exp1_fig3")]
    for r in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for er in (0.0, -0.5):
            for kf in (1.3, 2.0):
                S.append(dict(kind="exp2", name=f"exp2_r{r}_e{er}_k{kf}", r=r, etap_ratio=er, kf=kf, sigma1_over_sigma0=-r))
    for rho in RHO:
        for eta in (0.082, 0.085, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16):
            S.append(dict(kind="exp3", name=f"exp3_{rho}_eta{eta}", rho=rho, eta=eta))
    return S


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    npr = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 4
    with Pool(npr, maxtasksperchild=1) as p:
        for r in p.imap_unordered(job, specs()):
            print("done", r, flush=True)
