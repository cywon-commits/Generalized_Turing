"""v7 과제 A: 판정비 R 의 σ0 무관성. 연속체 (임계원 위 정확한) Bloch 셀에서 σ0 × {1/4, 1, 4}, η = κ_f η_r.
사용: python scripts/v7_A.py [--np 4]   -> results_v7/A/<case>_f<f>_k<kf>.json, results_v7/A_scaling.csv"""
import csv, json, os, sys, time, warnings
import numpy as np
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from ring_sim.model import Params, peak
from ring_sim.continuum import cont_setup
from ring_sim.bloch import find_states, seed_state, newton, classify_state
from v6_task1 import analyse, FS
from v3_stageA import A_SAT

DV_REF = 2.50309
S0_REF = peak(Params(Dv=DV_REF), 0)[0]
A_REF = A_SAT  # 줄무늬 진폭 (Dv=DV_REF)
CASES = {"oct": 2 * np.cos(np.pi / 8), "ctrlL": 1.82}
FSET = (1 / 64, 1 / 16, 0.25, 1.0, 4.0)
KFS = (1.05, 1.15, 1.3, 1.5)
OUT = "results_v7/A"


def dump(res, fn):
    with open(fn + ".tmp", "w") as fh:
        json.dump(res, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)


def Dv_of(f):
    from scipy.optimize import brentq
    return brentq(lambda D: peak(Params(Dv=D), 0)[0] - f * S0_REF, 1.6, 4.0, xtol=1e-13)


def stripe_of(c0, s0):
    X = seed_state(c0, A_REF * np.sqrt(s0 / S0_REF), ab_frac=0.0, rng=np.random.default_rng(3))
    X, r, _ = newton(c0, X)
    return X, r


def lam_rhombus(rho, Dv, eta):
    par, s0, k0, (c0, ck), _ = cont_setup(rho, Dv, eta)
    X, r = stripe_of(c0, s0)
    w = np.linalg.eigvals(c0.jacobian(X[:4], c0))
    nz = w.real[np.abs(w) > 1e-3 * s0]
    return nz.max() / s0


def eta_threshold(rho, Dv, f):
    g = 0.0815 * np.sqrt(f)
    return brentq(lambda e: lam_rhombus(rho, Dv, e), 0.6 * g, 1.6 * g, xtol=1e-8)


def eta_r_cached(case, f):
    fn = f"{OUT}/eta_r_{case}_f{f}.json"
    if os.path.exists(fn):
        return json.load(open(fn))["eta_r"]
    os.makedirs(OUT, exist_ok=True)
    e = eta_threshold(CASES[case], Dv_of(f), f)
    json.dump(dict(case=case, f=f, eta_r=e), open(fn, "w"))
    return e


def _thr(job):
    return job, eta_r_cached(*job)


def run(job):
    case, f, kf = job
    fn = f"{OUT}/{case}_f{f}_k{kf}.json"
    if os.path.exists(fn):
        return job, "cached"
    t0 = time.time()
    rho = CASES[case]; Dv = Dv_of(f)
    eta_r = eta_r_cached(case, f)
    eta = kf * eta_r
    par, s0, k0, (c0, ck), _ = cont_setup(rho, Dv, eta)
    k1 = k0 / rho
    Xs, rs = stripe_of(c0, s0)
    res = dict(case=case, f=f, kf=kf, Dv=Dv, sigma0=s0, eta_r=eta_r, eta=eta, stripe_resid=float(rs))
    st, _ = analyse(c0, ck, par, k0, k1, Xs, "stripe", False)
    res["stripe"] = {k: v for k, v in st.items() if not k.startswith("_")}
    T, h = 3e5 / f, 50.0 / f
    found = [x for x in find_states(c0, A_REF * np.sqrt(s0 / S0_REF), T=T, h=h) if x["info"]["type"] == "standing"]
    if not found:
        res["state"] = None
        dump(res, fn); return job, "no standing state"
    X = found[0]["X"]; info = found[0]["info"]
    res["info"] = info; res["s"] = float(info["Aa"] ** 2)
    o, _ = analyse(c0, ck, par, k0, k1, X, "standing", False)
    res["rhombus"] = {k: v for k, v in o.items() if not k.startswith("_")}
    # 진폭 스캔 (v6 과 같은 방법)
    jc0 = c0.lookup[(1, 1)]
    phi = np.angle(X[0, jc0]) - np.angle(Xs[0, jc0])
    Xa = Xs * np.exp(1j * phi * c0.m)[None, :]
    wgt = lambda ff: np.where(c0.m == c0.n, ff ** 2, ff)[None, :]
    scan = []
    for ff in FS:
        Xf = Xa + wgt(ff) * (X - Xa)
        sf = float(classify_state(c0, Xf)["Aa"] ** 2)
        oo, _ = analyse(c0, ck, par, k0, k1, Xf, "standing", False)
        scan.append(dict(f=ff, s=sf, alpha=oo["alpha"], beta=oo["beta"], qprod=oo["qprod"], lam_full=oo["lam_full"]))
    res["scan"] = scan
    res["time_s"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    dump(res, fn)
    return job, round(res["time_s"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    npr = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 4
    jobs = [(c, f, kf) for c in CASES for f in FSET for kf in KFS]
    with Pool(npr, maxtasksperchild=1) as p:
        for r in p.imap_unordered(_thr, [(c, f) for c in CASES for f in FSET]):
            print("eta_r", r, flush=True)
        for r in p.imap_unordered(run, jobs):
            print("done", r, flush=True)
