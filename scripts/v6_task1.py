"""v6 과제 1: fill-in 유효 계수 추출. 사용: python scripts/v6_task1.py [--np 4] [--only case]
각 (사례, η): 줄무늬/정상파형/나선형 상태에서 κ=b′ 셀의 선형화 J -> 섹터 기저 -> Schur 보수 유효 행렬 -> α, β, 결합 곱, 채널 분해.
출력: results_v6/task1/<case>_eta<η>.json"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from v3_stageA import setup, A_SAT, CASES
from ring_sim.bloch import find_states, seed_state, newton, classify_state, Cell
from ring_sim import fillin as F

ETAS = (0.082, 0.085, 0.09, 0.10, 0.11, 0.12, 0.13)
FS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stripe_state(c0):
    X = seed_state(c0, A_SAT, ab_frac=0.0, rng=np.random.default_rng(3))
    X, r, it = newton(c0, X)
    return X, r


def analyse(c0, ck, par, k0, k1, X0, kind, incl12, nev_full=True):
    """한 상태에서 유효 행렬 계수와 전체 fill-in 고유값. kind: 'standing' (b′ p=1,3) | 'p1' | 'p3' | 'stripe'"""
    s0 = None
    J = ck.jacobian(X0[:4], c0)
    U = F.sector_transform(ck.M)
    Jt = U @ J @ U.conj().T
    jc = ck.lookup[(1, 0)]; jb = ck.lookup[(0, 0)]
    picks = [(jc, 0)]
    bps = {"standing": [1, 3], "p1": [1], "p3": [3], "stripe": [1, 3]}[kind]
    picks += [(jb, p) for p in bps]
    if incl12:
        Ks = (2.0 / (k0 / k1)) ** 2 * (k0 ** 2)  # K* = (2/ρ)² q0, ρ = k0/k1
        j12 = min((j for j in range(ck.M) if j not in (jc, jb)), key=lambda j: abs(ck.q[j] - Ks))
        picks.append((j12, 0))
    R, Lf = F.basis_vectors(ck, par, picks)
    lam, M, parts, M0 = F.self_consistent(Jt, R, Lf)
    ib = list(range(1, 1 + len(bps)))
    co = F.coefficients(M, 0, ib)
    co0 = F.coefficients(M0, 0, ib)
    out = dict(kind=kind, incl12=incl12, d=len(picks), lam_self=lam, **co, static=co0)
    # 전체 고유값: P 성분 비중이 큰 고유벡터 중 최대
    w, V = np.linalg.eig(Jt)
    co_v = parts[5] @ V
    wt = (np.abs(co_v[:len(picks)]) ** 2).sum(0) / (np.abs(co_v) ** 2).sum(0)
    cand = np.where(wt > 0.3)[0]
    i = cand[np.argmax(w[cand].real)] if len(cand) else int(np.argmax(w.real))
    out["lam_full"] = float(w[i].real); out["lam_full_weight"] = float(wt[i]); out["lam_full_im"] = float(w[i].imag)
    out["lam_full_max_any"] = float(w.real.max())
    return out, (Jt, R, Lf, lam, picks, ib, M)


def run(job):
    case, eta = job
    fn = f"results_v6/task1/{case}_eta{eta}.json"
    if os.path.exists(fn):
        return job, "cached"
    t0 = time.time()
    par, s0, k0, (c0, ck), lat = setup(case, eta)
    k1 = k0 / CASES[case]
    ck._k1 = k1
    res = dict(case=case, eta=eta, sigma0=s0, k0=k0, k1=k1, lattice=lat, M0=c0.M, Mk=ck.M, states=[])
    Xs, rs = stripe_state(c0)
    states = [dict(type="stripe", X=Xs, info=classify_state(c0, Xs))]
    seen = []
    for f in find_states(c0, A_SAT, T=3e5):
        if f["info"]["type"] in ("standing", "spiral"):
            key = (f["info"]["type"], round(f["info"]["Aa"] ** 2, 7))
            if any(k[0] == key[0] and abs(k[1] - key[1]) < 0.02 * key[1] for k in seen):
                continue
            seen.append(key)
            states.append(dict(type=f["info"]["type"], X=f["X"], info=f["info"]))
    for st in states:
        info = st["info"]
        s = float(info["Aa"] ** 2) if st["type"] != "stripe" else 0.0
        ent = dict(type=st["type"], s=s, info=info, runs=[])
        kinds = ["stripe"] if st["type"] == "stripe" else (["standing"] if st["type"] == "standing" else ["p1", "p3"])
        for kind in kinds:
            for incl in (False, True):
                o, _ = analyse(c0, ck, par, k0, k1, st["X"], kind, incl)
                for key in ("alpha", "beta", "qprod", "lam_self", "lam_full", "lam_full_max_any"):
                    pass
                ent["runs"].append(o)
        # 진폭 스캔 (정상파형/나선형만): 줄무늬 상태와의 보간
        if st["type"] != "stripe":
            kind = "standing" if st["type"] == "standing" else "p1"
            scan = []
            for f in FS:
                Xf = Xs + f * (st["X"] - Xs)
                sf = float(classify_state(c0, Xf)["Aa"] ** 2)
                o, _ = analyse(c0, ck, par, k0, k1, Xf, kind, False)
                scan.append(dict(f=f, s=sf, **{k: o[k] for k in ("alpha", "beta", "qprod", "lam_full", "lam_self")}))
            ent["scan"] = scan
        res["states"].append(ent)
    res["time_s"] = time.time() - t0
    os.makedirs("results_v6/task1", exist_ok=True)
    json.dump(res, open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else (complex(o).real if isinstance(o, complex) else str(o)))
    return job, round(res["time_s"])


if __name__ == "__main__":
    npr = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 4
    cases = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["oct", "ctrlL", "ctrlH"]
    etas = [float(x) for x in sys.argv[sys.argv.index("--etas") + 1].split(",")] if "--etas" in sys.argv else ETAS
    jobs = [(c, e) for c in cases for e in etas]
    with Pool(npr, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run, jobs):
            print("done", r, flush=True)
