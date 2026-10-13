"""v5 A1: 비공명 대조 직접 시뮬레이션.
사용: python scripts/v5_A1.py [--np 4] [--only name,...]    -> results_v5/timeseries/<name>.json, runs_v5/<name>.ckpt.npz
사례 (영역은 정확 격자의 정수배, k0 = Cx dk):  oct: Cx=30 (v3 격자 ×3, N=288), ctrlL: Cx=26 (N=240), ctrlH: Cx=32 (N=294)."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from ring_sim.integrator import Solver
from ring_sim.initcond_v2 import make_state_v2
from ring_sim.diag_v2 import DiagV2
from ring_sim.invariant import Rings, closure
from ring_sim.arnoldi import embed
from v3_stageA import setup, A_SAT, CASES, LAT

DOM = {"oct": dict(m=3, N=288), "ctrlL": dict(m=1, N=240), "ctrlH": dict(m=1, N=294)}
DT = 8.0
OUT = "results_v5/timeseries"; RUN = "runs_v5"


def capture_mask(case, diag, par_rings, m):
    lat = LAT[case]["solutions"][0]
    a = np.array(lat["a"]) * m; b = np.array([lat["a"][0], -lat["a"][1]]) * m; bp = np.array(lat["bprime"]) * m
    Cx = LAT[case]["Cx"] * m
    dk = par_rings.k0 / Cx
    k1u = par_rings.k1 / dk
    Q = [tuple(int(round(k1u * f(np.radians(22.5 + 45 * j)))) for f in (np.cos, np.sin)) for j in range(4)]
    K = set()
    for seed in ([a, b], [a, b, bp], Q):
        K |= set(map(tuple, closure([tuple(map(int, s)) for s in seed], par_rings, dk, max_depth=6)["K"]))
    K = np.array(sorted(K))
    # compressed 모드 (kx,ky) 가 집합 원소 ±(1 칸 이내) 에 속하는가
    kv = np.stack([diag.kxi, diag.kyi], 1)
    mask = np.zeros(len(kv), bool)
    for s in (1, -1):
        for v in K * s:
            mask |= (np.abs(kv - v).max(1) <= 1)
    return mask, K


def init_state(case, eta, ic, seed, S, N):
    m = DOM[case]["m"]; Cx = LAT[case]["Cx"] * m
    rng = np.random.default_rng(seed)
    if ic == "R":
        par, s0, k0, (c0, ck), lat = setup(case, eta)
        from ring_sim.bloch import find_states
        st = [f for f in find_states(c0, A_SAT, T=3e5) if f["info"]["type"] == "standing"]
        assert st, "standing rhombus not found"
        a = np.array(lat["a"]) * m; b = np.array(lat["b"]) * m
        u, v = embed(c0, st[0]["X"], a, b, N)
        u = u + 1e-5 * rng.standard_normal(u.shape)
        V = S.state_from_real(u, v)
        info = dict(ic="R", c=[Cx, 0], a=list(map(int, a)), b=list(map(int, b)), amp=st[0]["info"])
        return V, info
    V, info = make_state_v2(S, CASES[case], ic, seed, Cx, A_SAT)
    return V, info


def run_one(job):
    case, eta, ic, seed = job
    name = f"{case}_eta{eta}_{ic}{seed}"
    os.makedirs(OUT, exist_ok=True); os.makedirs(RUN, exist_ok=True)
    fn = f"{OUT}/{name}.json"
    if os.path.exists(fn):
        return name, "cached"
    T = float(os.environ.get("A1_T", 1.2e5)); rec_dt = 400.0
    par, s0, k0, _, lat = setup(case, eta)
    N = DOM[case]["N"]; m = DOM[case]["m"]; Cx = lat["Cx"] * m
    L = 2 * np.pi * Cx / k0
    S = Solver(par, N=N, L=L, dt=DT)
    rings = Rings(CASES[case], eta=eta)
    diag = DiagV2(S, CASES[case], Cx, k0)
    mask, Kset = capture_mask(case, diag, rings, m)
    ck = f"{RUN}/{name}.ckpt.npz"
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        V, step, series = z["V"], int(z["step"]), list(z["series"]); info = json.loads(str(z["info"]))
        Pprev = z["Pprev"]
    else:
        V, info = init_state(case, eta, ic, seed, S, N)
        step, series, Pprev = 0, [], None
    diag.set_track(info)
    nsteps = int(round(T / DT)); rec_every = int(round(rec_dt / DT))

    def rec(V, t, Pprev):
        r = diag.record(V, t)
        P = (np.abs(V[0]) ** 2).sum(0); P[diag.kk < 1e-9] = 0
        r["capture"] = float(P[mask].sum() / P.sum())
        r["dP"] = float(np.abs(P - Pprev).sum() / P.sum()) if Pprev is not None else float("nan")
        r["Ptot"] = float(P.sum() / diag.norm ** 2)
        return r, P
    if step == 0:
        r, Pprev = rec(V, 0.0, None); series.append(r)
    t0 = time.time()
    while step < nsteps:
        V = S.step(V); step += 1
        if step % rec_every == 0:
            r, Pprev = rec(V, step * DT, Pprev); series.append(r)
        if step % (rec_every * 10) == 0 or step == nsteps:
            np.savez(ck + ".tmp.npz", V=V, step=step, series=np.array(series, dtype=object), info=json.dumps(info), Pprev=Pprev)
            os.replace(ck + ".tmp.npz", ck)
            print(f"[{name}] t={step*DT:.0f} N0={series[-1]['N0']} Nin={series[-1]['Nin']} cap={series[-1]['capture']:.3f} "
                  f"dP={series[-1]['dP']:.2e} {time.time()-t0:.0f}s", flush=True)
    fin = diag.final_analysis(V) if hasattr(diag, "final_analysis") else {}
    np.savez_compressed(f"{RUN}/{name}.final.npz", u=S.to_real(V[0]).astype(np.float32), V=V.astype(np.complex64))
    json.dump(dict(case=case, eta=eta, ic=ic, seed=seed, N=N, L=L, Cx=Cx, T=T, sigma0=s0, series=series, info=info,
                   capture_set=Kset.tolist(), final=fin), open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return name, "done"


def jobs_list():
    J = []
    for case in ("oct", "ctrlL", "ctrlH"):
        for seed in (0, 1):
            J.append((case, 0.16, "R", seed))           # 우선: R, η=0.16
    for eta in (0.15, 0.18):
        for case in ("oct", "ctrlL", "ctrlH"):
            for seed in (0, 1):
                J.append((case, eta, "R", seed))
    for case in ("oct", "ctrlL", "ctrlH"):
        for seed in (0, 1):
            J.append((case, 0.16, "S", seed))
    for case in ("oct", "ctrlL"):
        for seed in (0, 1):
            J.append((case, 0.16, "N", seed))
    return J


if __name__ == "__main__":
    npr = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 4
    J = jobs_list()
    if "--only" in sys.argv:
        keys = sys.argv[sys.argv.index("--only") + 1].split(",")
        J = [j for j in J if any(k in f"{j[0]}_eta{j[1]}_{j[2]}{j[3]}" for k in keys)]
    with Pool(npr, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_one, J):
            print("finished", r, flush=True)
