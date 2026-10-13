"""v3 단계 A: 비율 × η 스캔. 정확 격자 (축소 셀) 에서 정상상태들을 찾고 κ=0 / κ=b' 섹터와 (선택) 전 잉여류의 고유값을 계산.
사용: python scripts/v3_stageA.py [--coset]   -> results_v3/stageA/<case>_eta<η>.json"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.model import Params, peak, solve_kappa, RATIOS
from ring_sim.bloch import make_cells, find_states, eigen_sector, Cell, coset_reps

DV = 2.50309
A_SAT = 0.018153639577850524
CASES = {"oct": 2 * np.cos(np.radians(22.5)), "ctrlL": 1.82, "ctrlH": 1.87, "dodec": 2 * np.cos(np.radians(15))}
ETAS = (0.082, 0.085, 0.09, 0.10, 0.11, 0.12, 0.13)
LAT = json.load(open("results_v3/exact_lattices.json"))


def setup(case, eta, r=1.0, kcut_factor=3.0):
    base = Params(Dv=DV)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    rho = CASES[case]
    ku, kv = solve_kappa(rho, r, base)
    par = Params(Dv=DV, ku=ku, kv=kv, eta=eta, etap=-eta / 2)
    sol = LAT[case]["solutions"][0]
    Cx = LAT[case]["Cx"]
    a = tuple(sol["a"]); b = (a[0], -a[1]); bp = tuple(sol["bprime"])
    dk = k0 / Cx
    cells = make_cells(par, a, b, dk, kcut_factor * k0, [bp])
    for c in cells:
        c._k1 = k0 / rho
    return par, s0, k0, cells, dict(a=a, b=b, bprime=bp, Cx=Cx)


def analyze(args):
    case, eta, do_coset, kcut_factor = args
    t0 = time.time()
    par, s0, k0, (c0, ck), lat = setup(case, eta, kcut_factor=kcut_factor)
    T = 6e5 if eta < 0.09 else 3e5
    states = find_states(c0, A_SAT, T=T)
    out = dict(case=case, eta=eta, kcut_factor=kcut_factor, sigma0=s0, lattice=lat, M0=c0.M, Mk=ck.M, states=[])
    for f in states:
        info = f["info"]
        ent = dict(info=info, seed=f["seed"], newton_res=f["newton_res"])
        if info["type"] != "stripe":
            e0 = eigen_sector(c0, c0, f["X"], nev=6, core=((1, 0), (0, 1), (1, 1)), k0=k0)
            ek = eigen_sector(ck, c0, f["X"], nev=8, core=((0, 0), (1, 0)), k0=k0)
        else:
            e0 = eigen_sector(c0, c0, f["X"], nev=6, core=((1, 0), (0, 1), (1, 1)), k0=k0)
            ek = eigen_sector(ck, c0, f["X"], nev=8, core=((0, 0), (1, 0)), k0=k0)
        pack = lambda L: [dict(lam_over_s0=[e["lam"].real / s0, e["lam"].imag / s0], W_core=e["W_core"], W_ring=e.get("W_ring"),
                               top=[(m, n, k, w) for m, n, k, w in e["top_modes"]]) for e in L]
        ent["kappa0"], ent["kappa_bp"] = pack(e0), pack(ek)
        cand = [e for e in ek if e["W_core"] >= 0.5]
        ent["lambda_f_over_s0"] = max(e["lam"].real for e in cand) / s0 if cand else None
        ent["lambda_max_bp_over_s0"] = ek[0]["lam"].real / s0
        # κ=0 에서 중립(≈0) 모드를 제외한 최대 고유값
        nz = [e["lam"].real / s0 for e in e0 if abs(e["lam"].real / s0) > 1e-3]
        ent["kappa0_max_nonneutral"] = max(nz) if nz else None
        if do_coset and info["type"] != "stripe":
            best = (-1e9, None)
            for kap in coset_reps(lat["a"], lat["b"]):
                ck2 = Cell(par, lat["a"], lat["b"], k0 / lat["Cx"], 3 * k0, kappa=kap, P=c0.P)
                ck2._k1 = k0 / CASES[case]
                if ck2.Mx > (c0.P - 2) // 4:
                    ck2 = Cell(par, lat["a"], lat["b"], k0 / lat["Cx"], 3 * k0, kappa=kap)
                    if ck2.P != c0.P:
                        continue
                ev = eigen_sector(ck2, c0, f["X"], nev=1, core=(), k0=k0)
                lam = ev[0]["lam"].real / s0
                if kap != (0.0, 0.0) and lam > best[0]:
                    best = (lam, kap)
            ent["coset_max_nonzero_kappa_over_s0"] = best[0]
            ent["coset_argmax"] = best[1]
        out["states"].append(ent)
    out["time_s"] = time.time() - t0
    fn = f"results_v3/stageA/{case}_eta{eta}{'_kc%.1f' % kcut_factor if kcut_factor != 3.0 else ''}.json"
    json.dump(out, open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return case, eta, out["time_s"]


if __name__ == "__main__":
    coset = "--coset" in sys.argv
    if "--coset" in sys.argv:  # 전 잉여류 스캔 (η=0.10, 0.13 만)
        jobs = [(c, e, True, 3.0) for c in ("ctrlH", "ctrlL", "oct") for e in (0.10, 0.13)] + [("dodec", 0.10, True, 3.0)]
    else:
        jobs = [(c, e, False, 3.0) for c in ("oct", "ctrlL", "ctrlH") for e in ETAS] + [("dodec", 0.10, False, 3.0)]
    with Pool(4) as p:
        for r in p.imap_unordered(analyze, jobs):
            print("done", r, flush=True)
