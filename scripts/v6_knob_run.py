"""v6 과제 2: 사례 × 목표 |α0|/σ0 에서 γ 보정, 마름모 상태와 fill-in 고유값 λ_f(η).
사용: python scripts/v6_knob_run.py CASE TARGET [eta,eta,...]   -> results_v6/knob/<case>_t<TARGET>.json"""
import json, os, sys, time, warnings
import numpy as np
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_knob import design_knob, build, stripe, alpha0_of
from v3_stageA import A_SAT
from ring_sim.bloch import SEEDS, classify_state
from ring_sim.bloch12 import relax, newton, embed8
from ring_sim import fillin as F

ETAS = (0.082, 0.09, 0.10, 0.11, 0.13)


def calibrate(case, target, eta0=0.10):
    """γ 를 0 에서 시작해 기하급수로 늘려 α0/σ0 = -target 을 가두고 brentq. (α0 = c′ 계열 고유값 / σ0)"""
    kn, info = design_knob(case)

    def f(g):
        kn.gamma = g
        return alpha0_of(case, eta0, kn)[0]["alpha"] + target
    g_prev, f_prev = 0.0, f(0.0)
    g = -2e-5
    hist = [(0.0, f_prev - target)]
    for _ in range(14):
        fg = f(g); hist.append((g, fg - target))
        if fg * f_prev < 0:
            root = brentq(f, g, g_prev, xtol=1e-10, rtol=1e-7, maxiter=40)
            kn.gamma = root
            info["calib_history"] = hist
            return kn, info, root
        g_prev, f_prev = g, fg
        g *= 1.6
    raise RuntimeError(f"no bracket for {case} {target}: {hist}")


def fillin_analysis(par, s0, k0, c0, ck, X, kind):
    J = ck.jacobian(X, c0)
    U = F.sector_transform(ck.M, 16)
    Jt = U @ J @ U.conj().T
    jc = ck.lookup[(1, 0)]; jb = ck.lookup[(0, 0)]
    bps = {"standing": [1, 3], "p1": [1], "stripe": [1, 3]}[kind]
    picks = [(jc, 0)] + [(jb, p) for p in bps]
    R, Lf = F.basis_vectors(ck, par, picks, nc=16)
    lam, M, parts, M0 = F.self_consistent(Jt, R, Lf, lam0='bare')
    co = F.coefficients(M, 0, list(range(1, 1 + len(bps))))
    w = np.linalg.eigvals(Jt)
    return dict(lam_self=lam / s0, **{k: (v / s0 if k in ("alpha", "beta", "beta_max", "Lam2") else v) for k, v in co.items()}, lam_full_max=float(w.real.max()) / s0)


def run(case, target, etas):
    fn = f"results_v6/knob/{case}_t{target}.json"
    os.makedirs("results_v6/knob", exist_ok=True)
    t0 = time.time()
    gfn = f"results_v6/knob/{case}_t{target}.gamma.json"
    if os.path.exists(gfn):  # 재시작 이어받기: 보정된 γ 재사용
        from v6_knob import design_knob
        kn, info = design_knob(case); g = json.load(open(gfn))["gamma"]; kn.gamma = g
        info = json.load(open(gfn)).get("design", info)
    else:
        kn, info, g = calibrate(case, target)
        json.dump(dict(gamma=g, design=info), open(gfn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    res = dict(case=case, target=target, gamma=g, knob=dict(kn.__dict__), design=info, etas=[])
    if os.path.exists(fn):
        old = json.load(open(fn))
        res["etas"] = [e for e in old.get("etas", [])]
    done = {e["eta"] for e in res["etas"]}
    etas = [e for e in etas if e not in done]
    for eta in etas:
        par, s0, k0, (c0, ck) = build(case, eta, kn)
        ck_ = ck
        Xs, rs = stripe(c0)
        ent = dict(eta=eta, stripe_resid=float(rs))
        # 줄무늬: κ=b′ 셀에서 c′ 방향 성장 (α0), κ=0 셀에서 줄무늬 안정성 (최대 비중립 고유값)
        so = fillin_analysis(par, s0, k0, c0, ck, Xs, "stripe")
        ent["stripe"] = so
        J0 = c0.jacobian(Xs, c0); w0 = np.linalg.eigvals(J0)
        nz = w0.real[np.abs(w0) > 1e-3 * s0]
        ent["stripe_kappa0_max_nonneutral"] = float(nz.max()) / s0
        ent["states"] = []
        for sname, kind in (("sym_standing", "standing"), ("spiral_p1p3", "p1")):
            X8 = SEEDS[sname](c0, A_SAT, np.random.default_rng(7))
            X = embed8(X8, c0.M)
            X, r = relax(c0, X, T=3e5, h=50.0)
            if not np.all(np.isfinite(X)):
                ent["states"].append(dict(seed=sname, exists=False, note="relaxation diverged")); continue
            X, r, it = newton(c0, X)
            if r > 1e-9:
                ent["states"].append(dict(seed=sname, exists=False, resid=float(r))); continue
            info8 = classify_state(c0, X[:8])
            if info8["type"] == "stripe" or info8["Aa"] < 1e-5:
                ent["states"].append(dict(seed=sname, exists=False, note="went to stripe/other", type=info8["type"], Aa=info8["Aa"])); continue
            k = "standing" if info8["type"] == "standing" else "p1"
            fa = fillin_analysis(par, s0, k0, c0, ck, X, k)
            ent["states"].append(dict(seed=sname, exists=True, type=info8["type"], s=float(info8["Aa"] ** 2), Ac=info8["Ac"], resid=float(r), fillin=fa))
        res["etas"].append(ent)
        print(case, target, eta, [ (s.get("type"), s.get("fillin", {}).get("lam_full_max")) for s in ent["states"]], "stripe alpha0=%.3f" % so["alpha"], flush=True)
        json.dump(res, open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    res["time_s"] = time.time() - t0
    json.dump(res, open(fn, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    case, target = sys.argv[1], float(sys.argv[2])
    etas = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else ETAS
    run(case, target, etas)
