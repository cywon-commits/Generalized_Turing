"""v6 과제 2 도구: (w,y) 손잡이 설계 (선형), 줄무늬 상태의 |α0| 측정, 세기 γ 보정."""
import json, os, sys, time, warnings
import numpy as np
from scipy.optimize import fsolve, brentq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import CASES, LAT, A_SAT, DV
from ring_sim.model import Params, peak, solve_kappa
from ring_sim.bloch import seed_state, classify_state
from ring_sim.bloch12 import Cell12, Knob, w_growth, Lw_sector, relax, newton, embed8
from ring_sim import fillin as F
from scipy.fft import next_fast_len

THETA_STAR = {c: np.arccos(r / 2) for c, r in CASES.items()}


def design_knob(case, eps=3.0, kk=1.0, base=None):
    """p=0 의 (w,y) 최대 성장이 K₋ = 4 sin²θ* q0 에서 -eps·σ0 이 되도록 (aw, Dy) 를 푼다. 다른 섹터는 κ_w=κ_y=kk 로 강하게 안정."""
    base = base or Params(Dv=DV)
    s0, q0 = peak(base, 0)
    Km = 4 * np.sin(THETA_STAR[case]) ** 2 * q0 if False else 4 * np.sin(THETA_STAR[case]) ** 2 * q0
    # 의뢰서: K₋ = 4 sin²θ* · q0 (θ* 는 안쪽·바깥 사이각이 아니라 2θ* 사이각의 반: 8회 22.5° -> 0.586 q0)
    th = {"oct": 22.5, "ctrlL": 24.49464, "ctrlH": 20.77186}[case] if case in ("oct", "ctrlL", "ctrlH") else None
    Km = 4 * np.sin(np.radians(th)) ** 2 * q0
    lam = q0 / Km
    kn0 = Knob(aw=1.0, bw=1.6, dw=1.5, Dw=lam, Dy=lam * DV, kw=kk, ky=kk)

    def peakval(kn):
        qs = np.linspace(0.2 * Km, 2.5 * Km, 4001)
        g = np.array([w_growth(kn, 0, q) for q in qs])
        i = int(np.argmax(g))
        return g[i], qs[i]

    def eqs(x):
        aw, Dy = x
        kn = Knob(aw=aw, bw=1.6, dw=1.5, Dw=lam, Dy=Dy, kw=kk, ky=kk)
        pk, qp = peakval(kn)
        return [(pk + eps * s0) / s0, (qp - Km) / Km]
    sol = fsolve(eqs, [1.0 - 0.001, lam * DV], xtol=1e-12)
    kn = Knob(aw=sol[0], bw=1.6, dw=1.5, Dw=lam, Dy=sol[1], kw=kk, ky=kk)
    pk, qp = peakval(kn)
    sect = {p: max(w_growth(kn, p, q) for q in np.linspace(0, 3 * q0, 400)) / s0 for p in range(4)}
    return kn, dict(Km_over_q0=Km / q0, peak_over_s0=pk / s0, qpeak_over_Km=qp / Km, sector_max_over_s0=sect, lam=lam)


def build(case, eta, kn, kcut_factor=3.0, k0=None):
    base = Params(Dv=DV)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    rho = CASES[case]
    ku, kv = solve_kappa(rho, 1.0, base)
    par = Params(Dv=DV, ku=ku, kv=kv, eta=eta, etap=-eta / 2)
    sol = LAT[case]["solutions"][0]
    Cx = LAT[case]["Cx"]
    a = tuple(sol["a"]); b = (a[0], -a[1]); bp = tuple(sol["bprime"])
    dk = k0 / Cx
    cells = []
    for kap in [(0.0, 0.0), bp]:
        cells.append(Cell12(par, kn, a, b, dk, kcut_factor * k0, kappa=kap))
    P = next_fast_len(4 * max(c.Mx for c in cells) + 2)
    cells = [Cell12(par, kn, a, b, dk, kcut_factor * k0, kappa=c.kappa, P=P) for c in cells]
    for c in cells:
        c._k1 = k0 / rho
    return par, s0, k0, cells


def stripe(c0, A=A_SAT):
    X8 = seed_state(c0, A, ab_frac=0.0, rng=np.random.default_rng(3))
    X = embed8(X8, c0.M)
    X, r, it = newton(c0, X)
    return X, r


def alpha0_of(case, eta, kn, verbose=False):
    par, s0, k0, (c0, ck) = build(case, eta, kn)
    X, r = stripe(c0)
    if r > 1e-9:
        X, _ = relax(c0, X, T=1e5, h=40.0); X, r, _ = newton(c0, X)
    J = ck.jacobian(X, c0)
    Jt = F.sector_transform(ck.M, 16) @ J @ F.sector_transform(ck.M, 16).conj().T
    jc = ck.lookup[(1, 0)]
    R, Lf = F.basis_vectors(ck, par, [(jc, 0)], nc=16)
    lam, M, parts, M0 = F.self_consistent(Jt, R, Lf, lam0='bare')
    A_c = classify_state(c0, X[:8])["Ac"]
    out = dict(alpha=float(M[0, 0].real) / s0, resid=float(r), Ac=float(A_c))
    # 줄무늬 안정성: κ=0 섹터의 최대 고유값 (c0 셀 선형화)
    return out, (par, s0, k0, c0, ck, X)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "oct"
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    kn, info = design_knob(case)
    print(case, info)
    for g in (0.0, -0.002, -0.01, -0.05):
        kn.gamma = g
        t = time.time(); o, _ = alpha0_of(case, eta, kn); print("gamma", g, o, "%.0fs" % (time.time() - t), flush=True)
