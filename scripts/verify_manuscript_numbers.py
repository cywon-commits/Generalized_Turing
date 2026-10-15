"""v7 과제 E: 원고에 인용된 수치를 저장소 코드/결과에서 다시 계산해 비교한다. 출력: results_v7/E_numbers.csv"""
import csv, glob, json, os, sys, warnings
import numpy as np
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ring_sim.model import Params, peak, growth_rate, RATIOS
from ring_sim.bloch import Cell, relax, newton, seed_state
from v3_stageA import setup, A_SAT, CASES, DV

rows = []


def add(item, ms, rc, tol, note=""):
    diff = rc - ms if isinstance(ms, (int, float)) else float("nan")
    ok = abs(diff) <= tol if isinstance(ms, (int, float)) else None
    rows.append(dict(item=item, manuscript=ms, recomputed=rc, diff=diff, tolerance=tol, status="OK" if ok else ("DIFF" if ok is not None else "info"), note=note))
    print(f"{item:55s} manuscript={ms}  recomputed={rc:.6g}  diff={diff:.3g}  {rows[-1]['status']}", flush=True)


# 1. D_v = 1.00123 D_v^c  ->  σ0 = 4.11e-4
sig = lambda Dv: peak(Params(Dv=Dv), 0)[0]
Dvc = brentq(sig, 1.5, DV, xtol=1e-14)
s0, q0 = peak(Params(Dv=DV), 0)
add("Dv/Dv_c at Dv=2.50309", 1.00123, DV / Dvc, 1e-5, f"Dv_c={Dvc:.6f}; ratio={DV / Dvc:.7f} (원고는 5자리 반올림)")
add("sigma0 at Dv=1.00123 Dv_c (1e-4)", 4.11, 1e4 * peak(Params(Dv=1.00123 * Dvc), 0)[0], 0.02, "Dv/Dv_c=1.00123 정확히에서는 4.097; v3 의 Dv=2.50309(비 1.001236)에서 4.112")

# 2. 줄무늬 진폭 A/√σ0 (두 σ0: 4 배 차이)
def stripe_amp(Dv):
    par = Params(Dv=Dv, eta=0.1, etap=-0.05); s, q = peak(par, 0); k0 = np.sqrt(q)
    c = Cell(par, (1, 0), (1, 1), k0, 3 * k0)
    X = np.zeros((8, c.M), complex)
    j, jm = c.lookup[(1, 0)], c.lookup[(-1, 0)]
    X[:4, j] = 0.4475 * np.sqrt(s); X[:4, jm] = 0.4475 * np.sqrt(s)
    for jj in range(c.M): X[4:, jj] = X[:4, jj] / (par.d + par.Dv * c.q[jj])
    X, _ = relax(c, X, T=2e5, h=20.0 / (s / 4e-4))
    X, r, _ = newton(c, X)
    return 2 * abs(X[0, j]) / np.sqrt(s), s, r
A1, s1, r1 = stripe_amp(DV)
Dv4 = brentq(lambda D: sig(D) - 4 * s0, Dvc * 1.0000001, 3.5, xtol=1e-13)
A4, s4, r4 = stripe_amp(Dv4)
add("A/sqrt(sigma0) at sigma0=4.112e-4", 0.895, A1, 6e-4, f"resid {r1:.1e}")
add("A/sqrt(sigma0) at 4*sigma0", 0.896, A4, 6e-4, f"sigma0={s4:.3e}, resid {r4:.1e}")

# 3. 마름모 성장률 λ/σ0 = 33.5 (η - 0.081): 줄무늬 κ=0 섹터의 최대 비중립 고유값 (8회·L·H)
etas = (0.082, 0.085, 0.09, 0.10, 0.11, 0.12, 0.13)
fits = {}
for case in ("oct", "ctrlL", "ctrlH"):
    ys = []
    for eta in etas:
        par, s0c, k0, (c0, ck), lat = setup(case, eta)
        X = seed_state(c0, A_SAT, ab_frac=0.0, rng=np.random.default_rng(3)); X, r, _ = newton(c0, X)
        J = c0.jacobian(X[:4], c0); w = np.linalg.eigvals(J)
        nz = w.real[np.abs(w) > 1e-3 * s0c]
        ys.append(nz.max() / s0c)
    p = np.polyfit(etas, ys, 1); fits[case] = (p, ys)
    print(case, "rhombus growth/σ0:", np.round(ys, 3), "fit slope %.2f, zero at η=%.4f" % (p[0], -p[1] / p[0]), flush=True)
    add(f"rhombus growth slope d(lambda/s0)/d(eta) [{case}]", 33.5, p[0], 1.0)
    add(f"rhombus growth zero crossing eta_r [{case}]", 0.081, -p[1] / p[0], 0.001)

# 4. 1:2 여유 σ0(K*)/σ0, K* = (2/ρ)² q0
base = Params(Dv=DV)
for name, rho, ms in (("12-fold", 2 * np.cos(np.pi / 12), -0.00), ("10-fold", 2 * np.cos(np.pi / 10), -1.13), ("8-fold", 2 * np.cos(np.pi / 8), -4.44), ("ctrlL 1.82", 1.82, -6.82), ("ctrlH 1.87", 1.87, -2.88)):
    Ks = (2 / rho) ** 2 * q0
    add(f"1:2 margin sigma0(K*)/sigma0 [{name}]", ms, float(growth_rate(base, 0, Ks)) / s0, 0.02, "원고 값은 Dv=2.50309 (−4.429 등) 와 Dv/Dv_c=1.00123 정확히 (−4.456 등) 사이: Dv 반올림 차이")

# 5. A2a/Aa² (정상파형, η=0.10)
for case, ms in (("ctrlL", 10.58), ("oct", 15.15), ("ctrlH", 17.52), ("dodec", 27.23)):
    d = json.load(open(f"results_v3/stageA/{case}_eta0.1.json"))
    st = [s for s in d["states"] if s["info"]["type"] == "standing"][0]["info"]
    add(f"A2a/Aa^2 standing eta=0.10 [{case}]", ms, st["A2a"] / st["Aa"] ** 2, 0.01)

# 6. 8회 가지 (v4 결과 파일)
B = [r for r in csv.DictReader(open("results_v4/branches.csv")) if r["state"].startswith("N8-S")]
etas_b = [float(r["eta"]) for r in B]; lam_b = [float(r["max_nonneutral_eig_over_s0"]) for r in B]
add("N8-S lowest eta with converged state (fold ~0.113)", 0.113, min(etas_b), 0.002, "Newton failed at 0.112 (REPORT_v4)")
i = [k for k in range(len(B) - 1) if lam_b[k] > 0 >= lam_b[k + 1]][-1]
eta_c = etas_b[i] + (0 - lam_b[i]) * (etas_b[i + 1] - etas_b[i]) / (lam_b[i + 1] - lam_b[i])
add("N8-S stability boundary eta_c (linear interp of CSV)", 0.1364, eta_c, 0.0005, "finer scan in REPORT_v4: 0.1364+-0.0002")
R = json.load(open("results_v4/rhombus_ref.json"))
get = lambda e, t: [r for r in R if r["eta"] == e and r["type"] == t][0]
add("standing rhombus eta=0.14 max eigenvalue over cosets (stable)", -0.0467, get(0.14, "standing")["coset_max"], 5e-4)
add("standing rhombus eta=0.15 sideband eigenvalue (unstable)", 0.116, get(0.15, "standing")["coset_max"], 1e-3)

# 7. v6 수치
import ast
C = list(csv.DictReader(open("results_v6/coeffs.csv")))
Rs = [float(r["R"]) for r in C if r["incl12"] == "False" and r["type"] == "standing" and r["branch"] == "1" and float(r["R"]) > 0]
add("v6 standing R min (branch 1)", 0.059, min(Rs), 0.0006)
add("v6 standing R max (branch 1)", 0.092, max(Rs), 0.0006)
K = [r for r in csv.DictReader(open("results_v6/knob.csv")) if r["exists"] == "True"]
add("v6 knob max R_secant", 0.657, max(float(r["R_secant"]) for r in K if r["R_secant"] not in ("nan", "")), 0.001)
# g_cc'/g 와 Q 비 (줄무늬 + 섹터 2차 결합)
from ring_sim import fillin as F
from ring_sim.model import nonlinear_u
par, s0c, k0, (c0, ck), lat = setup("oct", 0.10)
X = seed_state(c0, A_SAT, ab_frac=0.0, rng=np.random.default_rng(3)); X, r, _ = newton(c0, X)
J = ck.jacobian(X[:4], c0); U = F.sector_transform(ck.M); Jt = U @ J @ U.conj().T
jc = ck.lookup[(1, 0)]; Rr, Lf = F.basis_vectors(ck, par, [(jc, 0)])
lam, M, parts, M0 = F.self_consistent(Jt, Rr, Lf)
sc = F.top_pair(par, 0, c0.q[c0.lookup[(1, 1)]])[0]; scp = F.top_pair(par, 0, ck.q[jc])[0]
add("g_cc'/g (8-fold, eta=0.10)", 2.009, (scp - lam) / sc, 0.0006)
pq = Params(Dv=DV, eta=0.1, etap=-0.05, c=0.0); n = np.arange(4); e = [np.exp(1j * np.pi * p * n / 2) for p in range(4)]
Bf = lambda x, y: nonlinear_u(x + y, pq) - nonlinear_u(x, pq) - nonlinear_u(y, pq)
add("Q_{1x3->0}/Q_{0x1->1}", 2.000, float((np.vdot(e[0], Bf(e[1], e[3])) / np.vdot(e[1], Bf(e[0], e[1]))).real), 5e-4)

os.makedirs("results_v7", exist_ok=True)
w = csv.DictWriter(open("results_v7/E_numbers.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("DIFF items:", [r["item"] for r in rows if r["status"] == "DIFF"])
