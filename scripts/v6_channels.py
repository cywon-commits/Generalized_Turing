"""v6 채널 분해: X (α 의 s 기울기), Y (β 의 기울기) 에 기여하는 Q 채널 (모드 (m,n), 섹터 p) 상위 5개.
줄무늬 (s=0) 와 마름모 (s*) 에서 같은 P (c′, b′) 로 2차 기여를 채널별로 계산하고 차이를 s 로 나눈다."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import setup, A_SAT, CASES
from ring_sim.bloch import find_states, seed_state, newton, classify_state
from ring_sim import fillin as F
import csv


def contribs(c0, ck, par, X0, kind, k0):
    J = ck.jacobian(X0[:4], c0)
    Jt = F.sector_transform(ck.M) @ J @ F.sector_transform(ck.M).conj().T
    jc = ck.lookup[(1, 0)]; jb = ck.lookup[(0, 0)]
    bps = [1, 3] if kind in ("standing", "stripe") else [1]
    picks = [(jc, 0)] + [(jb, p) for p in bps]
    R, Lf = F.basis_vectors(ck, par, picks)
    lam, M, parts, M0 = F.self_consistent(Jt, R, Lf)
    co = F.coefficients(M, 0, list(range(1, 1 + len(bps))))
    r, l = co["_rl"]; d = len(picks)
    eA = np.zeros(d, complex); eA[0] = 1
    lB = np.zeros(d, complex); rB = np.zeros(d, complex); lB[1:] = l; rB[1:] = r
    dA, chA, fA = F.channel_second_order(Jt, R, Lf, lam, eA, eA)
    dB, chB, fB = F.channel_second_order(Jt, R, Lf, lam, lB, rB)
    return dict(lam=lam, alpha=co["alpha"], beta=co["beta"], direct_A=dA, direct_B=dB, chA=chA, chB=chB, sumA=fA, sumB=fB)


def run(case, eta, rows):
    par, s0, k0, (c0, ck), lat = setup(case, eta)
    ck._k1 = k0 / CASES[case]
    Xs = seed_state(c0, A_SAT, ab_frac=0.0, rng=np.random.default_rng(3)); Xs, r, _ = newton(c0, Xs)
    cs = contribs(c0, ck, par, Xs, "stripe", k0)
    seen = set()
    for f in find_states(c0, A_SAT, T=3e5):
        t = f["info"]["type"]
        if t not in ("standing", "spiral") or t in seen:
            continue
        seen.add(t)
        s = f["info"]["Aa"] ** 2
        kind = "standing" if t == "standing" else "p1"
        cr = contribs(c0, ck, par, f["X"], kind, k0)
        for lab, dir_key, ch_key in (("X", "direct_A", "chA"), ("Y", "direct_B", "chB")):
            sgn = -1.0
            items = []
            keys = set(cr[ch_key]) | set(cs[ch_key])
            for k in keys:
                dv = (cr[ch_key].get(k, 0) - cs[ch_key].get(k, 0))
                items.append((abs(dv), k, sgn * dv.real / s / s0))
            items.sort(reverse=True)
            tot = sgn * ((cr[dir_key] - cs[dir_key]).real + sum((cr[ch_key].get(k, 0) - cs[ch_key].get(k, 0)).real for k in keys)) / s / s0
            for rank, (_, k, val) in enumerate(items[:5], 1):
                j, p = k
                rows.append(dict(case=case, eta=eta, type=t, coef=lab, rank=rank, m=int(ck.m[j]), n=int(ck.n[j]), k_over_k0=float(np.sqrt(ck.q[j]) / k0), sector=p,
                                 contribution_over_s0_per_s=val, direct_part=sgn * (cr[dir_key] - cs[dir_key]).real / s / s0, total=tot, s=float(s)))
    return rows


if __name__ == "__main__":
    rows = []
    for case in ("oct", "ctrlL", "ctrlH"):
        for eta in (0.10, 0.13):
            run(case, eta, rows); print(case, eta, len(rows), flush=True)
    w = csv.DictWriter(open("results_v6/channels.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
