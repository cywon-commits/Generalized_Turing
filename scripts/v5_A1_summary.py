"""A1_summary.csv + A1_peaks.json: 실행별 최종 상태 (피크 수, 포획률, 정상성, 피크 목록, A2 닫힘과의 일치)."""
import json, glob, os, csv, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.invariant import Rings, closure
rows, peaks_out = [], {}
for fn in sorted(glob.glob("results_v5/timeseries/*.json")):
    d = json.load(open(fn)); name = os.path.basename(fn)[:-5]
    s = d["series"]; n = len(s); q = s[int(0.75 * n):]
    N0 = [r["N0"] for r in q]; Nin = [r["Nin"] for r in q]
    z = np.load(f"runs_v5/{name}.final.npz"); u = z["u"]; N = u.shape[1]
    F = np.fft.fft2(u, axes=(1, 2)) / N ** 2
    P = (np.abs(F) ** 2).sum(0); P[0, 0] = 0; tot = P.sum()
    Cx = d["Cx"]; rho = {"oct": 2 * np.cos(np.pi / 8), "ctrlL": 1.82, "ctrlH": 1.87}[d["case"]]
    vecs = []
    for i, j in zip(*np.nonzero(P > 0.003 * P.max())):
        kx = i if i <= N // 2 else i - N; ky = j if j <= N // 2 else j - N
        r = np.hypot(kx, ky) / Cx  # k0 단위
        if not (abs(r - 1) < 0.1 or abs(r - 1 / rho) < 0.1):  # 임계원 위 피크만 (고조파·저주파 제외)
            continue
        if (ky > 0) or (ky == 0 and kx > 0):
            vecs.append((int(kx), int(ky), float(2 * P[i, j] / tot)))
    vecs.sort(key=lambda v: -v[2])
    # 정상성: 마지막 25% 에서 dP 평균, N0/Nin 범위
    dP = float(np.mean([r["dP"] for r in q]))
    sat = dP < 1e-4 and (max(Nin) == min(Nin)) and (max(N0) == min(N0))
    R = Rings(rho, eta=d["eta"]); dk = R.k0 / Cx
    allv = [(v[0], v[1]) for v in vecs]
    cl = closure(allv, R, dk, max_depth=6, max_size=3000)
    inK = {tuple(v) for v in d["capture_set"]}
    outK = [v for v in allv if v not in inK and (-v[0], -v[1]) not in inK]
    rows.append(dict(name=name, case=d["case"], eta=d["eta"], ic=d["ic"], seed=d["seed"], N=N, T=d["T"], N0_final=s[-1]["N0"], Nin_final=s[-1]["Nin"],
                     N0_range=f"{min(N0)}-{max(N0)}", Nin_range=f"{min(Nin)}-{max(Nin)}", capture_final=round(s[-1]["capture"], 4),
                     dP_last25=f"{dP:.2e}", saturated=bool(sat), n_peaks_0p3pct=len(vecs), closure_sizes="/".join(map(str, cl["sizes"])),
                     closure_closed=cl["closed"], n_peaks_outside_A2set=len(outK)))
    peaks_out[name] = dict(peaks=[dict(k=[v[0], v[1]], kk0=float(np.hypot(v[0], v[1]) / Cx), ang=float(np.degrees(np.arctan2(v[1], v[0]))), frac=v[2]) for v in vecs[:30]])
w = csv.DictWriter(open("results_v5/A1_summary.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(peaks_out, open("results_v5/A1_peaks.json", "w"), indent=1)
for r in rows: print({k: r[k] for k in ("name", "N0_final", "Nin_final", "capture_final", "dP_last25", "saturated", "n_peaks_0p3pct", "closure_sizes", "n_peaks_outside_A2set")})
