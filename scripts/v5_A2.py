"""v5 A2: 사례별 깊이에 따른 불변 집합 크기 -> results_v5/A2_closure.json
시드: R = 마름모 (a,b);  T = (a,b,b') (v3 정확 격자의 3파);  Q = 이상 8회 안쪽 파 4개 (22.5°+45°·j, j=0..3) 를 격자로 반올림.
영역 배율 m: 영역을 정확 격자의 m배로 키움 (dk = k0/(Cx*m)).  허용 tol: 성장률 기준 tol*σ0."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.invariant import Rings, closure

LAT = json.load(open("results_v3/exact_lattices.json"))
CASES = {"oct": 2 * np.cos(np.radians(22.5)), "ctrlL": 1.82, "ctrlH": 1.87}
out = {}
for case, rho in CASES.items():
    sol = LAT[case]["solutions"][0]; Cx = LAT[case]["Cx"]
    a = np.array(sol["a"]); b = np.array([a[0], -a[1]]); bp = np.array(sol["bprime"])
    for tol in (0.1, 0.25, 0.5):
        R = Rings(rho, tol=tol)
        for m in (1, 2, 3):
            dk = R.k0 / (Cx * m)
            k1u = R.k1 / dk
            Q = [tuple(int(round(k1u * f(np.radians(22.5 + 45 * j)))) for f in (np.cos, np.sin)) for j in range(4)]
            for name, seed in (("R", [m * a, m * b]), ("T", [m * a, m * b, m * bp]), ("Q", Q)):
                c = closure([tuple(map(int, s)) for s in seed], R, dk, max_depth=6, max_size=3000)
                out[f"{case}|tol{tol}|m{m}|{name}"] = dict(case=case, tol=tol, m=m, seed=name, seed_vecs=[list(map(int, s)) for s in seed],
                                                          Cx=Cx * m, sizes=c["sizes"], closed=c["closed"], n_out=c["n_out"], n_in=c["n_in"])
                print(case, tol, m, name, c["sizes"], c["closed"], flush=True)
os.makedirs("results_v5", exist_ok=True)
json.dump(out, open("results_v5/A2_closure.json", "w"), indent=1)
