"""보조 실험: 바깥 원의 초임계도(D_v)를 낮추어 3차 교차포화 대 매개변수 구동의 비를 본다.
작은 영역(128², 바깥 파장 14개). 안쪽 원의 선형 감쇠/성장률 Λ_in 을 깨끗한 지수 구간에서 측정."""
import itertools, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.runner import run
from ring_sim.model import Params, peak

def job(a):
    Dv, case, r, eta, seed = a
    s0 = peak(Params(Dv=Dv), 0)[0]
    T = float(min(12000, max(3000, 25 / s0)))
    cfg = dict(name=f"dv{Dv}_{case}_r{r}_eta{eta}_s{seed}", case=case, r=r, eta=eta, ic="O", seed=seed, Dv=Dv,
               N=128, R0=14, dt=1.0, T=T, rec_dt=20.0, ckpt_dt=1000.0, outdir="runs/dvscan")
    if os.path.exists(f"runs/dvscan/{cfg['name']}.json"):
        d = json.load(open(f"runs/dvscan/{cfg['name']}.json"))
    else:
        d = run(cfg, log=lambda s: None)
    s = d["series"]; t = np.array([x["t"] for x in s]); p = np.array([x["Pin"] for x in s])
    P0 = np.array([x["P0"] for x in s])
    return (Dv, case, r, eta, seed, s0, float(P0[-1]), t, p)

if __name__ == "__main__":
    jobs = list(itertools.product([2.55, 2.525, 2.5125], ["dodec"], [0.5], [0.5], [1, 2]))
    with Pool(2) as pl:
        res = pl.map(job, jobs)
    for Dv, case, r, eta, seed, s0, P0, t, p in res:
        m = (p < 1e-9) & (p > 1e-33) & (t > 200)
        lam = np.polyfit(t[m], np.log(p[m]), 1)[0] if m.sum() > 5 else float("nan")
        late = np.polyfit(t[t > t[-1] * 0.6], np.log(p[t > t[-1] * 0.6] + 1e-300), 1)[0]
        print(f"Dv={Dv} σ0={s0:.5f} seed{seed} P0={P0:.2e} Pin_max={p.max():.1e} Pin_end={p[-1]:.1e} "
              f"Λ_clean={lam:+.4f} (2σ1={-2*r*s0:+.4f}) Λ_late={late:+.4f}", flush=True)
