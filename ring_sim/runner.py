"""설정 파일(YAML/JSON) 로 한 실행을 수행. 체크포인트(.npz) 로 이어 돌리기 가능.

사용:  python -m ring_sim.runner config.yaml
"""
import json
import os
import sys
import time
import numpy as np
from .model import Params, peak, solve_kappa, RATIOS
from .integrator import Solver
from .initcond import make_state
from .diagnostics import Diag


def load_config(path):
    with open(path) as f:
        txt = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def build(cfg, workers=1):
    base = Params(Dv=cfg.get("Dv", 2.55))
    ratio = cfg.get("ratio", RATIOS[cfg["case"]])
    r = cfg.get("r", 1.0)
    if "ku" in cfg and "kv" in cfg:
        ku, kv = cfg["ku"], cfg["kv"]
    elif r == 0:  # δ = 0 기준선: κ_v/κ_u = D_v/D_u, S 는 비율에 맞춤
        ku, kv = solve_kappa(ratio, 0.0)[0], None
        # δ=0 -> kv = ku*Dv/Du ; S=Dv ku + Du kv 를 맞춰 q1 이동을 결정
        from scipy.optimize import brentq
        s0, q0 = peak(base, 0)
        qt = (np.sqrt(q0) / ratio) ** 2

        def f(x):
            return peak(Params(ku=x, kv=x * base.Dv / base.Du), 1)[1] - qt
        ku = brentq(f, 0.0, 0.2)
        kv = ku * base.Dv / base.Du
    else:
        ku, kv = solve_kappa(ratio, r)
    par = Params(eta=cfg.get("eta", 0.4), ku=ku, kv=kv)
    s0, q0 = peak(base, 0)
    k0 = np.sqrt(q0)
    R0 = cfg.get("R0", 56)
    N = cfg.get("N", 512)
    L = 2 * np.pi * R0 / k0
    S = Solver(par, N=N, L=L, dt=cfg.get("dt", 1.0), workers=workers)
    return par, S, ratio, R0, k0, s0


def run(cfg, workers=1, log=print):
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    name = cfg["name"]
    ck = os.path.join(out, name + ".ckpt.npz")
    par, S, ratio, R0, k0, s0 = build(cfg, workers)
    diag = Diag(S, ratio, R0, k0, tri_w=cfg.get("tri_w", 1.5))
    T = cfg["T"]
    rec_dt = cfg.get("rec_dt", 20.0)
    ckpt_dt = cfg.get("ckpt_dt", 500.0)
    rec_every = max(1, int(round(rec_dt / S.dt)))
    ck_every = max(1, int(round(ckpt_dt / S.dt)))
    nsteps = int(round(T / S.dt))
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        V, step = z["V"], int(z["step"])
        series = list(z["series"])
        info = json.loads(str(z["info"]))
        log(f"[{name}] resume at step {step}")
    else:
        V, info = make_state(S, cfg["case"], ratio, cfg["ic"], cfg["seed"], R0, k0)
        step, series = 0, []
        series.append(diag.record(V, 0.0))
    t0 = time.time()
    while step < nsteps:
        V = S.step(V)
        step += 1
        if step % rec_every == 0:
            series.append(diag.record(V, step * S.dt))
        if step % ck_every == 0 or step == nsteps:
            np.savez(ck + ".tmp.npz", V=V, step=step, series=np.array(series, dtype=object),
                     info=json.dumps(info))
            os.replace(ck + ".tmp.npz", ck)
            log(f"[{name}] step {step}/{nsteps} t={step * S.dt:.0f} Pin={series[-1]['Pin']:.3e} "
                f"P0={series[-1]['P0']:.3e} elapsed {time.time() - t0:.0f}s")
    fin = diag.final_analysis(V)
    res = dict(cfg=cfg, info=info, final=fin, kappa=[par.ku, par.kv], sigma0=s0,
               ntri_lattice=diag.ntri_lattice, series=series)
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # 최종 장 저장 (그림용): 층 0 의 u, 섹터장 m0 및 압축 스펙트럼
    np.savez_compressed(os.path.join(out, name + ".final.npz"), u=S.to_real(V[0]).astype(np.float32),
                        V=V.astype(np.complex64))
    return res


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    run(cfg)
