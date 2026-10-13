"""v4: 상태 가지의 η 연속 추적. 사용: python scripts/v4_branch.py STATE eta_start eta_end step [sym]
STATE: n8s (닫힌 8회 정상파형), c8_2, c8_3 (열린 사슬), r (마름모 기준), n10s, n10h, n6s, n6h ...
각 점: Newton-Krylov (대칭 사영 허용) -> 전체 공간 Arnoldi (대칭 제약 없음) -> JSON."""
import sys, time, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stageA import DV, A_SAT
from v4_probe import make
from ring_sim.model import Params

LAT = json.load(open("results_v4/network_lattices.json"))


def lattice(n):
    if os.environ.get("CX"):  # v5: Pell 근사체 (n=8 만): CX=10,24,58 -> 안쪽 (p,q), p=CX/2, q=round(p tan(π/8))
        from ring_sim.network import build_from_params, outer_from_inner, errors
        cx = int(os.environ["CX"]); p = cx // 2; q = int(round(p * np.tan(np.pi / 8)))
        I = build_from_params(8, (p, q)); c = outer_from_inner(I)
        return I, c, cx, int(os.environ.get("NGRID", {10: 96, 24: 240, 58: 576}[cx]))
    d = LAT[str(n)]["best"]
    return [np.array(v) for v in d["inner"]], [np.array(v) for v in d["outer"]], d["Cx"], LAT[str(n)]["N"]


def state_spec(name):
    """상태 이름 -> (n, 바깥 선택 j 목록, 안쪽 선택 m 목록, 대칭, 섹터 배정)"""
    if name == "n8s":
        return 8, [0, 1, 2, 3], [0, 1, 2, 3], "D4L", "standing"
    if name == "c8_2":
        return 8, [1, 2], [0, 1, 2], None, "standing"      # 사슬: c1=I0+I1, c2=I1+I2
    if name == "c8_3":
        return 8, [1, 2, 3], [0, 1, 2, 3], "MxL", "standing"
    if name == "c8h3":   # 나선형 교대 배정으로 섹터 좌절된 열린 사슬 (c0, c4 삼각형이 비공명)
        return 8, [1, 2, 3], [0, 1, 2, 3], "Mx*L", "spiral_alt"
    if name == "c8h2":
        return 8, [1, 2], [0, 1, 2], None, "spiral_alt"
    if name == "r8":
        return 8, [1], [0, 1], None, "standing"            # 마름모 (c1 = I0 + I1)
    if name == "n10s":
        return 10, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], "Mx+My+L", "standing"
    if name == "n10h":
        return 10, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], "Mx+My*L", "spiral_alt"
    if name == "n6s":
        return 6, [0, 1, 2], [0, 1, 2], "Mx+My+L", "standing"
    if name == "n6h":
        return 6, [0, 1, 2], [0, 1, 2], "Mx+My*L", "spiral_alt"
    raise ValueError(name)


def describe(net, vecr, inner, outer, k0, top=5):
    """실수 고유벡터 -> 격자 벡터별 가중치 (δu 의 층 평균 푸리에 세기) 상위 모드"""
    d = vecr.reshape(net.shape)[0]
    N = net.N
    F = np.fft.fft2(d, axes=(1, 2)) / (N * N)
    pw = (np.abs(F) ** 2).sum(0)
    tot = pw.sum()
    idx = np.argsort(pw.ravel())[::-1][:top * 2]
    out = []
    seen = set()
    for f in idx:
        i, j = divmod(int(f), N)
        kx = i if i <= N // 2 else i - N
        ky = j if j <= N // 2 else j - N
        key = tuple(sorted([(kx, ky), (-kx, -ky)]))
        if key in seen:
            continue
        seen.add(key)
        r = np.hypot(kx, ky) * net.dk / k0
        out.append(dict(k=(kx, ky), kk0=float(r), w=float(2 * pw[i, j] / tot)))
        if len(out) >= top:
            break
    return out


def amplitudes(net, inner, outer, n):
    mo = net.modes(None if False else net_x[0], [tuple(v) for v in outer[: n // 2]])
    return mo


if __name__ == "__main__":
    name = sys.argv[1]
    tag = os.environ.get('TAG', '')
    e0, e1, de = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    sym_over = sys.argv[5] if len(sys.argv) > 5 else None
    n, outs, ins, sym, sector = state_spec(name)
    sym = None if sym_over == "none" else (sym_over or sym)
    inner, outer, Cx, N = lattice(n)
    rho = 2 * np.cos(np.pi / n)
    etas = list(np.round(np.arange(e0, e1 - 1e-9 if de < 0 else e1 + 1e-9, de), 4))
    os.makedirs("results_v4/stages", exist_ok=True); os.makedirs("runs_v4", exist_ok=True)
    x_prev = np.load(os.environ['START_X']) if os.environ.get('START_X') else None
    results = []
    for eta in etas:
        net, s0 = make(rho, Cx, N, eta)
        net.set_sym(sym)
        t = time.time()
        if x_prev is None and os.environ.get("FROM_NET"):
            # 닫힌 연결망 해에서 바깥(과 고립된 안쪽) 모드를 제거한 씨앗
            xn = np.load(os.environ["FROM_NET"]).reshape(net.shape)
            drop = []
            for j in range(n // 2):
                if j not in outs:
                    drop.append(tuple(outer[j]))
            for m in range(n // 2):
                if m not in ins:
                    drop.append(tuple(inner[m]))
            F = np.fft.fft2(xn, axes=(2, 3))
            for v in drop:
                for sgn in (1, -1):
                    F[:, :, (sgn * v[0]) % N, (sgn * v[1]) % N] = 0
            x = np.fft.ifft2(F, axes=(2, 3)).real.reshape(-1)
            x = net.relax(x, float(os.environ.get('RELAX_T', 6e4)))
        elif x_prev is None:
            rng = np.random.default_rng(1)
            u = np.zeros((4, N, N))
            for j in outs:
                arg = 2 * np.pi * (outer[j][0] * net.X + outer[j][1] * net.Y) / net.L
                u += A_SAT * 0.45 * np.cos(arg)[None]
            for m in ins:
                if sector == "standing":
                    u += net.mode_field(inner[m], 0.25 * A_SAT, 0.0, 1, 1)
                else:  # 나선형 교대: s = 1, 3, 1, 3, ...
                    u += net.mode_field(inner[m], 0.5 * A_SAT, 0.0, 1 if m % 2 == 0 else 0, 0 if m % 2 == 0 else 1)
            V = net.v_from_u(u); x = net.to_vec(net.S.to_real(V[0]), net.S.to_real(V[1]))
            x = net.relax(x, 6e4)
        else:
            x = x_prev
        if x_prev is None:
            np.save(f"runs_v4/{name}{tag}_relaxed_eta{eta}.npy", x)
            mo = net.modes(x, [tuple(v) for v in outer[: n // 2]]); mi = net.modes(x, [tuple(v) for v in inner[: n // 2]])
            print(f"relaxed state η={eta}: outer|F0|={['%.2e' % a[0] for a in mo]} inner(p1,p3)={['(%.1e,%.1e)' % (a[1], a[3]) for a in mi]}", flush=True)
        try:
            x, res = net.newton(x, f_tol=1e-10, maxiter=15, inner_maxiter=30)
        except Exception as ex:
            res = float("nan"); print(f"η={eta}: Newton-Krylov failed ({type(ex).__name__})", flush=True)
            results.append(dict(eta=float(eta), converged=False)); break
        x = net.project_vec(x)
        ok = res < 1e-9
        mo = net.modes(x, [tuple(v) for v in outer[: n // 2 + (1 if n == 10 else 0)]][: n])  # 바깥 모드 (p=0..3)
        mi = net.modes(x, [tuple(v) for v in inner[: n // 2 + (0)]])
        rec = dict(eta=float(eta), converged=bool(ok), residual=float(res), outer_p0=[float(a[0]) for a in mo], inner_p1=[float(a[1]) for a in mi], inner_p3=[float(a[3]) for a in mi],
                   outer_modes=[tuple(map(int, v)) for v in outer[: len(mo)]], inner_modes=[tuple(map(int, v)) for v in inner[: len(mi)]])
        # 2k 모드 (p=0) 진폭
        m2 = net.modes(x, [tuple(2 * v) for v in inner[: n // 2]])
        rec["two_k_p0"] = [float(a[0]) for a in m2]
        if ok:
            net.set_sym(None)
            lam, w, Vv = net.eigenvalues(x, k=10, tol=1e-5)
            rec["eig"] = [float(l) for l in lam]
            nn = [l for l in lam if abs(l) > 2e-3]
            rec["max_nonneutral"] = float(max(nn)) if nn else None
            if rec["max_nonneutral"] is not None and rec["max_nonneutral"] > 1e-3:
                i0 = int(np.argmax([l if abs(l) > 2e-3 else -1e9 for l in lam]))
                rec["unstable_mode"] = describe(net, np.real(Vv[:, i0]), inner, outer, net.k0)
            x_prev = x
            np.save(f"runs_v4/{name}{tag}_eta{eta}.npy", x)
        rec["time_s"] = time.time() - t
        results.append(rec)
        print(f"η={eta}: res={res:.1e} ok={ok} outer={['%.2e' % a for a in rec['outer_p0']]} inner(p1)={['%.1e' % a for a in rec['inner_p1']]} maxλ={rec.get('max_nonneutral')} eig={['%.3f' % l for l in rec.get('eig', [])[:6]]} [{rec['time_s']:.0f}s]", flush=True)
        json.dump(results, open(f"results_v4/stages/branch_{name}{tag}_{e0}_{e1}.json", "w"))
        if not ok:
            break
