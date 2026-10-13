"""v4: n회 연결망 (닫힌 순환) 과 열린 사슬의 정확 격자와 씨앗.

안쪽 파 I_m (각도 θ_I + 360°·m/n, m=0..n-1; I_{m+n/2} = -I_m), 바깥 파 c_j = I_{j-1} + I_j (각도 360°·j/n).
정확 격자: 모든 I_m, c_j 가 정수 벡터. 대칭(x축 반사)으로 독립 매개변수만 탐색한다."""
import itertools
import numpy as np
from scipy.fft import next_fast_len


def _ang(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_from_params(n, P):
    """반사 대칭을 갖는 안쪽 벡터 I_0..I_{n-1} 구성."""
    if n == 8:
        p, q = P
        base = [(p, q), (q, p), (-q, p), (-p, q)]
    elif n == 10:
        p, q, r, t, s = P
        base = [(p, q), (r, t), (0, s), (-r, t), (-p, q)]
    elif n == 6:
        p, q, s = P
        base = [(p, q), (0, s), (-p, q)]
    else:
        raise ValueError(n)
    I = [np.array(v) for v in base] + [-np.array(v) for v in base]
    return I


def outer_from_inner(I):
    n = len(I)
    return [I[j - 1] + I[j] for j in range(n)]  # c_j = I_{j-1} + I_j (j=0 은 I_{-1}=I_{n-1})


def errors(n, I, c):
    """이상값 대비 오차: 안쪽/바깥 각도 오차(도), 반지름 상대오차. 기준: |c_0| = k0."""
    k0 = np.hypot(*c[0])
    th = np.degrees(np.arccos(2 * np.cos(np.pi / n) / 2 * 1.0)) if False else None
    rho = 2 * np.cos(np.pi / n)
    k1 = k0 / rho
    ang_err = []
    rad_err_out, rad_err_in = [], []
    for j in range(n):
        ang_err.append(_angdiff(_ang(c[j]), 360.0 * j / n))
        rad_err_out.append(abs(np.hypot(*c[j]) / k0 - 1))
        ang_err.append(_angdiff(_ang(I[j]), 360.0 * (j + 0.5) / n - 360.0 / n * 0 + 0.0 - 0.0) if False else 0)
    for m in range(n):
        ideal = 180.0 / n + 360.0 * m / n  # 이상 각도: θ_I = 180°/n (c_0 = I_{-1}+I_0 가 0° 이려면 I_0 = +180/n°)
        ang_err.append(_angdiff(_ang(I[m]), ideal))
        rad_err_in.append(abs(np.hypot(*I[m]) / k1 - 1))
    return max(ang_err), max(rad_err_out), max(rad_err_in)


def search_network(n, tol_deg=1.0, tol_out=0.011, tol_in=0.04, cx_max=80, n_keep=5):
    sols = []
    rho = 2 * np.cos(np.pi / n)
    for p in range(2, cx_max // 2 + 1):
        Cx = 2 * p
        R1 = Cx / rho
        q0 = p * np.tan(np.pi / n)
        qs = [q for q in range(1, p + 1) if abs(q - q0) < 1.6]
        for q in qs:
            if n == 8:
                cands = [((p, q),)]
            elif n == 6:
                cands = [((p, q, s),) for s in range(int(R1 * 0.95), int(R1 * 1.05) + 2) if s > 0]
            else:
                cands = []
                for s in range(int(R1 * 0.95), int(R1 * 1.05) + 2):
                    for r in range(1, int(R1) + 2):
                        for t in range(1, int(R1 * 1.05) + 2):
                            if abs(np.hypot(r, t) / R1 - 1) <= tol_in and _angdiff(_ang((r, t)), 54.0) <= tol_deg:
                                cands.append(((p, q, r, t, s),))
            for (Pp,) in cands:
                P = Pp if n != 8 else (p, q)
                I = build_from_params(n, P)
                c = outer_from_inner(I)
                a, ro, ri = errors(n, I, c)
                if a <= tol_deg and ro <= tol_out and ri <= tol_in:
                    sols.append(dict(n=n, Cx=Cx, params=[int(x) for x in P], max_ang_err=float(a), rel_out_err=float(ro), rel_in_err=float(ri),
                                     inner=[[int(x) for x in v] for v in I], outer=[[int(x) for x in v] for v in c]))
        if sols:
            break
    sols.sort(key=lambda s: s["max_ang_err"] / 1.0 + 100 * max(s["rel_out_err"], s["rel_in_err"]))
    return sols[:n_keep]


def n_grid(Cx, ppw=9.14):
    return int(next_fast_len(int(np.ceil(ppw * Cx))))
