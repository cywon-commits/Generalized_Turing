"""v6: fill-in 부분공간의 Schur 보수 유효 행렬과 판정식 계수 추출.

층 푸리에 섹터 기저 (모드 j, 종 f∈{u,v}, 섹터 p): 선형 연산자는 (j, p) 마다 2x2 블록 L_p(q) 로 대각화된다.
인덱스 (j, f, p) -> j*8 + 4*f + p.  P 기저는 선택한 (j, p) 의 2x2 최대 성장 고유벡터 (우·좌, 쌍직교 정규화 l^H r = 1)."""
import numpy as np
import scipy.linalg as sla
from .model import Params

NL = 4


def dft4():
    n = np.arange(NL)
    return np.exp(-1j * np.pi * np.outer(np.arange(NL), n) / 2) / 2.0  # (p, n); 유니터리 (행 정규직교)


def sector_transform(M, nc=8):
    F = dft4()
    B = np.zeros((nc, nc), complex)
    for k in range(nc // 4):
        B[4 * k:4 * k + 4, 4 * k:4 * k + 4] = F
    return np.kron(np.eye(M), B)  # (8M, 8M) 유니터리: 층 기저 -> 섹터 기저


def Lsec(par: Params, p, q):
    """섹터 p, 파수제곱 q 의 2x2 선형 블록 (u,v)"""
    c = 2 * np.cos(np.pi * p / 2)
    return np.array([[par.a - par.Du * q - 2 * par.ku + par.ku * c, -par.b],
                     [1.0, -par.d - par.Dv * q - 2 * par.kv + par.kv * c]])


def top_pair(par, p, q):
    """2x2 블록의 최대 성장 고유값과 쌍직교 정규화된 우·좌 고유벡터"""
    w, vl, vr = sla.eig(Lsec(par, p, q), left=True, right=True)
    i = int(np.argmax(w.real))
    r, l = vr[:, i], vl[:, i]
    l = l / np.conj(l @ r) if False else l / (np.vdot(l, r))  # l^H r = 1
    return w[i].real, r, l


def basis_vectors(cell, par, picks, nc=8):
    """picks: [(모드 인덱스 j, 섹터 p), ...] -> 우벡터 R (8M, d), 좌벡터 Lf (8M, d) (섹터 기저)"""
    d = len(picks)
    R = np.zeros((nc * cell.M, d), complex)
    Lf = np.zeros_like(R)
    for k, (j, p) in enumerate(picks):
        _, r, l = top_pair(par, p, cell.q[j])
        R[nc * j + p, k], R[nc * j + 4 + p, k] = r
        Lf[nc * j + p, k], Lf[nc * j + 4 + p, k] = l
    return R, Lf


def schur(Jt, R, Lf, lam=0.0):
    """M_eff(λ) = J_PP + J_PQ (λ - J_QQ)^{-1} J_QP.  (섹터 기저의 J, 쌍직교 P 기저)"""
    n, d = R.shape
    Qr = sla.null_space(Lf.conj().T)  # l^H Qr = 0
    V = np.concatenate([R, Qr], 1)
    T = np.linalg.inv(V)
    Jc = T @ Jt @ V
    Jpp, Jpq, Jqp, Jqq = Jc[:d, :d], Jc[:d, d:], Jc[d:, :d], Jc[d:, d:]
    G = np.linalg.solve(lam * np.eye(n - d) - Jqq, Jqp)
    return Jpp + Jpq @ G, (Jpp, Jpq, Jqp, Jqq, V, T)


def self_consistent(Jt, R, Lf, lam0=None, iters=1000, tol=1e-13):
    """λ = max Re eig M_eff(λ) 의 고정점. J_QQ 를 한 번만 고유분해해 λ 마다 O(d²n) 로 M_eff 를 계산."""
    n, d = R.shape
    Qr = sla.null_space(Lf.conj().T)
    V = np.concatenate([R, Qr], 1)
    T = np.linalg.inv(V)
    Jc = T @ Jt @ V
    Jpp, Jpq, Jqp, Jqq = Jc[:d, :d], Jc[:d, d:], Jc[d:, :d], Jc[d:, d:]
    mu, Vq = np.linalg.eig(Jqq)
    A = Jpq @ Vq
    B = np.linalg.solve(Vq, Jqp)

    def Meff(lam):
        return Jpp + (A / (lam - mu)[None, :]) @ B
    M = Meff(0.0)
    lam = float(np.max(np.linalg.eigvals(M).real)) if lam0 is None else lam0
    for _ in range(iters):
        new = float(np.max(np.linalg.eigvals(Meff(lam)).real))
        if abs(new - lam) < tol:
            lam = new
            break
        lam = 0.5 * lam + 0.5 * new
    return lam, Meff(lam), (Jpp, Jpq, Jqp, Jqq, V, T), Meff(0.0)


def coefficients(M, ic, ib):
    """M: 유효 행렬 (d x d). ic: c′ 인덱스, ib: b′ 인덱스 목록.
    α = M[c′,c′].  b′ 블록 B 의 고유쌍 (우 r_i, 좌 l_i, l_i^H r_i = 1) 중 c′ 와 가장 강하게 결합하는 것
    (|q_i| = |M_cb r_i · l_i^H M_bc| 최대, 반사 대칭 때문에 보통 하나만 결합) 을 지배 b′ 모드로 택한다: β = 그 고유값, q = 결합 곱.
    다른 고유쌍의 결합은 q_other 로 기록 (≈0 이어야 2x2 환원이 정확).  Λ2 = 2x2 환원의 최대 고유값."""
    alpha = M[ic, ic].real
    Bb = M[np.ix_(ib, ib)]
    w, vl, vr = sla.eig(Bb, left=True, right=True)
    qs = []
    for i in range(len(w)):
        r, l = vr[:, i], vl[:, i]
        l = l / np.vdot(l, r)
        qs.append(((M[ic, ib] @ r) * np.vdot(l, M[ib, ic])).real)
    qs = np.array(qs)
    i = int(np.argmax(np.abs(qs)))
    beta, qprod = w[i].real, float(qs[i])
    q_other = float(np.sum(np.abs(qs)) - abs(qs[i]))
    lam2 = 0.5 * (alpha + beta) + np.sqrt((0.5 * (alpha - beta)) ** 2 + qprod + 0j)
    rl = (vr[:, i], vl[:, i] / np.vdot(vl[:, i], vr[:, i]))
    return dict(alpha=float(alpha), beta=float(beta), qprod=float(qprod), q_other=q_other,
                beta_max=float(np.max(w.real)), Lam2=float(lam2.real), _rl=rl)


def channel_contributions(parts, d_idx, lam, Jt, V, T):
    """c′ (d_idx[0]) 대각 성분에서 Q 쪽 (j,p) 채널별 2차 기여: J_cq G_qq J_qc 의 채널 대각 블록.
    반환: 채널 (j, p) -> 기여값 (실수부). 채널은 섹터 기저의 (u,v) 2x2 블록 단위."""
    Jpp, Jpq, Jqp, Jqq, V_, T_ = parts
    d = Jpp.shape[0]
    n = Jqq.shape[0]
    G = np.linalg.inv(lam * np.eye(n) - Jqq)
    # Q 좌표 -> 섹터 기저 좌표: Q 열공간 = V[:, d:]
    Qcols = V[:, d:]
    Qrows = T[d:, :]
    out = {}
    return out, G, Qcols, Qrows


def channel_second_order(Jt, R, Lf, lam, lP, rP, nc=8):
    """스칼라 x_l^H (J + J G J) x_r 의 직접항과 Q 채널 (j,p) 별 2차 기여 (채널 = 섹터 기저의 (j,p) 2x2 블록).
    lP, rP: P 공간 좌·우 가중 (길이 d). 반환 direct, {(j,p): 기여}, 합 검증값."""
    n, d = R.shape
    Qr = sla.null_space(Lf.conj().T)
    V = np.concatenate([R, Qr], 1)
    T = np.linalg.inv(V)
    Jc = T @ Jt @ V
    Jqq = Jc[d:, d:]
    Gq = np.linalg.inv(lam * np.eye(n - d) - Jqq)
    G = V[:, d:] @ Gq @ T[d:, :]
    xl, xr = Lf @ lP, R @ rP
    A = xl.conj() @ Jt  # (n,)
    B = Jt @ xr
    direct = A @ xr
    out = {}
    M = n // nc
    tot = 0.0
    for j in range(M):
        for p in range(4):
            ix = [nc * j + p, nc * j + 4 + p]
            c = A[ix] @ G[np.ix_(ix, ix)] @ B[ix]
            out[(j, p)] = c
            tot += c
    full = A @ G @ B
    return direct, out, full
