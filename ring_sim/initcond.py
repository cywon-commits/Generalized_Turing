"""초기 조건 O (바깥만), F (완전 씨앗), N (잡음)."""
import numpy as np
from .model import NLAYER, s_p
from .lattice import design_vectors, NFOLD


def _cosmode(X, Y, L, vec, phase):
    return np.cos(2 * np.pi * (vec[0] * X + vec[1] * Y) / L + phase)


def make_state(S, case, ratio, ic, seed, R0, k0, A_out=0.04, A_in=0.04, in_noise=1e-4,
               white=1e-3):
    """Solver S 에 대한 초기 상태 V (2,4,K) 와 시드 벡터 정보 반환."""
    rng = np.random.default_rng(seed)
    N, L = S.N, S.L
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    nfold = NFOLD[case]
    res = abs(ratio - 2 * np.cos(np.pi / nfold)) < 1e-6
    out, inn = design_vectors(R0, ratio, nfold, res)
    u = np.zeros((NLAYER, N, N))
    info = dict(outer=out.tolist(), inner=inn.tolist(), resonant=bool(res), nfold=nfold)
    if ic in ("O", "F"):
        for v in out:
            u += A_out * _cosmode(X, Y, L, v, rng.uniform(0, 2 * np.pi))[None]
    if ic == "F":
        n = np.arange(NLAYER)[:, None, None]
        # 안쪽 벡터: 공명이면 A_j 및 -A_j 전부, 아니면 축 벡터. 나선형 적층 e^{±iπn/2}
        for v in inn:
            chir = rng.choice([-1, 1])
            ph = rng.uniform(0, 2 * np.pi)
            u += A_in * np.cos(2 * np.pi * (v[0] * X + v[1] * Y)[None] / L + ph + chir * np.pi * n / 2)
    if ic == "N":
        u += white * rng.standard_normal(u.shape)
    uh = S.to_hat(u)
    V = np.stack([uh, np.zeros_like(uh)])
    if ic in ("O", "F"):
        # 안쪽 원 대역(p=1,3)에 모드별 진폭 in_noise, 무작위 위상의 잡음
        k1 = k0 / ratio
        kk = np.sqrt(S.k2.ravel()[S.idx])
        band = np.abs(kk - k1) < 0.1 * k0
        norm = NLAYER * N * N
        for p in (1, 3):
            ph = rng.uniform(0, 2 * np.pi, band.sum())
            V[0, p, band] += in_noise * norm * np.exp(1j * ph) / 2
        # 아주 작은 백색 잡음으로 대칭 깨기
        V[0] += S.to_hat(1e-7 * rng.standard_normal(u.shape))
    # v 성분: 정상 관계 v = u / (d + s κv + Dv q)
    par = S.par
    q = S.k2.ravel()[S.idx]
    for p in range(NLAYER):
        V[1, p] = V[0, p] / (par.d + s_p(p) * par.kv + par.Dv * q)
    return V, info
