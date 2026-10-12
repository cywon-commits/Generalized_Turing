"""의사분광 + ETDRK4 (Cox-Matthews, 접촉 적분 계수) 적분기.

상태: (u, v) 는 (4, N, N) 실수장. 전 3축을 rfftn 하여 층 Fourier 섹터 p 와 파수를
동시에 대각화한다. 섹터·파수별 선형 연산자는 실수 2x2 행렬이다.
"""
import numpy as np
import scipy.fft as sfft
from .model import Params, lin_matrix, NLAYER


def _mm(A, B):
    return np.matmul(A, B)


def _expm2(A):
    """실수 2x2 행렬 지수 (…,2,2), 해석식."""
    tr = A[..., 0, 0] + A[..., 1, 1]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    D2 = tr * tr / 4 - det
    D = np.sqrt(D2.astype(complex))
    small = np.abs(D) < 1e-6
    shc = np.where(small, 1 + D2 / 6, np.sinh(np.where(small, 1, D)) / np.where(small, 1, D))
    ch = np.cosh(D)
    I = np.eye(2)
    M = A - (tr / 2)[..., None, None] * I
    E = np.exp(tr / 2)[..., None, None] * (ch[..., None, None] * I + shc[..., None, None] * M)
    return E.real


def _inv2(B):
    det = B[..., 0, 0] * B[..., 1, 1] - B[..., 0, 1] * B[..., 1, 0]
    out = np.empty_like(B)
    out[..., 0, 0] = B[..., 1, 1]
    out[..., 1, 1] = B[..., 0, 0]
    out[..., 0, 1] = -B[..., 0, 1]
    out[..., 1, 0] = -B[..., 1, 0]
    return out / det[..., None, None]


def etdrk4_coeffs(Lmat, h, M=64, R=1.0):
    """Lmat (K,2,2) 실수. 반환: E, E2, Q, f1, f2, f3 (K,2,2) 실수."""
    A = h * Lmat
    E = _expm2(A)
    E2 = _expm2(A / 2)
    th = np.pi * (np.arange(M) + 0.5) / M
    zs = R * np.exp(1j * th)  # 켤레쌍을 쓰므로 상반원 평균의 실수부 x2/2
    I = np.eye(2)
    Q = np.zeros(A.shape, complex)
    f1 = np.zeros(A.shape, complex)
    f2 = np.zeros(A.shape, complex)
    f3 = np.zeros(A.shape, complex)
    EA, EA2 = E.astype(complex), E2.astype(complex)
    for z in zs:
        B = A + z * I
        Bi = _inv2(B)
        Bi3 = _mm(Bi, _mm(Bi, Bi))
        ez = np.exp(z)
        EB = ez * EA
        EBh = np.exp(z / 2) * EA2
        B2 = _mm(B, B)
        Q += _mm(EBh - I, Bi)
        f1 += _mm(Bi3, -4 * I - B + _mm(EB, 4 * I - 3 * B + B2))
        f2 += _mm(Bi3, 2 * I + B + _mm(EB, -2 * I + B))
        f3 += _mm(Bi3, -4 * I - 3 * B - B2 + _mm(EB, 4 * I - B))
    sc = h / M
    return E, E2, (Q * sc).real, (f1 * sc).real, (f2 * sc).real, (f3 * sc).real


class Solver:
    def __init__(self, par: Params, N=512, L=None, dt=0.25, workers=1, dealias=2 / 3):
        self.par, self.N, self.L, self.dt = par, N, L, dt
        self.workers = workers
        dk = 2 * np.pi / L
        self.dk = dk
        kx = np.fft.fftfreq(N, 1.0 / N) * dk
        ky = np.fft.rfftfreq(N, 1.0 / N) * dk
        self.k2 = kx[:, None] ** 2 + ky[None, :] ** 2
        kmax = dealias * np.pi * N / L
        self.mask = (self.k2 <= kmax ** 2)
        self.shape_hat = (NLAYER, N, N // 2 + 1)
        # 섹터 p 의 층 FFT 인덱스: rfftn 축0 은 전체 fft 이므로 p = 0..3
        # 고유한 (p, q) 쌍만 계수 계산
        q_all = self.k2[self.mask]
        uq, inv = np.unique(np.round(q_all, 12), return_inverse=True)
        self._build(uq, inv, dt)

    def _build(self, uq, inv, h):
        par = self.par
        shp = self.k2.shape
        Eg = np.empty((NLAYER, 2, 2) + shp)
        E2g = np.empty_like(Eg)
        Qg = np.zeros((NLAYER, 2) + shp)
        F1, F2, F3 = np.zeros_like(Qg), np.zeros_like(Qg), np.zeros_like(Qg)
        full_q = self.k2.ravel()
        for p in range(NLAYER):
            A = h * lin_matrix(par, p, full_q)
            Eg[p] = np.moveaxis(_expm2(A), 0, -1).reshape((2, 2) + shp)
            E2g[p] = np.moveaxis(_expm2(A / 2), 0, -1).reshape((2, 2) + shp)
            _, _, Q, f1, f2, f3 = etdrk4_coeffs(lin_matrix(par, p, uq), h)
            for tgt, src in ((Qg, Q), (F1, f1), (F2, f2), (F3, f3)):
                for i in range(2):  # 비선형항은 u 성분(첫 열)에만 작용
                    t = np.zeros(shp)
                    t[self.mask] = src[inv, i, 0]
                    tgt[p, i] = t
        # (4,2,2,N,M) -> E[:, i, j] 를 (4,N,M) 로 쓰기 위해 축 정리
        m = self.mask
        self.E, self.E2 = np.ascontiguousarray(Eg[..., m]), np.ascontiguousarray(E2g[..., m])
        self.Q, self.f1, self.f2, self.f3 = [np.ascontiguousarray(np.swapaxes(x, 0, 1)[..., m])
                                             for x in (Qg, F1, F2, F3)]
        self.idx = np.flatnonzero(m.ravel())
        self.K = self.idx.size
        self._buf = np.zeros((NLAYER, self.N * (self.N // 2 + 1)), complex)

    # --- 변환 (상태는 dealias 마스크 안의 파수만 압축 저장: (2,4,K)) ---
    def pack(self, fh):
        """전체 rfftn 배열 (4,N,M) -> (4,K)"""
        return fh.reshape(NLAYER, -1)[:, self.idx]

    def to_hat(self, f):
        """실공간 (4,N,N) -> 압축 Fourier (4,K)"""
        return self.pack(sfft.rfftn(f, workers=self.workers))

    def to_real(self, fc):
        """압축 Fourier (4,K) -> 실공간 (4,N,N)"""
        self._buf[:, self.idx] = fc
        return sfft.irfftn(self._buf.reshape(self.shape_hat), s=(NLAYER, self.N, self.N),
                           workers=self.workers)

    def state_from_real(self, u, v):
        return np.stack([self.to_hat(u), self.to_hat(v)])

    def _lin(self, E, V):  # V (2,4,K) -> E·V
        out = np.empty_like(V)
        out[0] = E[:, 0, 0] * V[0] + E[:, 0, 1] * V[1]
        out[1] = E[:, 1, 0] * V[0] + E[:, 1, 1] * V[1]
        return out

    def nonlin(self, V):
        """V: (2,4,K) 압축 Fourier 상태 -> N(V) 의 u 성분 (4,N,M)."""
        u = self.to_real(V[0])
        eta, c = self.par.eta, self.par.c
        Nu = eta * u * u - c * u * u * u
        return self.to_hat(Nu)

    def _add_n(self, base, NV):
        # base (2,4,K) + coef * N  (N 은 u 성분 하나)
        return base

    def step(self, V):
        Q, f1, f2, f3 = self.Q, self.f1, self.f2, self.f3  
        N1 = self.nonlin(V)
        EV2 = self._lin(self.E2, V)
        a = EV2 + Q * N1
        N2 = self.nonlin(a)
        b = EV2 + Q * N2
        N3 = self.nonlin(b)
        c = self._lin(self.E2, a) + Q * (2 * N3 - N1)
        N4 = self.nonlin(c)
        return self._lin(self.E, V) + f1 * N1 + 2 * f2 * (N2 + N3) + f3 * N4
