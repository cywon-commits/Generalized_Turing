"""v3 단계 A 교차 검증: 전체 영역에서의 행렬 없는 선형화 흐름 사상 + Arnoldi (ARPACK).
정상상태(축소 셀의 푸리에 계수) 를 전체 격자에 심고, τ = 1/σ0 동안의 접선 흐름 (ETDRK4, 야코비안-벡터 곱) 의 고유값을 구한다."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from .integrator import Solver
from .model import NLAYER


def embed(cell, X, a, b, N):
    """축소 셀 푸리에 계수 X (8,M) -> 전체 N×N 격자의 실공간 (u, v) (4,N,N)."""
    gx = cell.m * a[0] + cell.n * b[0]
    gy = cell.m * a[1] + cell.n * b[1]
    out = []
    for f in (0, 1):
        arr = np.zeros((NLAYER, N, N), complex)
        for n in range(NLAYER):
            arr[n, gx % N, gy % N] = X[4 * f + n]
        out.append((np.fft.ifft2(arr, axes=(1, 2)) * N * N).real)
    return out[0], out[1]


class TangentFlow:
    def __init__(self, par, ustar, N, L, dt, tau):
        self.S = Solver(par, N=N, L=L, dt=dt)
        S = self.S
        p = par
        up, um = np.roll(ustar, -1, 0), np.roll(ustar, 1, 0)
        self.f1 = 2 * p.eta * ustar + p.etap * (up + um) - 3 * p.c * ustar ** 2
        self.f2 = p.etap * ustar
        self.nsteps = int(round(tau / dt))
        self.N = N
        self.shape = (2, NLAYER, N, N)

    def _nl(self, V):
        S = self.S
        d = S.to_real(V[0])
        Nd = self.f1 * d + self.f2 * (np.roll(d, -1, 0) + np.roll(d, 1, 0))
        return S.to_hat(Nd)

    def apply(self, x):
        S = self.S
        x = x.reshape(self.shape)
        V = S.state_from_real(x[0], x[1])
        for _ in range(self.nsteps):
            V = S.step(V, nl=self._nl)
        y = np.stack([S.to_real(V[0]), S.to_real(V[1])])
        return y.reshape(-1)

    def eigenvalues(self, k=10, ncv=None, tol=1e-6, seed=0, maxiter=300):
        n = int(np.prod(self.shape))
        op = LinearOperator((n, n), matvec=self.apply, dtype=float)
        rng = np.random.default_rng(seed)
        v0 = 1e-3 * rng.standard_normal(n)
        w, V = eigs(op, k=k, which="LM", ncv=ncv or max(2 * k + 1, 30), tol=tol, v0=v0, maxiter=maxiter)
        return w, V
