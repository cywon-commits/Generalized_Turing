"""v7 과제 D: ℤ2 이중층 (nl=2). 동위상 섹터 p=0, 역위상 섹터 p=1 (층간 결합 고유값 s=4)."""
import numpy as np
from scipy.optimize import fsolve, brentq, minimize_scalar
from .model import Params, growth_rate, peak
from .integrator import Solver

NL = 2


def peak_nl(par, p, qmax=1.0, n=4001):
    qs = np.linspace(0, qmax, n)
    g = growth_rate(par, p, qs, NL)
    i = int(np.argmax(g))
    r = minimize_scalar(lambda q: -float(growth_rate(par, p, q, NL)), bounds=(qs[max(i - 1, 0)], qs[min(i + 1, n - 1)]), method="bounded", options={"xatol": 1e-13})
    return float(-r.fun), float(r.x)


def solve_kappa_nl(ratio, r, base):
    """역위상 섹터 (p=1) 의 봉우리가 k0/ratio 에서 값 -r σ0(base) 가 되는 (κu, κv). σ0(base) = 동위상 봉우리 (음수 가능)."""
    s0, q0 = peak_nl(base, 0)
    k0 = np.sqrt(q0)
    tgt = (k0 / ratio) ** 2

    def f(x):
        par = Params(**{**base.__dict__, "ku": x[0], "kv": x[1]})
        s1, q1 = peak_nl(par, 1)
        return [q1 - tgt, s1 - (-r * s0)]
    sol = fsolve(f, [0.035, 0.085], xtol=1e-12)
    return float(sol[0]), float(sol[1])


def Dv_for_peak(sigma, lo=2.2, hi=2.9):
    return brentq(lambda D: peak_nl(Params(Dv=D), 0)[0] - sigma, lo, hi, xtol=1e-12)


def make_solver(par, L, N=128, dt=2.0):
    return Solver(par, N=N, L=L, dt=dt, nl=NL)


def noise_state(S, seed, amp=1e-3):
    rng = np.random.default_rng(seed)
    u = amp * rng.standard_normal((NL, S.N, S.N))
    uh = S.to_hat(u)
    V = np.stack([uh, np.zeros_like(uh)])
    S.hermitize(V)
    return V


def sector_spectra(S, V):
    u = S.to_real(V[0])
    N = S.N
    F = np.fft.fft2(u, axes=(1, 2)) / (N * N)
    f0 = 0.5 * (F[0] + F[1])
    f1 = 0.5 * (F[0] - F[1])
    return u, np.abs(f0) ** 2, np.abs(f1) ** 2


def peak_directions(P, N, L, k0, w=0.25, frac=0.15):
    """링 대역 안의 스펙트럼 피크 방향 (도, 0-180) 목록: 최대 세기의 frac 이상인 국소 극대"""
    k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / L
    KX, KY = np.meshgrid(k, k, indexing="ij")
    kk = np.hypot(KX, KY)
    band = np.abs(kk - k0) <= w * k0
    Pm = np.where(band, P, 0)
    m = Pm.max()
    if m <= 0: return [], 0.0
    loc = np.ones_like(Pm, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj: loc &= Pm >= np.roll(np.roll(Pm, di, 0), dj, 1)
    ok = loc & (Pm >= frac * m)
    ang = np.degrees(np.arctan2(KY[ok], KX[ok])) % 180.0
    return sorted(set(np.round(ang, 0))), float(Pm.sum())


# ---------------- 이중층 Bloch 셀 (Galerkin) ----------------
from scipy.fft import fft2, ifft2, next_fast_len
from .bloch import Cell
from .continuum import cont_vectors


class Cell2(Cell):
    """상태 X: (4, M) = [u0,u1,v0,v1]. 층간 결합 κ (n±1 이 같은 층): κ(2 u_other - 2 u_n). 2차항 η u², 3차항 -c u³ (η'=0)."""
    nc = 4

    def Lblock(self):
        p = self.par
        S2 = np.array([[0.0, 2.0], [2.0, 0.0]]); I = np.eye(2)
        L = np.zeros((self.M, 4, 4))
        for j in range(self.M):
            q = self.q[j]
            L[j, :2, :2] = (p.a - p.Du * q - 2 * p.ku) * I + p.ku * S2
            L[j, :2, 2:] = -p.b * I
            L[j, 2:, :2] = I
            L[j, 2:, 2:] = (-p.d - p.Dv * q - 2 * p.kv) * I + p.kv * S2
        return L

    def nonlinear(self, X):
        p = self.par
        u = self.to_real(X[:2])
        out = np.zeros_like(X)
        out[:2] = self.to_hat(p.eta * u * u - p.c * u ** 3)
        return out

    def jacobian(self, ustar, star_cell):
        p = self.par
        us = star_cell.to_real(ustar)
        fh = fft2(2 * p.eta * us - 3 * p.c * us ** 2) / (self.P ** 2)
        M = self.M
        J = np.zeros((M, 4, M, 4), complex)
        Lb = self.Lblock()
        for j in range(M):
            J[j, :, j, :] += Lb[j]
        for n in range(2):
            J[:, n, :, n] += self._conv(fh[n], self)
        return J.reshape(4 * M, 4 * M)


def relax2(cell, X, T, h):
    import scipy.linalg as sla
    Lb = cell.Lblock(); n = 4
    E = np.zeros_like(Lb); Ph = np.zeros_like(Lb)
    for j in range(cell.M):
        Z = np.zeros((2 * n, 2 * n)); Z[:n, :n] = Lb[j] * h; Z[:n, n:] = np.eye(n) * h
        EZ = sla.expm(Z); E[j] = EZ[:n, :n]; Ph[j] = EZ[:n, n:]
    for it in range(int(T / h)):
        X = np.einsum("jab,bj->aj", E, X) + np.einsum("jab,bj->aj", Ph, cell.nonlinear(X))
        if not np.all(np.isfinite(X)): return X
        if it % 200 == 0 and np.abs(cell.residual(X, Lb)).max() < 1e-13: break
    return X


def newton2(cell, X, iters=40, tol=1e-12):
    import scipy.linalg as sla
    Lb = cell.Lblock()
    for it in range(iters):
        F = cell.residual(X, Lb); r = np.abs(F).max()
        if r < tol: return X, r
        dx = sla.lstsq(cell.jacobian(X[:2], cell), -F.T.reshape(-1), cond=1e-9)[0]
        X = X + dx.reshape(cell.M, 4).T
    return X, np.abs(cell.residual(X, Lb)).max()


def cell2_setup(par, rho, nb=3, kcut_factor=3.0):
    """par: Params (ku,kv 포함). Λ = span{a,b} (연속체 정확 벡터, 길이는 k0 단위), nb x nb Bloch 오프셋."""
    s0, q0 = peak_nl(par, 0)
    k0 = np.sqrt(q0)
    a, b, bp = cont_vectors(rho)
    a_, b_ = np.array(a), np.array(b)
    kaps = [(0.0, 0.0)] + [tuple(i / nb * a_ + j / nb * b_) for i in range(nb) for j in range(nb) if (i, j) != (0, 0)]
    cells = [Cell2(par, a, b, k0, kcut_factor * k0, kappa=k) for k in kaps]
    P = next_fast_len(4 * max(c.Mx for c in cells) + 2)
    return [Cell2(par, a, b, k0, kcut_factor * k0, kappa=c.kappa, P=P) for c in cells], s0, k0


def layer_shift(cell, X):
    """층 0 과 층 1 의 실공간 장 (셀 좌표 = Λ 기본 셀의 분수 좌표) 의 교차상관 최대 위치 -> 분수 이동 (s,t), 상관값"""
    u = cell.to_real(X[:2]).real
    C = np.fft.ifft2(np.fft.fft2(u[0] - u[0].mean()) * np.conj(np.fft.fft2(u[1] - u[1].mean()))).real
    i, j = np.unravel_index(np.argmax(C), C.shape)
    P = cell.P
    norm = np.sqrt((u[0] - u[0].mean()) @ (u[0] - u[0].mean()) if False else (np.sum((u[0] - u[0].mean()) ** 2) * np.sum((u[1] - u[1].mean()) ** 2)))
    return (i / P % 1.0, j / P % 1.0), float(C[i, j] / norm)
