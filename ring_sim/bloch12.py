"""v6 과제 2: (w, y) 매개 종을 추가한 16성분 Bloch 셀. 상태 X: (16, M) = [u(4), v(4), w(4), y(4)] x 모드.

  ∂t w_n = a_w w_n - b_w y_n + β_w u_n² + D_w ∇² w_n + κ_w (w_{n+1}+w_{n-1}-2 w_n)
  ∂t y_n = w_n - d_w y_n + D_y ∇² y_n + κ_y (y_{n+1}+y_{n-1}-2 y_n)
  u 방정식에 -γ w_n u_n 추가.   (γ β_w < 0: 협동 부호 -> 바깥 교차계수 g_cc′ 감소)"""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as sla
from scipy.fft import fft2
from .bloch import Cell, _S
from .model import Params, nonlinear_u


@dataclass
class Knob:
    aw: float = 1.0
    bw: float = 1.6
    dw: float = 1.5
    Dw: float = 1.0
    Dy: float = 2.5
    kw: float = 0.0
    ky: float = 0.0
    betaw: float = 1.0
    gamma: float = 0.0


def Lw_sector(kn: Knob, p, q):
    """섹터 p, 파수제곱 q 의 (w,y) 2x2 선형 블록"""
    c = 2 * np.cos(np.pi * p / 2)
    return np.array([[kn.aw - kn.Dw * q - 2 * kn.kw + kn.kw * c, -kn.bw],
                     [1.0, -kn.dw - kn.Dy * q - 2 * kn.ky + kn.ky * c]])


def w_growth(kn: Knob, p, q):
    return np.linalg.eigvals(Lw_sector(kn, p, q)).real.max()


class Cell12(Cell):
    nc = 16

    def __init__(self, par: Params, kn: Knob, a, b, dk, kcut, kappa=(0.0, 0.0), P=None):
        self.kn = kn
        super().__init__(par, a, b, dk, kcut, kappa=kappa, P=P)

    def Lblock(self):
        p, k, S, I = self.par, self.kn, _S(), np.eye(4)
        L = np.zeros((self.M, 16, 16))
        for j in range(self.M):
            q = self.q[j]
            L[j, 0:4, 0:4] = (p.a - p.Du * q - 2 * p.ku) * I + p.ku * S
            L[j, 0:4, 4:8] = -p.b * I
            L[j, 4:8, 0:4] = I
            L[j, 4:8, 4:8] = (-p.d - p.Dv * q - 2 * p.kv) * I + p.kv * S
            L[j, 8:12, 8:12] = (k.aw - k.Dw * q - 2 * k.kw) * I + k.kw * S
            L[j, 8:12, 12:16] = -k.bw * I
            L[j, 12:16, 8:12] = I
            L[j, 12:16, 12:16] = (-k.dw - k.Dy * q - 2 * k.ky) * I + k.ky * S
        return L

    def nonlinear(self, X):
        p, k = self.par, self.kn
        u = self.to_real(X[0:4])
        w = self.to_real(X[8:12])
        up, um = np.roll(u, -1, 0), np.roll(u, 1, 0)
        Nu = p.eta * u * u + p.etap * u * (up + um) - p.c * u ** 3 - k.gamma * w * u
        Nw = k.betaw * u * u
        out = np.zeros_like(X)
        out[0:4] = self.to_hat(Nu)
        out[8:12] = self.to_hat(Nw)
        return out

    def jacobian(self, Xstar, star_cell):
        """정상상태 X* (16, M0) 주변 선형화 (조밀 16M x 16M)"""
        p, k = self.par, self.kn
        us = star_cell.to_real(Xstar[0:4])
        ws = star_cell.to_real(Xstar[8:12])
        usp, usm = np.roll(us, -1, 0), np.roll(us, 1, 0)
        f_uu = 2 * p.eta * us + p.etap * (usp + usm) - 3 * p.c * us ** 2 - k.gamma * ws  # d N_u / d u (대각 층)
        f_uw = -k.gamma * us                                                              # d N_u / d w
        f_wu = 2 * k.betaw * us                                                          # d N_w / d u
        P2 = self.P ** 2
        h_uu, h_uw, h_wu, h_us = (fft2(f) / P2 for f in (f_uu, f_uw, f_wu, us))
        M = self.M
        J = np.zeros((M, 16, M, 16), complex)
        Lb = self.Lblock()
        for j in range(M):
            J[j, :, j, :] += Lb[j]
        for n in range(4):
            J[:, n, :, n] += self._conv(h_uu[n], self)
            G = p.etap * self._conv(h_us[n], self)
            J[:, n, :, (n + 1) % 4] += G
            J[:, n, :, (n - 1) % 4] += G
            J[:, n, :, 8 + n] += self._conv(h_uw[n], self)
            J[:, 8 + n, :, n] += self._conv(h_wu[n], self)
        return J.reshape(16 * M, 16 * M)


# --- 이완·Newton (16 성분) ---
def relax(cell, X, T=60000.0, h=20.0, tol=1e-13):
    Lb = cell.Lblock()
    n = Lb.shape[1]
    E = np.zeros_like(Lb); Ph = np.zeros_like(Lb)
    for j in range(cell.M):
        Z = np.zeros((2 * n, 2 * n)); Z[:n, :n] = Lb[j] * h; Z[:n, n:] = np.eye(n) * h
        EZ = sla.expm(Z)
        E[j] = EZ[:n, :n]; Ph[j] = EZ[:n, n:]
    res = None
    for it in range(int(T / h)):
        N = cell.nonlinear(X)
        X = np.einsum("jab,bj->aj", E, X) + np.einsum("jab,bj->aj", Ph, N)
        if not np.all(np.isfinite(X)):
            return X, float("inf")
        if it % 100 == 0:
            res = np.abs(cell.residual(X, Lb)).max()
            if res < tol:
                break
    return X, res


def newton(cell, X, iters=30, tol=1e-13, cond=1e-9):
    Lb = cell.Lblock()
    for it in range(iters):
        F = cell.residual(X, Lb)
        r = np.abs(F).max()
        if r < tol:
            return X, r, it
        J = cell.jacobian(X, cell)
        dx = sla.lstsq(J, -F.T.reshape(-1), cond=cond)[0]
        X = X + dx.reshape(cell.M, 16).T
    return X, np.abs(cell.residual(X, Lb)).max(), iters


def embed8(X8, M):
    X = np.zeros((16, X8.shape[1]), complex)
    X[:8] = X8
    return X
