"""v4: 전체 영역 정상상태 (Newton-Krylov, 시간 τ 흐름 사상) 와 선형 안정성 (Arnoldi), 연결망/사슬 씨앗."""
import numpy as np
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, eigs
from .integrator import Solver
from .model import NLAYER, s_p
from .arnoldi import TangentFlow


class Net:
    def __init__(self, par, N, Cx, k0, sigma0, dt=4.0):
        self.par, self.N, self.Cx, self.k0, self.s0, self.dt = par, N, Cx, k0, sigma0, dt
        self.L = 2 * np.pi * Cx / k0
        self.S = Solver(par, N=N, L=self.L, dt=dt)
        self.tau = 1.0 / sigma0
        self.nsteps = int(round(self.tau / dt))
        self.shape = (2, NLAYER, N, N)
        self.dk = k0 / Cx
        x = np.arange(N) * self.L / N
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        self._ii, self._jj = ii, jj
        self.sym = None
        self._group = None

    # ---- 씨앗 ----
    def mode_field(self, vec, amp, phase, p1, p3, chi=0.0):
        """실수장 u_n = amp * [ p1 cos(k·x + φ + πn/2 + χ) + p3 cos(k·x + φ - πn/2 + χ') ] (층 n=0..3), (4,N,N)"""
        n = np.arange(NLAYER)[:, None, None]
        arg = 2 * np.pi * (vec[0] * self.X + vec[1] * self.Y)[None] / self.L + phase
        return amp * (p1 * np.cos(arg + np.pi * n / 2) + p3 * np.cos(arg - np.pi * n / 2))

    def v_from_u(self, u):
        S, p = self.S, self.par
        U = S.to_hat(u)
        q = S.k2.ravel()[S.idx]
        V = np.zeros((2,) + U.shape, complex)
        V[0] = U
        for pp in range(NLAYER):
            V[1, pp] = U[pp] / (p.d + s_p(pp) * p.kv + p.Dv * q)
        return V

    def to_vec(self, u, v):
        return np.stack([u, v]).reshape(-1)

    def from_vec(self, x):
        x = x.reshape(self.shape)
        return x[0], x[1]

    # ---- 대칭 사영 ----
    ALIAS = {"D4": "R+My", "D4L": "R+My+L", "Mx": "Mx", "My": "My", "I": "I", "MxL": "Mx+L", "MyL": "My+L"}

    def set_sym(self, name):
        """생성자들을 '+' 로 연결: R (90° 회전), Mx (x→-x), My (y→-y), I (반전), L (층 반사 n→-n),
        '*L' 이 붙은 공간 생성자는 층 반사와 결합된 생성자 (예: 'Mx+My*L' = Mx 와 (My∘L)). 별칭: D4, D4L, MxL, ..."""
        self.sym = name
        self._group = None
        if not name:
            return
        spec = self.ALIAS.get(name, name)
        N = self.N
        ii, jj = self._ii, self._jj
        sp = {"R": (jj, (-ii) % N), "Mx": ((-ii) % N, jj), "My": (ii, (-jj) % N), "I": ((-ii) % N, (-jj) % N)}
        gens = []
        for tok in spec.split("+"):
            tok = tok.strip()
            if tok == "L":
                gens.append((ii, jj, True))
            elif tok.endswith("*L"):
                gens.append((sp[tok[:-2]][0], sp[tok[:-2]][1], True))
            else:
                gens.append((sp[tok][0], sp[tok][1], False))
        key = lambda ia, ja, fl: (ia.tobytes(), ja.tobytes(), fl)
        elems = {key(ii, jj, False): (ii, jj, False)}
        frontier = [(ii, jj, False)]
        while frontier:
            new = []
            for (ia, ja, fl) in frontier:
                for (ga, gb, gf) in gens:
                    ca, cb, cf = ia[ga, gb], ja[ga, gb], (fl != gf)
                    k = key(ca, cb, cf)
                    if k not in elems:
                        elems[k] = (ca, cb, cf)
                        new.append((ca, cb, cf))
            frontier = new
        self._group = list(elems.values())

    def _project(self, u):
        """실공간 장 (...,4,N,N) 을 대칭 부분공간으로 사영 (군 평균)"""
        if not self.sym:
            return u
        acc = np.zeros_like(u)
        for (ia, ja, fl) in self._group:
            w = u[..., ia, ja]
            if fl:
                w = w[..., [0, 3, 2, 1], :, :]
            acc += w
        return acc / len(self._group)

    def project_vec(self, x):
        if not self.sym:
            return x
        x = x.reshape(self.shape)
        return np.stack([self._project(x[0]), self._project(x[1])]).reshape(-1)

    # ---- 흐름 ----
    def flow(self, x, nsteps=None):
        S = self.S
        u, v = self.from_vec(x)
        V = S.state_from_real(u, v)
        for _ in range(nsteps or self.nsteps):
            V = S.step(V)
        return np.stack([S.to_real(V[0]), S.to_real(V[1])]).reshape(-1)

    def relax(self, x, T, proj_every=10):
        S = self.S
        x = self.project_vec(x)
        u, v = self.from_vec(x)
        V = S.state_from_real(u, v)
        n = int(round(T / self.dt))
        for i in range(n):
            V = S.step(V)
            if self.sym and (i + 1) % proj_every == 0:
                V = S.state_from_real(self._project(S.to_real(V[0])), self._project(S.to_real(V[1])))
        return self.project_vec(np.stack([S.to_real(V[0]), S.to_real(V[1])]).reshape(-1))

    def residual(self, x):
        xp = self.project_vec(x)
        S = self.S
        u, v = self.from_vec(xp)
        V = S.state_from_real(u, v)
        for i in range(self.nsteps):
            V = S.step(V)
            if self.sym and (i + 1) % 10 == 0:
                V = S.state_from_real(self._project(S.to_real(V[0])), self._project(S.to_real(V[1])))
        y = self.project_vec(np.stack([S.to_real(V[0]), S.to_real(V[1])]).reshape(-1))
        return y - xp

    def newton(self, x0, f_tol=1e-10, maxiter=30, inner_maxiter=40, verbose=False):
        sol = newton_krylov(self.residual, x0, method="lgmres", f_tol=f_tol, maxiter=maxiter, inner_maxiter=inner_maxiter,
                            verbose=verbose, rdiff=1e-7)
        return sol, float(np.abs(self.residual(sol)).max())

    # ---- 안정성 ----
    def eigenvalues(self, x, k=10, tol=1e-5, ncv=None, maxiter=300, seed=0):
        u, v = self.from_vec(x)
        TF = TangentFlow(self.par, u, self.N, self.L, dt=self.dt, tau=self.tau)
        w, V = TF.eigenvalues(k=k, ncv=ncv or 3 * k + 10, tol=tol, maxiter=maxiter, seed=seed)
        lam = np.log(np.abs(w)) / self.tau / self.s0
        order = np.argsort(-lam)
        return lam[order], w[order], V[:, order]

    # ---- 진단: 푸리에 성분 ----
    def modes(self, x, vecs):
        """주어진 격자 벡터들의 층 섹터 진폭 |F_p| (p=0..3), 각 벡터별 (len,4)"""
        u, v = self.from_vec(x)
        N = self.N
        F = np.fft.fft2(u, axes=(1, 2)) / (N * N)  # (4,N,N)
        out = []
        for vec in vecs:
            col = F[:, vec[0] % N, vec[1] % N]  # u_n(k) 계수 (n=0..3) : u = Σ û e^{ikx}
            out.append([abs(np.sum(col * np.exp(-1j * np.pi * p * np.arange(4) / 2)) / 4) for p in range(4)])
        return np.array(out)


def composition(net, vecr, inner, outer, band=0.08):
    """실수 고유벡터 δ=(δu,δv) 의 구성: 범주별 δu 전력 비율과 층 섹터(p) 비율.
    범주: net_outer / net_inner (기존 연결망 모드 ±), ring_new_outer / ring_new_inner (링 대역의 새 격자 벡터),
          low (|k|<0.3k0), other."""
    d = vecr.reshape(net.shape)[0]
    N = net.N
    F = np.fft.fft2(d, axes=(1, 2)) / (N * N)           # (4,N,N) 층별 푸리에
    Fp = np.stack([np.sum(F * np.exp(-1j * np.pi * p * np.arange(4) / 2)[:, None, None], 0) / 4 for p in range(4)])  # 섹터 p
    pw = (np.abs(Fp) ** 2)                               # (4,N,N)
    tot = pw.sum()
    kx = np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kk = np.hypot(KX, KY) * net.dk / net.k0
    k1r = 1.0 / (2 * np.cos(np.pi / (len(inner))))
    netmask_o = np.zeros((N, N), bool)
    netmask_i = np.zeros((N, N), bool)
    for v in outer:
        netmask_o[v[0] % N, v[1] % N] = True
    for v in inner:
        netmask_i[v[0] % N, v[1] % N] = True
    ring_o = np.abs(kk - 1.0) < band
    ring_i = np.abs(kk - k1r) < band
    low = kk < 0.3
    cats = {
        "net_outer": pw[:, netmask_o].sum(), "net_inner": pw[:, netmask_i].sum(),
        "ring_new_outer": pw[:, ring_o & ~netmask_o].sum(), "ring_new_inner": pw[:, ring_i & ~netmask_i].sum(),
        "low": pw[:, low].sum(),
    }
    used = sum(cats.values())
    cats["other"] = max(tot - used, 0.0)
    out = {k: float(v / tot) for k, v in cats.items()}
    out["sector_p"] = [float(pw[p].sum() / tot) for p in range(4)]
    # 연결망 안쪽 모드의 p1/p3 불균형 (정상파형 -> 나선형 이행의 지표)
    imb = []
    for v in inner:
        a, b = pw[1, v[0] % N, v[1] % N], pw[3, v[0] % N, v[1] % N]
        imb.append(float((a - b) / (a + b + 1e-300)))
    out["inner_p1p3_imbalance_mean"] = float(np.mean(np.abs(imb)))
    return out
