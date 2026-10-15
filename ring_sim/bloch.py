"""v3 단계 A: 정상파형 마름모 정상상태와 Bloch 섹터별 선형 안정성 (축소 셀, Galerkin 절단).

정상상태 u*(x) 는 Λ = span{a, b} (격자 단위 정수 벡터) 위에서만 푸리에 성분을 갖는다. 모드를 (m,n) 으로 색인하여
g = m a + n b, 파수 k = κ + g (κ: Bloch 오프셋). 곱셈은 (m,n) 격자의 합성곱 → P×P FFT (비에일리어싱: P ≥ 4 M_max + 1).
상태 X: (8, M) = [u_0..u_3, v_0..v_3] x 모드. 선형 연산자는 모드별 8x8 블록(층 결합 포함),
비선형항 N(u)_n = η u_n² + η' u_n (u_{n+1}+u_{n-1}) - c u_n³ (u 성분만).
"""
import numpy as np
import scipy.linalg as sla
from scipy.fft import fft2, ifft2, next_fast_len
from .model import Params, NLAYER


def _S():
    S = np.zeros((4, 4))
    for n in range(4):
        S[n, (n + 1) % 4] += 1
        S[n, (n - 1) % 4] += 1
    return S


class Cell:
    def __init__(self, par: Params, a, b, dk, kcut, kappa=(0.0, 0.0), P=None):
        self.par, self.dk, self.kcut = par, dk, kcut
        self.a, self.b = np.array(a, float), np.array(b, float)
        self.kappa = np.array(kappa, float)
        nz = [abs(x) for x in (*self.a, *self.b) if abs(x) > 1e-12]
        Mr = int(np.ceil(kcut / dk / min(nz + [1.0]))) + 2  # v7: 비정수 (연속체) 벡터에서도 모든 모드를 포함
        mm, nn = np.meshgrid(np.arange(-Mr, Mr + 1), np.arange(-Mr, Mr + 1), indexing="ij")
        gx = mm * self.a[0] + nn * self.b[0] + self.kappa[0]
        gy = mm * self.a[1] + nn * self.b[1] + self.kappa[1]
        ok = np.hypot(gx, gy) * dk <= kcut
        self.m, self.n = mm[ok], nn[ok]
        self.kx, self.ky = gx[ok], gy[ok]
        self.M = self.m.size
        self.q = (self.kx ** 2 + self.ky ** 2) * dk ** 2
        self.Mx = int(max(np.abs(self.m).max(), np.abs(self.n).max()))
        self.P = P or next_fast_len(4 * self.Mx + 2)
        self.pos = (self.m % self.P) * self.P + (self.n % self.P)
        self.lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.m, self.n))}

    # --- 선형 블록 ---
    def Lblock(self):
        p, S, I = self.par, _S(), np.eye(4)
        L = np.zeros((self.M, 8, 8))
        for j in range(self.M):
            q = self.q[j]
            L[j, :4, :4] = (p.a - p.Du * q - 2 * p.ku) * I + p.ku * S
            L[j, :4, 4:] = -p.b * I
            L[j, 4:, :4] = I
            L[j, 4:, 4:] = (-p.d - p.Dv * q - 2 * p.kv) * I + p.kv * S
        return L

    # --- 변환 (이 셀의 모드 <-> P×P 격자) ---
    def to_real(self, Uh):  # (4,M) -> (4,P,P) 복소 격자값
        arr = np.zeros((Uh.shape[0], self.P * self.P), complex)
        arr[:, self.pos] = Uh
        return ifft2(arr.reshape(-1, self.P, self.P)) * (self.P * self.P)

    def to_hat(self, f):  # (4,P,P) -> (4,M)
        F = fft2(f) / (self.P * self.P)
        return F.reshape(f.shape[0], -1)[:, self.pos]

    def nonlinear(self, X):
        """X (8,M) -> N(X) (8,M): u 성분만 (Galerkin 사영)"""
        p = self.par
        u = self.to_real(X[:4])
        up, um = np.roll(u, -1, 0), np.roll(u, 1, 0)
        Nu = p.eta * u * u + p.etap * u * (up + um) - p.c * u ** 3
        out = np.zeros_like(X)
        out[:4] = self.to_hat(Nu)
        return out

    def apply_L(self, X, Lb=None):
        Lb = Lb if Lb is not None else self.Lblock()
        return np.einsum("jab,bj->aj", Lb, X)

    def residual(self, X, Lb=None):
        return self.apply_L(X, Lb) + self.nonlinear(X)

    def _conv(self, fhat, other):
        """곱셈 연산자의 행렬 G[i,j] = fhat[m_i-m_j, n_i-n_j] (self 행, other 열)"""
        P = self.P
        di = (self.m[:, None] - other.m[None, :]) % P
        dj = (self.n[:, None] - other.n[None, :]) % P
        return fhat[di, dj]

    def jacobian(self, ustar, star_cell):
        """정상상태 u* (star_cell 의 (4,M0) 푸리에 계수) 주변 선형화 연산자 A = L + N'(u*) (조밀 8M x 8M)."""
        p = self.par
        us = star_cell.to_real(ustar)  # (4,P0,P0), P0 = star_cell.P
        assert star_cell.P == self.P, "두 셀의 격자 크기가 같아야 한다"
        usp, usm = np.roll(us, -1, 0), np.roll(us, 1, 0)
        f1 = 2 * p.eta * us + p.etap * (usp + usm) - 3 * p.c * us ** 2
        fh1 = fft2(f1) / (self.P ** 2)
        fh2 = fft2(us) / (self.P ** 2)
        M = self.M
        J = np.zeros((M, 8, M, 8), complex)
        Lb = self.Lblock()
        for j in range(M):
            J[j, :, j, :] += Lb[j]
        for n in range(4):
            J[:, n, :, n] += self._conv(fh1[n], self)
            G = p.etap * self._conv(fh2[n], self)
            J[:, n, :, (n + 1) % 4] += G
            J[:, n, :, (n - 1) % 4] += G
        return J.reshape(8 * M, 8 * M)


def mode_index(cell, mn):
    return cell.lookup[(int(mn[0]), int(mn[1]))]


def seed_state(cell, A_sat, ab_frac=0.3, rng=None, standing=True, chir=None):
    """정상파형 마름모 씨앗: c = a + b -> (1,1), a -> (1,0), b -> (0,1)."""
    rng = rng or np.random.default_rng(0)
    X = np.zeros((8, cell.M), complex)
    n = np.arange(4)

    def put(mn, comp_fn):
        j, jm = cell.lookup[mn], cell.lookup[(-mn[0], -mn[1])]
        v = comp_fn  # (4,)
        X[:4, j] += v / 2
        X[:4, jm] += np.conj(v) / 2

    ph = lambda: np.exp(1j * rng.uniform(0, 2 * np.pi))
    put((1, 1), A_sat * ph() * np.ones(4))
    for k, mn in enumerate(((1, 0), (0, 1))):
        al, be = ab_frac * A_sat * ph(), ab_frac * A_sat * ph()
        if chir is not None:  # chir=('p1','p3'): a 는 p=1, b 는 p=3 성분만 ('both' 는 둘 다)
            if chir[k] == "p1":
                be = 0.0
            elif chir[k] == "p3":
                al = 0.0
        elif not standing:
            be = 0.0
        put(mn, al * np.exp(1j * np.pi * n / 2) + be * np.exp(-1j * np.pi * n / 2))
    par = cell.par
    for j in range(cell.M):
        X[4:, j] = X[:4, j] / (par.d + par.Dv * cell.q[j])
    return X


def relax(cell, X, T=60000.0, h=20.0, tol=1e-13):
    """ETD-오일러 (고정점이 h 에 무관: L x + N = 0) 로 정상상태까지 이완."""
    Lb = cell.Lblock()
    E = np.zeros_like(Lb); Ph = np.zeros_like(Lb)
    for j in range(cell.M):
        Z = np.zeros((16, 16)); Z[:8, :8] = Lb[j] * h; Z[:8, 8:] = np.eye(8) * h
        EZ = sla.expm(Z)
        E[j] = EZ[:8, :8]; Ph[j] = EZ[:8, 8:]  # = ∫0^h e^{Ls} ds
    res = None
    for it in range(int(T / h)):
        N = cell.nonlinear(X)
        X = np.einsum("jab,bj->aj", E, X) + np.einsum("jab,bj->aj", Ph, N)
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
        J = cell.jacobian(X[:4], cell)
        dx = sla.lstsq(J, -F.T.reshape(-1), cond=cond)[0]
        X = X + dx.reshape(cell.M, 8).T
    return X, np.abs(cell.residual(X, Lb)).max(), iters


# ---------------------------------------------------------------------------
def make_cells(par, a, b, dk, kcut, kappas):
    """정상상태 셀(κ=0) 과 Bloch 섹터 셀들을 같은 격자 크기 P 로 만든다."""
    cells = [Cell(par, a, b, dk, kcut, kappa=k) for k in [(0.0, 0.0)] + list(kappas)]
    P = next_fast_len(4 * max(c.Mx for c in cells) + 2)
    return [Cell(par, a, b, dk, kcut, kappa=c.kappa, P=P) for c in cells]


def standing_seed(cell, A_sat, ab_frac=0.3, rng=None):
    """n -> -n 대칭 부분공간의 정상파형 마름모 씨앗: a, b 의 p=1, p=3 성분이 같은 진폭·위상."""
    rng = rng or np.random.default_rng(0)
    X = np.zeros((8, cell.M), complex)
    n = np.arange(4)
    ph = lambda: np.exp(1j * rng.uniform(0, 2 * np.pi))

    def put(mn, v):
        j, jm = cell.lookup[mn], cell.lookup[(-mn[0], -mn[1])]
        X[:4, j] += v / 2
        X[:4, jm] += np.conj(v) / 2
    put((1, 1), A_sat * ph() * np.ones(4))
    for mn in ((1, 0), (0, 1)):
        put(mn, ab_frac * A_sat * ph() * 2 * np.cos(np.pi * n / 2))
    for j in range(cell.M):
        X[4:, j] = X[:4, j] / (cell.par.d + cell.par.Dv * cell.q[j])
    return X


def sector_amps(cell, X):
    """층 Fourier 섹터 성분 |F_p(g)| (p=0..3) : (4, M)"""
    n = np.arange(4)
    return np.stack([np.abs(np.sum(X[:4] * np.exp(-1j * np.pi * p * n / 2)[:, None], 0) / 4) for p in range(4)])


def stacking_type(cell, X, mn=(1, 0)):
    F = sector_amps(cell, X)[:, cell.lookup[mn]]
    p1, p3 = F[1], F[3]
    return dict(p1=float(p1), p3=float(p3), ratio=float(min(p1, p3) / max(p1, p3, 1e-300)),
                type="standing" if min(p1, p3) / max(p1, p3, 1e-300) > 0.7 else ("spiral" if min(p1, p3) / max(p1, p3, 1e-300) < 0.3 else "mixed"))


def eigen_sector(cell_k, cell0, X0, nev=10, core=((0, 0), (1, 0)), k0=None):
    """섹터 셀의 선형화 연산자 고유값(실수부 큰 순 nev 개) 과 모드 구성.
    core: (m,n) 인덱스들 (b', c'=a+b' 등)의 세기 비율 W_core, 링 대역(|k|∈k0, k1 ±5%) 비율 W_ring."""
    J = cell_k.jacobian(X0[:4], cell0)
    w, V = np.linalg.eig(J)
    order = np.argsort(-w.real)[:nev]
    M = cell_k.M
    res = []
    for o in order:
        v = V[:, o].reshape(M, 8)
        pw = (np.abs(v[:, :4]) ** 2).sum(1)  # u 성분 세기 / 모드
        pw = pw / pw.sum()
        wc = sum(pw[cell_k.lookup[c]] for c in core if c in cell_k.lookup)
        kk = np.sqrt(cell_k.q)
        ent = dict(lam=complex(w[o]), W_core=float(wc))
        if k0 is not None:
            band = (np.abs(kk - k0) < 0.06 * k0) | (np.abs(kk - k1_of(cell_k, k0)) < 0.06 * k0)
            ent["W_ring"] = float(pw[band].sum())
        top = np.argsort(-pw)[:3]
        ent["top_modes"] = [(int(cell_k.m[j]), int(cell_k.n[j]), float(kk[j] / k0) if k0 else None, float(pw[j])) for j in top]
        res.append(ent)
    return res


def k1_of(cell, k0):
    return getattr(cell, "_k1", k0 / 1.85)


def classify_state(cell, X, tol_stripe=1e-6):
    """정상상태 종류: stripe (a,b≈0), standing (a,b 모두 p1≈p3), spiral (a,b 모두 한 쪽 섹터만), mixed."""
    Fa = sector_amps(cell, X)[:, cell.lookup[(1, 0)]]
    Fb = sector_amps(cell, X)[:, cell.lookup[(0, 1)]]
    Fc = sector_amps(cell, X)[:, cell.lookup[(1, 1)]]
    amp_a, amp_b = float(np.hypot(Fa[1], Fa[3])), float(np.hypot(Fb[1], Fb[3]))
    info = dict(Ac=float(Fc[0]), Aa=amp_a, Ab=amp_b, a_p1=float(Fa[1]), a_p3=float(Fa[3]), b_p1=float(Fb[1]), b_p3=float(Fb[3]),
                A2a=float(sector_amps(cell, X)[0, cell.lookup[(2, 0)]]), A2b=float(sector_amps(cell, X)[0, cell.lookup[(0, 2)]]))
    if max(amp_a, amp_b) < tol_stripe:
        info["type"] = "stripe"
        return info
    ra = min(Fa[1], Fa[3]) / max(Fa[1], Fa[3], 1e-300)
    rb = min(Fb[1], Fb[3]) / max(Fb[1], Fb[3], 1e-300)
    info["ratio_a"], info["ratio_b"] = float(ra), float(rb)
    info["type"] = "standing" if min(ra, rb) > 0.7 else ("spiral" if max(ra, rb) < 0.1 else "mixed")
    return info


SEEDS = {
    "sym_standing": lambda c, A, rng: standing_seed(c, A, rng=rng),
    "std_phases": lambda c, A, rng: seed_state(c, A, rng=rng, standing=True),
    "spiral_p1p3": lambda c, A, rng: seed_state(c, A, rng=rng, chir=("p1", "p3")),
    "spiral_p3p1": lambda c, A, rng: seed_state(c, A, rng=rng, chir=("p3", "p1")),
    "spiral_p1p1": lambda c, A, rng: seed_state(c, A, rng=rng, chir=("p1", "p1")),
}


def find_states(cell0, A_sat, seeds=("sym_standing", "std_phases", "spiral_p1p3", "spiral_p1p1"), nrep=2, T=300000.0, h=50.0):
    """여러 씨앗에서 이완 + Newton 으로 정상상태들을 찾고 (종류, 진폭) 로 중복 제거."""
    found = []
    for sname in seeds:
        for rep in range(nrep):
            rng = np.random.default_rng(100 * rep + 7)
            X = SEEDS[sname](cell0, A_sat, rng)
            X, _ = relax(cell0, X, T=T, h=h)
            if not np.all(np.isfinite(X)):  # 이완이 발산한 씨앗은 건너뜀 (v5: 큰 η)
                continue
            try:
                X, r, it = newton(cell0, X)
            except ValueError:
                continue
            if r > 1e-10:
                continue
            info = classify_state(cell0, X)
            key = (info["type"], round(info["Aa"], 6), round(info["Ab"], 6))
            if not any(f["key"] == key for f in found):
                found.append(dict(key=key, X=X, info=info, seed=sname, newton_res=float(r)))
    return found


def coset_reps(a, b):
    """Λ = span{a,b} (a=(ax,ay), b=(ax,-ay)) 의 Z² 에서의 잉여류 대표 (|det| = 2 ax ay 개)."""
    ax, ay = int(a[0]), int(abs(a[1]))
    return [(float(x), float(y)) for x in range(2 * ax) for y in range(ay)]
