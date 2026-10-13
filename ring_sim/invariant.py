"""v5 A2: 유한 불변 모드 집합 반복.

격자 정수 벡터 (단위 dk) 위에서, 두 모드의 합이 임계원 (바깥 k0: 섹터 0, 안쪽 k1: 섹터 1) 위에 있으면 합 모드를 집합에 추가한다.
'임계원 위' = 해당 섹터 성장률이 봉우리 값에서 tol*σ0 (기본 0.5) 이내."""
import numpy as np
from .model import Params, peak, solve_kappa, growth_rate


class Rings:
    def __init__(self, ratio, r=1.0, Dv=2.50309, eta=0.15, tol=0.5):
        base = Params(Dv=Dv)
        self.s0, q0 = peak(base, 0)
        self.k0 = np.sqrt(q0)
        ku, kv = solve_kappa(ratio, r, base)
        self.par = Params(Dv=Dv, ku=ku, kv=kv, eta=eta, etap=-eta / 2)
        self.ratio, self.tol = ratio, tol
        self.k1 = self.k0 / ratio
        self.s1 = float(growth_rate(self.par, 1, self.k1 ** 2))  # 안쪽 봉우리 근처 값 (≈ -r σ0)
        s1pk, q1 = peak(self.par, 1, qmax=0.5)
        self.s1pk, self.k1pk = s1pk, np.sqrt(q1)

    def on_outer(self, k):
        return growth_rate(self.par, 0, np.asarray(k) ** 2) >= self.s0 * (1 - self.tol)

    def on_inner(self, k):
        return growth_rate(self.par, 1, np.asarray(k) ** 2) >= self.s1pk - self.tol * self.s0


def ring_points(rings, dk, scale):
    """dk 격자에서 바깥/안쪽 띠 안의 모든 정수 벡터 (dict: 벡터->'out'/'in')."""
    out = {}
    for name, kc, f in (("out", rings.k0, rings.on_outer), ("in", rings.k1pk, rings.on_inner)):
        R = int(np.ceil(1.15 * kc / dk)) + 2
        g = np.arange(-R, R + 1)
        X, Y = np.meshgrid(g, g, indexing="ij")
        kk = np.hypot(X, Y) * dk
        m = f(kk)
        for x, y in zip(X[m], Y[m]):
            out[(int(x), int(y))] = name
    return out


def closure(seed, rings, dk, max_depth=6, max_size=4000):
    """seed: 정수 벡터 목록 (±는 자동 추가). 반환: sizes[d] = 깊이 d 집합 크기, final set, closed 여부, 깊이별 (바깥,안쪽) 개수."""
    pts = ring_points(rings, dk, None)
    K = set()
    for v in seed:
        K.add((int(v[0]), int(v[1]))); K.add((-int(v[0]), -int(v[1])))
    sizes = [len(K)]
    comp = []
    closed = False
    for d in range(max_depth):
        A = np.array(sorted(K))
        S = (A[:, None, :] + A[None, :, :]).reshape(-1, 2)
        S = np.unique(S, axis=0)
        new = {(int(x), int(y)) for x, y in S if (int(x), int(y)) in pts} - K
        if not new:
            closed = True
            sizes.append(len(K))
            break
        K |= new
        sizes.append(len(K))
        if len(K) > max_size:
            break
    nout = sum(1 for v in K if pts.get(v) == "out"); nin = sum(1 for v in K if pts.get(v) == "in")
    return dict(sizes=sizes, closed=closed, K=sorted(K), n_out=nout, n_in=nin)


def closure_exact(seed, k0, k1, max_depth=6, max_size=3000, rtol=1e-7):
    """연속체 (격자 없음) 불변 집합: 복소수 파수, 반지름이 k0 또는 k1 (상대 rtol) 인 합만 추가. 반환 sizes[d], closed."""
    def key(z):
        return (round(z.real / k0, 6), round(z.imag / k0, 6))
    K = {}
    for z in seed:
        for w in (z, -z):
            K[key(w)] = w
    sizes = [len(K)]
    closed = False
    for d in range(max_depth):
        A = np.array(list(K.values()))
        S = (A[:, None] + A[None, :]).ravel()
        r = np.abs(S)
        ok = (np.abs(r / k0 - 1) < rtol) | (np.abs(r / k1 - 1) < rtol)
        new = {}
        for z in S[ok]:
            kz = key(z)
            if kz not in K and kz not in new:
                new[kz] = z
        if not new:
            closed = True
            sizes.append(len(K))
            break
        K.update(new)
        sizes.append(len(K))
        if len(K) > max_size:
            break
    return dict(sizes=sizes, closed=closed)
