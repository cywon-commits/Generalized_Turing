"""격자 파수 선택과 삼각형 (a + b = c) 열거."""
import itertools
import numpy as np

NFOLD = {"square": 4, "hex": 6, "oct": 8, "dec": 10, "dodec": 12,
         "ctrlA": 12, "ctrlB": 12}


def _cands(r, theta, box=3, top=6):
    v = np.array([r * np.cos(theta), r * np.sin(theta)])
    base = np.round(v).astype(int)
    pts = [(base[0] + i, base[1] + j) for i in range(-box, box + 1) for j in range(-box, box + 1)]
    pts = np.array(pts)
    rr = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    dang = np.angle(np.exp(1j * (ang - theta)))
    cost = (rr / r - 1) ** 2 + dang ** 2
    o = np.argsort(cost)[:top]
    return pts[o]


def _cost(v, r, theta):
    v = np.asarray(v, float)
    rr = np.hypot(v[..., 0], v[..., 1])
    ang = np.arctan2(v[..., 1], v[..., 0])
    return (rr / r - 1) ** 2 + np.angle(np.exp(1j * (ang - theta))) ** 2


def design_vectors(R0, ratio, nfold, resonant):
    """외부(outer) 축 nfold/2 개, 안쪽(inner) 축 nfold/2 개의 정수 격자 벡터.

    resonant=True 이면 안쪽 nfold 개 벡터 A_j 를 골라 C_m = A_m + A_{m-1} 로 외부 벡터를
    정의하여 (1,1,0) 삼각형이 격자 위에서 정확히 닫히게 한다 (반경 오차 최소화).
    그렇지 않으면 외부·안쪽 각각 독립적으로 이상적 방향에 가장 가까운 격자점.
    반환: outer (n/2,2), inner (n/2,2) — ± 쌍 중 한 쪽만.
    """
    n2 = nfold // 2
    rin = R0 / ratio
    th_out = [2 * np.pi * m / nfold for m in range(n2)]
    th_in = [2 * np.pi * (m + 0.5) / nfold for m in range(n2)]
    if not resonant:
        out = np.array([_cands(R0, t, top=1)[0] for t in th_out])
        inn = np.array([_cands(rin, t, top=1)[0] for t in th_in])
        return out, inn
    cand = [_cands(rin, t, top=6) for t in th_in]
    best, bc = None, 1e9
    for combo in itertools.product(*[range(len(c)) for c in cand]):
        A = np.array([cand[j][combo[j]] for j in range(n2)])
        full = np.concatenate([A, -A])  # A_0..A_{n-1}
        c = 0.0
        for j in range(n2):
            c += _cost(A[j], rin, th_in[j])
        for m in range(n2):
            C = full[m] + full[m - 1]
            c += _cost(C, R0, th_out[m])
        if c < bc:
            bc, best = c, A
    full = np.concatenate([best, -best])
    out = np.array([full[m] + full[m - 1] for m in range(n2)])
    return out, best


def band_points(R0, dk_ratio_lo, dk_ratio_hi, half_plane=False):
    """|v| in [lo,hi] (격자 단위) 인 정수 벡터."""
    m = int(np.ceil(dk_ratio_hi)) + 1
    ii, jj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    r = np.hypot(ii, jj)
    ok = (r >= dk_ratio_lo) & (r <= dk_ratio_hi)
    pts = np.stack([ii[ok], jj[ok]], 1)
    return pts


def enumerate_triangles(R0, ratio, w=1.5):
    """a (p=1, 안쪽 대역), b (p=3, 안쪽 대역), c = a+b (바깥 대역) 의 정확히 닫히는 삼각형.

    대역 반폭 w 는 격자 단위(dk). 반환 (T,6) int: ax,ay,bx,by,cx,cy
    """
    rin = R0 / ratio
    A = band_points(R0, rin - w, rin + w)
    out = []
    for a in A:
        c = a[None, :] + A
        rc = np.hypot(c[:, 0], c[:, 1])
        ok = np.abs(rc - R0) <= w
        for b, cc in zip(A[ok], c[ok]):
            out.append((a[0], a[1], b[0], b[1], cc[0], cc[1]))
    return np.array(out, dtype=int).reshape(-1, 6)
