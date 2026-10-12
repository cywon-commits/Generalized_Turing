"""진단량: 원별 세기, n회 차수 변수, 삼각형 위상 고정, 층 구조."""
import numpy as np
from .model import NLAYER
from .lattice import enumerate_triangles

HARM = (4, 6, 8, 10, 12)


class Diag:
    def __init__(self, S, ratio, R0, k0, tri_w=1.5):
        self.S, self.ratio, self.R0, self.k0 = S, ratio, R0, k0
        N = S.N
        self.norm = NLAYER * N * N
        kk = np.sqrt(S.k2.ravel()[S.idx])
        ky_i = (S.idx % (N // 2 + 1))
        kx_i = (S.idx // (N // 2 + 1))
        kx_i = np.where(kx_i > N // 2, kx_i - N, kx_i)
        theta = np.arctan2(ky_i, kx_i)  # half plane ky>=0 -> [0,pi]
        self.theta = theta
        k1 = k0 / ratio
        self.band_out = np.abs(kk - k0) < 0.1 * k0
        self.band_in = np.abs(kk - k1) < 0.1 * k0
        self.kk = kk
        self.e = {j: np.exp(1j * j * theta) for j in HARM}
        # compressed 인덱스 표
        tab = -np.ones(S.k2.shape, int)
        tab.ravel()[S.idx] = np.arange(S.K)
        self.tab = tab
        if tri_w and tri_w > 0:
            T = enumerate_triangles(R0, ratio, tri_w)
        else:  # v2 진단은 격자 삼각형 목록을 쓰지 않는다 (피크 기반)
            T = np.zeros((0, 6), int)
        self.ntri_lattice = len(T)
        self.tri = self._refs(T)

    def _ref(self, v, p):
        """격자 벡터 v (M,2), 섹터 p -> (p', 압축 인덱스, conj 플래그)"""
        N = self.S.N
        vx, vy = v[:, 0].copy(), v[:, 1].copy()
        flip = vy < 0
        vx = np.where(flip, -vx, vx)
        vy = np.where(flip, -vy, vy)
        pp = np.where(flip, (-p) % NLAYER, p)
        ci = self.tab[vx % N, vy]
        return pp, ci, flip

    def _refs(self, T):
        if len(T) == 0:
            return None
        a, b, c = T[:, 0:2], T[:, 2:4], T[:, 4:6]
        return (self._ref(a, 1), self._ref(b, 3), self._ref(c, 0))

    @staticmethod
    def _get(U, ref):
        pp, ci, fl = ref
        z = U[pp, ci]
        return np.where(fl, np.conj(z), z)

    def spectrum(self, V):
        F = V[0] / self.norm  # (4,K) 복소 진폭
        return F

    def harmonics(self, P, band):
        w = P[band]
        tot = w.sum()
        if tot <= 0:
            return {j: 0.0 for j in HARM}
        return {j: float(abs((w * self.e[j][band]).sum()) / tot) for j in HARM}

    def record(self, V, t):
        F = self.spectrum(V)
        P0 = np.abs(F[0]) ** 2
        P1 = np.abs(F[1]) ** 2
        P3 = np.abs(F[3]) ** 2
        P2 = np.abs(F[2]) ** 2
        Pin_k = P1 + P3
        out = dict(t=t,
                   P0=float(P0[self.band_out].sum()),
                   Pin=float(Pin_k[self.band_in].sum()),
                   P2=float(P2.sum()))
        for j, v in self.harmonics(P0, self.band_out).items():
            out[f"S0_{j}"] = v
        for j, v in self.harmonics(Pin_k, self.band_in).items():
            out[f"Sin_{j}"] = v
        out["amax0"] = float(np.sqrt(P0[self.band_out].max())) if self.band_out.any() else 0.0
        out["amaxin"] = float(np.sqrt(Pin_k[self.band_in].max())) if self.band_in.any() else 0.0
        if self.tri is not None:
            fa = self._get(F, self.tri[0]); fb = self._get(F, self.tri[1]); fc = self._get(F, self.tri[2])
            w = np.abs(fa * fb * fc)
            ph = np.exp(1j * np.angle(fa * fb * np.conj(fc)))
            sw = w.sum()
            out["R"] = float(abs((w * ph).sum()) / sw) if sw > 0 else 0.0
            # 활성 삼각형: 세 모드 모두 각 원 최대 진폭의 20% 이상
            act = (np.abs(fa) > 0.2 * out["amaxin"]) & (np.abs(fb) > 0.2 * out["amaxin"]) & \
                  (np.abs(fc) > 0.2 * out["amax0"])
            out["n_act"] = int(act.sum())
            out["R_act"] = float(abs(ph[act].mean())) if act.any() else 0.0
            out["Phi_act"] = float(np.angle(ph[act].mean())) if act.any() else 0.0
        else:
            out.update(R=0.0, n_act=0, R_act=0.0, Phi_act=0.0)
        return out

    def final_analysis(self, V):
        """최종 상태의 정성적 분석: 안쪽 원 봉우리 수, 적층 유형."""
        F = self.spectrum(V)
        Pin_k = np.abs(F[1]) ** 2 + np.abs(F[3]) ** 2
        res = {}
        # 각도 히스토그램 (반평면 [0,pi), 2도 간격)
        nb = 90
        ib = np.minimum((self.theta[self.band_in] / np.pi * nb).astype(int), nb - 1)
        H = np.bincount(ib, weights=Pin_k[self.band_in], minlength=nb)
        Hs = np.convolve(np.concatenate([H[-2:], H, H[:2]]), np.ones(3) / 3, mode="same")[2:-2]
        thr = 0.2 * Hs.max() if Hs.max() > 0 else np.inf
        peaks = [i for i in range(nb) if Hs[i] >= thr and Hs[i] >= Hs[i - 1] and Hs[i] > Hs[(i + 1) % nb]]
        # 인접 봉우리 병합 (5도 이내)
        merged = []
        for i in peaks:
            if not merged or (i - merged[-1]) * 2 > 5:
                merged.append(i)
        if len(merged) > 1 and (merged[0] + nb - merged[-1]) * 2 <= 5:
            merged.pop()
        res["inner_axes"] = len(merged)  # 파동 벡터 개수는 2배
        res["inner_peak_deg"] = [round(2.0 * i, 1) for i in merged]
        # 안쪽 원: 방향 점유 (전체 힘의 90% 를 담는 모드 수)
        pk = np.sort(Pin_k[self.band_in])[::-1]
        cs = np.cumsum(pk) / pk.sum() if pk.sum() > 0 else pk
        res["inner_modes90"] = int(np.searchsorted(cs, 0.9) + 1) if pk.sum() > 0 else 0
        # 층 구조: 강한 안쪽 모드에서 |F1|^2 vs |F3|^2 비대칭
        strong = Pin_k > 0.1 * Pin_k.max() if Pin_k.max() > 0 else np.zeros_like(Pin_k, bool)
        strong &= self.band_in
        if strong.any():
            p1, p3 = np.abs(F[1][strong]) ** 2, np.abs(F[3][strong]) ** 2
            asym = np.abs(p1 - p3) / (p1 + p3)
            frac = float((((p1 + p3) * (asym > 0.8)).sum()) / (p1 + p3).sum())
            res["spiral_frac"] = frac
            res["stacking"] = "spiral" if frac > 0.7 else ("standing" if frac < 0.3 else "mixed")
        else:
            res["spiral_frac"] = float("nan")
            res["stacking"] = "none"
        # 바깥 원 봉우리 수
        P0 = np.abs(F[0]) ** 2
        ib0 = np.minimum((self.theta[self.band_out] / np.pi * nb).astype(int), nb - 1)
        H0 = np.bincount(ib0, weights=P0[self.band_out], minlength=nb)
        H0s = np.convolve(np.concatenate([H0[-2:], H0, H0[:2]]), np.ones(3) / 3, mode="same")[2:-2]
        thr0 = 0.2 * H0s.max()
        pk0 = [i for i in range(nb) if H0s[i] >= thr0 and H0s[i] >= H0s[i - 1] and H0s[i] > H0s[(i + 1) % nb]]
        res["outer_axes"] = len(pk0)
        return res
