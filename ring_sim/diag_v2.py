"""v2 진단: 피크 수 N0(t), N_in(t), 피크 방향 집합과 연쇄 궤도 {θ0 + 2kθ*} 잔차, 삼각형 연결망, 위상 고정 R."""
import numpy as np
from .model import NLAYER
from .diagnostics import Diag

KS = (2, 4, 8)


class DiagV2(Diag):
    def __init__(self, S, ratio, R0, k0, peak_frac=0.05, tol_tri=1.0):
        super().__init__(S, ratio, R0, k0, tri_w=0)
        self.peak_frac, self.tol_tri = peak_frac, tol_tri
        N = S.N
        self.N = N
        self.theta_star = np.arccos(ratio / 2)  # 안쪽 파와 바깥 파 사이 각
        kx = np.fft.fftfreq(N, 1.0 / N)
        ky = np.arange(N // 2 + 1)
        self.kxi = np.where(S.idx // (N // 2 + 1) > N // 2, S.idx // (N // 2 + 1) - N, S.idx // (N // 2 + 1))
        self.kyi = S.idx % (N // 2 + 1)
        kkl = np.hypot(self.kxi, self.kyi)  # 격자 단위 |k|
        self.rout = np.abs(kkl - R0) < 0.1 * R0
        self.rin = np.abs(kkl - R0 / ratio) < 0.1 * R0
        self.R0 = R0

    def set_track(self, info):
        """초기 조건 정보의 c, a, b 격자 벡터를 추적 (줄무늬/마름모 진폭과 '나머지' 전력)."""
        self.track = {k: np.array(info[k], int)[None] for k in ("c", "a", "b") if k in info}

    def _amp(self, F, vec, ps):
        z = [self._get(F, self._ref(vec, p))[0] for p in ps]
        return float(np.sqrt(sum(abs(x) ** 2 for x in z)))

    def _full(self, P, band):
        """압축 세기 -> 전평면 (N,N) 세기 (kx,ky 정수 인덱스 mod N), 대역 밖은 0"""
        N = self.N
        full = np.zeros((N, N))
        i = self.kxi[band] % N
        j = self.kyi[band]
        full[i, j] = P[band]
        full[(-self.kxi[band]) % N, (-j) % N] = P[band]
        return full

    def peaks(self, P, band):
        """국소 최대(8-이웃)이며 대역 최대의 peak_frac 이상인 전평면 피크 벡터와 세기. ±k 쌍을 모두 반환."""
        full = self._full(P, band)
        m = full.max()
        if m <= 0:
            return np.zeros((0, 2), int), np.zeros(0)
        loc = np.ones_like(full, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    loc &= full >= np.roll(np.roll(full, di, 0), dj, 1)
        ok = loc & (full >= self.peak_frac * m)
        ii, jj = np.nonzero(ok)
        # 같은 세기의 인접 동률 제거 (평평한 봉우리): 이웃과 동률이고 사전식으로 앞선 것만 유지
        vec = np.stack([np.where(ii > self.N // 2, ii - self.N, ii), np.where(jj > self.N // 2, jj - self.N, jj)], 1)
        pw = full[ii, jj]
        keep = np.ones(len(vec), bool)
        for a in range(len(vec)):
            if not keep[a]:
                continue
            d = np.abs(vec - vec[a]).max(1)
            near = (d <= 1) & (np.arange(len(vec)) != a) & (np.abs(pw - pw[a]) <= 1e-12 * pw[a])
            keep[near & (np.arange(len(vec)) > a)] = False
        return vec[keep], pw[keep]

    @staticmethod
    def _half(vec):
        """±k 쌍을 하나로: ky>0 또는 (ky=0, kx>0)"""
        return vec[(vec[:, 1] > 0) | ((vec[:, 1] == 0) & (vec[:, 0] > 0))]

    def angles(self, vec):
        h = self._half(vec)
        return np.sort(np.degrees(np.arctan2(h[:, 1], h[:, 0])) % 180.0) if len(h) else np.zeros(0)

    def orbit_residual(self, ang, th0, offset):
        """각도(도) 집합이 {th0 + offset + 2kθ*, |k|≤K} (mod 180) 에 얼마나 맞는가: K 별 rms 잔차(도)"""
        if len(ang) == 0:
            return {K: float("nan") for K in KS}
        ts = np.degrees(self.theta_star)
        out = {}
        for K in KS:
            orb = (th0 + offset + 2 * ts * np.arange(-K, K + 1)) % 180.0
            d = np.abs(((ang[:, None] - orb[None, :] + 90.0) % 180.0) - 90.0).min(1)
            out[K] = float(np.sqrt((d ** 2).mean()))
        return out

    def connectivity(self, vin, vout):
        """안쪽 피크 벡터(전평면, ±) 쌍 a,b 와 바깥 피크 c 의 |a+b-c|<=tol 삼각형. 안쪽 모드별 소속 삼각형 수의 분포."""
        if len(vin) == 0 or len(vout) == 0:
            return {"0": len(vin), "1": 0, "2": 0, "3+": 0}, 0, []
        cnt = np.zeros(len(vin), int)
        tris = []
        for ia in range(len(vin)):
            for ib in range(ia, len(vin)):
                s = vin[ia] + vin[ib]
                d = np.abs(vout - s).max(1)
                for ic in np.nonzero(d <= self.tol_tri)[0]:
                    tris.append((ia, ib, ic, int(d[ic])))
                    cnt[ia] += 1
                    if ib != ia:
                        cnt[ib] += 1
        dist = {"0": int((cnt == 0).sum()), "1": int((cnt == 1).sum()), "2": int((cnt == 2).sum()), "3+": int((cnt >= 3).sum())}
        return dist, len(tris), tris

    def record(self, V, t):
        out = super().record(V, t)
        F = self.spectrum(V)
        P0 = np.abs(F[0]) ** 2
        Pin = np.abs(F[1]) ** 2 + np.abs(F[3]) ** 2
        vout, pout = self.peaks(P0, self.rout)
        vin, pin = self.peaks(Pin, self.rin)
        out["N0"] = len(self._half(vout))
        out["Nin"] = len(self._half(vin))
        aout, ain = self.angles(vout), self.angles(vin)
        out["ang_out"] = [round(float(a), 2) for a in aout]
        out["ang_in"] = [round(float(a), 2) for a in ain]
        if len(aout):
            top = vout[np.argmax(pout)]
            th0 = float(np.degrees(np.arctan2(top[1], top[0])) % 180.0)
            ro = self.orbit_residual(aout, th0, 0.0)
            ri = self.orbit_residual(ain, th0, float(np.degrees(self.theta_star)))
            for K in KS:
                out[f"res_out_K{K}"] = ro[K]
                out[f"res_in_K{K}"] = ri[K]
        tr = getattr(self, "track", None)
        if tr:
            Ac = self._amp(F, tr["c"], (0,))
            Aa = self._amp(F, tr["a"], (1, 3))
            Ab = self._amp(F, tr["b"], (1, 3))
            mult = 2 if int(tr["c"][0, 1]) == 0 else 1  # ky=0 열에서는 ±c 가 모두 반평면에 저장됨
            out.update(Ac=Ac, Aa=Aa, Ab=Ab,
                       P0_rest=float(max(P0[self.rout].sum() - mult * Ac ** 2, 0.0)),
                       Pin_rest=float(max(Pin[self.rin].sum() - Aa ** 2 - Ab ** 2, 0.0)))
        dist, ntri, tris = self.connectivity(vin, vout)
        out["conn"] = dist
        out["ntri_peak"] = ntri
        # 삼각형 위상 고정 (정확히 닫히는 것: d=0), 가중 R: Φ = arg(F1[a] F3[b] conj F0[c])
        ph, w = [], []
        for ia, ib, ic, d in tris:
            if d == 0:
                for (A, B) in ((vin[ia], vin[ib]), (vin[ib], vin[ia])):
                    fa = self._get(F, self._ref(A[None], 1)); fb = self._get(F, self._ref(B[None], 3))
                    fc = self._get(F, self._ref(vout[ic][None], 0))
                    z = fa * fb * np.conj(fc)
                    ph.append(np.angle(z[0])); w.append(abs(z[0]))
        if w and sum(w) > 0:
            w = np.array(w); ph = np.array(ph)
            out["R_peak"] = float(abs((w * np.exp(1j * ph)).sum() / w.sum()))
            out["Phi_peak"] = float(np.angle((w * np.exp(1j * ph)).sum()))
        else:
            out["R_peak"], out["Phi_peak"] = float("nan"), float("nan")
        return out
