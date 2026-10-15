"""v7 과제 C: 8회 상태의 스펙트럼 진단. 입력 u (4,N,N) 실공간 층별 u 장, Cx (격자 단위 바깥 반지름), ρ.
바깥 원 = p=0 섹터, 대역 |k|/Cx ∈ [1-w, 1+w];  안쪽 원 = p=1,3 섹터 (세기 합), 대역 |k|/(Cx/ρ) ∈ [1-w, 1+w]."""
import numpy as np

RHO8 = 2 * np.cos(np.pi / 8)


def sector_power(u):
    N = u.shape[1]
    F = np.fft.fft2(u, axes=(1, 2)) / (N * N)
    n = np.arange(4)
    Fp = [np.tensordot(np.exp(-1j * np.pi * p * n / 2), F, axes=(0, 0)) / 4 for p in range(4)]
    return [np.abs(x) ** 2 for x in Fp], N


def _modes(P, N, Cx, kc, w):
    k = np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    kk = np.hypot(KX, KY)
    band = np.abs(kk - kc) <= w * kc
    return KX[band], KY[band], P[band]


def ring_diag(kx, ky, pw, ideal_deg, window_deg, top=16):
    tot = pw.sum()
    order = np.argsort(pw)[::-1]
    W = float(pw[order[:top]].sum() / tot)
    ang = np.degrees(np.arctan2(ky, kx)) % 180.0
    res, peaks, widths = [], [], []
    for th in ideal_deg:
        d = (ang - th + 90.0) % 180.0 - 90.0
        sel = np.abs(d) <= window_deg
        if sel.sum() == 0 or pw[sel].sum() <= 0:
            peaks.append(0.0); widths.append(np.nan); continue
        wgt = pw[sel] / pw[sel].sum()
        m = (wgt * d[sel]).sum()
        widths.append(float(np.sqrt((wgt * (d[sel] - m) ** 2).sum())))
        peaks.append(float(np.sqrt(pw[sel].max())))
    return dict(W16=W, widths_deg=widths, peak_amps=peaks, total=float(tot))


def diagnose(u, Cx, rho=RHO8, w=0.2, top=16):
    P, N = sector_power(u)
    kx, ky, p0 = _modes(P[0], N, Cx, Cx, w)
    out = ring_diag(kx, ky, p0, [0, 45, 90, 135], 22.5 / 1.0, top)
    pa = out["peak_amps"]
    axis, diag = np.mean([pa[0], pa[2]]), np.mean([pa[1], pa[3]])
    kxi, kyi, pin = _modes(P[1] + P[3], N, Cx, Cx / rho, w)
    inn = ring_diag(kxi, kyi, pin, [22.5 + 45 * m for m in range(4)], 22.5, top)
    ia = np.array(inn["peak_amps"])
    return dict(outer_W16=out["W16"], outer_width_deg=float(np.nanmean(out["widths_deg"])), outer_diag_over_axis=float(diag / axis),
                inner_W16=inn["W16"], inner_width_deg=float(np.nanmean(inn["widths_deg"])), inner_peak_std_over_mean=float(ia.std() / ia.mean()),
                outer_total=out["total"], inner_total=inn["total"])
