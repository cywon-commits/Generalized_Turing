"""실행별 최종 상태 그림, 원별 세기 시계열, 단계 2 문턱 그림.
사용: python scripts/make_figures.py"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.model import peak, Params, RATIOS

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
S0, Q0 = peak(Params(), 0)
K0 = np.sqrt(Q0)


def full_power(Vc, N, L, kind):
    """압축 스펙트럼 -> shift 된 (N,N) 전력 배열 (|F|^2)."""
    dk = 2 * np.pi / L
    kx = np.fft.fftfreq(N, 1.0 / N) * dk
    ky = np.fft.rfftfreq(N, 1.0 / N) * dk
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    mask = k2 <= (2 / 3 * np.pi * N / L) ** 2
    idx = np.flatnonzero(mask.ravel())
    F = Vc[0] / (4 * N * N)
    P = np.abs(F[0]) ** 2 if kind == "out" else np.abs(F[1]) ** 2 + np.abs(F[3]) ** 2
    half = np.zeros(k2.shape)
    half.ravel()[idx] = P
    full = np.zeros((N, N))
    full[:, : N // 2 + 1] = half
    ii = (-np.arange(N)) % N
    jj = np.arange(1, N // 2)
    full[np.ix_(ii, N - jj)] = half[:, jj]
    return np.fft.fftshift(full), dk


def run_fig(path):
    d = json.load(open(path))
    c = d["cfg"]
    fz = path.replace(".json", ".final.npz")
    if not os.path.exists(fz):
        return
    z = np.load(fz)
    u, V = z["u"], z["V"]
    N = u.shape[-1]
    L = 2 * np.pi * c["R0"] / K0
    ratio = RATIOS[c["case"]]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    crop = slice(N // 2 - 60, N // 2 + 60)
    ax[0].imshow(u[0][:256, :256].T, origin="lower", cmap="RdBu_r"); ax[0].set_title("layer 0 u (crop 256²)")
    ax[1].imshow(u.mean(0)[:256, :256].T, origin="lower", cmap="RdBu_r"); ax[1].set_title("sector field m0 (crop)")
    th = np.linspace(0, 2 * np.pi, 400)
    for a, kind, lab in ((ax[2], "out", "outer |F0|²"), (ax[3], "in", "inner |F1|²+|F3|²")):
        P, dk = full_power(V.astype(complex), N, L, kind)
        ext = dk * N / 2
        a.imshow(np.log10(P[crop, crop].T + 1e-30), origin="lower", cmap="magma",
                 extent=[-60 * dk, 60 * dk, -60 * dk, 60 * dk], vmin=np.log10(P.max() + 1e-30) - 8)
        for k, ls in ((K0, ":"), (K0 / ratio, "--")):
            a.plot(k * np.cos(th), k * np.sin(th), "c" + ls, lw=0.8)
        a.set_title(lab + f" (max {P.max():.1e})")
    fig.suptitle(f"{c['name']}   verdict: {d['final']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, c["name"] + ".png"), dpi=80)
    plt.close(fig)


def series_fig(stage, files):
    groups = {}
    for f in files:
        d = json.load(open(f)); c = d["cfg"]
        groups.setdefault((c["case"], c["ic"]), []).append(d)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")
    for gi, (key, ds) in enumerate(sorted(groups.items())):
        for d in ds:
            t = [s["t"] for s in d["series"]]
            axs[0].semilogy(t, [s["P0"] for s in d["series"]], color=cmap(gi), alpha=0.7,
                            label="/".join(key) if d is ds[0] else None)
            axs[1].semilogy(t, [max(s["Pin"], 1e-30) for s in d["series"]], color=cmap(gi), alpha=0.7)
    axs[0].set_title(f"stage {stage}: outer ring power P0(t)"); axs[1].set_title("inner ring power Pin(t)")
    for a in axs: a.set_xlabel("t")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"timeseries_stage{stage}.png"), dpi=90); plt.close(fig)


def threshold_fig():
    import csv
    rows = [r for r in csv.DictReader(open("results/summary.csv")) if r["stage"] == "2"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r_val, col in ((0.5, "C0"), (1.0, "C1"), (2.0, "C2")):
        pts = sorted((float(x["eta"]), float(x["Lam_in"])) for x in rows if float(x["r"]) == r_val)
        if pts:
            ax.plot(*zip(*pts), "o", color=col, alpha=0.6)
            e = sorted(set(p[0] for p in pts))
            ax.plot(e, [np.mean([p[1] for p in pts if p[0] == v]) for v in e], "-", color=col, label=f"r={r_val}")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("η"); ax.set_ylabel("Λ_in = d ln Pin/dt (late window)")
    ax.legend(); ax.set_title("stage 2: parametric-drive threshold (12-fold, O)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "threshold_stage2.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    for f in sorted(glob.glob("runs/stage*/*.json")):
        run_fig(f)
    for st in range(1, 6):
        fs = sorted(glob.glob(f"runs/stage{st}/*.json"))
        if fs:
            series_fig(st, fs)
    threshold_fig()
