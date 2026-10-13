"""v2 그림: (i) N0(t), N_in(t) 시계열(두 비율 겹침), (ii) 시점별 스펙트럼(저장된 경우), (iii) 최종 상태."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.model import peak, Params, RATIOS

FIG = "results_v2/figures"
os.makedirs(FIG, exist_ok=True)
DV = 2.50309
S0, Q0 = peak(Params(Dv=DV), 0)
K0 = np.sqrt(Q0)


def peaks_fig(stage, label):
    files = sorted(glob.glob(f"runs_v2/stage{stage}/*.json"))
    if not files:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    col = {"dodec": "C3", "ctrlA": "C0"}
    for f in files:
        d = json.load(open(f)); c = d["cfg"]; s = d["series"]
        t = np.array([x["t"] for x in s]) * S0  # t σ0 (무차원)
        ls = {0.09: "-", 0.1: "-", 0.11: "--", 0.12: ":", 0.083: "-."}.get(c["eta"], "-")
        lab = f"{c['case']} η={c['eta']} s{c['seed']}"
        ax[0].plot(t, [x["N0"] for x in s], ls, color=col[c["case"]], alpha=0.8, label=lab)
        ax[1].plot(t, [x["Nin"] for x in s], ls, color=col[c["case"]], alpha=0.8)
        ax[2].semilogy(t, [max(x["Pin"], 1e-30) for x in s], ls, color=col[c["case"]], alpha=0.8)
    ax[0].set_title("outer peak count N0(t)"); ax[1].set_title("inner peak count N_in(t)  (±k pair = 1)")
    ax[2].set_title("inner ring power Pin(t)")
    for a in ax: a.set_xlabel("t·σ0")
    ax[0].legend(fontsize=6)
    fig.suptitle(label); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"peaks_stage{stage}.png"), dpi=90); plt.close(fig)


def final_fig(path):
    d = json.load(open(path)); c = d["cfg"]
    fz = path.replace(".json", ".final.npz")
    if not os.path.exists(fz):
        return
    z = np.load(fz); u, V = z["u"], z["V"].astype(complex)
    N = u.shape[-1]; L = 2 * np.pi * c["R0"] / K0; ratio = RATIOS[c["case"]]
    dk = 2 * np.pi / L
    kx = np.fft.fftfreq(N, 1.0 / N) * dk; ky = np.fft.rfftfreq(N, 1.0 / N) * dk
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2; mask = k2 <= (2 / 3 * np.pi * N / L) ** 2; idx = np.flatnonzero(mask.ravel())
    F = V[0] / (4 * N * N)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(u[0][:N, :N].T, origin="lower", cmap="RdBu_r"); ax[0].set_title("layer 0 u")
    ax[1].imshow(u.mean(0).T, origin="lower", cmap="RdBu_r"); ax[1].set_title("sector field m0")
    th = np.linspace(0, 2 * np.pi, 400)
    for a, P, lab in ((ax[2], np.abs(F[0]) ** 2, "outer |F0|²"), (ax[3], np.abs(F[1]) ** 2 + np.abs(F[3]) ** 2, "inner |F1|²+|F3|²")):
        half = np.zeros(k2.shape); half.ravel()[idx] = P
        full = np.zeros((N, N)); full[:, : N // 2 + 1] = half
        ii = (-np.arange(N)) % N; jj = np.arange(1, N // 2); full[np.ix_(ii, N - jj)] = half[:, jj]
        full = np.fft.fftshift(full); w = min(int(1.2 * K0 / dk), N // 2 - 1)
        cc = N // 2
        a.imshow(np.log10(full[cc - w:cc + w, cc - w:cc + w].T + 1e-30), origin="lower", cmap="magma",
                 extent=[-w * dk, w * dk, -w * dk, w * dk], vmin=np.log10(full.max() + 1e-30) - 8)
        for k, ls in ((K0, ":"), (K0 / ratio, "--")):
            a.plot(k * np.cos(th), k * np.sin(th), "c" + ls, lw=0.8)
        a.set_title(lab)
    x = d["series"][-1]
    fig.suptitle(f"{c['name']}  t={x['t']:.0f}  N0={x['N0']} Nin={x['Nin']} conn={x['conn']}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, c["name"] + ".png"), dpi=75); plt.close(fig)


if __name__ == "__main__":
    for st, lab in (("0", "stage 0 (validation 4, 128², R seeds)"), ("2", "stage 2 (η=0.10, 256², S seeds)"),
                    ("2b", "stage 2b (η=0.11, 0.12, 256², S seeds)")):
        peaks_fig(st, lab)
    for f in sorted(glob.glob("runs_v2/stage*/*.json")):
        final_fig(f)
