import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def orient(u, k0, L, w=0.3):
    """층 섹터별 링 대역 파워의 방향 질서도 ψ_n = |Σ P e^{i n θ}| / Σ P  (n=2,4,6; θ 는 mod π 방향)"""
    N = u.shape[1]
    F = np.fft.fft2(u, axes=(1, 2)) / (N * N)
    k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / L
    KX, KY = np.meshgrid(k, k, indexing="ij"); kk = np.hypot(KX, KY); th = np.arctan2(KY, KX)
    band = np.abs(kk - k0) <= w * k0
    out = {}
    for name, f in (("in", 0.5 * (F[0] + F[1])), ("anti", 0.5 * (F[0] - F[1]))):
        P = np.abs(f) ** 2 * band
        tot = P.sum()
        out[name] = dict(P=float(tot), **{f"psi{n}": float(abs((P * np.exp(1j * n * th)).sum()) / tot) if tot > 1e-20 else float("nan") for n in (2, 4, 6)})
    return out


def main():
    rows = []
    files = sorted(glob.glob("results_v7/D/D1_*_s*.json"))
    fig, ax = plt.subplots(2, max(len(files), 2), figsize=(3.3 * max(len(files), 2), 6.8))
    for j, fn in enumerate(files):
        d = json.load(open(fn)); u = np.load(fn.replace(".json", "_u.npy")).astype(float)
        o = orient(u, d["k0"], d["L"])
        rows.append((d["case"], d["seed"], o))
        print(d["case"], d["seed"], "in: P=%.2e psi2=%.2f psi4=%.2f psi6=%.2f | anti: P=%.2e psi2=%.2f psi4=%.2f psi6=%.2f" % tuple([o["in"]["P"], o["in"]["psi2"], o["in"]["psi4"], o["in"]["psi6"], o["anti"]["P"], o["anti"]["psi2"], o["anti"]["psi4"], o["anti"]["psi6"]]))
        for r_, lay in enumerate((0, 1)):
            ax[r_, j].imshow(u[lay].T, origin="lower", cmap="RdBu_r"); ax[r_, j].axis("off")
        ax[0, j].set_title(f"{d['case'][:9]} s{d['seed']}", fontsize=8)
    os.makedirs("results_v7/figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("results_v7/figures/D1_final_states.png", dpi=80)
    json.dump([dict(case=c, seed=s, **o) for c, s, o in rows], open("results_v7/D/D1_orientation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
