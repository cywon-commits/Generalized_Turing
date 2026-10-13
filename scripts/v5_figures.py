import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("results_v5/figures", exist_ok=True)
COL = {"oct": "#1f77b4", "ctrlL": "#d62728", "ctrlH": "#2ca02c"}
def load(pat): return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"results_v5/timeseries/{pat}.json"))}
# (i) 시계열
for tag, pat in (("R_eta0.16", "*_eta0.16_R*"), ("S_eta0.16", "*_eta0.16_S*"), ("N_eta0.16", "*_eta0.16_N*"), ("R_eta0.15", "*_eta0.15_R*"), ("R_eta0.18", "*_eta0.18_R*")):
    D = load(pat)
    if not D: continue
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for name, d in D.items():
        s = d["series"]; t = np.arange(len(s)) * 400.0
        for a, key in zip(ax, ("N0", "Nin", "capture")):
            a.plot(t, [r[key] for r in s], color=COL[d["case"]], alpha=0.8, label=name.split("_")[0] if name.endswith("0") else None)
    for a, tl in zip(ax, ("N0 (outer peaks)", "Nin (inner peaks)", "capture")): a.set_title(tl); a.set_xlabel("t")
    ax[0].legend(); fig.suptitle(tag); fig.tight_layout(); fig.savefig(f"results_v5/figures/timeseries_{tag}.png", dpi=110); plt.close(fig)
# (ii) 최종 상태 + 스펙트럼
runs = ["oct_eta0.16_R0", "ctrlL_eta0.16_R0", "ctrlH_eta0.16_R0", "oct_eta0.16_S0", "ctrlL_eta0.16_S0", "oct_eta0.16_T0"]
runs = [r for r in runs if os.path.exists(f"runs_v5/{r}.final.npz")]
fig, ax = plt.subplots(2, len(runs), figsize=(3.2 * len(runs), 6.4))
for j, r in enumerate(runs):
    u = np.load(f"runs_v5/{r}.final.npz")["u"][0]; N = u.shape[0]
    ax[0, j].imshow(u.T, origin="lower", cmap="RdBu_r"); ax[0, j].set_title(r.replace("_eta", " η"), fontsize=8); ax[0, j].axis("off")
    F = np.fft.fftshift(np.abs(np.fft.fft2(np.load(f"runs_v5/{r}.final.npz")["u"], axes=(1, 2))).sum(0))
    c = N // 2; w = N // 6
    ax[1, j].imshow(np.log10(F[c - w:c + w, c - w:c + w].T + 1e-12), origin="lower", cmap="magma"); ax[1, j].axis("off")
fig.tight_layout(); fig.savefig("results_v5/figures/final_states.png", dpi=100); plt.close(fig)
# (iii) 불변 집합 크기 대 깊이
A = json.load(open("results_v5/A2_closure.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, v in A.items():
    if k == "continuum_exact": continue
    if v["tol"] == 0.5 and v["m"] == 3: ax[0].plot(range(len(v["sizes"])), v["sizes"], "o-", color=COL[v["case"]], ls={"R": ":", "T": "--", "Q": "-"}[v["seed"]], alpha=.8)
ax[0].set_title("lattice closure (tol 0.5σ0, domain x3); line=Q, dashed=T, dotted=R"); ax[0].set_xlabel("depth"); ax[0].set_ylabel("set size")
for k, v in A["continuum_exact"].items():
    case, seed = k.split("|")
    if seed in ("Q", "Q5"): ax[1].plot(range(len(v["sizes"])), v["sizes"], "o-", color=COL.get(case, "gray"), ls="-" if seed == "Q" else "--", label=k)
ax[1].set_title("exact continuum closure"); ax[1].legend(fontsize=6); ax[1].set_xlabel("depth")
fig.tight_layout(); fig.savefig("results_v5/figures/closure_vs_depth.png", dpi=110); plt.close(fig)
print("ok")
