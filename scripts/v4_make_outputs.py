"""results_v4: branches.csv, figures (가지, 고유값, 대표 상태, 쌍안정 경로)"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_probe import make
from v4_branch import lattice

rows = []
re = {}
for f in sorted(glob.glob("results_v4/stages/n8s_reeval_*.json")):
    d = json.load(open(f)); re[d["eta"]] = d
    c = d.get("unstable_comp") or {}
    rows.append(dict(state="N8-S (closed 8-fold, standing)", eta=d["eta"], exists=True, outer_mean=np.mean(d["outer_p0"]), outer_axis=d["outer_p0"][0], outer_diag=d["outer_p0"][1],
                     inner_p1=np.mean(d["inner_p1"]), inner_p3=np.mean(d["inner_p3"]), two_k_p0=np.mean(d["two_k_p0"]), residual=d["residual_full"],
                     max_nonneutral_eig_over_s0=d["max_nonneutral"], stable=bool(d["max_nonneutral"] < -1e-3),
                     unstable_new_ring_modes=(c.get("ring_new_outer", 0) + c.get("ring_new_inner", 0)) if c else None, note="full 96^2 domain Arnoldi; symmetric NK (D4+layer reflection)"))
for r in json.load(open("results_v4/rhombus_ref.json")):
    rows.append(dict(state=f"R (rhombus, {r['type']}; v3 Galerkin)", eta=r["eta"], exists=True, outer_mean=r["Ac"], inner_p1=r["a_p1"], inner_p3=r["a_p3"], two_k_p0=r["A2a"],
                     max_nonneutral_eig_over_s0=max(r["kappa0_max"], r["coset_max"]), stable=bool(max(r["kappa0_max"], r["coset_max"]) < -1e-3), note="all cosets (20) of reduced lattice"))
rows.append(dict(state="R (standing rhombus, v4 full-domain Arnoldi)", eta=0.14, exists=True, outer_mean=0.01508, inner_p1=0.00782, inner_p3=0.00782, max_nonneutral_eig_over_s0=-0.0467, stable=True, note="reproduces v3 coset max -0.0467"))
for eta in (0.12, 0.15, 0.18):
    rows.append(dict(state="C8-2 / C8-3 (open chain by removing c0 [c3,I3])", eta=eta, exists=False, note="Newton-Krylov from chain seeds converged to the closed network (all four outer modes regrow); relaxation also regrows them"))
rows.append(dict(state="C8-3h (open chain by sector frustration, spiral alternating)", eta=0.15, exists=False, outer_mean=None, max_nonneutral_eig_over_s0=0.1157, stable=False,
                 note="relaxation collapses to a single rhombus (c2); that rhombus is unstable (+0.116)"))
rows.append(dict(state="C8-2h (spiral alternating chain)", eta=0.15, exists=False, note="Newton-Krylov did not converge"))
rows.append(dict(state="N10-S (10-fold, standing)", eta=0.18, exists=None, note="relaxed in Mx+My+L subspace to a 5-outer/5-inner state whose amplitudes keep drifting (T=9e4 more); Newton-Krylov failed: no steady state found"))
rows.append(dict(state="N10-S (10-fold, standing)", eta=0.15, exists=None, note="relaxed state with weak c2,c3 (5e-4); Newton-Krylov did not finish/converge"))
rows.append(dict(state="N10-H (10-fold, spiral alternating)", eta=0.15, exists=False, note="relaxation (Mx+My*L) collapses to a single rhombus (c0 only)"))
rows.append(dict(state="N10-H (10-fold, spiral alternating)", eta=0.18, exists=False, note="relaxation: rhombus (c0, 7e-3) plus weak residual modes; not a network"))
keys = sorted({k for r in rows for k in r})
with open("results_v4/branches.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

os.makedirs("results_v4/figures", exist_ok=True)
# (i),(ii) 가지 그림
etas = sorted(re)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
stab = [re[e]["max_nonneutral"] < -1e-3 for e in etas]
def seg(a, y, label, c):
    for i in range(len(etas) - 1):
        a.plot(etas[i:i + 2], y[i:i + 2], "-" if (stab[i] and stab[i + 1]) else "--", color=c, label=label if i == 0 else None)
seg(ax[0], [np.mean(re[e]["outer_p0"]) * 1e3 for e in etas], "N8-S outer |F0| (mean)", "C3")
seg(ax[0], [np.mean(re[e]["inner_p1"]) * 1e3 for e in etas], "N8-S inner |F1|=|F3|", "C0")
rr = [r for r in json.load(open("results_v4/rhombus_ref.json")) if r["type"] == "standing" and r["Ac"] > 1e-6 and r["Aa"] > 1e-3 and r["Aa"] < 0.03]
ax[0].plot([r["eta"] for r in rr], [r["Ac"] * 1e3 for r in rr], "k^", label="rhombus c (standing, v3)")
ax[0].plot([r["eta"] for r in rr], [r["a_p1"] * 1e3 for r in rr], "ks", mfc="none", label="rhombus inner |F1|")
ax[0].axvline(0.113, color="gray", ls=":"); ax[0].axvline(0.1364, color="gray", ls="-.")
ax[0].set_xlabel("η"); ax[0].set_ylabel("amplitude ×10³"); ax[0].set_title("N8-S branch: solid = stable, dashed = unstable\n(dotted: fold η≈0.113; dash-dot: stability η_c≈0.1364)"); ax[0].legend(fontsize=7)
ax[1].plot(etas, [re[e]["max_nonneutral"] for e in etas], "o-", color="C3", label="N8-S")
ax[1].plot([r["eta"] for r in rr], [max(r["kappa0_max"], r["coset_max"]) for r in rr], "k^-", label="rhombus (standing)")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylim(-1, 1); ax[1].set_xlabel("η"); ax[1].set_ylabel("max non-neutral λ/σ0"); ax[1].set_title("linear stability (full 96² domain)"); ax[1].legend(fontsize=7)
for e in etas:
    ev = re[e]["eig"]; ax[2].plot([e] * len(ev), ev, "k.", ms=3)
ax[2].set_ylim(-1.5, 1); ax[2].axhline(0, color="r", lw=0.6); ax[2].set_xlabel("η"); ax[2].set_ylabel("top-10 eigenvalues λ/σ0"); ax[2].set_title("N8-S spectrum")
fig.tight_layout(); fig.savefig("results_v4/figures/n8s_branch_and_eigenvalues.png", dpi=90); plt.close(fig)

# (iii) 대표 안정 상태 (η=0.15)
inner, outer, Cx, N = lattice(8)
net, s0 = make(2 * np.cos(np.pi / 8), Cx, N, 0.15)
x = np.load("runs_v4/n8s_proj_eta0.15.npy").reshape(net.shape)
u = x[0]
fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
ax[0].imshow(u[0].T, origin="lower", cmap="RdBu_r"); ax[0].set_title("N8-S η=0.15: layer 0 u")
ax[1].imshow(u.mean(0).T, origin="lower", cmap="RdBu_r"); ax[1].set_title("sector field m0")
F = np.fft.fftshift(np.fft.fft2(u, axes=(1, 2)) / (N * N), axes=(1, 2))
lay = lambda p: np.abs(np.sum(F * np.exp(-1j * np.pi * p * np.arange(4) / 2)[:, None, None], 0) / 4) ** 2
w = 16
for a, p, lab in ((ax[2], 0, "outer |F0|²"), (ax[3], 1, "inner |F1|² (|F3|² equal)")):
    P = lay(p); c = N // 2
    a.imshow(np.log10(P[c - w:c + w + 1, c - w:c + w + 1].T + 1e-30), origin="lower", cmap="magma", extent=[-w - .5, w + .5, -w - .5, w + .5], vmin=np.log10(P.max()) - 8)
    th = np.linspace(0, 2 * np.pi, 300)
    for r, ls in ((Cx, ":"), (Cx / (2 * np.cos(np.pi / 8)), "--")): a.plot(r * np.cos(th), r * np.sin(th), "c" + ls, lw=0.8)
    a.set_title(lab + " (lattice units)")
fig.tight_layout(); fig.savefig("results_v4/figures/n8s_state_eta0.15.png", dpi=85); plt.close(fig)

# (iv) 쌍안정 경로
d = json.load(open("results_v4/bistable_traj_eta0.14.json"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for p in d["paths"]:
    t = np.array(p["t"]); o = np.array(p["outer"]); ii = np.array(p["inner_p1"])
    col = "C3" if p["final"] == "network" else "C0"
    ax[0].plot(t, o.max(1) * 1e3, color=col, alpha=0.8); ax[1].plot(t, (o > 1e-3).sum(1), color=col, alpha=0.8, label=f"s={p['s']} → {p['final']}")
ax[0].set_xlabel("t"); ax[0].set_ylabel("max outer |F0| ×10³"); ax[0].set_title("bistability η=0.14: interpolated initial conditions")
ax[1].set_xlabel("t"); ax[1].set_ylabel("# outer modes > 1e-3 (of 4)"); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig("results_v4/figures/bistability_paths_eta0.14.png", dpi=90); plt.close(fig)
print(len(rows), "branch rows")
