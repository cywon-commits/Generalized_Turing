import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ring_sim.bilayer import *
from v7_D2 import params, seed, RHO
par = params(); cells, s0, k0 = cell2_setup(par, RHO, nb=3); c0 = cells[0]
X = seed(c0, "rhombus", np.random.default_rng(3)); X = relax2(c0, X, T=4e4 * 4.1e-4 / s0 * 4, h=10.0); X, r = newton2(c0, X)
sh, corr = layer_shift(c0, X)
u = c0.to_real(X[:2]).real                      # 분수 좌표 격자 (P x P) -> 실공간으로 변환해 3x3 셀 타일
a = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]) / np.sqrt(3); b = np.array([np.cos(np.radians(30)), -np.sin(np.radians(30))]) / np.sqrt(3)
# 실공간 기본 셀: a*, b* 이중 기저 (a·A=2π δ ...). 분수 좌표 (s,t) -> x = s A1 + t A2
M = np.array([a, b]); Ainv = 2 * np.pi * np.linalg.inv(M).T   # 행: A1, A2 (실공간)
P = c0.P; n = 3
ss, tt = np.meshgrid(np.arange(n * P) / P, np.arange(n * P) / P, indexing="ij")
xx = ss * Ainv[0, 0] + tt * Ainv[1, 0]; yy = ss * Ainv[0, 1] + tt * Ainv[1, 1]
U0 = np.tile(u[0], (n, n)); U1 = np.tile(u[1], (n, n))
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for a_, U, t in ((ax[0], U0, "layer 0 (Bloch-cell rhombus)"), (ax[1], U1, "layer 1")):
    a_.pcolormesh(xx * k0, yy * k0, U, cmap="RdBu_r", shading="auto"); a_.set_aspect("equal"); a_.set_title(t, fontsize=9)
for k, s in enumerate((0, 1)):
    uu = np.load(f"results_v7/D/D2_pde_s{s}_u.npy"); a_ = ax[2 + k]
    a_.imshow(np.concatenate([uu[0], uu[1]], 1).T, origin="lower", cmap="RdBu_r"); a_.axis("off"); a_.set_title(f"PDE noise seed {s}: layer 0 | layer 1", fontsize=9)
fig.tight_layout(); fig.savefig("results_v7/figures/D2_states.png", dpi=90)
print("shift", sh, "corr", corr)
