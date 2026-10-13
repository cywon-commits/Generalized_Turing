"""v4 N8-S (Cx=10, η=0.15) 고유값 재현: results_v4/branches.csv 의 max_nonneutral (-0.2306σ0) 과 비교."""
import sys, os, numpy as np, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_probe import make
rho = 2 * np.cos(np.pi / 8)
net, s0 = make(rho, 10, 96, 0.15)
x = np.load("runs_v4/n8s_eta0.15.npy")
print("residual", net.residual(x))
lam, w, V = net.eigenvalues(x, k=10, tol=1e-5)
nn = [l for l in lam if abs(l) > 2e-3]
print("max non-neutral", max(nn)); 
ref = [float(r["max_nonneutral_eig_over_s0"]) for r in csv.DictReader(open("results_v4/branches.csv")) if r["eta"] == "0.15" and r["state"].startswith("N8-S")][0]
print("v4 reference", ref, "diff", max(nn) - ref)
