import sys, glob, os, csv, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.specdiag import diagnose
rows = []
for f in sorted(glob.glob("runs_v4/n8s_proj_eta*.npy"), key=lambda s: float(re.findall(r"eta([0-9.]+)\.npy", s)[0])):
    eta = float(re.findall(r"eta([0-9.]+)\.npy", f)[0])
    x = np.load(f).reshape(2, 4, 96, 96)
    d = diagnose(x[0], 10); d.update(Cx=10, eta=eta); rows.append(d)
    print("eta=%.3f outerW16=%.3f innerW16=%.3f outer_width=%.1f° inner_width=%.1f° diag/axis=%.3f inner_std/mean=%.3f" % (eta, d["outer_W16"], d["inner_W16"], d["outer_width_deg"], d["inner_width_deg"], d["outer_diag_over_axis"], d["inner_peak_std_over_mean"]))
os.makedirs("results_v7", exist_ok=True)
w = csv.DictWriter(open("results_v7/C_baseline_v4.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
