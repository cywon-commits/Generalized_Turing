import json, glob, sys, numpy as np, os
for f in sorted(glob.glob("runs_v2/stage%s/*.ckpt.npz" % sys.argv[1])):
    z = np.load(f, allow_pickle=True); s = list(z["series"]); x = s[-1]
    print("%-36s t=%7.0f P0=%.2e Pin=%.2e N0=%2d Nin=%3d conn=%s Sin12=%.2f res_in_K4=%.1f" % (
        os.path.basename(f)[:-9], x["t"], x["P0"], x["Pin"], x["N0"], x["Nin"], x["conn"], x["Sin_12"], x.get("res_in_K4", float("nan"))))
