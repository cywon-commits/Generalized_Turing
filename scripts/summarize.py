"""runs/stage*/*.json -> results/summary.csv"""
import csv, glob, json, os
import numpy as np

NF = {"square": 4, "hex": 6, "oct": 8, "dec": 10, "dodec": 12, "ctrlA": 12, "ctrlB": 12}


def fit_rate(t, y, frac=0.25, tmin=500.0):
    t, y = np.asarray(t), np.asarray(y)
    t1 = t[-1]
    w = max(tmin, frac * t1)
    m = (t >= t1 - w) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(t[m], np.log(y[m]), 1)[0])


def classify(rec, nfold):
    P0, Pin, lam, axes = rec["P0"], rec["Pin"], rec["Lam_in"], rec["inner_axes"]
    rho = Pin / P0 if P0 > 0 else 0.0
    if rho < 1e-3 and lam < 0:
        return "inner extinct"
    if rho > 1e-2 and lam > -0.3 * abs(2 * rec["sigma1"]):
        full = axes >= nfold // 2 - 1
        return f"inner sustained: {'ring-like' if full else 'partial'} ({axes} axes)"
    return "inner decaying/ambiguous"


def rows():
    out = []
    for f in sorted(glob.glob("runs/stage*/*.json")):
        d = json.load(open(f))
        c, s, fin = d["cfg"], d["series"], d["final"]
        t = [x["t"] for x in s]
        last = s[-1]
        sigma0 = d["sigma0"]
        r = c["r"]
        rec = dict(
            stage=c["stage"], case=c["case"], r=r, eta=c["eta"], ic=c["ic"], seed=c["seed"], T=t[-1],
            P0=last["P0"], Pin=last["Pin"], P0_init=s[0]["P0"], Pin_init=s[0]["Pin"],
            Lam_in=fit_rate(t, [x["Pin"] for x in s]), Lam_0=fit_rate(t, [x["P0"] for x in s]),
            sigma1=-r * sigma0 if r else 0.0,
            S0_12=last["S0_12"], S0_8=last["S0_8"], S0_6=last["S0_6"], S0_10=last["S0_10"], S0_4=last["S0_4"],
            Sin_12=last["Sin_12"], Sin_8=last["Sin_8"], Sin_6=last["Sin_6"], Sin_10=last["Sin_10"], Sin_4=last["Sin_4"],
            ntri_lattice=d["ntri_lattice"], n_act=last["n_act"], R=last["R"], R_act=last["R_act"],
            Phi_act=last["Phi_act"], inner_axes=fin["inner_axes"], outer_axes=fin["outer_axes"],
            inner_modes90=fin["inner_modes90"], stacking=fin["stacking"], name=c["name"])
        rec["verdict"] = classify(rec, NF[c["case"]])
        out.append(rec)
    return out


if __name__ == "__main__":
    R = rows()
    os.makedirs("results", exist_ok=True)
    if R:
        keys = [k for k in R[0] if k not in ("sigma1",)]
        with open("results/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(R)
    for r in R:
        print(f"{r['name']:42s} P0={r['P0']:.2e} Pin={r['Pin']:.2e} Lam_in={r['Lam_in']:+.4f} "
              f"axes={r['inner_axes']:2d} R_act={r['R_act']:.2f} n_act={r['n_act']:4d} {r['verdict']}")
