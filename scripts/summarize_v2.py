"""runs_v2/stage*/*.json -> results_v2/summary.csv, results_v2/timeseries/<name>.json"""
import csv, glob, json, os
import numpy as np

S0 = 0.00041166089609179224


def classify(x, nfold_pairs=6):
    """마지막 시점의 피크 구조로 상태 분류 (판정 규칙 7절)."""
    n0, nin = x["N0"], x["Nin"]
    if nin <= 1:
        return "stripe (no inner pair)"
    if n0 == 1 and nin <= 6:
        return f"rhombus-like (N0={n0}, Nin={nin})"
    if n0 >= 4 or nin >= 10:
        return f"proliferated (N0={n0}, Nin={nin})"
    return f"intermediate (N0={n0}, Nin={nin})"


def saturation_time(t, y, tol=0.05, frac=0.25):
    """y(t) 가 마지막 값의 ±tol 안에 마지막으로 들어간 이후 계속 머문 시작 시각 (포화 시각); 마지막 frac 구간 안정 필요."""
    y = np.asarray(y, float)
    if len(y) < 4:
        return float("nan")
    last = y[-1]
    ok = np.abs(y - last) <= tol * max(abs(last), 1)
    if not ok[-max(2, int(len(y) * frac)):].all():
        return float("nan")
    i = len(y) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return float(t[i])


def main():
    os.makedirs("results_v2/timeseries", exist_ok=True)
    rows = []
    for f in sorted(glob.glob("runs_v2/stage*/*.json")):
        d = json.load(open(f)); c = d["cfg"]; s = d["series"]
        t = np.array([x["t"] for x in s])
        N0 = np.array([x["N0"] for x in s]); Nin = np.array([x["Nin"] for x in s])
        Pin_t = np.array([x["Pin"] for x in s])
        est = Pin_t >= 1e-7  # 안쪽 원이 초기 잡음 수준을 넘어선 시점 (그 전의 피크 수는 잡음 피크)
        # 지수 성장 구간 (1e-9 < Pin < 1e-6) 의 진폭 성장률 / σ0
        g = (Pin_t > 1e-9) & (Pin_t < 1e-6) & (t > t[0])
        lam = float(np.polyfit(t[g], np.log(Pin_t[g]), 1)[0] / 2 / S0) if g.sum() >= 4 else float("nan")
        last = s[-1]
        keys = ["t", "P0", "Pin", "P2", "N0", "Nin", "ang_out", "ang_in", "conn", "ntri_peak", "R_peak", "Phi_peak",
                "S0_4", "S0_6", "S0_8", "S0_10", "S0_12", "Sin_4", "Sin_6", "Sin_8", "Sin_10", "Sin_12",
                "res_out_K2", "res_out_K4", "res_out_K8", "res_in_K2", "res_in_K4", "res_in_K8"]
        json.dump({k: [x.get(k) for x in s] for k in keys}, open(f"results_v2/timeseries/{c['name']}.json", "w"))
        conn = last["conn"]
        rows.append(dict(
            stage=c["stage"], case=c["case"], r=c["r"], eta=c["eta"], ic=c["ic"], seed=c["seed"], N=c["N"], R0=c["R0"],
            T_target=c["T"], T_reached=last["t"], N0=int(N0[-1]), Nin=int(Nin[-1]), N0_max=int(N0[est].max()) if est.any() else -1, Nin_max=int(Nin[est].max()) if est.any() else -1, growth_amp_over_sigma0=lam,
            Nin_saturated_since=saturation_time(t, Nin), N0_saturated_since=saturation_time(t, N0),
            S4=last["Sin_4"], S6=last["Sin_6"], S8=last["Sin_8"], S10=last["Sin_10"], S12=last["Sin_12"],
            S12_outer=last["S0_12"], conn_0=conn["0"], conn_1=conn["1"], conn_2=conn["2"], conn_3p=conn["3+"],
            ntri_peak=last["ntri_peak"], R=last["R_peak"], stacking=d["final"]["stacking"],
            P0=last["P0"], Pin=last["Pin"], res_in_K4=last.get("res_in_K4"), verdict=classify(last), name=c["name"]))
    if rows:
        with open("results_v2/summary.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['name']:38s} T={r['T_reached']:7.0f} N0={r['N0']} Nin={r['Nin']:3d} (max {r['Nin_max']:3d}) λ/σ0={r['growth_amp_over_sigma0']:+.2f} "
              f"conn(0/1/2/3+)={r['conn_0']}/{r['conn_1']}/{r['conn_2']}/{r['conn_3p']} S12={r['S12']:.2f} {r['verdict']}")


if __name__ == "__main__":
    main()
