"""results/summary.csv + runs/ -> results/tables.md (REPORT.md 에 삽입할 표)"""
import csv, glob, json
import numpy as np

rows = list(csv.DictReader(open("results/summary.csv")))
out = []

def f(x): return float(x)

def cls(r):
    v = r["verdict"]
    return "소멸" if "extinct" in v else ("유지" if "sustained" in v else "애매")

# 단계 1,3,4,5 집계
out.append("| 단계 | 사례 | 초기 | n | 안쪽 원 소멸/유지/애매 | Pin 범위 | P0 범위 |")
out.append("|---|---|---|---|---|---|---|")
for st in ("1", "3", "4", "5"):
    groups = {}
    for r in rows:
        if r["stage"] == st:
            groups.setdefault((r["case"], r["ic"]), []).append(r)
    for (case, ic), rs in sorted(groups.items()):
        c = [cls(r) for r in rs]
        pin = [f(r["Pin"]) for r in rs]; p0 = [f(r["P0"]) for r in rs]
        out.append(f"| {st} | {case} | {ic} | {len(rs)} | {c.count('소멸')}/{c.count('유지')}/{c.count('애매')} | "
                   f"{min(pin):.1e}–{max(pin):.1e} | {min(p0):.1e}–{max(p0):.1e} |")
out.append("")

# 단계 2: 깨끗한 선형 감쇠 구간의 Λ_in 과 2σ1 비교
s0 = 0.00657937074526993
res = {}
for f_ in sorted(glob.glob("runs/stage2/*.json")):
    d = json.load(open(f_)); c = d["cfg"]; s = d["series"]
    t = np.array([x["t"] for x in s]); p = np.array([x["Pin"] for x in s])
    m = (p < 1e-14) & (p > 1e-33) & (t > 200)
    if m.sum() < 6: continue
    res.setdefault((c["r"], c["eta"]), []).append((np.polyfit(t[m], np.log(p[m]), 1)[0],
                                                  np.mean([x["P0"] for x in s if x["t"] > 1000])))
out.append("| r | η | 측정 Λ_in | 구동 없는 2σ₁ | 비 Λ/2σ₁ | 초과 감쇠 Δ=Λ/2−σ₁ | P0 | n |")
out.append("|---|---|---|---|---|---|---|---|")
for (r, eta), v in sorted(res.items()):
    lam = np.mean([x[0] for x in v]); p0 = np.mean([x[1] for x in v]); s1 = -r * s0
    out.append(f"| {r} | {eta} | {lam:+.4f} | {2*s1:+.4f} | {lam/(2*s1):.1f} | {abs(lam/2 - s1):.4f} | {p0:.1e} | {len(v)} |")
open("results/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
