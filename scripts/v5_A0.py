"""v5 A0: 대조군/8회 마름모 안정성 (η=0.15,0.16,0.18), v3 Galerkin + 전 잉여류. 결과 -> results_v3/stageA/ (v3 형식) 후 v5_A0_table.py 가 A0.csv 생성."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from v3_stageA import analyze
if __name__ == "__main__":
    jobs = [(c, e, True, 3.0) for e in (0.16, 0.15, 0.18) for c in ("ctrlL", "ctrlH", "oct")]
    jobs = [j for j in jobs if not os.path.exists(f'results_v3/stageA/{j[0]}_eta{j[1]}.json')]
    with Pool(int(os.environ.get("NP", 4))) as p:
        for r in p.imap_unordered(analyze, jobs):
            print("done", r, flush=True)
