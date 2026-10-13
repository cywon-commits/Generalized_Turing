"""특정 설정 파일 목록을 4프로세스로 실행: python scripts/run_v2_list.py cfg1.yaml cfg2.yaml ..."""
import os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.runner import load_config, run

def job(p):
    cfg = load_config(p)
    run(cfg, log=lambda s: print(s, flush=True))
    return cfg["name"]

if __name__ == "__main__":
    with Pool(4) as pool:
        for n in pool.imap_unordered(job, sys.argv[1:]):
            print("FINISHED", n, flush=True)
