"""configs/stage{n} 의 실행을 프로세스 풀로 병렬 수행 (끝난 것은 건너뜀, 체크포인트에서 재개).
사용: python scripts/run_batch.py STAGE [nproc]"""
import glob, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.runner import load_config, run


def job(path):
    cfg = load_config(path)
    if os.path.exists(os.path.join(cfg["outdir"], cfg["name"] + ".json")):
        return cfg["name"], "skip"
    tag = cfg["name"]
    run(cfg, log=lambda s: print(s, flush=True))
    return tag, "done"


if __name__ == "__main__":
    stage = sys.argv[1]
    cfgdir = sys.argv[3] if len(sys.argv) > 3 else "configs"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    paths = sorted(glob.glob(f"{cfgdir}/stage{stage}/*.yaml"))
    with Pool(n) as p:
        for r in p.imap_unordered(job, paths):
            print("FINISHED", r, flush=True)
