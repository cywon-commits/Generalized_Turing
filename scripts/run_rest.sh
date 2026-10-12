#!/bin/bash
# 단계 2-5 를 순차 실행 (단계 내부는 4프로세스 병렬)
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/.."
for s in 2 3 4 5; do python3 scripts/run_batch.py $s 4 > runs/stage$s.log 2>&1; done
echo ALLDONE > runs/rest.done
