"""의뢰서 5절 실행 행렬 -> configs/stage{n}/*.yaml"""
import os, sys, itertools
import yaml

T_DEFAULT = float(os.environ.get("RING_T", 6000))
OUT = "runs"


def cfg(stage, case, r, eta, ic, seed, T=None):
    name = f"s{stage}_{case}_r{r}_eta{eta}_{ic}_seed{seed}"
    return dict(name=name, stage=stage, case=case, r=r, eta=eta, ic=ic, seed=seed,
                N=512, R0=56, dt=1.0, T=T or T_DEFAULT, rec_dt=20.0, ckpt_dt=500.0,
                outdir=f"{OUT}/stage{stage}")


def matrix():
    M = []
    for case, ic, seed in itertools.product(["dodec", "ctrlA", "ctrlB"], ["O", "F"], range(1, 4)):
        M.append(cfg(1, case, 1, 0.4, ic, seed))
    for r, eta, seed in itertools.product([0.5, 1, 2], [0.2, 0.3, 0.4, 0.5], range(1, 3)):
        M.append(cfg(2, "dodec", r, eta, "O", seed, T=4000))
    for case, ic, seed in itertools.product(["hex", "oct", "dec"], ["O", "F"], range(1, 3)):
        M.append(cfg(3, case, 1, 0.4, ic, seed))
    for case, seed in itertools.product(["dodec", "ctrlA"], range(1, 4)):
        M.append(cfg(4, case, 1, 0.4, "N", seed, T=6000))
    for seed in range(1, 3):
        M.append(cfg(5, "dodec", 0, 0.4, "F", seed))
    return M


if __name__ == "__main__":
    for c in matrix():
        d = os.path.join("configs", f"stage{c['stage']}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, c["name"] + ".yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
    print(len(matrix()), "configs")
