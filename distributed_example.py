from pathlib import Path
import numpy as np

from optimizers import (AcceleratedExtraGradient, OGMG,
                        AcceleratedGradientDescent, GradientDescent)
from problems import RandomDistributedRidgeRegression

if __name__ == "__main__":
    NUM_WORKERS: int = 10
    problem = RandomDistributedRidgeRegression(3000, 3000, lmbd=0.1,
                                               gaussian_sigma=0.5, num_workers=NUM_WORKERS)
    X_sample = problem.x_clean
    x_init = np.zeros(X_sample.shape[1])
    print(f"Initial error: {problem.q(x_init) + problem.p(x_init)}")

    mu: float = 0.1
    Lq: float = 1.
    Lp: float = 10.

    if Path("x_solution.npy").exists():
        x_best = np.load("x_solution.npy")
    else:
        opt = GradientDescent(problem.r, problem.grad_r, x_init,
                            liepshitz_const=Lp + Lq, log=True)
        x_best = opt.optimize(3000)
        np.save("x_solution.npy", x_best)

    opt = AcceleratedExtraGradient(problem.q, problem.grad_q,
                                   problem.p, problem.grad_p,
                                   OGMG, x_init, mu, Lq, Lp, True)
    x_opt = opt.optimize(1000)
    print(f"Final error of AEGD: {np.linalg.norm(x_best - x_opt[-1])}")
    np.save("x_AEGD.npy", np.stack(x_opt))
    opt = AcceleratedGradientDescent(problem.r, problem.grad_r,
                                     x_init, (Lq + Lp), mu, True)
    x_opt = opt.optimize(1000)
    print(f"Final error of AGD: {np.linalg.norm(x_best - x_opt[-1])}")
    np.save("x_AGD.npy", np.stack(x_opt))
