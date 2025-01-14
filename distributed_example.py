import numpy as np
from pathlib import Path

from optimizers import (AcceleratedExtraGradient, OGMG,
                        AcceleratedGradientDescent, GradientDescent)

class RandomDistributedRidgeProblem():
    def __init__(self, dim: int, data_size: int, lmbd: float, 
                 gaussian_sigma: float=1., num_workers: int=1, seed: int=42):
        np.random.seed(seed)
        self.x_clean = np.random.rand(data_size, dim) * 2. - 1.
        self.y_clean = np.random.randn(data_size) * 2. - 1.

        self.x_noise = np.random.randn(num_workers, data_size, dim) * gaussian_sigma
        self.y_noise = np.random.randn(num_workers, data_size) * gaussian_sigma

        self.x = self.x_noise + self.x_clean
        self.y = self.y_noise + self.y_clean

        self.data_size = data_size
        self.num_workers = num_workers
        self.lmbd = lmbd

    def f_at_node(self, x: np.ndarray, node: int) -> np.ndarray:
        diff = np.dot(self.x[node], x) - self.y[node]
        return np.mean(diff ** 2) / 2. + self.lmbd / 2. * np.dot(x, x)

    def grad_f_at_node(self, x: np.ndarray, node: int) -> np.ndarray:
        diff = np.dot(self.x[node], x) - self.y[node]
        return np.dot(diff, self.x[node]) / self.data_size + self.lmbd * x

    def q(self, x: np.ndarray) -> np.ndarray:
        return self.f_at_node(x, 0)

    def p(self, x: np.ndarray) -> np.ndarray:
        value: float = 0.
        f_master = self.f_at_node(x, 0)
        for node in range(1, self.num_workers):
            value += self.f_at_node(x, node) - f_master
        return value / self.num_workers

    def r(self, x: np.ndarray) -> np.ndarray:
        value: float = 0.
        for node in range(self.num_workers):
            value += self.f_at_node(x, node)
        return value / self.num_workers

    def grad_q(self, x: np.ndarray) -> np.ndarray:
        return self.grad_f_at_node(x, 0)

    def grad_p(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x, dtype=float)
        grad_master = self.grad_f_at_node(x, 0)
        for node in range(1, self.num_workers):
            grad += self.grad_f_at_node(x, node) - grad_master
        return grad / self.num_workers

    def grad_r(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x, dtype=float)
        for node in range(self.num_workers):
            grad += self.grad_f_at_node(x, node)
        return grad / self.num_workers

class DistributedRidgeProblem():
    def __init__(self, x_clean: np.ndarray, y_clean: np.ndarray, lmbd: float,
                 gaussian_sigma: float=0.01, num_workers: int=1, seed: int=42):
        np.random.seed(seed)
        # x_clean = x_clean[:300, :300]
        # y_clean = y_clean[:300]
        self.data_size, self.dim = x_clean.shape

        self.x_noise = np.random.randn(num_workers, self.data_size, self.dim) * gaussian_sigma
        self.y_noise = np.random.randn(num_workers, self.data_size) * gaussian_sigma

        self.x = self.x_noise + x_clean
        self.y = self.y_noise + y_clean

        self.data_size = self.data_size
        self.num_workers = num_workers
        self.lmbd = lmbd

    def f_at_node(self, x: np.ndarray, node: int) -> np.ndarray:
        diff = np.dot(self.x[node], x) - self.y[node]
        return np.mean(diff ** 2) / 2. + self.lmbd / 2. * np.dot(x, x)

    def grad_f_at_node(self, x: np.ndarray, node: int) -> np.ndarray:
        diff = np.dot(self.x[node], x) - self.y[node]
        return np.dot(diff, self.x[node]) / self.data_size + self.lmbd * x

    def q(self, x: np.ndarray) -> np.ndarray:
        return self.f_at_node(x, 0)

    def p(self, x: np.ndarray) -> np.ndarray:
        value: float = 0.
        f_master = self.f_at_node(x, 0)
        for node in range(1, self.num_workers):
            value += self.f_at_node(x, node) - f_master
        return value / self.num_workers

    def r(self, x: np.ndarray) -> np.ndarray:
        value: float = 0.
        for node in range(self.num_workers):
            value += self.f_at_node(x, node)
        return value / self.num_workers

    def grad_q(self, x: np.ndarray) -> np.ndarray:
        return self.grad_f_at_node(x, 0)

    def grad_p(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x, dtype=float)
        grad_master = self.grad_f_at_node(x, 0)
        for node in range(1, self.num_workers):
            grad += self.grad_f_at_node(x, node) - grad_master
        return grad / self.num_workers

    def grad_r(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x, dtype=float)
        for node in range(self.num_workers):
            grad += self.grad_f_at_node(x, node)
        return grad / self.num_workers

if __name__ == "__main__":
    NUM_WORKERS: int = 10
    # X_sample = np.load("sample_data/X.npy")
    # y_sample = np.load("sample_data/y.npy")
    # print(X_sample.min(), X_sample.max(), y_sample.min(), y_sample.max())
    # problem = DistributedRidgeProblem(X_sample, y_sample, lmbd=0.1,
    #                                   gaussian_sigma=0.5, num_workers=NUM_WORKERS)
    problem = RandomDistributedRidgeProblem(3000, 3000, lmbd=0.1,
                                            gaussian_sigma=0.5, num_workers=NUM_WORKERS)
    X_sample = problem.x_clean
    x_init = np.zeros(X_sample.shape[1])
    print(f"Initial error: {problem.q(x_init) + problem.p(x_init)}")

    mu: float = 0.1
    Lp: float = 10.
    Lq: float = 1.
    # opt = OGMG(problem.q, problem.prad_f, x_init, L)
    if Path("x_solution.npy").exists():
        x_best = np.load("x_solution.npy")
    else:
        opt = GradientDescent(problem.r, problem.grad_r, x_init,
                            liepshitz_const=Lp + Lq, log=True)
        x_best = opt.optimize(3000)
        np.save("x_solution.npy", x_best)

    # opt = DANE(problem.f_at_node, problem.grad_f_at_node, AcceleratedGradientDescent,
    #            x_init, mu, Lp, 0.01, 1., NUM_WORKERS, True)
    # x_opt = opt.optimize(200)
    # print(f"Final error of DANE: {np.linalg.norm(x_best - x_opt)}")
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
    # for node in range(20):
    #     print(problem.grad_f_at_node(x_init, node))
