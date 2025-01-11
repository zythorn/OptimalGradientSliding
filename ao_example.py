import numpy as np

from optimizers import OGMG, AcceleratedExtragradient

class RidgeRegression():
    def __init__(self, X: np.ndarray, y: np.ndarray, lmbd: float):
        self.X = X
        self.y = y
        self.lmbd = lmbd

    def f(self, x: np.ndarray) -> np.ndarray:
        diff = np.dot(self.X, x) - self.y
        return np.dot(diff, diff)

    def g(self, x: np.ndarray) -> np.ndarray:
        return self.lmbd / 2. * np.dot(x, x)

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        diff = self.X @ x - self.y
        return 2 * (self.X.T @ diff)

    def grad_g(self, x: np.ndarray) -> np.ndarray:
        return self.lmbd * x

if __name__ == "__main__":
    X_sample = np.load("sample_data/X.npy")
    y_sample = np.load("sample_data/y.npy")
    problem = RidgeRegression(X_sample, y_sample, 0.1)
    x_init = np.zeros(3000)
    print(f"Initial error: {problem.f(x_init) + problem.g(x_init)}")

    mu: float = 1e2
    Lp : float = 1e4
    Lq: float = 1e5
    # opt = OGMG(problem.f, problem.grad_f, x_init, L)
    opt = AcceleratedExtragradient(problem.f, problem.grad_f,
                                   problem.g, problem.grad_g,
                                   OGMG, x_init, mu, Lq, Lp, True)
    x_opt = opt.optimize(500)
    print(f"Final error: {problem.f(x_opt) + problem.g(x_opt)}")
