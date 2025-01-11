import numpy as np
from optimizers import OGMG

# class RandomLinearProblem():
#     def __init__(self, dim: int, seed: int=42):
#         np.random.seed(seed)
#         self.A = np.random.randn(dim, dim)
#         self.b = np.random.randn(dim)

#     def f(self, x: np.ndarray) -> np.ndarray:
#         return self.A @ x + self.b

#     def grad_f(self, x: np.ndarray) -> np.ndarray:
#         return self.A

class QuadraticProblem():
    def __init__(self, X: np.ndarray, y: np.ndarray, lmbd: float):
        self.X = X
        self.y = y
        self.lmbd = lmbd

    def f(self, x: np.ndarray) -> np.ndarray:
        diff = np.dot(self.X, x) - self.y
        return np.dot(diff, diff) + self.lmbd / 2. * np.dot(x, x)

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        diff = self.X @ x - self.y
        return 2 * (self.X.T @ diff) + self.lmbd * x

if __name__ == "__main__":
    X_sample = np.load("sample_data/X.npy")
    y_sample = np.load("sample_data/y.npy")
    problem = QuadraticProblem(X_sample, y_sample, 0.1)
    x_init = np.zeros(3000)
    print(f"Initial error: {problem.f(x_init)}")

    L: float = 5e4
    # opt = OGMG(problem.f, problem.grad_f, x_init, L)
    opt = OGMG(problem.f, problem.grad_f, x_init, L)
    x_opt = opt.optimize(1000)
    print(f"Final error: {problem.f(x_opt)}")
