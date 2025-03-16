import numpy as np


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


class RandomDistributedRidgeRegression():
    def __init__(self, dim: int, data_size: int, lmbd: float,
                 gaussian_sigma: float = 1., num_workers: int = 1, seed: int = 42):
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


class DistributedRidgeRegression():
    def __init__(self, x_clean: np.ndarray, y_clean: np.ndarray, lmbd: float,
                 gaussian_sigma: float = 0.01, num_workers: int = 1, seed: int = 42):
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
