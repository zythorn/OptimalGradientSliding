from collections.abc import Callable
from abc import ABC, abstractmethod
import numpy as np

class BaseFirstOrderOptimizer(ABC):
    def __init__(self, f: Callable[[np.ndarray], np.ndarray],
                 grad_f: Callable[[np.ndarray], np.ndarray],
                 x_init: np.ndarray, liepshitz_const: float,
                 log: bool=False):
        self.f = f
        self.grad_f = grad_f
        self.x = x_init
        self.L = liepshitz_const
        self.log = log

    @abstractmethod
    def optimize(self, n_iters: int=10**3):
        raise NotImplementedError()

class GradientDescent(BaseFirstOrderOptimizer):
    def optimize(self, n_iters=10**3):
        for _ in range(n_iters):
            self.x = self.x - 1. / self.L * self.grad_f(self.x)
        return self.x

class OGMG(BaseFirstOrderOptimizer):
    def __init__(self, f: Callable[[np.ndarray], np.ndarray],
                 grad_f: Callable[[np.ndarray], np.ndarray],
                 x_init: np.ndarray, liepshitz_const: float,
                 log: bool=False):
        super().__init__(f, grad_f, x_init, liepshitz_const, log)
        self.y = x_init

    @staticmethod
    def _schedule_theta(n_iters: int):
        theta = np.ones((n_iters + 1))
        for i in range(n_iters - 1, 0, -1):
            theta[i] = (1 + np.sqrt(1 + 4 * theta[i + 1] ** 2)) / 2
        theta[0] = (1 + np.sqrt(1 + 8 * theta[1] ** 2)) / 2
        return theta

    def optimize(self, n_iters: int=10**3):
        theta = self._schedule_theta(n_iters)

        for i in range(n_iters):
            y_new = self.x - 1. / self.L * self.grad_f(self.x)

            y_coef = (theta[i] - 1) * (2 * theta[i + 1] - 1) / (theta[i] * (2 * theta[i] - 1))
            x_coef = (2 * theta[i + 1] - 1) / (2 * theta[i] - 1)

            self.x = y_new + y_coef * (y_new - self.y) + x_coef * (y_new - self.x)
            self.y = y_new
            if self.log and i % 100 == 0:
                print(f"Error after {i} steps: {self.f(self.x)}")

        return self.x
