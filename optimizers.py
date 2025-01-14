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
        for i in range(n_iters):
            self.x = self.x - 1. / self.L * self.grad_f(self.x)
            if self.log and (i + 1) % 50 == 0:
                print(f"Function value after {i + 1} steps: {self.f(self.x)}")
        return self.x

class AcceleratedGradientDescent(BaseFirstOrderOptimizer):
    def __init__(self, f: Callable[[np.ndarray], np.ndarray],
                 grad_f: Callable[[np.ndarray], np.ndarray],
                 x_init: np.ndarray, liepshitz_const: float,
                 mu: float, log: bool=False):
        super().__init__(f, grad_f, x_init, liepshitz_const, log)
        self.beta = (np.sqrt(self.L / mu) - 1) / (np.sqrt(self.L / mu) + 1)
        self.momentum = x_init
        self.y = x_init
        if self.log:
            self.x_history = [x_init]

    def optimize(self, n_iters=10**3):
        for i in range(n_iters):
            self.y = self.x + self.beta * (self.x - self.momentum)
            self.momentum = self.x
            self.x = self.y - 1. / self.L * self.grad_f(self.y)
            if self.log:
                self.x_history.append(self.x)
                if (i + 1) % 50 == 0:
                    print(f"Function value after {i + 1} steps: {self.f(self.x)}")

        if self.log:
            return self.x_history
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
            if self.log and i % 50 == 0:
                print(f"Function value after {i} steps: {self.f(self.x)}")

        return self.x

class AcceleratedExtraGradient():
    def __init__(self, q: Callable[[np.ndarray], np.ndarray],
                 grad_q: Callable[[np.ndarray], np.ndarray],
                 p: Callable[[np.ndarray], np.ndarray],
                 grad_p: Callable[[np.ndarray], np.ndarray],
                 auxiliary_opt: BaseFirstOrderOptimizer,
                 x_init: np.ndarray, mu: float,
                 liepshitz_q: float, liepshitz_p: float,
                 log: bool=False):
        self.q = q
        self.grad_q = grad_q
        self.p = p
        self.grad_p = grad_p

        self.liepshitz_q = liepshitz_q
        self.liepshitz_p = liepshitz_p
        self.mu = mu

        self.x = x_init
        self.x_g = self.x_f = x_init
        self.auxiliary_opt = auxiliary_opt

        self.tau = np.min((1., np.sqrt(mu) / (2 * np.sqrt(liepshitz_p))))
        self.theta = 1. / (2 * liepshitz_p)
        self.eta = np.min((1. / (2 * mu), (1. / (2 * np.sqrt(mu * liepshitz_p)))))
        self.alpha = mu

        self.log = log

        if self.log:
            self.x_history = [x_init]

    def set_parameters(self, tau: float | None=None, theta: float | None=None,
                       eta: float | None=None, alpha: float | None=None):
        if tau is not None:
            self.tau = tau
        if theta is not None:
            self.theta = theta
        if eta is not None:
            self.eta = eta
        if alpha is not None:
            self.alpha = alpha

    def _auxiliary_problem(self) -> tuple[Callable[[np.ndarray], np.ndarray],
                                          Callable[[np.ndarray], np.ndarray]]:
        current_p = self.p(self.x_g)
        current_grad_p = self.grad_p(self.x_g)

        def f(x: np.ndarray) -> np.ndarray:
            diff = x - self.x_g
            return (current_p + np.dot(current_grad_p, diff) +
                    1. / (2 * self.theta) * np.dot(diff, diff) + self.q(x))

        def grad_f(x: np.ndarray) -> np.ndarray:
            diff = x - self.x_g
            return current_grad_p + 1. / self.theta * diff + self.grad_q(x)

        return f, grad_f

    def optimize(self, n_iters: int=10**3) -> np.ndarray:
        for i in range(n_iters):
            self.x_g = self.tau * self.x + (1. - self.tau) * self.x_f

            f, grad_f= self._auxiliary_problem()
            aux_opt = self.auxiliary_opt(f, grad_f, self.x_g,
                                         2 * self.liepshitz_p + self.liepshitz_q, None)
            self.x_f = aux_opt.optimize(4)

            self.x = (self.x + self.eta * self.alpha * (self.x_f - self.x) -
                      self.eta * (self.grad_p(self.x_f) + self.grad_q(self.x_f)))
            if self.log:
                self.x_history.append(self.x)
                if (i + 1) % 50 == 0:
                    print(f"Function value after {i + 1} steps: {self.p(self.x) + self.q(self.x)}")

        if self.log:
            return self.x_history
        return self.x
