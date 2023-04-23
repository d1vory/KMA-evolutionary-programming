import models


class LinearRank(models.Function):
    def __init__(self, beta: float, n: int):
        super().__init__()

        self._beta = beta
        self._n = n

    def _f(self, arg):
        return (2 - self._beta) / self._n \
               + (2 * arg * (self._beta - 1)) \
               / (self._n * (self._n - 1))


class LinearScaling(models.Function):
    def __init__(self, a: int, b: int):
        super().__init__()

        self._a = a
        self._b = b

    def _f(self, arg):
        return self._a * arg + self._b
