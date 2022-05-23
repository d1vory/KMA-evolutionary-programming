from core import fitness_functions, utils
from generators import BaseGenerator


class RealGenerator(BaseGenerator):
    def __init__(
            self, *, n: int, length: int,
            optimal: str,
            fitness_fn: fitness_functions.FitnessFunction,
            low_range: float,
            high_range: float,
            generate_optimal: bool = False,
            **kwargs
    ):
        super().__init__(n=n, length=length, optimal=optimal, generate_optimal=generate_optimal, **kwargs)

        self._fitness_fn = fitness_fn
        self._low_range = low_range
        self._high_range = high_range
        self._dist = utils.generate_norm_dist(self._low_range, self._high_range, n)
        self._index = 0

    def generate_individual(self) -> str:
        item = self._dist[self._index]
        item = self._fitness_fn.get_x(item)
        item = round(item, 2)
        item = self._fitness_fn.encode(item)

        self._index += 1

        return item
