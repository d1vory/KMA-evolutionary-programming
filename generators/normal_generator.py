import random

from generators import base_generator


class NormalGenerator(base_generator.BaseGenerator):
    def __init__(
            self, *, n: int, length: int,
            optimal: str,
            generate_optimal: bool = False,
    ):
        super().__init__(n=n, length=length, optimal=optimal, generate_optimal=generate_optimal)

    def generate_optimal_individual(self) -> str:
        return self._optimal

    def generate_individual(self) -> str:
        return "".join(["1" if random.random() < 0.5 else "0" for _ in range(self._length)])
