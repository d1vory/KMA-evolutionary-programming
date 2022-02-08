import random

from generators import base_generator


class NormalGenerator(base_generator.BaseGenerator):
    def __init__(
            self, *, n: int, length: int,
            optimal: bool = False,
            optimal_sign: str = "0"
    ):
        super().__init__(n=n, length=length, optimal=optimal)

        self._optimal_sign: str = optimal_sign

    @property
    def optimal_sign(self) -> str:
        return self._optimal_sign

    @optimal_sign.setter
    def optimal_sign(self, value: str):
        self._optimal_sign = value

    def generate_optimal_individual(self) -> str:
        return self._optimal_sign * self._length

    def generate_individual(self) -> str:
        return "".join(["1" if random.random() < 0.5 else "0" for _ in range(self._length)])
        # return "".join(random.choices("01", k=self._length))
