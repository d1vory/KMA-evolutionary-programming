import random

from generators import base_generator


class DefaultGenerator(base_generator.BaseGenerator):
    def __init__(
            self, *, n: int, length: int,
            optimal: str,
            generate_optimal: bool = False,
            **kwargs
    ):
        super().__init__(n=n, length=length, optimal=optimal, generate_optimal=generate_optimal, **kwargs)

    def generate_individual(self) -> str:
        return "".join(["1" if random.random() < 0.5 else "0" for _ in range(self._length)])
