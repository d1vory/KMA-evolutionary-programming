import random

from generators import base_generator


class ConstGenerator(base_generator.BaseGenerator):
    def generate_optimal_individual(self) -> str:
        return self.generate_individual()

    def generate_individual(self) -> str:
        return ("0" if random.random() < 0.5 else "1") * self._length
