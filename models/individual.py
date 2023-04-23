import typing


class Individual:
    def __init__(self, genotype: str, fitness: float, scaled_fitness: float | None = None):
        self._genotype: str = genotype
        self._fitness: float = fitness
        self._scaled_fitness: float = scaled_fitness

    @property
    def genotype(self) -> str:
        return self._genotype

    @genotype.setter
    def genotype(self, value: str):
        self._genotype = value

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float):
        self._fitness = value

    @property
    def scaled_fitness(self) -> float:
        if self._scaled_fitness is None:
            return self._fitness

        return self._scaled_fitness

    @scaled_fitness.setter
    def scaled_fitness(self, value: float):
        self._scaled_fitness = value

    def is_zero(self):
        return list(set(self._genotype)) == ["0"]

    def __repr__(self):
        return f"Individual(genotype='{self._genotype}', fitness={self._fitness:.2f}, scaled_fitness={self._scaled_fitness:.2f})"

    def __eq__(self, other) -> bool:
        return self._genotype == other.genotype

    def __len__(self) -> int:
        return len(self._genotype)
