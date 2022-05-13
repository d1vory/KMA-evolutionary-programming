import typing


class Individual:
    def __init__(self, genotype: str, fitness: float, rank: typing.Union[float, None] = None):
        self._genotype: str = genotype
        self._fitness: float = fitness
        self._rank: float = rank

    @property
    def genotype(self) -> str:
        return self._genotype

    @property
    def fitness(self) -> float:
        return self._fitness

    @property
    def rank(self) -> float:
        if self._rank is None:
            return self._fitness

        return self._rank

    @rank.setter
    def rank(self, value: float):
        self._rank = value

    def is_zero(self):
        return list(set(self._genotype)) == ["0"]

    def __repr__(self):
        return f"Individual(genotype='{self._genotype}', fitness={self._fitness:.2f}, rank={self._rank}:.2f)"

    def __eq__(self, other) -> bool:
        return self._genotype == other.genotype
