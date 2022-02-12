class Individual:
    def __init__(self, genotype: str, fitness: float, rank: float):
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
        return self._rank

    def __repr__(self):
        return f"Individual(genotype='{self._genotype}', fitness={self._fitness:.2f}, rank={self._rank}:.2f)"
