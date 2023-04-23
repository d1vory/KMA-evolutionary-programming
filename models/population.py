import typing

import numpy as np

import models


class Population:
    def __init__(self, individuals: typing.List[models.Individual]):
        self._individuals: typing.List[models.Individual] = individuals

        self._score: typing.Union[float, None] = None
        self._avg_score: typing.Union[float, None] = None
        self._std_score: typing.Union[float, None] = None
        self._fitness_arr: typing.Union[typing.List[float], None] = None
        self._scaled_fitness = None

    @property
    def individuals(self) -> typing.List[models.Individual]:
        return self._individuals

    @property
    def score(self) -> float:
        if self._score is None:
            self._score = sum(self.fitness_arr)

        return self._score

    @property
    def avg_score(self) -> float:
        if self._avg_score is None:
            self._avg_score = np.mean(self.fitness_arr)

        return self._avg_score

    @property
    def best_score(self) -> float:
        return np.max(self.fitness_arr)

    @property
    def std_score(self) -> float:
        if self._std_score is None:
            self._std_score = np.std(self.fitness_arr)

        return self._std_score

    @property
    def fitness_arr(self) -> typing.List[float]:
        if self._fitness_arr is None:
            self._fitness_arr = [individual.fitness for individual in self._individuals]

        return self._fitness_arr

    @property
    def scaled_fitness(self) -> float:
        if self._scaled_fitness is None:
            self._scaled_fitness = sum([individual.scaled_fitness for individual in self._individuals])
        return self._scaled_fitness

    def sort(self):
        self._individuals: typing.List[models.Individual] = sorted(
            self._individuals, key=lambda x: x.fitness
        )

    def get_fittest(self, n: int) -> typing.List[models.Individual]:
        return self._individuals[-n:]

    def count(self, individual: models.Individual) -> int:
        counter = 0

        for i in self._individuals:
            if i == individual:
                counter += 1

        return counter

    def optimal(self, optimal: str) -> bool:
        if self.convergence():
            if self._individuals[0].genotype == optimal:
                return True

        return False

    def convergence(self) -> bool:
        return len(set(
            [individual.genotype for individual in self._individuals]
        )) == 1

    def homogenity(self, threshold=0.99) -> bool:
        unique_individuals = len(set([individual.genotype for individual in self._individuals]))
        unique_to_all_ratio = unique_individuals / len(self._individuals)
        homogenity_score = 1.0 - unique_to_all_ratio
        return homogenity_score >= threshold

    def invalidate(self):
        self._score = None
        self._avg_score = None
        self._std_score = None
        self._fitness_arr = None
        self._scaled_fitness = None

    def __repr__(self):
        return f"Population(individuals={len(self._individuals)}, total_score={self.score})"

    def __contains__(self, item) -> bool:
        return item in self._individuals

    def __eq__(self, other) -> bool:
        return self._individuals == other.individuals

    def __len__(self) -> int:
        return len(self._individuals)
