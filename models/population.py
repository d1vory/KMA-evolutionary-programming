import typing

import models


class Population:
    def __init__(self, individuals: typing.List[models.Individual]):
        self._individuals: typing.List[models.Individual] = sorted(
            individuals, key=lambda x: x.fitness, reverse=True
        )
        self._score: typing.Union[float, None] = None

    @property
    def individuals(self) -> typing.List[models.Individual]:
        return self._individuals

    def get_fittest(self, n: int) -> typing.List[models.Individual]:
        return self._individuals[:n]

    def score(self) -> float:
        if self._score is None:
            self._score = sum([individual.fitness for individual in self._individuals])

        return self._score

    def __repr__(self):
        return f"Population(individuals={len(self._individuals)}, total_score={self.score()})"
