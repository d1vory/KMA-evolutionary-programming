import logging
import typing

import matplotlib.pyplot as plt
import numpy as np

import generators
import models


def convergence(population: models.Population) -> bool:
    return len(set(
        [individual.genotype for individual in population.individuals]
    )) == 1


class GeneticAlgorithm:
    def __init__(
            self, *,
            generator: generators.BaseGenerator,
            fitness_function: typing.Callable[
                [typing.Union[str, typing.List[str]]],
                typing.Union[float, typing.List[float]]
            ],
            scale_function: typing.Callable[
                [typing.Union[float, typing.List[float]]],
                typing.Union[float, typing.List[float]]
            ],
            selection_algo: typing.Callable[
                [models.Population], models.Population
            ],
            max_iteration: int = 10_000_000,
            draw_step: typing.Union[None, int] = None,
            draw_total_steps: bool = False,
    ):
        self._generator: generators.BaseGenerator = generator

        self._fitness_function: typing.Callable[
            [typing.Union[str, typing.List[str]]],
            typing.Union[float, typing.List[float]]
        ] = fitness_function
        self._scale_function: typing.Callable[
            [typing.Union[float, typing.List[float]]],
            typing.Union[float, typing.List[float]]
        ] = scale_function
        self._selection_algo: typing.Callable[
            [models.Population], models.Population
        ] = selection_algo

        self._max_iteration: int = max_iteration
        self._draw_step: int = draw_step
        self._draw_total_steps = draw_total_steps
        self._iteration: int = 0

        self._populations: typing.List[models.Population] = []
        self._total_scores: typing.List[float] = []

        self._population: models.Population = self._evaluate_population(self._generator.generate_population())

    @property
    def generator(self) -> generators.BaseGenerator:
        return self._generator

    @property
    def max_iteration(self) -> int:
        return self._max_iteration

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def populations(self) -> typing.List[models.Population]:
        return self._populations

    @property
    def total_scores(self) -> typing.List[float]:
        return self._total_scores

    @property
    def population(self) -> models.Population:
        return self._population

    @property
    def fitness_function(self) -> typing.Callable:
        return self._fitness_function

    def _evaluate_population(self, population: typing.List[str]) -> models.Population:
        individuals = []

        for individual in population:
            fitness = self._fitness_function(individual)
            rank = self._scale_function(fitness)
            individuals.append(models.Individual(individual, fitness, rank))

        return models.Population(individuals)

    def fit(self):
        logging.info("Starting fitting")

        while True:
            total_score: float = self._population.score()

            msg = f"Iteration #{self._iteration}. Total score: {total_score}"
            logging.info(msg)

            if self._draw_step and self._iteration % self._draw_step == 0:
                self._draw_scores([
                    individual.fitness for individual in self._population.individuals
                ], msg)

            self._populations.append(self._population)
            self._total_scores.append(total_score)

            if self._iteration == self._max_iteration:
                logging.info(f"Max iteration exceeded, iterations - {self._iteration}")
                break

            if convergence(self._population):
                logging.info(f"Convergence of the population, iterations - {self._iteration}")
                break

            self._population = self._selection_algo(self._population)

            self._iteration += 1

        msg = f"Finished at Iteration #{self._iteration}. Total score: {total_score}"
        logging.info(msg)

        if self._draw_total_steps:
            self._draw_total_scores(self._total_scores, "Total score difference")

    @staticmethod
    def _generate_next_population(population: models.Population) -> models.Population:

        return population

    @staticmethod
    def _draw_total_scores(scores: typing.List[float], title: str):
        x = np.arange(0., len(scores), 1.)
        plt.plot(x, scores, color="b")
        plt.xlabel("Iterations")
        plt.xlabel("Total score")
        plt.title(title)
        plt.show()

    @staticmethod
    def _draw_scores(scores: typing.List[float], title: str):
        plt.hist(scores, bins=25, color="b")
        plt.xlabel("Scores")
        plt.ylabel("Number of Individuals")
        plt.title(title)
        plt.show()
