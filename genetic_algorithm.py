import logging
import math
import random
import typing

import matplotlib.pyplot as plt

import generators


def convergence(population: typing.List[str]):
    return len(set(population)) == 1


class GeneticAlgorithm:
    def __init__(
            self, *,
            generator: generators.BaseGenerator,
            fitness_function: typing.Callable,
            max_iteration: int = 10_000_000,
    ):
        self._generator: generators.BaseGenerator = generator

        self._fitness_function = fitness_function

        self._max_iteration: int = max_iteration
        self._iteration: int = 0

        self._populations: typing.List[typing.List[str]] = []
        self._scores: typing.List[int] = []

        self._population: typing.List[str] = self._generator.generate_population()

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
    def populations(self) -> typing.List[typing.List[str]]:
        return self._populations

    @property
    def scores(self) -> typing.List[int]:
        return self._scores

    @property
    def population(self) -> typing.List[str]:
        return self._population

    @property
    def fitness_function(self) -> typing.Callable:
        return self._fitness_function

    def fit(self):
        logging.info("Starting fitting")

        while True:
            if self._iteration == self._max_iteration:
                logging.info(f"Max iteration exceeded, iterations - {self._iteration}")
                break

            if convergence(self._population):
                logging.info(f"Convergence of the population, iterations - {self._iteration}")
                break

            scores: typing.List[int] = self._fitness_function(self._population)
            total_score: int = sum(scores)

            msg = f"Iteration #{self._iteration}. Total score: {total_score}"
            logging.info(msg)
            self._draw_scores(scores, msg)

            self._populations.append(self._population)
            self._scores.append(total_score)
            self._population = self._generate_next_population(
                self._population, scores
            )

            self._iteration += 1

        logging.info(f"Finished fitting after {self._iteration} iterations.")

    @staticmethod
    def _generate_next_population(population, scores) -> typing.List[str]:
        beta = 1.2
        n = len(population)

        new_population = []

        for individual, score in zip(population, scores):
            p = (2 - beta) / n + (2 * score * (beta - 1)) / (n * (n - 1))

            lower_bound = math.floor(p * n)
            upper_bound = math.ceil(p * n)
            msg = f"Individual - {individual}\nScore - {score}\nRank - {p}\nSelected [{lower_bound}, {upper_bound}]"
            # print(msg)

            new_population.extend(random.randint(lower_bound, upper_bound) * [individual])

        print(len(population))
        return new_population

    @staticmethod
    def _draw_scores(scores: typing.List[int], title: str):
        plt.hist(scores, bins=25, color="b")
        plt.xlabel("Scores")
        plt.ylabel("Number of Individuals")
        plt.title(title)
        plt.show()
