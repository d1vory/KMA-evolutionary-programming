import logging
import typing

import matplotlib.pyplot as plt
import numpy as np

import models
import utils


class GeneticAlgorithm:
    def __init__(
            self, *,
            base_population: typing.List[str],
            fitness_function: models.Function,
            scale_function: models.Function,
            selection_algo: typing.Callable[
                [models.Population], models.Population
            ],
            modified_selection_algo: bool = False,
            stats_mode: str = "full",
            max_iteration: int = 10_000_000,
            draw_step: typing.Union[None, int] = None,
            draw_total_steps: bool = False,
    ):
        self._base_population: typing.List[str] = base_population

        self._fitness_function: models.Function = fitness_function
        self._scale_function: models.Function = scale_function
        self._selection_algo: typing.Callable[
            [models.Population], models.Population
        ] = selection_algo
        self._modified_selection_algo = modified_selection_algo

        self._stats_mode: str = stats_mode
        self._max_iteration: int = max_iteration
        self._draw_step: int = draw_step
        self._draw_total_steps = draw_total_steps
        self._iteration: int = 0

        self._populations: typing.List[models.Population] = []
        self._total_scores: typing.List[float] = []

        self._population: models.Population = self._evaluate_population(self._base_population)

        self._stats: typing.Dict[str, float] = {}
        self._selection_differences: typing.List[float] = []
        self._reproduction_coeffs: typing.List[float] = []
        self._loss_of_diversity_coeffs: typing.List[float] = []
        self._selection_intensities: typing.List[float] = []
        self._growth_rates: typing.List[float] = []
        self._convergence_iteration: typing.Union[None, float] = None

    @property
    def base_population(self) -> typing.List[str]:
        return self._base_population

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
    def stats(self) -> typing.Dict[str, float]:
        return self._stats

    def _update_noise_stats(self):
        self._stats["NI"] = self._convergence_iteration or -1
        self._stats["ConvTo"] = 0 if self._population.individuals[0].is_zero() else 1

    def _update_stats(self):
        if self._stats_mode == "noise":
            return

        previous_population: models.Population = self._populations[-2]
        current_population: models.Population = self._populations[-1]

        selection_difference = current_population.avg_score - previous_population.avg_score
        self._selection_differences.append(selection_difference)

        if "s_min" not in self._stats or selection_difference < self._stats["s_min"]:
            self._stats["s_min"] = selection_difference
            self._stats["NI_s_min"] = self._iteration
        if "s_max" not in self._stats or selection_difference > self._stats["s_max"]:
            self._stats["s_max"] = selection_difference
            self._stats["NI_s_max"] = self._iteration

        selection_intensity = selection_difference / np.std(previous_population.fitness_arr)
        self._selection_intensities.append(selection_intensity)

        if "I_min" not in self._stats or selection_intensity < self._stats["I_min"]:
            self._stats["I_min"] = selection_intensity
            self._stats["NI_I_min"] = self._iteration
        if "I_max" not in self._stats or selection_intensity > self._stats["I_max"]:
            self._stats["I_max"] = selection_intensity
            self._stats["NI_I_max"] = self._iteration

        in_parent_pool = 0
        best = current_population.get_fittest(1)[0]
        num_of_best = 0
        best_in_previous = previous_population.get_fittest(1)[0]
        num_of_best_in_previous = 0

        for individual in previous_population.individuals:
            if individual == best_in_previous:
                num_of_best_in_previous += 1
            if individual in current_population:
                in_parent_pool += 1

        for individual in current_population.individuals:
            if individual == best:
                num_of_best += 1
                continue
            break

        growth_rate = 0
        if num_of_best >= num_of_best_in_previous:
            growth_rate = num_of_best / num_of_best_in_previous
        self._growth_rates.append(growth_rate)

        if self._iteration == 2:
            self._stats["GR_early"] = growth_rate
        if "GR_late" not in self._stats and num_of_best >= len(current_population.individuals) / 2:
            self._stats["GR_late"] = growth_rate
            self._stats["NI_GR_late"] = self._iteration

        reproduction = in_parent_pool / len(previous_population.individuals)
        loss_of_diversity = 1 - reproduction

        self._reproduction_coeffs.append(reproduction)
        self._loss_of_diversity_coeffs.append(loss_of_diversity)

        if "RR_min" not in self._stats or reproduction < self._stats["RR_min"]:
            self._stats["RR_min"] = reproduction
            self._stats["NI_RR_min"] = self._iteration
        if "RR_max" not in self._stats or reproduction > self._stats["RR_max"]:
            self._stats["RR_max"] = reproduction
            self._stats["NI_RR_max"] = self._iteration
        if "Teta_min" not in self._stats or loss_of_diversity < self._stats["Teta_min"]:
            self._stats["Teta_min"] = loss_of_diversity
            self._stats["NI_Teta_min"] = self._iteration
        if "Teta_max" not in self._stats or loss_of_diversity > self._stats["Teta_max"]:
            self._stats["Teta_max"] = loss_of_diversity
            self._stats["NI_Teta_max"] = self._iteration

    def _update_final_stats(self):
        if self._stats_mode == "noise":
            self._update_noise_stats()
            return

        self._stats["s_avg"] = np.mean(self._selection_differences)
        self._stats["RR_avg"] = np.mean(self._reproduction_coeffs)
        self._stats["Teta_avg"] = np.mean(self._loss_of_diversity_coeffs)
        self._stats["F_avg"] = self._population.avg_score
        self._stats["F_found"] = self._population.get_fittest(1)[0].fitness
        self._stats["I_avg"] = np.mean(self._selection_intensities)
        self._stats["GR_avg"] = np.mean(self._growth_rates)
        self._stats["NI"] = self._convergence_iteration or -1

    def _evaluate_population(self, population: typing.List[str]) -> models.Population:
        individuals = []

        for individual in population:
            fitness = self._fitness_function(individual)
            individuals.append(models.Individual(individual, fitness))

        return models.Population(individuals)

    def _calculate_ranks(self, population: models.Population):
        individuals: typing.List[models.Individual] = []

        i = 0
        while i < len(population):
            j = i + 1

            if self._modified_selection_algo:
                r = [i]
                while j < len(population):
                    if population.individuals[j] == population.individuals[i]:
                        r.append(j)
                    else:
                        break

                    j += 1

                rank = np.mean(r)
            else:
                rank = i

            while i < j:
                individual = population.individuals[i]

                individuals.append(models.Individual(
                    individual.genotype, individual.fitness, self._scale_function(rank)
                ))

                i += 1

        # for index, individual in enumerate(population.individuals):
        #     individuals.append(models.Individual(
        #         individual.genotype, individual.fitness, self._scale_function(index)
        #     ))

        return models.Population(individuals, sort_on_init=False)

    def fit(self):
        logging.info("Starting fitting")

        while True:
            total_score: float = self._population.score

            self._populations.append(self._population)
            self._total_scores.append(total_score)

            if self._iteration > 0:
                self._update_stats()

            if self._iteration == self._max_iteration:
                logging.info(f"Max iteration exceeded, iterations - {self._iteration}")
                break

            if utils.convergence(self._population):
                self._convergence_iteration = self.iteration
                logging.info(f"Convergence of the population, iterations - {self._iteration}")
                break

            msg = f"Iteration #{self._iteration}. Total score: {total_score}"
            logging.info(msg)

            if self._draw_step and self._iteration % self._draw_step == 0:
                self._draw_scores([
                    individual.fitness for individual in self._population.individuals
                ], msg)

            self._population = self._calculate_ranks(self._population)
            self._population = self._selection_algo(self._population)

            self._iteration += 1

        msg = f"Finished at Iteration #{self._iteration}. Total score: {total_score}"
        logging.info(msg)

        if self._draw_step:
            self._draw_scores([
                individual.fitness for individual in self._population.individuals
            ], msg)

        self._update_final_stats()

        if self._draw_total_steps:
            self._draw_total_scores(self._total_scores, "Total score difference")

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
