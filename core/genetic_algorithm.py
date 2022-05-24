import collections
import json
import logging
import math
import random
import re
import typing

import numpy as np

import models
from core import fitness_functions, utils


class GeneticAlgorithm:
    def __init__(
            self, *,
            base_population: typing.List[str],
            fitness_function: fitness_functions.FitnessFunction,
            scale_function: models.Function,
            selection_algo: typing.Callable[
                [models.Population], models.Population
            ],
            optimal: str,
            modified_selection_algo: bool = False,
            stats_mode: str = "full",
            max_iteration: int = 10_000_000,
            mutation_rate: typing.Union[None, float] = None,
            early_stopping: typing.Union[None, int] = None,
            draw_step: typing.Union[None, int] = None,
            draw_total_steps: bool = False,
            graphics_dir: str = None
    ):
        self._base_population: typing.List[str] = base_population

        self._fitness_function: fitness_functions.FitnessFunction = fitness_function
        self._scale_function: models.Function = scale_function
        self._selection_algo: typing.Callable[
            [models.Population], models.Population
        ] = selection_algo
        self._modified_selection_algo = modified_selection_algo
        self._optimal: str = optimal

        self._stats_mode: str = stats_mode
        self._max_iteration: int = max_iteration
        self._draw_step: int = draw_step
        self._draw_total_steps = draw_total_steps
        self._iteration: int = 0
        self._mutation_rate: float = mutation_rate
        self._early_stopping: int = early_stopping
        self._early_stopping_iteration: int = 0  # works only with mutation rate

        self._populations: typing.List[models.Population] = []
        self._total_scores: typing.List[float] = []

        self._population: models.Population = self._evaluate_population(self._base_population)
        self._population_len: int = len(self._population)
        self._individual_len: int = len(self._population.individuals[0])
        self._total_genes: int = self._population_len * self._individual_len

        self._stats: typing.Dict[str, typing.Union[float, str]] = {}
        self._selection_differences: typing.List[float] = []
        self._reproduction_coeffs: typing.List[float] = []
        self._loss_of_diversity_coeffs: typing.List[float] = []
        self._selection_intensities: typing.List[float] = []
        self._growth_rates: typing.List[float] = []
        self._convergence_iteration: typing.Union[None, float] = None

        self._graphics_dir: str = graphics_dir
        self._graphics_data: dict = collections.defaultdict(list)

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

        if selection_difference < self._stats.get("s_min", math.inf):
            self._stats["s_min"] = selection_difference
            self._stats["NI_s_min"] = self._iteration
        if selection_difference > self._stats.get("s_max", -math.inf):
            self._stats["s_max"] = selection_difference
            self._stats["NI_s_max"] = self._iteration

        selection_intensity = selection_difference / previous_population.std_score
        self._selection_intensities.append(selection_intensity)

        if selection_intensity < self._stats.get("I_min", math.inf):
            self._stats["I_min"] = selection_intensity
            self._stats["NI_I_min"] = self._iteration
        if selection_intensity > self._stats.get("I_max", -math.inf):
            self._stats["I_max"] = selection_intensity
            self._stats["NI_I_max"] = self._iteration

        in_parent_pool = 0
        best = current_population.get_fittest(1)[0]
        num_of_best = 0
        best_in_previous = previous_population.get_fittest(1)[0]
        num_of_best_in_previous = 0

        for individual in previous_population.individuals[::-1]:
            if individual == best_in_previous:
                num_of_best_in_previous += 1
            if individual in current_population:
                in_parent_pool += 1

        for individual in current_population.individuals[::-1]:
            if individual == best:
                num_of_best += 1
                continue
            break

        growth_rate = 0
        if num_of_best >= num_of_best_in_previous:
            growth_rate = num_of_best / num_of_best_in_previous
        self._growth_rates.append(growth_rate)

        if self._iteration == 1:
            self._stats["GR_early"] = growth_rate
        if "GR_late" not in self._stats and num_of_best >= self._population_len / 2:
            self._stats["GR_late"] = growth_rate
            self._stats["NI_GR_late"] = self._iteration

        reproduction = in_parent_pool / self._population_len
        loss_of_diversity = 1 - reproduction

        self._reproduction_coeffs.append(reproduction)
        self._loss_of_diversity_coeffs.append(loss_of_diversity)

        if reproduction < self._stats.get("RR_min", math.inf):
            self._stats["RR_min"] = reproduction
            self._stats["NI_RR_min"] = self._iteration
        if reproduction > self._stats.get("RR_max", -math.inf):
            self._stats["RR_max"] = reproduction
            self._stats["NI_RR_max"] = self._iteration
        if loss_of_diversity < self._stats.get("Teta_min", math.inf):
            self._stats["Teta_min"] = loss_of_diversity
            self._stats["NI_Teta_min"] = self._iteration
        if loss_of_diversity > self._stats.get("Teta_max", -math.inf):
            self._stats["Teta_max"] = loss_of_diversity
            self._stats["NI_Teta_max"] = self._iteration

        # graphics data
        self._graphics_data["avg_score"].append(current_population.avg_score)
        self._graphics_data["intensity"].append(selection_intensity)
        self._graphics_data["difference"].append(selection_difference)
        self._graphics_data["std_score"].append(current_population.std_score)
        self._graphics_data["best_part"].append(num_of_best / self._population_len)
        self._graphics_data["growth_rate"].append(growth_rate)
        self._graphics_data["reproduction"].append(reproduction)
        self._graphics_data["loss_of_diversity"].append(loss_of_diversity)

    def _update_final_stats(self):
        if self._stats_mode == "noise":
            self._update_noise_stats()
            return

        self._stats["s_avg"] = np.mean(self._selection_differences)
        self._stats["RR_avg"] = np.mean(self._reproduction_coeffs)
        self._stats["Teta_avg"] = np.mean(self._loss_of_diversity_coeffs)
        self._stats["F_avg"] = self._population.avg_score
        self._stats["F_found"] = self._population.get_fittest(1)[0].fitness
        self._stats["F"] = self._population.get_fittest(1)[0].genotype
        self._stats["I_avg"] = np.mean(self._selection_intensities)
        self._stats["GR_avg"] = np.mean(self._growth_rates)
        self._stats["NI"] = self._convergence_iteration or -1

    def _draw_hists(self, iteration=None):
        iteration = "final" if iteration is None else iteration
        ones_in_genotypes = [
            len(re.findall("1", individual.genotype))
            for individual in self._population.individuals
        ]
        utils.draw_hist(
            ones_in_genotypes,
            f"Amount of '1' in chromosomes: {iteration} iteration",
            "'1' in chromosomes",
            "Individuals",
            filename=f"{self._graphics_dir}/ones_in_genotypes_hist_{iteration}.png"
        )

        if self._fitness_function.is_arg_real():
            fitness_arr = self._population.fitness_arr
            x_vals = [
                self._fitness_function.decode(individual.genotype)
                for individual in self._population.individuals
            ]
            utils.draw_hist(
                fitness_arr,
                f"Fitness Score Hist: {iteration} iteration",
                "Score",
                "Individuals",
                filename=f"{self._graphics_dir}/fitness_score_hist_{iteration}.png"
            )
            utils.draw_hist(
                x_vals,
                f"X values: {iteration} iteration",
                "X - real number",
                "Individuals",
                filename=f"{self._graphics_dir}/x_vals_hist_{iteration}.png"
            )

    def _draw_graphics(self):
        with open(f"{self._graphics_dir}/data.json", 'w', encoding='utf-8') as f:
            json.dump(self._graphics_data, f, ensure_ascii=False, indent=4)

        utils.draw_graphics(
            self._graphics_data["avg_score"], "AVG FITNESS", "N generation", "Avg fitness",
            filename=f"{self._graphics_dir}/avg_score.png"
        )
        utils.draw_graphics(
            self._graphics_data["intensity"], "INTENSITY", "N generation", "Intensity",
            filename=f"{self._graphics_dir}/intensity.png"
        )
        utils.draw_graphics(
            self._graphics_data["difference"], "DIFFERENCE", "N generation", "Difference",
            filename=f"{self._graphics_dir}/difference.png"
        )
        utils.draw_graphics(
            self._graphics_data["std_score"], "STANDARD DEVIATION", "N generation", "Std",
            filename=f"{self._graphics_dir}/std.png"
        )
        utils.draw_multiple(
            [self._graphics_data["intensity"], self._graphics_data["difference"]], "INTENSITY AND DIFFERENCE",
            "N generation", "Intensity and Difference",
            filename=f"{self._graphics_dir}/intensity_difference.png"
        )
        utils.draw_graphics(
            self._graphics_data["best_part"], "BEST PARTITION", "N generation", "Best part",
            filename=f"{self._graphics_dir}/best_count.png"
        )
        utils.draw_graphics(
            self._graphics_data["growth_rate"], "GROWTH RATE", "N generation", "Growth rate",
            filename=f"{self._graphics_dir}/growth_rate.png"
        )
        utils.draw_multiple(
            [self._graphics_data["reproduction"], self._graphics_data["loss_of_diversity"]],
            "REPRODUCTION AND LOSS OF DIVERSITY", "N generation", "Reproduction and Loss of Diversity",
            filename=f"{self._graphics_dir}/reproduction_loss_of_diversity.png"
        )

    def _mutate(self):
        population_changed = False

        for individual in self._population.individuals:
            changed = False

            for gene_index in range(self._individual_len):
                if random.random() < self._mutation_rate:
                    changed = True
                    genotype = list(individual.genotype)
                    genotype[gene_index] = "1" if genotype[gene_index] == "0" else "0"
                    individual.genotype = "".join(genotype)

            if changed:
                population_changed = True
                individual.fitness = self._fitness_function(individual.genotype)

        if population_changed:
            self._population.invalidate()

    def _evaluate_population(self, population: typing.List[str]) -> models.Population:
        individuals = []

        for individual in population:
            fitness = self._fitness_function(individual)
            individuals.append(models.Individual(individual, fitness))

        return models.Population(individuals)

    def _calculate_ranks(self):
        i = 0
        while i < len(self._population):
            j = i + 1

            if self._modified_selection_algo:
                r = [i]
                while j < len(self._population):
                    if self._population.individuals[j] == self._population.individuals[i]:
                        r.append(j)
                    else:
                        break

                    j += 1

                rank = np.mean(r)
            else:
                rank = i

            while i < j:
                self._population.individuals[i].rank = self._scale_function(rank)

                i += 1

    def fit(self):
        logging.info("Starting fitting")

        while True:
            self._population.sort()

            total_score: float = self._population.score

            self._populations.append(self._population)
            self._total_scores.append(total_score)

            if self._iteration > 0:
                self._update_stats()

            if self._iteration == self._max_iteration:
                logging.info(f"Max iteration exceeded, iterations - {self._iteration}")
                break

            if self._mutation_rate is None and self._population.convergence():
                # if self._population.convergence():
                self._convergence_iteration = self._iteration
                logging.info(f"Convergence of the population, iterations - {self._iteration}")
                break

            msg = f"Iteration #{self._iteration}. Total score: {total_score}"
            logging.info(msg)
            # print(msg)

            if self._draw_step and self._iteration % self._draw_step == 0:
                utils.draw_hist(self._population.fitness_arr, msg, "Scores", "Number of Individuals")

            if self._graphics_dir and self._iteration < 5:
                self._draw_hists(self._iteration)

            self._calculate_ranks()
            self._population = self._selection_algo(self._population)
            if self._mutation_rate is not None:
                previous_population = self._populations[-1]

                if self._population.avg_score - previous_population.avg_score <= 0.0001:
                    self._early_stopping_iteration += 1
                else:
                    self._early_stopping_iteration = 0

                if self._early_stopping_iteration == self._early_stopping:
                    self._convergence_iteration = self._iteration
                    logging.info(
                        f"Early stopping on I#{self._iteration} after {self._early_stopping_iteration} iterations"
                    )
                    break

                self._mutate()

            self._iteration += 1

        msg = f"Finished at Iteration #{self._iteration}. Total score: {total_score}. " \
              f"Convergence: {self._convergence_iteration}"
        logging.info(msg)
        # print(msg)

        if self._draw_step:
            utils.draw_hist(self._population.fitness_arr, msg, "Scores", "Number of Individuals")

        self._update_final_stats()

        if self._draw_total_steps:
            utils.draw_graphics(
                self._total_scores, "Total score", "N generation", "Score",
            )

        if self._graphics_dir:
            self._draw_hists()
            self._draw_graphics()
