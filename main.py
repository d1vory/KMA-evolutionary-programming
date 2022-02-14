# TODO:
# -1. Function refactoring, command line args, report generation
# 0. comments
# 1. Logging
import logging

import fitness_functions
import generators
import genetic_algorithm
import scale_functions
import selection_algorithms


def evaluate_stats(stats: dict):
    print("\n\n\n====GA STATS====\n")
    for key, value in stats.items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"{key}={value}")
    print("================")


def main():
    n = 500
    length = 100
    beta = 1.2
    generator = generators.NormalGenerator(n=n, length=length, optimal=True)
    algo = genetic_algorithm.GeneticAlgorithm(
        generator=generator,
        fitness_function=fitness_functions.fh,
        scale_function=scale_functions.linear_rank(beta, n),
        selection_algo=selection_algorithms.sus,
        max_iteration=1000,
        draw_step=None,
        draw_total_steps=True,
    )
    algo.fit()

    evaluate_stats(algo.stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
