# TODO:
# 0. comments
# 1. Logging
import logging

import fitness_functions
import generators
import genetic_algorithm


def main():
    n = 500
    length = 100
    generator = generators.NormalGenerator(n=n, length=length, optimal=True)
    algo = genetic_algorithm.GeneticAlgorithm(
        generator=generator,
        fitness_function=fitness_functions.fh,
        max_iteration=10
    )
    algo.fit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
