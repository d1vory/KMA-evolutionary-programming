import copy
import random
import typing

import models
import numpy as np


def rws(
        population: models.Population, pointers: typing.List[float]
) -> models.Population:
    keep = []
    for p in pointers:
        i = 0
        # use individual's coeff of
        sum_ = population.individuals[i].scaled_fitness
        while sum_ < p:
            i += 1
            sum_ += population.individuals[i].scaled_fitness
        keep.append(copy.copy(population.individuals[i]))
    return models.Population(keep)

def my_rws(population: models.Population):
    try:
        probabilities = [chromosome.scaled_fitness / population.scaled_fitness for chromosome in population.individuals]
        keep = [np.random.choice(population.individuals, p=probabilities) for _ in range(0, len(population))]
    except:
        kijh = 2+2
        print('asd')
    kek = models.Population(keep)
    return kek

def my_sus(population: models.Population) -> models.Population:
    f = population.scaled_fitness  # sum of all probabilities = 1
    n = len(population)
    p = f / n
    start = random.uniform(0, p)
    pointers = [start + i * p for i in range(n)]
    return rws(population, pointers)


def sus(population: models.Population) -> models.Population:
    total_fitness = 0
    fitness_scale = []
    for index, individual in enumerate(population.individuals):
        total_fitness += individual.scaled_fitness
        if index == 0:
            fitness_scale.append(individual.scaled_fitness)
        else:
            fitness_scale.append(individual.scaled_fitness + fitness_scale[index - 1])

    # Store the selected parents
    mating_pool = []
    # Equal to the size of the population
    number_of_parents = len(population.individuals)
    # How fast we move along the fitness scale
    fitness_step = total_fitness / number_of_parents
    random_offset = random.uniform(0, fitness_step)

    # Iterate over the parents size range and for each:
    # - generate pointer position on the fitness scale
    # - pick the parent who corresponds to the current pointer position and add them to the mating pool
    current_fitness_pointer = random_offset
    last_fitness_scale_position = 0
    for index in range(len(population.individuals)):
        for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
            if fitness_scale[fitness_scale_position] >= current_fitness_pointer:
                mating_pool.append(population.individuals[fitness_scale_position])
                last_fitness_scale_position = fitness_scale_position
                break
        current_fitness_pointer += fitness_step

    return models.Population(mating_pool)
