import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import models


def convergence(population: models.Population) -> bool:
    return len(set(
        [individual.genotype for individual in population.individuals]
    )) == 1


def draw_population(p: typing.List[int], title: str):
    if not p:
        return

    if not title:
        title = f"Population of {len(p)} individuals"

    plt.figure()
    plt.hist(p, bins=25, density=True, color="b")

    # Fit a normal distribution to the population
    mu, std = norm.fit(p)
    # Plot the PDF.
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    plt.title(title)
    plt.show()
