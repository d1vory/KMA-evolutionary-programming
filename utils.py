import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import models


def decode_sampling(a, b, x, m):
    return a + x * ((b - a) / (2 ** m - 1))


def get_bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    str
    """
    return format(x, 'b').zfill(n)


def get_dec(x):
    return int(x, 2)


def encode_gray(x):
    return x ^ (x >> 1)


def decode_gray(x):
    mask = x

    while mask:
        mask >>= 1
        x ^= mask

    return x


def decode(x, a, b, m):
    x = get_dec(x)
    x = decode_gray(x)
    return decode_sampling(a, b, x, m)


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
