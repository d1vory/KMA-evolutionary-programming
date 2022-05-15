import collections
import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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


def aggregate_runs_data(runs_data: list, stats_mode: str) -> dict:
    total_data = collections.defaultdict(dict)
    result = {}

    for epoch_data in runs_data:
        for selection_fn in epoch_data:
            if selection_fn not in total_data:
                total_data[selection_fn] = collections.defaultdict(list)

            total_data[selection_fn]["Suc"].append(epoch_data[selection_fn]["NI"] != -1)
            total_data[selection_fn]["NI"].append(epoch_data[selection_fn]["NI"])

            if stats_mode == "noise":
                total_data[selection_fn]["Num0"].append(epoch_data[selection_fn]["ConvTo"] == 0)
                total_data[selection_fn]["Num1"].append(epoch_data[selection_fn]["ConvTo"] == 1)

    for key in total_data:
        data = total_data[key]
        total = len(data["Suc"])
        suc = sum(data["Suc"])

        result[key] = {
            "Suc": suc / total,
            "Min_NI": min(data["NI"]),
            "Max_NI": max(data["NI"]),
            "Avg_NI": np.mean(data["NI"]),
        }

        if stats_mode == "noise":
            num0 = sum(data["Num0"])
            num1 = sum(data["Num1"])

            result[key].update({
                "Num0": num0 / total,
                "Num1": num1 / total,
            })

    return result
