import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def generate_norm_dist(low, high, length):
    variation = abs(low - high) / 2
    std_dev = variation / 5  # 4
    mean = (low + high) / 2
    dist = np.random.normal(mean, std_dev, length)
    dist = list(filter(lambda x: low <= x <= high, dist))
    for _ in range(length - len(dist)):
        dist.append(random.choice(dist))
    return dist


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def decode_sampling(a, b, x, m):
    return a + x * ((b - a) / (2 ** m - 1))


def encode_sampling(a, b, x, m):
    return int(((x - a) / (b - a)) * (2 ** m - 1))


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


def encode(x, a, b, m):
    x = int(encode_sampling(a, b, x, m))
    x = encode_gray(x)
    return get_bin(x, m)


def decode_binary(x, a, b, m):
    x = get_dec(x)
    return decode_sampling(a, b, x, m)


def encode_binary(x, a, b, m):
    x = int(encode_sampling(a, b, x, m))
    return get_bin(x, m)


def aggregate_runs_data(
        runs_data: list, stats_mode: str, optimal: str, fitness_fn
) -> dict:
    total_data = collections.defaultdict(dict)
    result = {}

    for epoch_data in runs_data:
        for selection_fn in epoch_data:
            if selection_fn not in total_data:
                total_data[selection_fn] = collections.defaultdict(list)

            if stats_mode == "full":
                if epoch_data[selection_fn]["NI"] == -1:
                    total_data[selection_fn]["Suc"].append(False)
                    print("ERROR --- RUN WAS NOT SUCCESSFUL, POPULATION WAS NOT CONVERGENCE")
                    continue

                if fitness_fn.is_arg_real():
                    x_val = fitness_fn.decode(optimal)
                    y_val = fitness_fn(optimal)
                    x_found = fitness_fn.decode(epoch_data[selection_fn]["F"])
                    y_found = fitness_fn(epoch_data[selection_fn]["F"])

                    if abs(x_found - x_val) > 0.01 or y_found < y_val - 0.01:
                        total_data[selection_fn]["Suc"].append(False)
                        print("ERROR --- RUN WAS NOT SUCCESSFUL, REAL ARG DONT MEET OPTIMAL CONDITIONS")
                        continue

                else:
                    if epoch_data[selection_fn]["F"] != optimal:
                        total_data[selection_fn]["Suc"].append(False)
                        print("ERROR --- RUN WAS NOT SUCCESSFUL, POPULATION WAS NOT OPTIMAL")
                        continue

                total_data[selection_fn]["Suc"].append(True)

            else:
                total_data[selection_fn]["Suc"].append(epoch_data[selection_fn]["NI"] != -1)

            total_data[selection_fn]["NI"].append(epoch_data[selection_fn]["NI"])

            if stats_mode == "noise":
                total_data[selection_fn]["Num0"].append(epoch_data[selection_fn]["ConvTo"] == 0)
                total_data[selection_fn]["Num1"].append(epoch_data[selection_fn]["ConvTo"] == 1)

                # rr
                total_data[selection_fn]["RR_min"].append(epoch_data[selection_fn]["RR_min"])
                total_data[selection_fn]["NI_RR_min"].append(epoch_data[selection_fn]["NI_RR_min"])
                total_data[selection_fn]["RR_max"].append(epoch_data[selection_fn]["RR_max"])
                total_data[selection_fn]["NI_RR_max"].append(epoch_data[selection_fn]["NI_RR_max"])
                total_data[selection_fn]["RR_avg"].append(epoch_data[selection_fn]["RR_avg"])
                # loss of diversity
                total_data[selection_fn]["Teta_min"].append(epoch_data[selection_fn]["Teta_min"])
                total_data[selection_fn]["NI_Teta_min"].append(epoch_data[selection_fn]["NI_Teta_min"])
                total_data[selection_fn]["Teta_max"].append(epoch_data[selection_fn]["Teta_max"])
                total_data[selection_fn]["NI_Teta_max"].append(epoch_data[selection_fn]["NI_Teta_max"])
                total_data[selection_fn]["Teta_avg"].append(epoch_data[selection_fn]["Teta_avg"])
            else:
                # intensity
                total_data[selection_fn]["I_min"].append(epoch_data[selection_fn]["I_min"])
                total_data[selection_fn]["NI_I_min"].append(epoch_data[selection_fn]["NI_I_min"])
                total_data[selection_fn]["I_max"].append(epoch_data[selection_fn]["I_max"])
                total_data[selection_fn]["NI_I_max"].append(epoch_data[selection_fn]["NI_I_max"])
                total_data[selection_fn]["I_avg"].append(epoch_data[selection_fn]["I_avg"])
                # growth rate
                total_data[selection_fn]["GR_avg"].append(epoch_data[selection_fn]["GR_avg"])
                total_data[selection_fn]["GR_early"].append(epoch_data[selection_fn]["GR_early"])
                total_data[selection_fn]["GR_late"].append(epoch_data[selection_fn].get("GR_late", 1.03))
                # rr
                total_data[selection_fn]["RR_min"].append(epoch_data[selection_fn]["RR_min"])
                total_data[selection_fn]["NI_RR_min"].append(epoch_data[selection_fn]["NI_RR_min"])
                total_data[selection_fn]["RR_max"].append(epoch_data[selection_fn]["RR_max"])
                total_data[selection_fn]["NI_RR_max"].append(epoch_data[selection_fn]["NI_RR_max"])
                total_data[selection_fn]["RR_avg"].append(epoch_data[selection_fn]["RR_avg"])
                # loss of diversity
                total_data[selection_fn]["Teta_min"].append(epoch_data[selection_fn]["Teta_min"])
                total_data[selection_fn]["NI_Teta_min"].append(epoch_data[selection_fn]["NI_Teta_min"])
                total_data[selection_fn]["Teta_max"].append(epoch_data[selection_fn]["Teta_max"])
                total_data[selection_fn]["NI_Teta_max"].append(epoch_data[selection_fn]["NI_Teta_max"])
                total_data[selection_fn]["Teta_avg"].append(epoch_data[selection_fn]["Teta_avg"])
                # selection difference
                total_data[selection_fn]["s_min"].append(epoch_data[selection_fn]["s_min"])
                total_data[selection_fn]["NI_s_min"].append(epoch_data[selection_fn]["NI_s_min"])
                total_data[selection_fn]["s_max"].append(epoch_data[selection_fn]["s_max"])
                total_data[selection_fn]["NI_s_max"].append(epoch_data[selection_fn]["NI_s_max"])
                total_data[selection_fn]["s_avg"].append(epoch_data[selection_fn]["s_avg"])

    for key in total_data:
        data = total_data[key]
        total = len(data["Suc"])
        suc = sum(data["Suc"])

        if not suc:
            result[key] = {
                "Suc": suc / total,
            }
            continue

        result[key] = {
            "Suc": suc / total,
            "Min_NI": min(data["NI"]),
            "Max_NI": max(data["NI"]),
            "Avg_NI": np.mean(data["NI"]),
            "Sigma_NI": np.std(data["NI"])
        }

        if stats_mode == "noise":
            num0 = sum(data["Num0"])
            num1 = sum(data["Num1"])

            result[key].update({
                "Num0": num0 / total,
                "Num1": num1 / total,
            })
        else:
            # intensity
            min_i_min = min(data["I_min"])
            ni_i_min = data["NI_I_min"][data["I_min"].index(min_i_min)]
            max_i_max = max(data["I_max"])
            ni_i_max = data["NI_I_max"][data["I_max"].index(max_i_max)]

            avg_i_min = np.mean(data["I_min"])
            avg_i_max = np.mean(data["I_max"])
            avg_i_avg = np.mean(data["I_avg"])

            std_i_min = np.std(data["I_min"])
            std_i_max = np.std(data["I_max"])
            std_i_avg = np.std(data["I_avg"])

            # growth rate
            avg_gr_early = np.mean(data["GR_early"])
            min_gr_early = min(data["GR_early"])
            max_gr_early = max(data["GR_early"])

            avg_gr_late = np.mean(data["GR_late"])
            min_gr_late = min(data["GR_late"])
            max_gr_late = max(data["GR_late"])

            avg_gr_avg = np.mean(data["GR_avg"])
            min_gr_avg = min(data["GR_avg"])
            max_gr_avg = max(data["GR_avg"])

            # rr
            min_rr_min = min(data["RR_min"])
            ni_rr_min = data["NI_RR_min"][data["RR_min"].index(min_rr_min)]
            max_rr_max = max(data["RR_max"])
            ni_rr_max = data["NI_RR_max"][data["RR_max"].index(max_rr_max)]

            avg_rr_min = np.mean(data["RR_min"])
            avg_rr_max = np.mean(data["RR_max"])
            avg_rr_avg = np.mean(data["RR_avg"])

            std_rr_min = np.std(data["RR_min"])
            std_rr_max = np.std(data["RR_max"])
            std_rr_avg = np.std(data["RR_avg"])

            # loss of diversity
            min_teta_min = min(data["Teta_min"])
            ni_teta_min = data["NI_Teta_min"][data["Teta_min"].index(min_teta_min)]
            max_teta_max = max(data["Teta_max"])
            ni_teta_max = data["NI_Teta_max"][data["Teta_max"].index(max_teta_max)]

            avg_teta_min = np.mean(data["Teta_min"])
            avg_teta_max = np.mean(data["Teta_max"])
            avg_teta_avg = np.mean(data["Teta_avg"])

            std_teta_min = np.std(data["Teta_min"])
            std_teta_max = np.std(data["Teta_max"])
            std_teta_avg = np.std(data["Teta_avg"])

            # selection difference
            min_s_min = min(data["s_min"])
            ni_s_min = data["NI_s_min"][data["s_min"].index(min_s_min)]
            max_s_max = max(data["s_max"])
            ni_s_max = data["NI_s_max"][data["s_max"].index(max_s_max)]

            avg_s_min = np.mean(data["s_min"])
            avg_s_max = np.mean(data["s_max"])
            avg_s_avg = np.mean(data["s_avg"])

            result[key].update({
                "Min_I_min": min_i_min, "NI_I_min": ni_i_min, "Max_I_max": max_i_max,
                "NI_I_max": ni_i_max, "Avg_I_min": avg_i_min, "Avg_I_max": avg_i_max, "Avg_I_avg": avg_i_avg,
                "Sigma_I_max": std_i_max, "Sigma_I_min": std_i_min, "Sigma_I_avg": std_i_avg,
                "AvgGR_early": avg_gr_early, "MinGR_early": min_gr_early, "MaxGR_early": max_gr_early,
                "AvgGR_late": avg_gr_late, "MinGR_late": min_gr_late, "MaxGR_late": max_gr_late,
                "AvgGR_avg": avg_gr_avg, "MinGR_avg": min_gr_avg, "MaxGR_avg": max_gr_avg,
                "Min_RR_min": min_rr_min, "NI_RR_min": ni_rr_min, "Max_RR_max": max_rr_max,
                "NI_RR_max": ni_rr_max, "Avg_RR_min": avg_rr_min, "Avg_RR_max": avg_rr_max, "Avg_RR_avg": avg_rr_avg,
                "Sigma_RR_max": std_rr_max, "Sigma_RR_min": std_rr_min, "Sigma_RR_avg": std_rr_avg,
                "Min_Teta_min": min_teta_min, "NI_Teta_min": ni_teta_min, "Max_Teta_max": max_teta_max,
                "NI_Teta_max": ni_teta_max, "Avg_Teta_min": avg_teta_min, "Avg_Teta_max": avg_teta_max,
                "Avg_Teta_avg": avg_teta_avg, "Sigma_Teta_max": std_teta_max, "Sigma_Teta_min": std_teta_min,
                "Sigma_Teta_avg": std_teta_avg, "Min_s_min": min_s_min, "NI_s_min": ni_s_min, "Max_s_max": max_s_max,
                "NI_s_max": ni_s_max, "Avg_s_min": avg_s_min, "Avg_s_max": avg_s_max, "Avg_s_avg": avg_s_avg
            })

    return result


def _draw(
        values: list,
        title: str,
        x_label: str = None,
        y_label: str = None,
        color: str = "b",
        filename: str = None,
        mode: str = "one"
):
    if filename:
        plt.ioff()

    if mode == "multiple":
        x = np.arange(0., len(values[0]), 1.)
        for value in values:
            plt.plot(x, value)
    elif mode == "hist":
        plt.hist(values, bins=25, color=color)
    else:
        x = np.arange(0., len(values), 1.)
        plt.plot(x, values, color=color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if filename:
        plt.savefig(filename)
        plt.close()
        plt.ion()
    else:
        plt.show()


def draw_graphics(
        values: list, title: str, x_label: str = None, y_label: str = None, color: str = "b", filename: str = None
):
    _draw(values, title, x_label, y_label, color, filename)


def draw_multiple(
        values: list, title: str, x_label: str = None, y_label: str = None, filename: str = None
):
    _draw(values, title, x_label, y_label, filename=filename, mode="multiple")


def draw_hist(
        values: list, title: str, x_label: str = None, y_label: str = None, color: str = "b", filename: str = None
):
    _draw(values, title, x_label, y_label, color, filename, mode="hist")


if __name__ == "__main__":
    x_ = 10.23
    enc = encode(x_, 0, 10.23, 10)
    print(enc)
    dec = decode(enc, 0, 10.23, 10)
    print(dec)
    print(dec == x_)
