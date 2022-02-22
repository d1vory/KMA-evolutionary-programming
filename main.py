# TODO:
# -1. command line args, report generation
# 0. comments
# 1. Logging
# 2. Drawing
import collections
import logging

import fitness_functions
import generators
import genetic_algorithm
import scale_functions
import selection_algorithms
import xlsx


def generate_report(epochs: int, n: int, beta: float, length: int, max_iteration: int, writer: xlsx.XLSX, stats: dict):
    writer.text(f"Experiment (n={n}, beta={beta}, length={length}, max_iteration={max_iteration})", style="bold_bg")
    stats_id = f"{n}_{beta}_{length}_{max_iteration}"

    stats_keys = ['s_min', 'NI_s_min', 's_max', 'NI_s_max', 'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'RR_min',
                  'NI_RR_min', 'RR_max', 'NI_RR_max', 'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'GR_early',
                  'GR_late', 'NI_GR_late', 's_avg', 'RR_avg', 'Teta_avg', 'F_avg', 'F_found', 'I_avg', 'GR_avg', 'NI']
    writer.row(["Criteria \\ N_epoch", *[x + 1 for x in range(epochs)]])
    writer.col(stats_keys, style="bold_bg")

    for key in stats_keys:
        lst = []

        for epoch in range(epochs):
            lst.append(stats[epoch][stats_id][key])

        writer.row(lst)

    writer.set_pos(col=1)
    writer.skip(row=10)


def main():
    epochs = 10
    n_vals = [100, 500, 1000]
    length = 100
    beta_vals = [1.2, 1.6, 2.0]
    max_iteration = 10_000_000
    stats = collections.defaultdict(dict)

    for epoch in range(epochs):
        for n in n_vals:
            generator = generators.NormalGenerator(n=n, length=length, optimal=True)
            population = generator.generate_population()

            for beta in beta_vals:
                algo = genetic_algorithm.GeneticAlgorithm(
                    base_population=population,
                    fitness_function=fitness_functions.FH(),
                    scale_function=scale_functions.LinearRank(beta, n),
                    selection_algo=selection_algorithms.sus,
                    max_iteration=max_iteration,
                    draw_step=None,
                    draw_total_steps=False,
                )
                algo.fit()

                stats[epoch][f"{n}_{beta}_{length}_{max_iteration}"] = algo.stats

    w = xlsx.XLSX("report_1.xlsx", "Linear Rank (without mutations)")

    for n in n_vals:
        for beta in beta_vals:
            generate_report(epochs, n, beta, length, max_iteration, w, stats)

    w.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
