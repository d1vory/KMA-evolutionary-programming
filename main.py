# TODO:
# -1. command line args, report generation
# 0. comments
# 1. Logging
# 2. Drawing
import collections
import datetime
import logging

import fitness_functions
import generators
import genetic_algorithm
import scale_functions
import selection_algorithms
import xlsx


def generate_report(
        epochs: int,
        n: int,
        selection_functions: list,
        length: int,
        max_iteration: int,
        writer: xlsx.XLSX, stats: dict
):
    writer.text(f"Experiment (n={n}, length={length}, max_iteration={max_iteration})", style="bold_bg")

    stats_keys = ['NI', 'F_found', 'F_avg', 'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'I_avg', 'GR_early', 'GR_avg',
                  'GR_late', 'NI_GR_late', 'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg', 'Teta_min',
                  'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg', 's_min', 'NI_s_min', 's_max', 'NI_s_max',
                  's_avg']
    # writer.row(["Criteria \\ N_epoch", *[x + 1 for x in range(epochs)]])
    writer.col(['Epochs', 'Selection \\ Criteria', *selection_functions], style="bold_bg")
    # writer.col(stats_keys, style="bold_bg")

    for epoch in range(epochs):
        epoch_to_write = True
        for stat in stats_keys:
            lst = [f"epoch {epoch + 1}" if epoch_to_write else '', stat]

            for selection_function in selection_functions:
                stats_id = f"{n}_{selection_function}_{length}_{max_iteration}"

                lst.append(stats[epoch][stats_id][stat])

            writer.col(lst, style="bold_bg" if epoch_to_write else "normal")

            epoch_to_write = False

    # for key in stats_keys:
    #     lst = []
    #
    #     for epoch in range(epochs):
    #         lst.append(stats[epoch][stats_id][key])
    #
    #     writer.row(lst)

    writer.set_pos(col=1)

    skip_rows = 5 + len(selection_functions) + 2
    writer.skip(row=skip_rows)


def main():
    epochs = 10
    # n_vals = [100, 500, 1000]
    n_vals = [100]
    length = 100

    selection_functions = ['linear 1.2', 'linear 1.6', 'linear 2', 'modified_linear 1.2', 'modified_linear 1.6',
                           'modified_linear 2']
    max_iteration = 10_000_000
    stats = collections.defaultdict(dict)

    for epoch in range(epochs):
        for n in n_vals:
            generator = generators.NormalGenerator(n=n, length=length, optimal=True)
            population = generator.generate_population()

            for selection_function in selection_functions:
                func_name, beta = selection_function.split()
                beta = float(beta)

                if func_name == 'linear':
                    scale_function = scale_functions.LinearRank(beta, n)
                    modified_selection_algo = False
                elif func_name == 'modified_linear':
                    scale_function = scale_functions.LinearRank(beta, n)
                    modified_selection_algo = True
                else:
                    raise Exception

                algo = genetic_algorithm.GeneticAlgorithm(
                    base_population=population,
                    fitness_function=fitness_functions.FH(),
                    scale_function=scale_function,
                    selection_algo=selection_algorithms.my_sus,
                    modified_selection_algo=modified_selection_algo,
                    max_iteration=max_iteration,
                    draw_step=None,
                    draw_total_steps=False,
                )
                algo.fit()

                stats[epoch][f"{n}_{selection_function}_{length}_{max_iteration}"] = algo.stats

    w = xlsx.XLSX(f"report_{datetime.datetime.now().isoformat()}.xlsx", "EP Report Sheet")

    for n in n_vals:
        generate_report(epochs, n, selection_functions, length, max_iteration, w, stats)

    w.save()


if __name__ == "__main__":
    # n = 100
    # length = 100
    # beta = 1.2
    # max_iteration = 1000
    # generator = generators.NormalGenerator(n=n, length=length, optimal=True)
    # algo = genetic_algorithm.GeneticAlgorithm(
    #     base_population=generators.NormalGenerator(n=n, length=length, optimal=True).generate_population(),
    #     fitness_function=fitness_functions.FH(),
    #     scale_function=scale_functions.LinearRank(beta, n),
    #     selection_algo=selection_algorithms.sus,
    #     max_iteration=max_iteration,
    #     draw_step=3,
    #     draw_total_steps=True,
    # )
    # algo.fit()
    #
    # logging.basicConfig(level=logging.INFO)
    main()
