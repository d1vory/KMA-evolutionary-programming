# TODO:
# -1. command line args, report generation
# 0. comments
# 1. Logging
# 2. Drawing
import collections
import datetime
import itertools
import logging
import time

import fitness_functions
import generators
import genetic_algorithm
import scale_functions
import selection_algorithms
import xlsx


def generate_report(
        epochs: int,
        n: int,
        length: int,
        fitness_fn: str,
        selection_fns: list,
        max_iteration: int,
        writer: xlsx.XLSX,
        stats: dict,
):
    writer.text(f"{fitness_fn}, n={n}, length={length}, max_iteration={max_iteration}", style="bold_bg")

    full_stats_keys = ['NI', 'F_found', 'F_avg', 'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'I_avg', 'GR_early',
                       'GR_avg', 'GR_late', 'NI_GR_late', 'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
                       'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg', 's_min', 'NI_s_min', 's_max',
                       'NI_s_max', 's_avg']
    noise_stats_keys = ['NI', 'ConvTo']

    stats_keys = noise_stats_keys if fitness_fn == 'fconst' else full_stats_keys
    # writer.row(["Criteria \\ N_epoch", *[x + 1 for x in range(epochs)]])
    writer.col([
        'Epochs',
        'Selection \\ Criteria',
        *[f"linear{' modified' if modified else ''}, beta={beta}"
          for beta, modified in selection_fns]],
        style="bold_bg"
    )
    # writer.col(stats_keys, style="bold_bg")

    for epoch in range(epochs):
        epoch_to_write = True
        for stat in stats_keys:
            lst = [f"epoch {epoch + 1}" if epoch_to_write else '', stat]

            for beta, modified in selection_fns:
                stats_id = f"{epoch}_{n}_{fitness_fn}_{beta}_{modified}"

                lst.append(stats[stats_id][stat])

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

    skip_rows = 5 + len(selection_fns) + 2
    writer.skip(row=skip_rows)


def main():
    beta_vals = [1.2, 1.6, 2.0]
    modified_vals = [True, False]
    selection_fns = list(itertools.product(beta_vals, modified_vals))

    epochs = 10
    # n_vals = [100, 500, 1000]
    n_vals = [100]
    length = 100
    m = 10

    fitness_fns = {
        'fh': fitness_functions.FH(),
        'fhd(theta=10)': fitness_functions.FHD(theta=10),
        'fhd(theta=50)': fitness_functions.FHD(theta=50),
        'fhd(theta=150)': fitness_functions.FHD(theta=150),
        'f=x^2': fitness_functions.FX('x^2', 0, 10.23, m),
        'f=x': fitness_functions.FX('x', 0, 10.23, m),
        'f=x^4': fitness_functions.FX('x^4', 0, 10.23, m),
        'f=2x^2': fitness_functions.FX('2x^2', 0, 10.23, m),
        'f=(5.12)^2-x^2': fitness_functions.FX('(5.12)^2-x^2', -5.11, 5.12, m),
        'f=(5.12)^4-x^4': fitness_functions.FX('(5.12)^4-x^4', -5.11, 5.12, m),
        'f=e^(0.25*x)': fitness_functions.FECX(0.25, 0, 10.23, m),
        'f=e^(1*x)': fitness_functions.FECX(1, 0, 10.23, m),
        'f=e^(2*x)': fitness_functions.FECX(2, 0, 10.23, m),
    }
    length_100_fn = ['fh', 'fhd(theta=10)', 'fhd(theta=50)', 'fhd(theta=150)', 'fconst']
    max_iteration = 10_000_000
    stats = collections.defaultdict(dict)

    for n in n_vals:
        for epoch in range(epochs):
            generator = generators.ConstGenerator(n=n, length=length)
            population = generator.generate_population()

            for beta, modified in selection_fns:
                print(
                    f"Fitting GA: epoch={epoch}, n={n}, fitness=fconst, "
                    f"linear(beta={beta}, modified={modified})"
                )

                algo = genetic_algorithm.GeneticAlgorithm(
                    base_population=population,
                    fitness_function=fitness_functions.FConst(),
                    scale_function=scale_functions.LinearRank(beta, n),
                    selection_algo=selection_algorithms.my_sus,
                    stats_mode="noise",
                    modified_selection_algo=modified,
                    max_iteration=max_iteration,
                    draw_step=None,
                    draw_total_steps=False,
                )
                algo.fit()

                stats[f"{epoch}_{n}_fconst_{beta}_{modified}"] = algo.stats
        continue
        for fitness_fn, fn in fitness_fns.items():
            generator = generators.NormalGenerator(
                n=n,
                length=length if fitness_fn in length_100_fn else m,
                optimal=True
            )
            for epoch in range(epochs):
                population = generator.generate_population()

                for beta, modified in selection_fns:
                    print(
                        f"Fitting GA: epoch={epoch}, n={n}, fitness={fitness_fn}, "
                        f"linear(beta={beta}, modified={modified})"
                    )

                    algo = genetic_algorithm.GeneticAlgorithm(
                        base_population=population,
                        fitness_function=fn,
                        scale_function=scale_functions.LinearRank(beta, n),
                        selection_algo=selection_algorithms.my_sus,
                        modified_selection_algo=modified,
                        max_iteration=max_iteration,
                        draw_step=None,
                        draw_total_steps=False,
                    )
                    algo.fit()

                    stats[f"{epoch}_{n}_{fitness_fn}_{beta}_{modified}"] = algo.stats

    w = xlsx.XLSX(f"report_{datetime.datetime.now().isoformat()}.xlsx", "EP Report Sheet")

    for n in n_vals:
        generate_report(epochs, n, length, 'fconst', selection_fns, max_iteration, w, stats)
        continue
        for fitness_fn in fitness_fns:
            generate_report(
                epochs,
                n,
                length if fitness_fn in length_100_fn else m,
                fitness_fn,
                selection_fns,
                max_iteration,
                w,
                stats
            )

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
    start = time.monotonic()
    main()
    print(f"Finished in {time.monotonic() - start:.3f}s.")
