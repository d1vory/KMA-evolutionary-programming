# TODO:
# -1. command line args
# 0. comments
# 1. Logging
# 2. Drawing
import time

from core import evaluator_config, report_builder, evaluator


def main():
    working_dir = "new_reports"

    n_vals = [100]  # [100, 500, 1000]
    # epochs = 10
    # max_iteration = 10_000_000
    # beta = [1.2, 1.6, 2.0]
    # modified = [True, False]
    # fitness_fns = ['fconst', 'fh', 'fhd(theta=10)', 'fhd(theta=50)', 'fhd(theta=150)', 'f=x^2',
    # 'f=x', 'f=x^4', 'f=2x^2', 'f=(5.12)^2-x^2', 'f=(5.12)^4-x^4', 'f=e^(0.25*x)', 'f=e^(1*x)', 'f=e^(2*x)']
    fitness_fns = ['fconst', 'f=x^2']
    evaluator.Evaluator(
        evaluator_config.get_config(n_vals=n_vals, writing_dir=working_dir, fitness_fns=fitness_fns)
    ).evaluate()
    report_builder.ReportBuilder(working_dir).generate()
    # TODO: graphics


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

    start = time.monotonic()
    main()
    print(f"Finished in {time.monotonic() - start:.3f}s.")
