import datetime
import functools
import json
import multiprocessing
import pathlib

from core import evaluator_config, genetic_algorithm, scale_functions, selection_algorithms, utils


class Evaluator:
    def __init__(self, config: evaluator_config.EvaluatorConfig, cpu_count: int = None):
        self._config: evaluator_config.EvaluatorConfig = config
        self._writing_dir: pathlib.Path = pathlib.Path(f"./{self._config.writing_dir}/data")
        self._graphics_dir: pathlib.Path = pathlib.Path(f"./{self._config.writing_dir}/graphics")

        self._writing_dir.mkdir(parents=True, exist_ok=True)

        self._cpu_count = cpu_count or multiprocessing.cpu_count() - 1

    def _write_report(self, name, data):
        with open(self._writing_dir / f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def run_epoch(
            self, epoch: int, n: int, max_iteration: int, fitness_fn: evaluator_config.FitnessFunctionConfig
    ):
        fn = fitness_fn.handler(**fitness_fn.values)
        stats_mode = fitness_fn.stats_mode
        length = fitness_fn.length

        optimal = True
        if fitness_fn.mutation_rate is not None and epoch >= 5:  # epochs from 0
            optimal = False

        generator = fitness_fn.generator(
            n=n,
            length=length,
            optimal=fitness_fn.optimal,
            generate_optimal=optimal,
            fitness_fn=fn,
            low_range=fitness_fn.values.get("low"),
            high_range=fitness_fn.values.get("high"),
        )
        run_data = {}

        population = generator.generate_population()

        for selection_fn in self._config.selection_fns:
            print(
                f"Fitting GA: epoch={epoch}, n={n}, fitness={fitness_fn.name}, "
                f"linear(beta={selection_fn.beta}, modified={selection_fn.modified})"
            )
            selection_fn_name = f"{selection_fn.beta}${selection_fn.modified}"

            graphics_dir = self._graphics_dir / fitness_fn.name / str(n) / selection_fn_name / str(epoch)
            graphics_dir.mkdir(parents=True, exist_ok=True)

            algo = genetic_algorithm.GeneticAlgorithm(
                base_population=population,
                fitness_function=fn,
                scale_function=scale_functions.LinearRank(selection_fn.beta, n),
                selection_algo=selection_algorithms.my_sus,
                optimal=fitness_fn.optimal,
                stats_mode=stats_mode,
                modified_selection_algo=selection_fn.modified,
                max_iteration=max_iteration,
                mutation_rate=fitness_fn.mutation_rate,
                early_stopping=fitness_fn.early_stopping,
                draw_step=None,
                draw_total_steps=False,
                graphics_dir=str(graphics_dir)
            )
            algo.fit()

            run_data[selection_fn_name] = algo.stats

        return run_data

    def evaluate_experiment(
            self, epochs: int, n: int, max_iteration: int, fitness_fn: evaluator_config.FitnessFunctionConfig
    ):
        fn = fitness_fn.handler(**fitness_fn.values)
        stats_mode = fitness_fn.stats_mode
        length = fitness_fn.length

        report_data = []

        with multiprocessing.Pool(self._cpu_count) as p:
            for result in p.imap(functools.partial(
                    self.run_epoch, n=n, max_iteration=max_iteration, fitness_fn=fitness_fn
            ), range(epochs)):
                report_data.append(result)

        report_meta = {
            "n": n,
            "epochs": epochs,
            "max_iteration": max_iteration,
            "selection_fns": {
                "beta": list({fn.beta for fn in self._config.selection_fns}),
                "modified": list({fn.modified for fn in self._config.selection_fns})
            },
            "length": length,
            "fitness_fn": fitness_fn.name,
            "fitness_fn_values": fitness_fn.values,
            "stats_mode": stats_mode,
            "data": report_data,
            "total_data": utils.aggregate_runs_data(report_data, stats_mode, fitness_fn.optimal, fn)
        }
        current_time = datetime.datetime.now().strftime("%d.%m.%yT%H:%M:%S")
        name = f"data${current_time}${n}${fitness_fn.name}${epochs}"
        self._write_report(name, report_meta)

        print(f"Report successfully generated: {name}.json")

    def evaluate(self):
        epochs = self._config.epochs
        max_iteration = self._config.max_iteration

        for n in self._config.n_vals:
            for fitness_fn in self._config.fitness_fns:
                self.evaluate_experiment(epochs, n, max_iteration, fitness_fn)


if __name__ == "__main__":
    Evaluator(evaluator_config.get_config([100])).evaluate()
