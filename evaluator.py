import collections
import datetime
import itertools
import json
import pathlib

import numpy as np

import fitness_functions
import generators
import genetic_algorithm
import scale_functions
import selection_algorithms


class Evaluator:
    def __init__(self, config: dict, writing_dir: str = 'reports'):
        self._config: dict = config
        self._writing_dir: pathlib.Path = pathlib.Path(f"./{writing_dir}")

        self._writing_dir.mkdir(parents=True, exist_ok=True)

    def _evaluate_run(self, run_data: list, stats_mode: str) -> dict:
        total_data = collections.defaultdict(dict)
        result = {}

        if stats_mode == "noise":
            for epoch_data in run_data:
                for selection_fn in epoch_data:
                    if selection_fn not in total_data:
                        total_data[selection_fn] = collections.defaultdict(list)

                    total_data[selection_fn]["Suc"].append(epoch_data[selection_fn]["NI"] != -1)
                    total_data[selection_fn]["Num0"].append(epoch_data[selection_fn]["ConvTo"] == 0)
                    total_data[selection_fn]["Num1"].append(epoch_data[selection_fn]["ConvTo"] == 1)
                    total_data[selection_fn]["NI"].append(epoch_data[selection_fn]["NI"])

            for key in total_data:
                data = total_data[key]
                total = len(data["Suc"])
                suc = sum(data["Suc"])
                num0 = sum(data["Num0"])
                num1 = sum(data["Num1"])

                result[key] = {
                    "Suc": suc / total,
                    "Num0": num0 / total,
                    "Num1": num1 / total,
                    "Min_NI": min(data["NI"]),
                    "Max_NI": max(data["NI"]),
                    "Avg_NI": np.mean(data["NI"]),
                }

        else:
            pass

        return result

    def _write_report(self, name, data):
        with open(self._writing_dir / f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def evaluate(self):
        epochs = self._config["epochs"]
        max_iteration = self._config["max_iteration"]
        beta_vals = self._config["selection_fns"]["beta"]
        modified_vals = self._config["selection_fns"]["modified"]

        for n in self._config["n_vals"]:
            for fitness_fn, fn_config in self._config["fitness_fns"].items():
                report_data = []

                fn = fn_config["handler"](**fn_config["values"])
                stats_mode = fn_config["stats_mode"]
                length = fn_config["length"]

                generator = fn_config["generator"](n=n, length=length, optimal=True)

                for epoch in range(epochs):
                    run_data = {}

                    population = generator.generate_population()

                    for beta, modified in itertools.product(beta_vals, modified_vals):
                        print(
                            f"Fitting GA: epoch={epoch}, n={n}, fitness={fitness_fn}, "
                            f"linear(beta={beta}, modified={modified})"
                        )

                        algo = genetic_algorithm.GeneticAlgorithm(
                            base_population=population,
                            fitness_function=fn,
                            scale_function=scale_functions.LinearRank(beta, n),
                            selection_algo=selection_algorithms.my_sus,
                            stats_mode=stats_mode,
                            modified_selection_algo=modified,
                            max_iteration=max_iteration,
                            draw_step=None,
                            draw_total_steps=False,
                        )
                        algo.fit()

                        run_data[f"{beta}${modified}"] = algo.stats

                    report_data.append(run_data)

                report_meta = {
                    "n": n,
                    "epochs": epochs,
                    "max_iteration": max_iteration,
                    "selection_fns": self._config["selection_fns"],
                    "length": length,
                    "fitness_fn": fitness_fn,
                    "fitness_fn_values": fn_config["values"],
                    "stats_mode": stats_mode,
                    "data": report_data,
                    "total_data": self._evaluate_run(report_data, stats_mode)
                }
                current_time = datetime.datetime.now().strftime("%d.%m.%yT%H:%M:%S")
                name = f"{n}${fitness_fn}${epochs}${current_time}"
                self._write_report(name, report_meta)

                print(f"Report successfully generated: {name}.json")


if __name__ == "__main__":
    config_ = {
        "epochs": 10,
        "n_vals": [100],
        "max_iteration": 10_000_000,
        "selection_fns": {
            "beta": [1.2, 1.6, 2.0],
            "modified": [True, False],
        },
        "fitness_fns": {
            "fconst": {
                "generator": generators.ConstGenerator,
                "stats_mode": "noise",
                "length": 100,
                "handler": fitness_functions.FConst,
                "values": {}
            },
            "fh": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 100,
                "handler": fitness_functions.FH,
                "values": {}
            },
            "fhd(theta=10)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 100,
                "handler": fitness_functions.FHD,
                "values": {"theta": 10}
            },
            "fhd(theta=50)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 100,
                "handler": fitness_functions.FHD,
                "values": {"theta": 50}
            },
            "fhd(theta=150)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 100,
                "handler": fitness_functions.FHD,
                "values": {"theta": 150}
            },
            "f=x^2": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "x^2", "a": 0, "b": 10.23, "m": 10},
            },
            "f=x": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "x", "a": 0, "b": 10.23, "m": 10},
            },
            "f=x^4": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "x^4", "a": 0, "b": 10.23, "m": 10},
            },
            "f=2x^2": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "2x^2", "a": 0, "b": 10.23, "m": 10},
            },
            "f=(5.12)^2-x^2": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10},
            },
            "f=(5.12)^4-x^4": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FX,
                "values": {"mode": "(5.12)^4-x^4", "a": -5.11, "b": 5.12, "m": 10},
            },
            "f=e^(0.25*x)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FECX,
                "values": {"c": 0.25, "a": 0, "b": 10.23, "m": 10},
            },
            "f=e^(1*x)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FECX,
                "values": {"c": 1.0, "a": 0, "b": 10.23, "m": 10},
            },
            "f=e^(2*x)": {
                "generator": generators.NormalGenerator,
                "stats_mode": "full",
                "length": 10,
                "handler": fitness_functions.FECX,
                "values": {"c": 2.0, "a": 0, "b": 10.23, "m": 10},
            },
        }
    }

    Evaluator(config_).evaluate()
