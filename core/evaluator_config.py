import itertools
from dataclasses import dataclass, field
from typing import List

from core import fitness_functions, utils
import generators
import models

WRITING_DIR_DEFAULT = '../reports'
N_DEFAULT_VALUES = [100, 500, 1000]
MAX_ITERATION_DEFAULT = 10_000_000
EPOCHS_DEFAULT = 10
BETA_DEFAULT_VALUES = [1.2, 1.6, 2.0]
MODIFIED_DEFAULT_VALUES = [True, False]


@dataclass
class SelectionFunctionConfig:
    beta: float
    modified: bool


@dataclass
class FitnessFunctionConfig:
    name: str
    generator: type(generators.BaseGenerator)
    stats_mode: str
    length: int
    handler: type(models.Function)
    optimal: str
    values: dict = field(default_factory=dict)
    mutation_rate: float = None
    early_stopping: int = None


@dataclass
class EvaluatorConfig:
    epochs: int
    n_vals: List[int]
    max_iteration: int
    selection_fns: List[SelectionFunctionConfig]
    fitness_fns: List[FitnessFunctionConfig]
    writing_dir: str


EARLY_STOPPING = 10
MUTATION_COEFF = 10
FITNESS_FN_TABLE = {
    "fconst": FitnessFunctionConfig(
        "fconst", generators.ConstGenerator, "noise", 100, fitness_functions.FConst, "1" * 100, {}
    ),
    "fh": FitnessFunctionConfig("fh", generators.NormalGenerator, "full", 100, fitness_functions.FH, "0" * 100, {}),
    # "fhd": FitnessFunctionConfig("fhd", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100, {}),
    "fhd(theta=10)": FitnessFunctionConfig(
        "fhd(theta=10)", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100, {"theta": 10}
    ),
    "fhd(theta=50)": FitnessFunctionConfig(
        "fhd(theta=50)", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100, {"theta": 50}
    ),
    "fhd(theta=150)": FitnessFunctionConfig(
        "fhd(theta=150)", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100, {"theta": 150}
    ),
    "f=x^2": FitnessFunctionConfig(
        "f=x^2", generators.NormalGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
        {"mode": "x^2", "a": 0, "b": 10.23, "m": 10}
    ),
    "f=x": FitnessFunctionConfig(
        "f=x", generators.NormalGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
        {"mode": "x", "a": 0, "b": 10.23, "m": 10}
    ),
    "f=x^4": FitnessFunctionConfig(
        "f=x^4", generators.NormalGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
        {"mode": "x^4", "a": 0, "b": 10.23, "m": 10},
    ),
    "f=2x^2": FitnessFunctionConfig(
        "f=2x^2", generators.NormalGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
        {"mode": "2x^2", "a": 0, "b": 10.23, "m": 10}
    ),
    "f=(5.12)^2-x^2": FitnessFunctionConfig(
        "f=(5.12)^2-x^2", generators.NormalGenerator, "full", 10, fitness_functions.FX,
        utils.encode(0, -5.11, 5.12, 10),
        {"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10}
    ),
    "f=(5.12)^4-x^4": FitnessFunctionConfig(
        "f=(5.12)^4-x^4", generators.NormalGenerator, "full", 10, fitness_functions.FX,
        utils.encode(0, -5.11, 5.12, 10),
        {"mode": "(5.12)^4-x^4", "a": -5.11, "b": 5.12, "m": 10}
    ),
    "f=e^(0.25*x)": FitnessFunctionConfig(
        "f=e^(0.25*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX,
        utils.encode(10.23, 0, 10.23, 10),
        {"c": 0.25, "a": 0, "b": 10.23, "m": 10}
    ),
    "f=e^(1*x)": FitnessFunctionConfig(
        "f=e^(1*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX, utils.encode(10.23, 0, 10.23, 10),
        {"c": 1.0, "a": 0, "b": 10.23, "m": 10}
    ),
    "f=e^(2*x)": FitnessFunctionConfig(
        "f=e^(2*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX, utils.encode(10.23, 0, 10.23, 10),
        {"c": 2.0, "a": 0, "b": 10.23, "m": 10}
    ),
    "fh | mutated": FitnessFunctionConfig(
        "fh | mutated", generators.NormalGenerator, "full", 100, fitness_functions.FH,
        "0" * 100, {}, 0.0000672757925523407 * MUTATION_COEFF, EARLY_STOPPING
    ),
    # "fhd | mutated": FitnessFunctionConfig(
    #     "fhd | mutated", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100,
    #     {}, 0.0000660220531651611 * MUTATION_COEFF, EARLY_STOPPING
    # ),
    "fhd(theta=10) | mutated": FitnessFunctionConfig(
        "fhd(theta=10) | mutated", generators.NormalGenerator, "full", 100, fitness_functions.FHD, "0" * 100,
        {"theta": 10}, 0.0000660220531651611 * MUTATION_COEFF, EARLY_STOPPING
    ),
    "f=x^2 | mutated": FitnessFunctionConfig(
        "f=x^2 | mutated", generators.NormalGenerator, "full", 10, fitness_functions.FX,
        utils.encode(10.23, 0, 10.23, 10),
        {"mode": "x^2", "a": 0, "b": 10.23, "m": 10}, 0.00107915462143049 * MUTATION_COEFF, EARLY_STOPPING
    ),
    "f=(5.12)^2-x^2 | mutated": FitnessFunctionConfig(
        "f=(5.12)^2-x^2 | mutated", generators.NormalGenerator, "full", 10, fitness_functions.FX,
        utils.encode(0, -5.11, 5.12, 10),
        {"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10}, 0.00107915462143049 * MUTATION_COEFF, EARLY_STOPPING
    ),
}


def get_fitness_fns_config(fns=None):
    fns = fns or FITNESS_FN_TABLE.keys()
    return [FITNESS_FN_TABLE[fn] for fn in fns]


def get_selection_fns_config(beta=None, modified=None):
    beta = beta or BETA_DEFAULT_VALUES
    modified = modified or MODIFIED_DEFAULT_VALUES

    return [
        SelectionFunctionConfig(b, m)
        for b, m in itertools.product(beta, modified)
    ]


def get_config(
        n_vals=None,
        selection_fns=None,
        fitness_fns=None,
        epochs=None,
        max_iteration=None,
        writing_dir=None,
):
    epochs = epochs or EPOCHS_DEFAULT
    max_iteration = max_iteration or MAX_ITERATION_DEFAULT
    n_vals = n_vals or N_DEFAULT_VALUES
    writing_dir = writing_dir or WRITING_DIR_DEFAULT

    if selection_fns:
        beta = selection_fns['beta']
        modified = selection_fns['modified']
    else:
        beta = None
        modified = None
    selection_fns = get_selection_fns_config(beta, modified)
    fitness_fns = get_fitness_fns_config(fitness_fns)

    return EvaluatorConfig(epochs, n_vals, max_iteration, selection_fns, fitness_fns, writing_dir)


if __name__ == "__main__":
    print(list(FITNESS_FN_TABLE.keys()))
