import itertools
from dataclasses import dataclass, field
from typing import List

from core import fitness_functions, utils
import generators
import models

WRITING_DIR_DEFAULT = '../reports'
#N_DEFAULT_VALUES = [100, 1000]
N_DEFAULT_VALUES = [100]
#MAX_ITERATION_DEFAULT = 10_000_000
MAX_ITERATION_DEFAULT = 100_000
EPOCHS_DEFAULT = 10
#BETA_DEFAULT_VALUES = [1.2, 1.6, 2.0]
#BETA_DEFAULT_VALUES = [1.2]
#MODIFIED_DEFAULT_VALUES = [True, False]

# A_VALUES = [1, 1, 2, 2]
# B_VALUES = [1, -1, 1, -1]

AB_VALUES = [(1,1), (1,-1), (2,1), (2,-1)]
#AB_VALUES = [(1, 1)]

MUTATION_RATE=0.00001

@dataclass
class SelectionFunctionConfig:
    a: int
    b: int


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
    use_crossingover: bool = False



@dataclass
class EvaluatorConfig:
    epochs: int
    n_vals: List[int]
    max_iteration: int
    selection_fns: List[SelectionFunctionConfig]
    fitness_fns: List[FitnessFunctionConfig]
    writing_dir: str


EARLY_STOPPING = 10
MUTATION_COEFF = 1
FITNESS_FN_TABLE = {
    "fconst__no_mut__no_cross": FitnessFunctionConfig(
        name="fconst__no_mut__no_cross",
        generator=generators.ConstGenerator,
        stats_mode="noise",
        length=100,
        handler=fitness_functions.FConst,
        optimal="1" * 100,
        values={}
    ),
    "fconst__mut__no_cross": FitnessFunctionConfig(
        name="fconst__mut__no_cross",
        generator=generators.ConstGenerator,
        stats_mode="noise",
        length=100,
        handler=fitness_functions.FConst,
        optimal="1" * 100,
        values={},
        mutation_rate=MUTATION_RATE
    ),
    "fconst__no_mut__cross": FitnessFunctionConfig(
        name="fconst__no_mut__cross",
        generator=generators.ConstGenerator,
        stats_mode="noise",
        length=100,
        handler=fitness_functions.FConst,
        optimal="1" * 100,
        values={},
        use_crossingover=True
    ),
    "fconst__mut__cross": FitnessFunctionConfig(
        name="fconst__mut__cross",
        generator=generators.ConstGenerator,
        stats_mode="noise",
        length=100,
        handler=fitness_functions.FConst,
        optimal="1" * 100,
        values={},
        mutation_rate=MUTATION_RATE,
        use_crossingover=True
    ),
####################################################################################################################################
    "fhd(theta=100)__no_mut__no_cross": FitnessFunctionConfig(
        name="fhd(theta=100)__no_mut__no_cross",
        generator=generators.NormalGenerator,
        stats_mode="full",
        length=100,
        handler=fitness_functions.FHD,
        optimal="0" * 100,
        values={"theta": 100}
    ),
    "fhd(theta=100)__no_mut__cross": FitnessFunctionConfig(
        name="fhd(theta=100)__no_mut__cross",
        generator=generators.NormalGenerator,
        stats_mode="full",
        length=100,
        handler=fitness_functions.FHD,
        optimal="0" * 100,
        values={"theta": 100},
        use_crossingover=True
    ),

    "fhd(theta=100)__mut__no_cross": FitnessFunctionConfig(
        name="fhd(theta=100)__mut__no_cross",
        generator=generators.NormalGenerator,
        stats_mode="full",
        length=100,
        handler=fitness_functions.FHD,
        optimal="0" * 100,
        values={"theta": 100},
        early_stopping=EARLY_STOPPING,
        mutation_rate=MUTATION_RATE,
    ),
    "fhd(theta=100)__mut__cross": FitnessFunctionConfig(
        name="fhd(theta=100)__mut__cross",
        generator=generators.NormalGenerator,
        stats_mode="full",
        length=100,
        handler=fitness_functions.FHD,
        optimal="0" * 100,
        values={"theta": 100},
        mutation_rate=MUTATION_RATE,
        early_stopping=EARLY_STOPPING,
        use_crossingover=True
    ),
    #######
    "f=x^2__no_mut__no_cross": FitnessFunctionConfig(
        name="f=x^2__no_mut__no_cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(x=10.23, a=0, b=10.23, m=10),
        values={"mode": "x^2", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 104.6529},
    ),
    "f=x^2__no_mut__cross": FitnessFunctionConfig(
        name="f=x^2__no_mut__cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(x=10.23, a=0, b=10.23, m=10),
        values={"mode": "x^2", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 104.6529},
        use_crossingover=True
    ),
    "f=x^2__mut__no_cross": FitnessFunctionConfig(
        name="f=x^2__mut__no_cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(x=10.23, a=0, b=10.23, m=10),
        values={"mode": "x^2", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 104.6529},
        mutation_rate=MUTATION_RATE,
        early_stopping=EARLY_STOPPING
    ),
    "f=x^2__mut__cross": FitnessFunctionConfig(
        name="f=x^2__mut__cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(x=10.23, a=0, b=10.23, m=10),
        values={"mode": "x^2", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 104.6529},
        mutation_rate=MUTATION_RATE,
        early_stopping=EARLY_STOPPING,
        use_crossingover=True
    ),

    "f=(5.12)^2-x^2__no_mut__no_cross": FitnessFunctionConfig(
        name="f=(5.12)^2-x^2__no_mut__no_cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(0, -5.11, 5.12, 10),
        values={"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 26.2144}
    ),
    "f=(5.12)^2-x^2__no_mut__cross": FitnessFunctionConfig(
        name="f=(5.12)^2-x^2__no_mut__cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(0, -5.11, 5.12, 10),
        values={"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 26.2144},
        use_crossingover=True
    ),
    "f=(5.12)^2-x^2__mut__no_cross": FitnessFunctionConfig(
        name="f=(5.12)^2-x^2__mut__no_cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(0, -5.11, 5.12, 10),
        values={"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 26.2144},
        mutation_rate=MUTATION_RATE,
        early_stopping=EARLY_STOPPING
    ),
    "f=(5.12)^2-x^2__mut__cross": FitnessFunctionConfig(
        name="f=(5.12)^2-x^2__mut__cross",
        generator=generators.RealGenerator,
        stats_mode="full",
        length=10,
        handler=fitness_functions.FX,
        optimal=utils.encode(0, -5.11, 5.12, 10),
        values={"mode": "(5.12)^2-x^2", "a": -5.12, "b": 5.12, "m": 10, "low": 0, "high": 26.2144},
        mutation_rate=MUTATION_RATE,
        early_stopping=EARLY_STOPPING,
        use_crossingover=True
    ),
    # "f=x": FitnessFunctionConfig(
    #     "f=x", generators.RealGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
    #     {"mode": "x", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 10.23}
    # ),
    # "f=x^4": FitnessFunctionConfig(
    #     "f=x^4", generators.RealGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
    #     {"mode": "x^4", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 10952.22947841},
    # ),
    # "f=2x^2": FitnessFunctionConfig(
    #     "f=2x^2", generators.NormalGenerator, "full", 10, fitness_functions.FX, utils.encode(10.23, 0, 10.23, 10),
    #     {"mode": "2x^2", "a": 0, "b": 10.23, "m": 10}
    # ),
    # "f=(5.12)^2-x^2": FitnessFunctionConfig(
    #     "f=(5.12)^2-x^2", generators.RealGenerator, "full", 10, fitness_functions.FX,
    #     utils.encode(0, -5.11, 5.12, 10),
    #     {"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 26.2144}
    # ),
    # "f=(5.12)^4-x^4": FitnessFunctionConfig(
    #     "f=(5.12)^4-x^4", generators.RealGenerator, "full", 10, fitness_functions.FX,
    #     utils.encode(0, -5.11, 5.12, 10),
    #     {"mode": "(5.12)^4-x^4", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 687.19476736}
    # ),
    # "f=e^(0.25*x)": FitnessFunctionConfig(
    #     "f=e^(0.25*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX,
    #     utils.encode(10.23, 0, 10.23, 10),
    #     {"c": 0.25, "a": 0, "b": 10.23, "m": 10}
    # ),
    # "f=e^(1*x)": FitnessFunctionConfig(
    #     "f=e^(1*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX, utils.encode(10.23, 0, 10.23, 10),
    #     {"c": 1.0, "a": 0, "b": 10.23, "m": 10}
    # ),
    # "f=e^(2*x)": FitnessFunctionConfig(
    #     "f=e^(2*x)", generators.NormalGenerator, "full", 10, fitness_functions.FECX, utils.encode(10.23, 0, 10.23, 10),
    #     {"c": 2.0, "a": 0, "b": 10.23, "m": 10}
    # ),
    # "fh | mutated": FitnessFunctionConfig(
    #     "fh | mutated", generators.NormalGenerator, "full", 100, fitness_functions.FH,
    #     "0" * 100, {},
    #     0.000005620502213 * MUTATION_COEFF,  # n=1000
    #     # 0.0000672757925523407 * MUTATION_COEFF,  # n=100
    #     EARLY_STOPPING
    # ),
    # "fhd(theta=10) | mutated": FitnessFunctionConfig(
    #     "fhd(theta=10) | mutated",
    #     generators.NormalGenerator, "full", 100,
    #     fitness_functions.FHD, "0" * 100, {"theta": 10},
    #     0.000006059621816 * MUTATION_COEFF,  # n=1000
    #     # 0.0000660220531651611 * MUTATION_COEFF, # n=100
    #     EARLY_STOPPING
    # ),
    # "f=x^2___mutated": FitnessFunctionConfig(
    #     name="f=x^2___mutated",
    #     generator=generators.RealGenerator,
    #     stats_mode="full",
    #     length=10,
    #     handler=fitness_functions.FX,
    #     optimal=utils.encode(x=10.23, a=0, b=10.23, m=10),
    #     use_crossingover=True,
    #     values={"mode": "x^2", "a": 0, "b": 10.23, "m": 10, "low": 0, "high": 104.6529},
    #     mutation_rate=0.000148257805588131 * MUTATION_COEFF,  # n=1000
    #     # 0.00107915462143049 * MUTATION_COEFF,  # n=100
    #     early_stopping=EARLY_STOPPING
    # ),
    # "f=(5.12)^2-x^2 | mutated": FitnessFunctionConfig(
    #     "f=(5.12)^2-x^2 | mutated", generators.RealGenerator, "full", 10, fitness_functions.FX,
    #     utils.encode(0, -5.11, 5.12, 10),
    #     {"mode": "(5.12)^2-x^2", "a": -5.11, "b": 5.12, "m": 10, "low": 0, "high": 26.2144},
    #     0.000148257805588131 * MUTATION_COEFF,  # n=1000
    #     # 0.00107915462143049 * MUTATION_COEFF,  # n=100
    #     EARLY_STOPPING
    # ),
}


def get_fitness_fns_config(fns=None):
    fns = fns or FITNESS_FN_TABLE.keys()
    return [FITNESS_FN_TABLE[fn] for fn in fns]


def get_selection_fns_config(ab_values=None):
    ab_values = ab_values or AB_VALUES
    return [SelectionFunctionConfig(a, b) for a, b in ab_values]


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

    selection_fns = selection_fns or get_selection_fns_config()
    fitness_fns = get_fitness_fns_config(fitness_fns)

    return EvaluatorConfig(epochs, n_vals, max_iteration, selection_fns, fitness_fns, writing_dir)


if __name__ == "__main__":
    print(list(FITNESS_FN_TABLE.keys()))
