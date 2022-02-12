import typing


def linear_rank(beta: float, n: int) -> typing.Callable[[float], float]:
    def f(
            value: typing.Union[float, typing.List[float]]
    ) -> typing.Union[float, typing.List[float]]:
        def f_helper(score: float) -> float:
            return (2 - beta) / n + (2 * score * (beta - 1)) / (n * (n - 1))

        if isinstance(value, float):
            return f_helper(value)
        elif isinstance(value, list):
            return [f_helper(x) for x in value]
        else:
            raise NotImplementedError

    return f
