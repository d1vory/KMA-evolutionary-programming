import re
import typing


def fh(value: typing.Union[str, typing.List[str]]) -> typing.Union[float, typing.List[float]]:
    def f(x: str) -> float:
        return float(len(re.findall("0", x)))

    if isinstance(value, str):
        return f(value)
    elif isinstance(value, list):
        return [f(x) for x in value]
    else:
        raise NotImplementedError
