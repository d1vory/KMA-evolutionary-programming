import math
import re

import models
from core import utils


class FConst(models.Function):
    def _f(self, arg):
        return len(arg)


class FH(models.Function):
    @property
    def name(self):
        return "fh"

    def _f(self, arg):
        return float(len(re.findall("0", arg)))


class FHD(models.Function):
    def __init__(self, theta):
        self._theta = theta

    @property
    def name(self):
        return f"fhd(theta:{self._theta})"

    def _f(self, arg):
        k = float(len(re.findall("0", arg)))

        return (len(arg) - k) + k * self._theta


class FX(models.Function):
    def __init__(self, mode: str, a: float, b: float, m: int):
        handlers_table = {
            'x^2': FX._f_x_squared,
            'x': FX._f_x,
            'x^4': FX._f_x_fourthed,
            '2x^2': FX._f_2x_squared,
            '(5.12)^2-x^2': FX._f_512_x_squared,
            '(5.12)^4-x^4': FX._f_512_x_fourthed,
        }
        self._mode = mode
        self._handler = handlers_table[self._mode]

        self._a: float = a
        self._b: float = b
        self._m: int = m

    @property
    def name(self):
        return f"f_{self._mode}, {self._a}<=x<={self._b}, m={self._m}"

    @staticmethod
    def _f_x_squared(arg):
        return arg ** 2

    @staticmethod
    def _f_x(arg):
        return arg

    @staticmethod
    def _f_x_fourthed(arg):
        return arg ** 4

    @staticmethod
    def _f_2x_squared(arg):
        return 2 * arg ** 2

    @staticmethod
    def _f_512_x_squared(arg):
        return 5.12 ** 2 - arg ** 2

    @staticmethod
    def _f_512_x_fourthed(arg):
        return 5.12 ** 4 - arg ** 4

    @staticmethod
    def _f_e_c_x(arg):
        return arg

    def _f(self, arg):
        return self._handler(utils.decode(arg, self._a, self._b, self._m))


class FECX(models.Function):
    def __init__(self, c: float, a: float, b: float, m: int):
        self._c = c
        self._a = a
        self._b = b
        self._m = m

    @property
    def name(self):
        return f"f_e^({self._c}*x), {self._a}<=x<={self._b}, m={self._m}"

    def _f(self, arg):
        x = utils.decode(arg, self._a, self._b, self._m)
        return math.exp(self._c * x)
