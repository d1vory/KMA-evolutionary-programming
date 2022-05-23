import abc
import math
import random
import re

import models
from core import utils


class FitnessFunction(models.Function):
    def __init__(self, **kwargs):
        pass

    @property
    def name(self):
        return self._name()

    @abc.abstractmethod
    def _name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _f(self, arg):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, arg):
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, arg):
        raise NotImplementedError

    @abc.abstractmethod
    def get_x(self, arg):
        raise NotImplementedError

    def is_arg_real(self):
        return False


class FConst(FitnessFunction):
    def decode(self, arg):
        raise NotImplementedError

    def encode(self, arg):
        raise NotImplementedError

    def get_x(self, arg):
        raise NotImplementedError

    def _name(self):
        return "fconst"

    def _f(self, arg):
        return len(arg)


class FH(FitnessFunction):
    def decode(self, arg):
        raise NotImplementedError

    def encode(self, arg):
        raise NotImplementedError

    def get_x(self, arg):
        raise NotImplementedError

    def _name(self):
        return "fh"

    def _f(self, arg):
        return float(len(re.findall("0", arg)))


class FHD(FitnessFunction):
    def __init__(self, theta=0, **kwargs):
        super().__init__(**kwargs)
        self._theta = theta

    def decode(self, arg):
        raise NotImplementedError

    def encode(self, arg):
        raise NotImplementedError

    def get_x(self, arg):
        raise NotImplementedError

    def _name(self):
        return f"fhd(theta:{self._theta})"

    def _f(self, arg):
        k = float(len(re.findall("0", arg)))

        return (len(arg) - k) + k * self._theta


class FX(FitnessFunction):
    def __init__(self, mode: str, a: float, b: float, m: int, **kwargs):
        super().__init__(**kwargs)
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

    def _name(self):
        return f"f_{self._mode}, {self._a}<=x<={self._b}, m={self._m}"

    def is_arg_real(self):
        return True

    @staticmethod
    def _format_double_sign(arg, reverse_tuple=False):
        tuple_ = arg, -arg
        if reverse_tuple:
            return tuple_
        return random.choice(tuple_)

    @staticmethod
    def _f_x_squared(arg, reverse=False, reverse_tuple=False):
        if reverse:
            return FX._format_double_sign(arg ** 0.5, reverse_tuple=reverse_tuple)
        return arg ** 2

    # noinspection PyUnusedLocal
    @staticmethod
    def _f_x(arg, reverse=False, reverse_tuple=False):
        return arg

    @staticmethod
    def _f_x_fourthed(arg, reverse=False, reverse_tuple=False):
        if reverse:
            return FX._format_double_sign(arg ** 0.25, reverse_tuple=reverse_tuple)
        return arg ** 4

    @staticmethod
    def _f_2x_squared(arg, reverse=False, reverse_tuple=False):
        if reverse:
            return FX._format_double_sign((arg / 2) ** 0.5, reverse_tuple=reverse_tuple)
        return 2 * arg ** 2

    @staticmethod
    def _f_512_x_squared(arg, reverse=False, reverse_tuple=False):
        if reverse:
            return FX._format_double_sign((5.12 ** 2 - arg) ** 0.5, reverse_tuple=reverse_tuple)
        return 5.12 ** 2 - arg ** 2

    @staticmethod
    def _f_512_x_fourthed(arg, reverse=False, reverse_tuple=False):
        if reverse:
            return FX._format_double_sign((5.12 ** 4 - arg) ** 0.25, reverse_tuple=reverse_tuple)
        return 5.12 ** 4 - arg ** 4

    def get_x(self, arg):
        return self._handler(arg, reverse=True)

    def encode(self, arg):
        return utils.encode(arg, self._a, self._b, self._m)

    def decode(self, arg):
        return utils.decode(arg, self._a, self._b, self._m)

    def _f(self, arg):
        return self._handler(self.decode(arg))


class FECX(FitnessFunction):
    def __init__(self, c: float, a: float, b: float, m: int, **kwargs):
        super().__init__(**kwargs)
        self._c = c
        self._a = a
        self._b = b
        self._m = m

    def _name(self):
        return f"f_e^({self._c}*x), {self._a}<=x<={self._b}, m={self._m}"

    def encode(self, arg):
        return utils.encode(arg, self._a, self._b, self._m)

    def decode(self, arg):
        return utils.decode(arg, self._a, self._b, self._m)

    def is_arg_real(self):
        return True

    def get_x(self, arg):
        raise NotImplementedError

    def _f(self, arg):
        return math.exp(self._c * self.decode(arg))
