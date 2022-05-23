import abc
import typing


class BaseGeneratorException(Exception):
    """
    Base exception class for all population generators classes.
    """
    pass


class BaseGenerator:
    """
    Base interface for all population generators classes.
    """

    def __init__(
            self, *, n: int, length: int, optimal: str, generate_optimal: bool = False, **kwargs
    ):
        """
        Initialize population generator
        :param n: Number of individuals in population to generate
        :param length: Length of each individual in population
        :param optimal: Flag to generate optimal individual
        at first place in population
        """
        self._n: int = n
        self._length: int = length
        self._optimal: str = optimal
        self._generate_optimal: bool = generate_optimal

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value

    @property
    def length(self) -> int:
        return self._length

    @length.setter
    def length(self, value: int):
        self._length = value

    @property
    def optimal(self) -> str:
        return self._optimal

    @optimal.setter
    def optimal(self, value: str):
        self._optimal = value

    @abc.abstractmethod
    def generate_individual(self) -> str:
        """
        Generates individual.
        :return: Individual as string of bits (e.g. 10101011)
        """
        pass

    def generate_optimal_individual(self) -> str:
        """
        Generates optimal individual.
        :return: Individual as string of bits (e.g. 10101011)
        """
        return self._optimal

    def generate_population(self) -> typing.List[str]:
        """
        Generates new population.
        :return: Population (list of strings/individuals)
        """
        optimal = []
        n = self._n

        if self._generate_optimal:
            optimal.append(self.generate_optimal_individual())
            n = self._n - 1

        return optimal + [self.generate_individual() for _ in range(n)]
