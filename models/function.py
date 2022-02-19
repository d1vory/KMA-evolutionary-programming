import abc


class Function(abc.ABC):
    def __call__(self, args):
        if isinstance(args, list):
            return [self._f(x) for x in args]
        else:
            return self._f(args)

    @abc.abstractmethod
    def _f(self, arg):
        raise NotImplementedError
