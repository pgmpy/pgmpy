from abc import ABCMeta, abstractmethod, abstractproperty


class BaseDistribution(object):
    """
    @abstractproperty
    def pdf(self):
        pass

    @abstractproperty
    def variables(self):
        pass

    @abstractmethod
    def assignment(self, *args, **kwargs):
        pass

    @abstractmethod
    def copy(self):
        pass
    @abstractmethod
    def discretize(self, method, *args, **kwargs):
        pass

    @abstractmethod
    def reduce(self, values, inplace=True):
        pass

    @abstractmethod
    def marginalize(self, variables, inplace=True):
        pass

    @abstractmethod
    def normalize(self, inplace=True):
         pass

    @abstractmethod
    def product(self, other, inplace=True):
        pass

    @abstractmethod
    def divide(self, other, inplace=True):
        pass
    """
    pass
