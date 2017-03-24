from abc import ABCMeta, abstractmethod, abstractproperty


class BaseDistribution(object):
    #__metaclass__ = ABCMeta
    # TODO: Fix this
    # @abstractproperty
    # def _pdf(self):
    #     pass
    #
    # @abstractproperty
    # def variables(self):
    #     pass

    @abstractmethod
    def get_pdf(self):
        pass

    @abstractmethod
    def get_scope(self):
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

