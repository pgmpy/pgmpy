__author__ = 'navin'
class FactorNodesNotInClique(Exception):
    def __init__(self, *missing):
        self.missing = missing

    def __str__(self):
        return repr("The nodes in the factor are not in a single clique in the graph: " + str(self.missing))

class ObservationNotFound(Exception):
    def __init__(self, *missing):
        self.missing = missing

    def __str__(self):
        return repr("The observation provided was not given as a valid state for the node: " + str(self.missing))
