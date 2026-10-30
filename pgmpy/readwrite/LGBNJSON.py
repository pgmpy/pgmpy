import json
import math

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite._base import BaseReader, BaseWriter


class LGBNJSONReader(BaseReader):
    """
    Reads a Linear Gaussian Bayesian Network from the bnlearn-compatible JSON format.

    The JSON object has the keys ``nodes``, ``arcs`` and ``cpds``. Each CPD stores the ``parents``, the
    ``coefficients`` (including ``(Intercept)``) and the ``variance`` of a node.

    Parameters
    ----------
    path : str or os.PathLike
        Path of the JSON file.

    string : str
        The JSON data as a string.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.readwrite import LGBNJSONReader, LGBNJSONWriter
    >>> ecoli = load_model("bnlearn/ecoli70")
    >>> json_str = str(LGBNJSONWriter(ecoli))
    >>> model = LGBNJSONReader(string=json_str).read()
    >>> print(model)
    LinearGaussianBayesianNetwork with 46 nodes and 70 edges
    """

    format_name = "json"
    file_extensions = ["json"]

    def __init__(self, path=None, string=None):
        super().__init__(path=path, string=string)
        if path is not None:
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = json.loads(string)

    def read(self) -> LinearGaussianBayesianNetwork:
        """
        Returns the Linear Gaussian Bayesian Network read from the file/string.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import LGBNJSONReader, LGBNJSONWriter
        >>> LGBNJSONWriter(load_model("bnlearn/ecoli70")).write("ecoli70.json")
        >>> model = LGBNJSONReader("ecoli70.json").read()
        """
        nodes = self.data.get("nodes")
        edges = self.data.get("arcs")
        cpds_data = self.data.get("cpds")

        model = LinearGaussianBayesianNetwork(edges)
        model.add_nodes_from(nodes)

        cpds = []
        for node, cpd_info in cpds_data.items():
            coefficients = cpd_info["coefficients"]
            var = cpd_info["variance"][0]
            parents = cpd_info["parents"]

            intercept = coefficients["(Intercept)"][0]
            parent_coeffs = [coefficients[parent][0] for parent in parents]

            cpd = LinearGaussianCPD(
                variable=node,
                beta=[intercept] + parent_coeffs,
                std=math.sqrt(var),
                evidence=parents,
            )
            cpds.append(cpd)

        model.add_cpds(*cpds)
        return model


class LGBNJSONWriter(BaseWriter):
    """
    Writes a Linear Gaussian Bayesian Network to the bnlearn-compatible JSON format.

    Parameters
    ----------
    model : LinearGaussianBayesianNetwork
        The model to write.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.readwrite import LGBNJSONWriter
    >>> writer = LGBNJSONWriter(load_model("bnlearn/ecoli70"))
    >>> writer.write("ecoli70.json")
    """

    format_name = "json"
    file_extensions = ["json"]
    supported_models = (LinearGaussianBayesianNetwork,)

    def __str__(self):
        """
        Return the JSON as string.
        """
        model_data = {
            "nodes": list(self.model.nodes()),
            "arcs": list(self.model.edges()),
            "cpds": {},
        }

        for cpd in self.model.get_cpds():
            coeffs_dict = {"(Intercept)": [float(cpd.beta[0])]}
            for idx, parent in enumerate(cpd.evidence):
                coeffs_dict[parent] = [float(cpd.beta[idx + 1])]

            cpd_data = {
                "coefficients": coeffs_dict,
                "variance": [float(cpd.std**2)],
                "parents": list(cpd.evidence),
            }
            model_data["cpds"][cpd.variable] = cpd_data

        return json.dumps(model_data, indent=4)
