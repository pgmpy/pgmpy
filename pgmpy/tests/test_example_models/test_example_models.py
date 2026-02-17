from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.example_models import list_models, load_model
from pgmpy.example_models._base import _BaseExampleModel
from pgmpy.models import (
    DiscreteBayesianNetwork,
    FunctionalBayesianNetwork,
    LinearGaussianBayesianNetwork,
)

DISCRETE_MODELS = [
    "bnlearn/asia",
    "bnlearn/alarm",
    "bnlearn/cancer",
    "bnlearn/earthquake",
    "bnlearn/pathfinder",
    "bnlearn/pigs",
    "bnlearn/water",
    "bnlearn/munin",
    "bnlearn/munin1",
    "bnlearn/munin2",
    "bnlearn/munin3",
    "bnlearn/munin4",
    "bnlearn/andes",
    "bnlearn/diabetes",
    "bnlearn/link",
    "bnlearn/hailfinder",
    "bnlearn/hepar2",
    "bnlearn/win95pts",
    "bnlearn/insurance",
    "bnlearn/child",
    "bnlearn/barley",
    "bnlearn/sachs",
    "bnlearn/mildew",
    "bnlearn/survey",
]

CONTINUOUS_MODELS = [
    "bnlearn/arth150",
    "bnlearn/ecoli70",
    "bnlearn/magic_niab",
    "bnlearn/magic_irri",
]

HYBRID_MODELS = []

DAGS = [
    "dagitty/acid_1996",
    "dagitty/confounding",
    "dagitty/didelez_2010",
    "dagitty/kampen_2014",
    "dagitty/mediator",
    "dagitty/m_bias",
    "dagitty/paths",
    "dagitty/polzer_2012",
    "dagitty/schipf_2010",
    "dagitty/sebastiani_2005",
    "dagitty/shrier_2008",
    "dagitty/thoemmes_2013",
]

ALL_MODELS = DISCRETE_MODELS + CONTINUOUS_MODELS + HYBRID_MODELS + DAGS


def test_list_models():
    assert set(list_models()) == set(ALL_MODELS)

    assert set(list_models(name="bnlearn/alarm")) == {"bnlearn/alarm"}

    assert "bnlearn/alarm" in set(list_models(is_parameterized=True))
    assert "bnlearn/arth150" in set(list_models(is_parameterized=True))

    assert "bnlearn/alarm" in set(list_models(is_discrete=True))
    assert "bnlearn/arth150" in set(list_models(is_continuous=True))


def test_tags():
    for model_name in ALL_MODELS:
        tags = all_objects(
            object_types=_BaseExampleModel,
            package_name="pgmpy.example_models",
            filter_tags={"name": model_name},
            return_names=False,
        )[0]._tags
        assert isinstance(tags, dict)
        assert "name" in tags
        assert "n_nodes" in tags
        assert "n_edges" in tags
        assert "is_parameterized" in tags

        if tags["is_parameterized"]:
            assert "is_discrete" in tags
            assert "is_continuous" in tags
            assert "is_hybrid" in tags


def test_load_model():
    for model_name in ALL_MODELS:
        model = load_model(model_name)

        assert isinstance(
            model,
            (
                DAG,
                DiscreteBayesianNetwork,
                LinearGaussianBayesianNetwork,
                FunctionalBayesianNetwork,
            ),
        )

        model_tags = all_objects(
            object_types=_BaseExampleModel,
            package_name="pgmpy.example_models",
            filter_tags={"name": model_name},
            return_names=False,
        )[0]._tags

        assert model_tags["n_nodes"] == len(model.nodes())
        assert model_tags["n_edges"] == len(model.edges())
        if model_tags["is_parameterized"]:
            assert hasattr(model, "cpds")
            assert model_tags["is_discrete"] == isinstance(
                model, DiscreteBayesianNetwork
            )
            assert model_tags["is_continuous"] == isinstance(
                model, LinearGaussianBayesianNetwork
            )
            assert model_tags["is_hybrid"] == isinstance(
                model, FunctionalBayesianNetwork
            )
        else:
            assert isinstance(model, DAG)
