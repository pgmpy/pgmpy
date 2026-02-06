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
    "asia",
    "alarm",
    "cancer",
    "earthquake",
    "water",
    "munin",
    "munin1",
    "munin2",
    "munin3",
    "munin4",
    "andes",
    "diabetes",
    "link",
    "hailfinder",
    "hepar2",
    "win95pts",
    "insurance",
    "child",
    "barley",
    "sachs",
    "mildew",
    "survey",
]

CONTINUOUS_MODELS = [
    "arth150",
    "ecoli70",
    "magic_niab",
]

HYBRID_MODELS = []

DAGS = [
    "acid_1996",
    "m_bias",
    "confounding",
]

ALL_MODELS = DISCRETE_MODELS + CONTINUOUS_MODELS + HYBRID_MODELS + DAGS


def test_list_models():
    assert set(list_models()) == set(ALL_MODELS)

    assert set(list_models(name="alarm")) == {"alarm"}

    assert "alarm" in set(list_models(is_parameterized=True))
    assert "arth150" in set(list_models(is_parameterized=True))

    assert "alarm" in set(list_models(is_discrete=True))
    assert "arth150" in set(list_models(is_continuous=True))


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
