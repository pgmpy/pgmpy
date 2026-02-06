from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.example_models import list_models, load_model
from pgmpy.example_models._base import _BaseExampleModel
from pgmpy.models import (
    DiscreteBayesianNetwork,
    FunctionalBayesianNetwork,
    LinearGaussianBayesianNetwork,
)

ALL_MODELS = [
    "alarm",
    "arth150",
    "acid_1996",
]


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
