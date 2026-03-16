import re

import numpy as np
import pytest
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

BNREP_DISCRETE_MODELS = [
    "bnrep/APSsystem",
    "bnrep/BOPfailure1",
    "bnrep/BOPfailure2",
    "bnrep/BOPfailure3",
    "bnrep/GDIpathway1",
    "bnrep/GDIpathway2",
    "bnrep/accidents",
    "bnrep/adhd",
    "bnrep/adversarialbehavior",
    "bnrep/aerialvehicles",
    "bnrep/agropastoral1",
    "bnrep/agropastoral2",
    "bnrep/agropastoral3",
    "bnrep/agropastoral4",
    "bnrep/agropastoral5",
    "bnrep/aircrash",
    "bnrep/airegulation1",
    "bnrep/airegulation2",
    "bnrep/airegulation3",
    "bnrep/algal1",
    "bnrep/algalactivity1",
    "bnrep/algalactivity2",
    "bnrep/algorithms3",
    "bnrep/algorithms4",
    "bnrep/arcticwaters",
    "bnrep/argument",
    "bnrep/asia",
    "bnrep/augmenting",
    "bnrep/bank",
    "bnrep/bankruptcy",
    "bnrep/beam1",
    "bnrep/beam2",
    "bnrep/beatles",
    "bnrep/blacksea",
    "bnrep/blockchain",
    "bnrep/bullet",
    "bnrep/burglar",
    "bnrep/cardiovascular",
    "bnrep/case",
    "bnrep/catchment",
    "bnrep/charleston",
    "bnrep/chds",
    "bnrep/cng",
    "bnrep/compaction",
    "bnrep/conasense",
    "bnrep/concrete1",
    "bnrep/concrete2",
    "bnrep/concrete3",
    "bnrep/concrete4",
    "bnrep/concrete5",
    "bnrep/concrete6",
    "bnrep/concrete7",
    "bnrep/consequenceCovid",
    "bnrep/constructionproductivity",
    "bnrep/coral1",
    "bnrep/coral2",
    "bnrep/coral3",
    "bnrep/coral4",
    "bnrep/coral5",
    "bnrep/corical",
    "bnrep/corrosion",
    "bnrep/corticosteroid",
    "bnrep/covid1",
    "bnrep/covid2",
    "bnrep/covid3",
    "bnrep/covidfear",
    "bnrep/covidrisk",
    "bnrep/covidtech",
    "bnrep/crimescene",
    "bnrep/criminal1",
    "bnrep/criminal2",
    "bnrep/criminal3",
    "bnrep/criminal4",
    "bnrep/crypto",
    "bnrep/curacao1",
    "bnrep/curacao2",
    "bnrep/curacao3",
    "bnrep/curacao4",
    "bnrep/curacao5",
    "bnrep/diabetes",
    "bnrep/dioxins",
    "bnrep/disputed1",
    "bnrep/disputed2",
    "bnrep/disputed3",
    "bnrep/disputed4",
    "bnrep/dragline",
    "bnrep/drainage",
    "bnrep/dustexplosion",
    "bnrep/earthquake",
    "bnrep/ecosystem",
    "bnrep/electricvehicle",
    "bnrep/electrolysis",
    "bnrep/emergency",
    "bnrep/engines",
    "bnrep/enrollment",
    "bnrep/estuary",
    "bnrep/ets",
    "bnrep/fingermarks1",
    "bnrep/fingermarks2",
    "bnrep/fire",
    "bnrep/firealarm",
    "bnrep/firerisk",
    "bnrep/flood",
    "bnrep/fluids1",
    "bnrep/fluids2",
    "bnrep/fluids3",
    "bnrep/foodallergy1",
    "bnrep/foodallergy2",
    "bnrep/foodallergy3",
    "bnrep/forest",
    "bnrep/fundraising",
    "bnrep/gasexplosion",
    "bnrep/gasifier",
    "bnrep/gonorrhoeae",
    "bnrep/greencredit",
    "bnrep/grounding",
    "bnrep/humanitarian",
    "bnrep/hydraulicsystem",
    "bnrep/income",
    "bnrep/intensification",
    "bnrep/intentionalattacks",
    "bnrep/inverters",
    "bnrep/knowledge",
    "bnrep/kosterhavet",
    "bnrep/lawschool",
    "bnrep/lidar",
    "bnrep/liquidity",
    "bnrep/lithium",
    "bnrep/macrophytes",
    "bnrep/medicaltest",
    "bnrep/megacities",
    "bnrep/metal",
    "bnrep/moodstate",
    "bnrep/mountaingoat",
    "bnrep/nanomaterials1",
    "bnrep/nanomaterials2",
    "bnrep/navigation",
    "bnrep/nuclearwaste",
    "bnrep/nuisancegrowth",
    "bnrep/oildepot",
    "bnrep/onlinerisk",
    "bnrep/orbital",
    "bnrep/oxygen",
    "bnrep/perioperative",
    "bnrep/permaBN",
    "bnrep/phdarticles",
    "bnrep/pilot",
    "bnrep/pneumonia",
    "bnrep/polymorphic",
    "bnrep/poultry",
    "bnrep/project",
    "bnrep/projectmanagement",
    "bnrep/propellant",
    "bnrep/rainstorm",
    "bnrep/rainwater",
    "bnrep/realestate1",
    "bnrep/realestate2",
    "bnrep/realestate3",
    "bnrep/redmeat",
    "bnrep/resilience",
    "bnrep/ricci",
    "bnrep/rockburst",
    "bnrep/rockquality",
    "bnrep/ropesegment",
    "bnrep/safespeeds",
    "bnrep/sallyclark",
    "bnrep/salmonella1",
    "bnrep/salmonella2",
    "bnrep/seismic",
    "bnrep/shipping",
    "bnrep/simulation",
    "bnrep/softwarelogs1",
    "bnrep/softwarelogs2",
    "bnrep/softwarelogs3",
    "bnrep/softwarelogs4",
    "bnrep/soil",
    "bnrep/soillead",
    "bnrep/soilliquefaction1",
    "bnrep/soilliquefaction2",
    "bnrep/soilliquefaction3",
    "bnrep/soilliquefaction4",
    "bnrep/student1",
    "bnrep/student2",
    "bnrep/tastingtea",
    "bnrep/tbm",
    "bnrep/theft1",
    "bnrep/theft2",
    "bnrep/titanic",
    "bnrep/trajectories",
    "bnrep/transport",
    "bnrep/tubercolosis",
    "bnrep/twinframework",
    "bnrep/urinary",
    "bnrep/vaccine",
    "bnrep/vessel1",
    "bnrep/vessel2",
    "bnrep/volleyball",
    "bnrep/waterlead",
    "bnrep/wheat",
    "bnrep/windturbine",
    "bnrep/witness",
    "bnrep/yangtze",
]

BNREP_CONTINUOUS_MODELS = [
    "bnrep/algal2",
    "bnrep/algorithms1",
    "bnrep/algorithms2",
    "bnrep/building",
    "bnrep/cachexia1",
    "bnrep/cachexia2",
    "bnrep/diagnosis",
    "bnrep/expenditure",
    "bnrep/foodsecurity",
    "bnrep/lexical",
    "bnrep/liquefaction",
    "bnrep/stocks",
    "bnrep/suffocation",
    "bnrep/turbine1",
    "bnrep/turbine2",
]

ALL_MODELS = (
    DISCRETE_MODELS
    + CONTINUOUS_MODELS
    + HYBRID_MODELS
    + DAGS
    + BNREP_DISCRETE_MODELS
    + BNREP_CONTINUOUS_MODELS
)


def test_list_models():
    assert set(list_models()) == set(ALL_MODELS)

    assert set(list_models(name="bnlearn/alarm")) == {"bnlearn/alarm"}

    assert "bnlearn/alarm" in set(list_models(is_parameterized=True))
    assert "bnlearn/arth150" in set(list_models(is_parameterized=True))

    assert "bnlearn/alarm" in set(list_models(is_discrete=True))
    assert "bnlearn/arth150" in set(list_models(is_continuous=True))


def test_invalid_tag():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_models(is_paraterized=True)  # typo

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_models(num_nodes=10)  # wrong key name entirely


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
    for model_name in np.random.choice(ALL_MODELS, 5):
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


def test_load_model_invalid_name():
    msg = "Model with name 'bnrep/soilead' not found. Please use list_models() to see available datasets."
    with pytest.raises(ValueError, match=re.escape(msg)):
        load_model("bnrep/soilead")
