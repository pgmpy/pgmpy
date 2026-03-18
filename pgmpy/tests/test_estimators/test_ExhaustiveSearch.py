import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import BIC, K2, BDeu, ExhaustiveSearch


@pytest.fixture
def setup_data():
    np.random.seed(42)
    rand_data = pd.DataFrame(
        np.random.randint(0, 5, size=(5000, 2)),
        columns=list("AB"),
        dtype="category",
    )
    rand_data["C"] = rand_data["B"]
    est_rand = ExhaustiveSearch(rand_data)
    est_rand_bdeu = ExhaustiveSearch(rand_data, scoring_method=BDeu(rand_data))
    est_rand_bic = ExhaustiveSearch(rand_data, scoring_method=BIC(rand_data))

    # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
    titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
    titanic_data2 = titanic_data[["Survived", "Sex", "Pclass"]].astype("category")
    est_titanic = ExhaustiveSearch(titanic_data2)

    data = {
        "rand_data": rand_data,
        "est_rand": est_rand,
        "est_rand_bdeu": est_rand_bdeu,
        "est_rand_bic": est_rand_bic,
        "titanic_data": titanic_data,
        "titanic_data2": titanic_data2,
        "est_titanic": est_titanic,
    }

    yield data


def test_all_dags(setup_data):
    data = setup_data
    assert len(list(data["est_rand"].all_dags(["A", "B", "C", "D"]))) == 543
    # self.assertEqual(len(list(self.est_rand.all_dags(nodes=range(5)))), 29281)  # takes ~30s

    abc_dags = set(map(tuple, [sorted(dag.edges()) for dag in data["est_rand"].all_dags()]))
    abc_dags_ref = {
        (("A", "B"), ("C", "A"), ("C", "B")),
        (("A", "C"), ("B", "C")),
        (("B", "A"), ("B", "C")),
        (("C", "B"),),
        (("A", "C"), ("B", "A")),
        (("B", "C"), ("C", "A")),
        (("A", "B"), ("B", "C")),
        (("A", "C"), ("B", "A"), ("B", "C")),
        (("A", "B"),),
        (("A", "B"), ("C", "A")),
        (("B", "A"), ("C", "A"), ("C", "B")),
        (("A", "C"), ("C", "B")),
        (("A", "B"), ("A", "C"), ("C", "B")),
        (("B", "A"), ("C", "B")),
        (("A", "B"), ("A", "C")),
        (("C", "A"), ("C", "B")),
        (("A", "B"), ("A", "C"), ("B", "C")),
        (("C", "A"),),
        (("B", "A"), ("B", "C"), ("C", "A")),
        (("B", "A"),),
        (("A", "B"), ("C", "B")),
        (),
        (("B", "A"), ("C", "A")),
        (("A", "C"),),
        (("B", "C"),),
    }
    assert abc_dags == abc_dags_ref


def test_estimate_rand(setup_data):
    data = setup_data
    est_k2 = ExhaustiveSearch(data["rand_data"], scoring_method=K2(data["rand_data"]))
    est = est_k2.estimate()
    assert set(est.nodes()) == {"A", "B", "C"}
    assert set(est.edges()) == {("B", "A"), ("B", "C"), ("C", "A")}

    est_bdeu = data["est_rand_bdeu"].estimate()
    assert set(est_bdeu.edges()) == {("B", "C")}

    est_bic = data["est_rand_bic"].estimate()
    assert set(est_bic.edges()) == {("B", "C")}


def test_estimate_titanic(setup_data):
    data = setup_data
    est_k2 = ExhaustiveSearch(data["titanic_data2"], scoring_method=K2(data["titanic_data2"]))
    e1 = est_k2.estimate()
    assert set(e1.edges()) == {
        ("Survived", "Pclass"),
        ("Sex", "Pclass"),
        ("Sex", "Survived"),
    }


def test_all_scores(setup_data):
    data = setup_data
    est_k2 = ExhaustiveSearch(data["titanic_data2"], scoring_method=K2(data["titanic_data2"]))
    scores = est_k2.all_scores()
    scores_ref = [
        (-2072.9132364404695, []),
        (-2069.071694164769, [("Pclass", "Sex")]),
        (-2069.0144197068785, [("Sex", "Pclass")]),
        (-2025.869489762676, [("Survived", "Pclass")]),
        (-2025.8559302273054, [("Pclass", "Survived")]),
        (-2022.0279474869753, [("Pclass", "Sex"), ("Survived", "Pclass")]),
        (-2022.0143879516047, [("Pclass", "Sex"), ("Pclass", "Survived")]),
        (-2021.9571134937144, [("Pclass", "Survived"), ("Sex", "Pclass")]),
        (-2017.5258065853768, [("Sex", "Pclass"), ("Survived", "Pclass")]),
        (-1941.3075053892837, [("Survived", "Sex")]),
        (-1941.2720031713893, [("Sex", "Survived")]),
        (-1937.4304608956886, [("Pclass", "Sex"), ("Sex", "Survived")]),
        (-1937.4086886556927, [("Sex", "Pclass"), ("Survived", "Sex")]),
        (-1937.3731864377983, [("Sex", "Pclass"), ("Sex", "Survived")]),
        (-1934.1344850608882, [("Pclass", "Sex"), ("Survived", "Sex")]),
        (-1894.2637587114903, [("Survived", "Pclass"), ("Survived", "Sex")]),
        (-1894.2501991761198, [("Pclass", "Survived"), ("Survived", "Sex")]),
        (-1894.2282564935958, [("Sex", "Survived"), ("Survived", "Pclass")]),
        (-1891.0630673606006, [("Pclass", "Survived"), ("Sex", "Survived")]),
        (
            -1887.2215250849,
            [("Pclass", "Sex"), ("Pclass", "Survived"), ("Sex", "Survived")],
        ),
        (
            -1887.1642506270096,
            [("Pclass", "Survived"), ("Sex", "Pclass"), ("Sex", "Survived")],
        ),
        (
            -1887.0907383830947,
            [("Pclass", "Sex"), ("Survived", "Pclass"), ("Survived", "Sex")],
        ),
        (
            -1887.0771788477243,
            [("Pclass", "Sex"), ("Pclass", "Survived"), ("Survived", "Sex")],
        ),
        (
            -1885.9200755341915,
            [("Sex", "Pclass"), ("Survived", "Pclass"), ("Survived", "Sex")],
        ),
        (
            -1885.884573316297,
            [("Sex", "Pclass"), ("Sex", "Survived"), ("Survived", "Pclass")],
        ),
    ]

    assert [sorted(model.edges()) for score, model in scores] == [edges for score, edges in scores_ref]

    # use assertAlmostEqual point wise to avoid rounding issues
    for (score, _), (ref_score, _) in zip(scores, scores_ref):
        assert score == pytest.approx(ref_score)


def test_estimate_rand_bic_default(setup_data):
    """
    Tests the new default BIC scoring method.
    """
    data = setup_data
    est_bic = data["est_rand"].estimate()
    assert set(est_bic.nodes()) == {"A", "B", "C"}
    assert set(est_bic.edges()) == {("B", "C")}
    assert nx.is_directed_acyclic_graph(est_bic)


def test_estimate_titanic_bic_default(setup_data):
    """
    Tests the new default BIC scoring method on the titanic dataset.
    """
    data = setup_data
    e1_bic = data["est_titanic"].estimate()
    assert set(e1_bic.edges()) == {
        ("Sex", "Survived"),
        ("Pclass", "Survived"),
        ("Pclass", "Sex"),
    }
    assert nx.is_directed_acyclic_graph(e1_bic)
