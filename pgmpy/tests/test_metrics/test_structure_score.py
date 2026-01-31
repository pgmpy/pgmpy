import pytest

from pgmpy.metrics import StructureScore
from pgmpy.utils import get_example_model


@pytest.fixture(scope="module")
def alarm_and_data():
    alarm = get_example_model("alarm")
    data = alarm.simulate(int(1e4), show_progress=False)

    alarm_no_cpd = alarm.copy()
    alarm_no_cpd.cpds = []

    return alarm, data, alarm_no_cpd


class TestStructureScore:
    def test_discrete_network(self, alarm_and_data):
        alarm, data, alarm_no_cpd = alarm_and_data

        for model in (alarm, alarm_no_cpd):
            for scoring_method in (None, "k2", "bdeu", "bds", "bic-d", "ll-d"):
                scorer = StructureScore(scoring_method=scoring_method)
                metric = scorer(X=data, causal_graph=model)
                assert isinstance(metric, float)

            for scoring_method in ("bdeu", "bds"):
                scorer = StructureScore(scoring_method=scoring_method)
                metric = scorer(X=data, causal_graph=model, equivalent_sample_size=10)
                assert isinstance(metric, float)

    def test_input(self, alarm_and_data):
        alarm, data, _ = alarm_and_data

        with pytest.raises(ValueError):
            scorer = StructureScore(scoring_method="random scoring")
            scorer(X=data, causal_graph=alarm)

        with pytest.raises(ValueError):
            scorer = StructureScore(scoring_method="k2")
            scorer(X=data, causal_graph="I am wrong model type")

        with pytest.raises(ValueError):
            scorer = StructureScore(scoring_method="k2")
            scorer(X=data.values, causal_graph=alarm)

        df_wrong_columns = data.copy()
        df_wrong_columns.columns = range(len(data.columns))
        with pytest.raises(ValueError):
            scorer = StructureScore(scoring_method="k2")
            scorer(X=df_wrong_columns, causal_graph=alarm)
