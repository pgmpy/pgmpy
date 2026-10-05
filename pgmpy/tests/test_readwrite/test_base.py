import pytest

from pgmpy.base import DAG
from pgmpy.example_models import load_model
from pgmpy.readwrite import FORMATS


@pytest.fixture(scope="module")
def models():
    return {"discrete": load_model("bnlearn/asia"), "json": load_model("bnlearn/ecoli70")}


@pytest.mark.parametrize("name", FORMATS)
def test_reader_writer_input_validation(name):
    reader, writer = FORMATS[name]
    with pytest.raises(ValueError, match="exactly one"):
        reader()
    with pytest.raises(ValueError, match="exactly one"):
        reader(path="model", string="model")
    with pytest.raises(TypeError, match=writer.__name__):
        writer(DAG([("A", "B")]))


@pytest.mark.parametrize("name", FORMATS)
def test_writer_str_is_repeatable_string(models, name):
    writer = FORMATS[name][1](models.get(name, models["discrete"]))
    text = str(writer)
    assert isinstance(text, str)
    assert str(writer) == text


@pytest.mark.parametrize("name", FORMATS)
def test_get_model_deprecated_alias(models, name):
    reader_class, writer_class = FORMATS[name]
    reader = reader_class(string=str(writer_class(models.get(name, models["discrete"]))))
    with pytest.warns(FutureWarning, match=rf"{reader_class.__name__}\.read"):
        assert set(reader.get_model().edges()) == set(reader.read().edges())
