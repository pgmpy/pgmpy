import pytest
from skbase.lookup import all_objects

from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import BaseCITest
from pgmpy.registry import OBJECT_TYPES, TAG_REGISTER, all_tags, check_tag_is_valid
from pgmpy.structure_score import BaseStructureScore

OBJECTS = [
    cls
    for base, package in [
        (BaseCausalDiscovery, "pgmpy.causal_discovery"),
        (BaseCITest, "pgmpy.ci_tests"),
        (BaseStructureScore, "pgmpy.structure_score"),
    ]
    for cls in all_objects(object_types=base, package_name=package, return_names=False)
]


def test_tag_register_is_well_formed():
    for tag_name, object_type, tag_type, description in TAG_REGISTER:
        assert isinstance(tag_name, str) and isinstance(description, str) and description
        assert object_type in OBJECT_TYPES
        assert tag_type in ("bool", "str") or (tag_type[0] in ("str", "list") and isinstance(tag_type[1], list))

    pairs = [(tag[0], tag[1]) for tag in TAG_REGISTER]
    assert len(pairs) == len(set(pairs))


@pytest.mark.parametrize("cls", OBJECTS, ids=lambda cls: f"{cls.__module__}.{cls.__name__}")
def test_object_tags_are_registered_and_valid(cls):
    tags = cls.get_class_tags()
    assert tags.keys() == {tag[0] for tag in all_tags(tags["object_type"])}
    assert tags["name"] == tags["name"].lower()
    for tag_name, tag_value in tags.items():
        check_tag_is_valid(tag_name, tag_value)


@pytest.mark.parametrize("object_type", OBJECT_TYPES)
def test_object_names_are_unique(object_type):
    names = [cls.get_class_tag("name") for cls in OBJECTS if cls.get_class_tag("object_type") == object_type]
    assert names and len(names) == len(set(names))


def test_check_tag_is_valid():
    check_tag_is_valid("assumption:linearity", False)
    check_tag_is_valid("default_for", None)
    with pytest.raises(KeyError):
        check_tag_is_valid("not_a_tag", True)
    with pytest.raises(ValueError, match="must be a bool"):
        check_tag_is_valid("assumption:linearity", None)
    with pytest.raises(ValueError, match="must be one of"):
        check_tag_is_valid("supported_datatype", "text")


def test_all_tags():
    assert {tag[1] for tag in all_tags()} == set(OBJECT_TYPES)
    assert {tag[1] for tag in all_tags(["ci_test", "structure_score"])} == {"ci_test", "structure_score"}
    df = all_tags("causal_discovery", as_dataframe=True)
    assert list(df.columns) == ["name", "object_type", "type", "description"]
    assert "capability:multivariate" in df["name"].values
