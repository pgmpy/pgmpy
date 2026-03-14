from pgmpy.utils.filtering import apply_filter, split_filter_tags


def test_exact_match():
    assert apply_filter(10, 10) is True
    assert apply_filter("alarm", "alarm") is True
    assert apply_filter(True, True) is True
    assert apply_filter(10, 5) is False


def test_valid_comparator():
    assert apply_filter(15, ">10") is True
    assert apply_filter(15, ">=15") is True
    assert apply_filter(15, "==15") is True
    assert apply_filter(5, ">10") is False


def test_invalid_comparator_returns_false():
    assert apply_filter(15, ">>10") is False
    assert apply_filter(15, ">abc") is False


def test_split_exact_vs_comparator():
    skbase_filters, custom_filters = split_filter_tags(
        {
            "n_nodes": ">10",
            "is_discrete": True,
            "name": "bnlearn/alarm",
            "n_edges": ">=20",
        }
    )

    assert skbase_filters == {
        "is_discrete": True,
        "name": "bnlearn/alarm",
    }
    assert custom_filters == {
        "n_nodes": ">10",
        "n_edges": ">=20",
    }
