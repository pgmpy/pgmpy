import pytest

from pgmpy.independencies import IndependenceAssertion, Independencies


@pytest.fixture
def assertion():
    return IndependenceAssertion()


@pytest.fixture
def eq_assertions():
    return {
        "i1": IndependenceAssertion("a", "b", "c"),
        "i2": IndependenceAssertion("a", "b"),
        "i3": IndependenceAssertion("a", ["b", "c", "d"]),
        "i4": IndependenceAssertion("a", ["b", "c", "d"], "e"),
        "i5": IndependenceAssertion("a", ["d", "c", "b"], "e"),
        "i6": IndependenceAssertion("a", ["d", "c"], ["e", "b"]),
        "i7": IndependenceAssertion("a", ["c", "d"], ["b", "e"]),
        "i8": IndependenceAssertion("a", ["f", "d"], ["b", "e"]),
        "i9": IndependenceAssertion("a", ["d", "k", "b"], "e"),
        "i10": IndependenceAssertion(["k", "b", "d"], "a", "e"),
    }


@pytest.fixture
def independencies():
    return {
        "ind3": Independencies(["a", ["b", "c", "d"], ["e", "f", "g"]], ["c", ["d", "e", "f"], ["g", "h"]]),
        "ind4": Independencies([["f", "d", "e"], "c", ["h", "g"]], [["b", "c", "d"], "a", ["f", "g", "e"]]),
        "ind5": Independencies(["a", ["b", "c", "d"], ["e", "f", "g"]], ["c", ["d", "e", "f"], "g"]),
    }


class TestIndependenceAssertion:
    def test_return_list_if_not_collection(self, assertion):
        assert assertion._return_list_if_not_collection("U") == ["U"]
        assert assertion._return_list_if_not_collection(["U", "V"]) == ["U", "V"]

    def test_get_assertion(self):
        assert IndependenceAssertion("U", "V", "Z").get_assertion() == (
            {"U"},
            {"V"},
            {"Z"},
        )
        assert IndependenceAssertion("U", "V").get_assertion() == (
            {"U"},
            {"V"},
            set(),
        )

    def test_init(self):
        a = IndependenceAssertion("U", "V", "Z")
        assert a.event1 == {"U"}
        assert a.event2 == {"V"}
        assert a.event3 == {"Z"}

        b = IndependenceAssertion(["U", "V"], ["Y", "Z"], ["A", "B"])
        assert b.event1 == {"U", "V"}
        assert b.event2 == {"Y", "Z"}
        assert b.event3 == {"A", "B"}

    def test_init_exceptions(self):
        with pytest.raises(ValueError):
            IndependenceAssertion(event2=["U"], event3="V")
        with pytest.raises(ValueError):
            IndependenceAssertion(event2=["U"])
        with pytest.raises(ValueError):
            IndependenceAssertion(event3=["Z"])
        with pytest.raises(ValueError):
            IndependenceAssertion(event1=["U"])
        with pytest.raises(ValueError):
            IndependenceAssertion(event1=["U"], event3=["Z"])


class TestIndependenciesAssertionEq:
    def test_eq1(self, eq_assertions):
        i1, i2, i4, i6 = (
            eq_assertions["i1"],
            eq_assertions["i2"],
            eq_assertions["i4"],
            eq_assertions["i6"],
        )
        assert not (i1 == "a")
        assert not (i2 == 1)
        assert not (i4 == [2, "a"])
        assert not (i6 == "c")

    def test_eq2(self, eq_assertions):
        i1, i2, i3, i4, i6 = (
            eq_assertions["i1"],
            eq_assertions["i2"],
            eq_assertions["i3"],
            eq_assertions["i4"],
            eq_assertions["i6"],
        )
        assert not (i1 == i2)
        assert not (i1 == i3)
        assert not (i2 == i4)
        assert not (i3 == i6)

    def test_eq3(self, eq_assertions):
        i4, i5, i6, i7, i8, i9, i10 = (
            eq_assertions["i4"],
            eq_assertions["i5"],
            eq_assertions["i6"],
            eq_assertions["i7"],
            eq_assertions["i8"],
            eq_assertions["i9"],
            eq_assertions["i10"],
        )
        assert i4 == i5
        assert i6 == i7
        assert not (i7 == i8)
        assert not (i4 == i9)
        assert not (i5 == i9)
        assert i10 == i9
        assert i10 != i8


class TestIndependencies:
    def test_init(self):
        ind1 = Independencies(["X", "Y", "Z"])
        assert ind1 == Independencies(["X", "Y", "Z"])
        ind2 = Independencies()
        assert ind2 == Independencies()

    def test_add_assertions(self):
        ind1 = Independencies(["X", "Y", "Z"])
        assert ind1 == Independencies(["X", "Y", "Z"])
        ind2 = Independencies(["A", "B", "C"], ["D", "E", "F"])
        assert ind2 == Independencies(["A", "B", "C"], ["D", "E", "F"])

    def test_get_assertions(self):
        ind1 = Independencies(["X", "Y", "Z"])
        assert ind1.independencies == ind1.get_assertions()
        ind2 = Independencies(["A", "B", "C"], ["D", "E", "F"])
        assert ind2.independencies == ind2.get_assertions()

    def test_get_all_variables(self, independencies):
        ind3, ind4, ind5 = (
            independencies["ind3"],
            independencies["ind4"],
            independencies["ind5"],
        )
        assert ind3.get_all_variables() == frozenset(("a", "b", "c", "d", "e", "f", "g", "h"))
        assert ind4.get_all_variables() == frozenset(("f", "d", "e", "c", "h", "g", "b", "c", "a"))
        assert ind5.get_all_variables() == frozenset(("a", "b", "c", "d", "e", "f", "g"))

    def test_closure(self):
        ind1 = Independencies(("A", ["B", "C"], "D"))
        assert ind1.closure() == Independencies(
            ("A", ["B", "C"], "D"),
            ("A", "B", ["C", "D"]),
            ("A", "C", ["B", "D"]),
            ("A", "B", "D"),
            ("A", "C", "D"),
        )
        ind2 = Independencies(("W", ["X", "Y", "Z"]))
        assert ind2.closure() == Independencies(
            ("W", "Y"),
            ("W", "Y", "X"),
            ("W", "Y", "Z"),
            ("W", "Y", ["X", "Z"]),
            ("W", ["Y", "X"]),
            ("W", "X", ["Y", "Z"]),
            ("W", ["X", "Z"], "Y"),
            ("W", "X"),
            ("W", ["X", "Z"]),
            ("W", ["Y", "Z"], "X"),
            ("W", ["Y", "X", "Z"]),
            ("W", "X", "Z"),
            ("W", ["Y", "Z"]),
            ("W", "Z", "X"),
            ("W", "Z"),
            ("W", ["Y", "X"], "Z"),
            ("W", "X", "Y"),
            ("W", "Z", ["Y", "X"]),
            ("W", "Z", "Y"),
        )
        ind3 = Independencies(
            ("c", "a", ["b", "e", "d"]),
            (["e", "c"], "b", ["a", "d"]),
            (["b", "d"], "e", "a"),
            ("e", ["b", "d"], "c"),
            ("e", ["b", "c"], "d"),
            (["e", "c"], "a", "b"),
        )
        assert len(ind3.closure().get_assertions()) == 78

    def test_entails(self):
        ind1 = Independencies([["A", "B"], ["C", "D"], "E"])
        ind2 = Independencies(["A", "C", "E"])
        assert ind1.entails(ind2)
        assert not ind2.entails(ind1)
        ind3 = Independencies(("W", ["X", "Y", "Z"]))
        assert ind3.entails(ind3.closure())
        assert ind3.closure().entails(ind3)

    def test_is_equivalent(self):
        ind1 = Independencies(["X", ["Y", "W"], "Z"])
        ind2 = Independencies(["X", "Y", "Z"], ["X", "W", "Z"])
        ind3 = Independencies(["X", "Y", "Z"], ["X", "W", "Z"], ["X", "Y", ["W", "Z"]])
        assert not ind1.is_equivalent(ind2)
        assert ind1.is_equivalent(ind3)

    def test_eq(self, independencies):
        ind3, ind4, ind5 = (
            independencies["ind3"],
            independencies["ind4"],
            independencies["ind5"],
        )
        assert ind3 == ind4
        assert not (ind3 != ind4)
        assert ind3 != ind5
        assert not (ind4 == ind5)
        assert not (Independencies() == Independencies(["A", "B", "C"]))
        assert not (Independencies(["A", "B", "C"]) == Independencies())
        assert Independencies() == Independencies()

    def test_reduce(self):
        ind1 = Independencies(["X", "Y", "Z"], ["X", "Y", "Z"])
        assert len(ind1.reduce().independencies) == 1

        ind2 = Independencies(["A", "B", "C"], ["D", "E", "F"])
        reduced = ind2.reduce().independencies
        assert len(reduced) == 2
        assert all(assertion in reduced for assertion in ind2.get_assertions())

        ind3 = Independencies(["W", ["X", "Y", "Z"]], ["W", "X", "Y"])
        reduced = ind3.reduce()
        assert len(reduced.independencies) == 1
        assert reduced.independencies[0] == IndependenceAssertion("W", ["X", "Y", "Z"])

        ind4 = Independencies(
            ["A", ["B", "C"], "D"],
            ["A", "B", "D"],
            ["A", "C", "D"],
            ["E", "F", "G"],
        )
        reduced = ind4.reduce()
        assert len(reduced.independencies) == 2
        assert IndependenceAssertion("A", ["B", "C"], "D") in reduced.independencies
        assert IndependenceAssertion("E", "F", "G") in reduced.independencies

        ind5 = Independencies(["X", "Y", "Z"], ["X", "Y", "Z"], ["A", "B", "C"])
        original_assertions = ind5.get_assertions()
        ind5.reduce(inplace=True)
        assert len(original_assertions) != len(ind5.get_assertions())
        assert len(ind5.get_assertions()) == 2

        ind6 = Independencies()
        assert len(ind6.reduce().independencies) == 0

        ind7 = Independencies(["X", "Y", "Z"])
        reduced = ind7.reduce().independencies
        assert len(reduced) == 1
        assert reduced[0] == IndependenceAssertion("X", "Y", "Z")
