import pytest

from pgmpy.identification.probability_expression import (
    DivisionNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
    _TreeNode,
)


@pytest.fixture
def prob_y():
    return ProbabilityNode(frozenset({"Y"}))


@pytest.fixture
def prob_x():
    return ProbabilityNode(frozenset({"X"}))


@pytest.fixture
def prob_y_cond_x():
    return ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))


@pytest.fixture
def frontdoor_expr():
    """
    ProbabilityExpressionTree for the frontdoor formula:
        sum_M [ P(M|X) * sum_X [ P(Y|M,X) P(X) ] ]

    P(Y|do(X)) in a model with mediator M and a hidden confounder between X and Y.
    Reference: Pearl (2009), Causality, Example 3.3.2.
    """
    inner_product = ProductNode(
        [
            ProbabilityNode(frozenset({"Y"}), cond=frozenset({"M", "X"})),
            ProbabilityNode(frozenset({"X"})),
        ]
    )
    inner_sum = MarginalNode(inner_product, sumset=frozenset({"X"}))
    outer_product = ProductNode(
        [
            ProbabilityNode(frozenset({"M"}), cond=frozenset({"X"})),
            inner_sum,
        ]
    )
    return ProbabilityExpressionTree(root=MarginalNode(outer_product, sumset=frozenset({"M"})))


@pytest.fixture
def bowarc_expr():
    """
    ProbabilityExpressionTree for the bow-arc formula with a do-intervention:
        sum_Z [ P(Z | do(X)) * sum_X [ P(Y | do(X), Z) P(X) ] ]
    """
    inner_product = ProductNode(
        [
            ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"})),
            ProbabilityNode(frozenset({"X"})),
        ]
    )
    inner_sum = MarginalNode(inner_product, sumset=frozenset({"X"}))
    outer_product = ProductNode(
        [
            ProbabilityNode(frozenset({"Z"}), do=frozenset({"X"})),
            inner_sum,
        ]
    )
    return ProbabilityExpressionTree(root=MarginalNode(outer_product, sumset=frozenset({"Z"})))


@pytest.fixture
def conditional_id_expr():
    """
    A conditional-identification-style expression that nests a DivisionNode under a
    MarginalNode -- the ratio-of-products shape the IDC algorithm produces:

        sum_Z [ ( P(Y | X, Z) P(Z) ) / P(Z | X) ]

    Not claimed to be the identification result for any particular graph; it exists to
    exercise DivisionNode inside a realistic composition.
    """
    numerator = ProductNode(
        [
            ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X", "Z"})),
            ProbabilityNode(frozenset({"Z"})),
        ]
    )
    denominator = ProbabilityNode(frozenset({"Z"}), cond=frozenset({"X"}))
    ratio = DivisionNode(numerator, denominator)
    return ProbabilityExpressionTree(root=MarginalNode(ratio, sumset=frozenset({"Z"})))


class TestTreeNodeAbstract:
    def test_concrete_subclasses_are_tree_nodes(self, prob_y, prob_x):
        assert isinstance(prob_y, _TreeNode)
        assert isinstance(MarginalNode(prob_y, sumset=frozenset({"Y"})), _TreeNode)
        assert isinstance(ProductNode([prob_y, prob_x]), _TreeNode)
        assert isinstance(DivisionNode(prob_y, prob_x), _TreeNode)


class TestMarginalize:
    """Summing over an expression, and the exact simplifications each node type applies."""

    def test_marginalize_over_nothing_is_the_identity(self):
        node = ProbabilityNode(frozenset({"X", "Y"}))
        assert node.marginalize(frozenset()) is node

    def test_marginalize_shrinks_a_plain_joint(self):
        """sum_X P(X, Y) is P(Y), not a MarginalNode wrapping P(X, Y). This keeps an estimand a plain joint for as
        long as it genuinely is one, which is what lets the ID algorithm emit atomic conditionals instead of ratios."""
        assert ProbabilityNode(frozenset({"X", "Y"})).marginalize({"X"}) == ProbabilityNode(frozenset({"Y"}))

    def test_marginalize_does_not_shrink_a_conditional(self):
        node = ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))
        assert node.marginalize({"Y"}) == MarginalNode(node, sumset=frozenset({"Y"}))

    def test_marginalize_does_not_shrink_an_interventional_term(self):
        node = ProbabilityNode(frozenset({"Y", "Z"}), do=frozenset({"X"}))
        assert node.marginalize({"Z"}) == MarginalNode(node, sumset=frozenset({"Z"}))

    def test_marginalize_flattens_nested_sums(self):
        inner = MarginalNode(ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})), sumset=frozenset({"X"}))
        result = inner.marginalize({"Y"})
        assert result.sumset == frozenset({"X", "Y"})
        assert result.children[0] == inner.children[0]

    def test_marginalize_wraps_a_composite_expression(self, prob_y_cond_x, prob_x):
        product = ProductNode([prob_y_cond_x, prob_x])
        assert product.marginalize({"X"}) == MarginalNode(product, sumset=frozenset({"X"}))

    def test_marginalize_coerces_and_leaves_the_original_untouched(self, prob_y_cond_x):
        result = prob_y_cond_x.marginalize(["X"])
        assert result.sumset == frozenset({"X"})
        assert prob_y_cond_x.cond == frozenset({"X"})


class TestProbabilityNode:
    def test_init(self):
        p = ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"}))
        assert p.children == []
        assert p.variables == frozenset({"Y"})
        assert p.do == frozenset({"X"})
        assert p.cond == frozenset({"Z"})

        p_defaults = ProbabilityNode(frozenset({"Y"}))
        assert p_defaults.do == frozenset()
        assert p_defaults.cond == frozenset()

        p_coerced = ProbabilityNode(["Y"], do=["X"], cond=["Z"])
        assert isinstance(p_coerced.variables, frozenset)
        assert isinstance(p_coerced.do, frozenset)
        assert isinstance(p_coerced.cond, frozenset)

        with pytest.raises(TypeError):
            ProbabilityNode(["Y"], sumset=["W"])

        with pytest.raises(ValueError, match="at least one variable"):
            ProbabilityNode(frozenset())

    def test_to_latex(self):
        assert ProbabilityNode(frozenset({"Y"})).to_latex() == "P(Y)"
        assert ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})).to_latex() == r"P(Y \mid X)"
        assert ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"})).to_latex() == r"P(Y \mid do(X))"
        assert (
            ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"})).to_latex()
            == r"P(Y \mid do(X), Z)"
        )
        assert ProbabilityNode(frozenset({"Z", "Y"})).to_latex() == "P(Y, Z)"
        assert ProbabilityNode(frozenset({"Y"}), cond=frozenset({"Z", "X"})).to_latex() == r"P(Y \mid X, Z)"

        latex = MarginalNode(ProbabilityNode(frozenset({"Y", "Z"})), sumset=frozenset({"Z"})).to_latex()
        assert latex == r"\sum_{Z} P(Y, Z)"

        latex_multi = MarginalNode(ProbabilityNode(frozenset({"X", "Y", "Z"})), sumset=frozenset({"X", "Z"})).to_latex()
        assert latex_multi.startswith(r"\sum_{X, Z}")


class TestMarginalNode:
    def test_init(self, prob_y):
        m = MarginalNode(prob_y, sumset=frozenset({"Y"}))
        assert isinstance(m, MarginalNode)
        assert len(m.children) == 1
        assert m.children[0] is prob_y
        assert isinstance(m.sumset, frozenset)
        assert m.sumset == frozenset({"Y"})

        with pytest.raises(ValueError, match="sum over"):
            MarginalNode(prob_y, sumset=frozenset())

    def test_to_latex(self):
        inner = ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))
        assert MarginalNode(inner, sumset=frozenset({"X"})).to_latex() == r"\sum_{X} P(Y \mid X)"

        inner2 = ProbabilityNode(frozenset({"Y"}))
        latex = MarginalNode(inner2, sumset=frozenset({"Z", "X"})).to_latex()
        assert latex.startswith(r"\sum_{X, Z}")


class TestProductNode:
    def test_init(self, prob_y, prob_x, prob_y_cond_x):
        prod = ProductNode([prob_y, prob_x])
        assert isinstance(prod, ProductNode)
        assert len(prod.children) == 2
        assert prod.children[0] is prob_y
        assert prod.children[1] is prob_x

        assert len(ProductNode([prob_y, prob_x, prob_y_cond_x]).children) == 3

        with pytest.raises(ValueError, match="at least two factors"):
            ProductNode([prob_y])

        with pytest.raises(ValueError, match="at least two factors"):
            ProductNode([])

    def test_to_latex(self, prob_y, prob_x):
        p1 = ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))
        p2 = ProbabilityNode(frozenset({"X"}))
        assert ProductNode([p1, p2]).to_latex() == r"P(Y \mid X) P(X)"

        assert r"\left[" not in ProductNode([prob_y, prob_x]).to_latex()

        inner = MarginalNode(ProbabilityNode(frozenset({"Y"})), sumset=frozenset({"Y"}))
        latex = ProductNode([ProbabilityNode(frozenset({"X"})), inner]).to_latex()
        assert r"\left[" in latex
        assert r"\right]" in latex


class TestDivisionNode:
    def test_init(self, prob_y, prob_x):
        d = DivisionNode(prob_y, prob_x)
        assert isinstance(d, DivisionNode)
        assert len(d.children) == 2
        assert d.children[0] is prob_y
        assert d.children[1] is prob_x

    def test_to_latex(self):
        num = ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}))
        den = ProbabilityNode(frozenset({"Z"}), do=frozenset({"X"}))
        assert DivisionNode(num, den).to_latex() == r"\frac{P(Y \mid do(X))}{P(Z \mid do(X))}"

        num2 = ProbabilityNode(frozenset({"Y", "Z"}), do=frozenset({"X"}))
        den2 = ProbabilityNode(frozenset({"Z"}), do=frozenset({"X"}))
        latex = DivisionNode(num2, den2).to_latex()
        assert latex.startswith(r"\frac{")
        assert r"P(Y, Z \mid do(X))" in latex
        assert r"P(Z \mid do(X))" in latex


class TestProbabilityExpressionTree:
    def test_init(self, prob_y):
        expr = ProbabilityExpressionTree(root=prob_y)
        assert expr.root is prob_y

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root="not a node")

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root=42)

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root=None)

    def test_to_latex(self, prob_y_cond_x):
        expr = ProbabilityExpressionTree(root=prob_y_cond_x)
        assert expr.to_latex() == prob_y_cond_x.to_latex()
        assert repr(expr) == repr(prob_y_cond_x)

    def test_collect_node_types(self, frontdoor_expr, bowarc_expr, prob_y):
        single = ProbabilityExpressionTree(root=prob_y)
        assert single.collect_node_types() == [ProbabilityNode]

        assert frontdoor_expr.collect_node_types() == [
            MarginalNode,
            ProductNode,
            ProbabilityNode,
            MarginalNode,
            ProductNode,
            ProbabilityNode,
            ProbabilityNode,
        ]

        assert bowarc_expr.collect_node_types() == [
            MarginalNode,
            ProductNode,
            ProbabilityNode,
            MarginalNode,
            ProductNode,
            ProbabilityNode,
            ProbabilityNode,
        ]

    def test_find_leaves(self, frontdoor_expr, bowarc_expr, prob_y, prob_x):
        assert all(leaf.children == [] for leaf in frontdoor_expr.find_leaves())
        assert len(frontdoor_expr.find_leaves()) == 3
        assert len(bowarc_expr.find_leaves()) == 3

        single = ProbabilityExpressionTree(root=prob_y)
        assert single.find_leaves() == [prob_y]

        div_tree = ProbabilityExpressionTree(root=DivisionNode(prob_y, prob_x))
        leaves = div_tree.find_leaves()
        assert len(leaves) == 2
        assert leaves[0] is prob_y
        assert leaves[1] is prob_x


class TestFrontdoorFormula:
    """
    Tests for the frontdoor formula: sum_M [ P(M|X) * sum_X [ P(Y|M,X) P(X) ] ]

    P(Y|do(X)) in a model with mediator M and hidden confounder between X and Y.
    Reference: Pearl (2009), Causality, Example 3.3.2.
    """

    def test_tree_structure(self, frontdoor_expr):
        assert isinstance(frontdoor_expr.root, MarginalNode)
        assert frontdoor_expr.root.sumset == frozenset({"M"})

    def test_to_latex(self, frontdoor_expr):
        assert frontdoor_expr.to_latex() == (r"\sum_{M} P(M \mid X) \left[ \sum_{X} P(Y \mid M, X) P(X) \right]")


class TestBowArcFormula:
    """
    Tests for the bow-arc formula with do-intervention:
        sum_Z [ P(Z | do(X)) * sum_X [ P(Y | do(X), Z) P(X) ] ]

    P(Y|do(X)) for the bow-arc graph (X->Z->Y, X<->Z). Role-based methods cannot derive
    this; it requires the ID algorithm.
    Reference: Shpitser & Pearl (2006), AAAI-06.
    """

    def test_tree_structure(self, bowarc_expr):
        assert isinstance(bowarc_expr.root, MarginalNode)
        assert bowarc_expr.root.sumset == frozenset({"Z"})

    def test_to_latex(self, bowarc_expr):
        assert bowarc_expr.to_latex() == (r"\sum_{Z} P(Z \mid do(X)) \left[ \sum_{X} P(Y \mid do(X), Z) P(X) \right]")


class TestNodeEquality:
    def test_eq_and_hash(self, frontdoor_expr):
        # Same type + same attributes (modulo input coercion) => equal and hash-equal.
        a = ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"}))
        b = ProbabilityNode(["Y"], do=["X"], cond=["Z"])
        assert a == b and hash(a) == hash(b)
        assert len({a, b}) == 1

        # Differing attributes, differing type, and non-nodes are all unequal (no error).
        assert a != ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}))
        assert ProductNode([a, b]) != DivisionNode(a, b)  # identical children, different type
        assert a != "not a node"

        # Equality is structural and recursive: an independently rebuilt frontdoor tree matches.
        rebuilt = ProbabilityExpressionTree(
            root=MarginalNode(
                ProductNode(
                    [
                        ProbabilityNode(frozenset({"M"}), cond=frozenset({"X"})),
                        MarginalNode(
                            ProductNode(
                                [
                                    ProbabilityNode(frozenset({"Y"}), cond=frozenset({"M", "X"})),
                                    ProbabilityNode(frozenset({"X"})),
                                ]
                            ),
                            sumset=frozenset({"X"}),
                        ),
                    ]
                ),
                sumset=frozenset({"M"}),
            )
        )
        assert frontdoor_expr == rebuilt
        assert hash(frontdoor_expr) == hash(rebuilt)

        # Products are commutative: factor order changes neither equality nor hash...
        px, py = ProbabilityNode(frozenset({"X"})), ProbabilityNode(frozenset({"Y"}))
        assert ProductNode([px, py]) == ProductNode([py, px])
        assert hash(ProductNode([px, py])) == hash(ProductNode([py, px]))
        # ...but factor multiplicity still matters, and division stays order-sensitive.
        assert ProductNode([px, px, py]) != ProductNode([px, py])
        assert DivisionNode(px, py) != DivisionNode(py, px)


class TestConditionalIdFormula:
    def test_tree_and_latex(self, conditional_id_expr):
        # Structure: a marginal over a ratio whose numerator is a product.
        assert conditional_id_expr.collect_node_types() == [
            MarginalNode,
            DivisionNode,
            ProductNode,
            ProbabilityNode,
            ProbabilityNode,
            ProbabilityNode,
        ]
        assert len(conditional_id_expr.find_leaves()) == 3
        # DivisionNode renders the numerator product un-bracketed (the \frac already groups it).
        assert conditional_id_expr.to_latex() == r"\sum_{Z} \frac{P(Y \mid X, Z) P(Z)}{P(Z \mid X)}"
