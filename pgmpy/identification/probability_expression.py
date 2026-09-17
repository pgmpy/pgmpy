from collections import Counter
from itertools import permutations


class _TreeNode:
    """
    Private base class for all expression tree nodes.

    Not part of the public API. Provides the shared interface that makes uniform tree traversal
    possible: every node exposes ``children`` (list of _TreeNode), ``to_latex()`` (str), and
    ``__repr__()`` (str).

    Concrete subclasses: ProbabilityNode, MarginalNode, ProductNode, DivisionNode.

    Do not instantiate directly.
    """

    @staticmethod
    def _vars_to_latex(var_set):
        """Sort and join variable names for deterministic LaTeX output."""
        return ", ".join(sorted((str(v) for v in var_set), key=str))

    def marginalize(self, sumset):
        r"""Return :math:`\sum_{sumset} self`.

        Returns a new expression; nodes are immutable. Summing over nothing is the identity, and a sum that is exactly
        a smaller node is returned as one rather than wrapped: a plain joint ``P(A)`` shrinks to ``P(A \ sumset)``, and
        nested sums collapse into one.

        Parameters
        ----------
        sumset: iterable of hashable
            The variables being summed out. Assumed to be a subset of the variables this expression ranges over.

        Returns
        -------
        expression: _TreeNode
            The marginalised expression.

        Examples
        --------
        >>> from pgmpy.identification.probability_expression import ProbabilityNode
        >>> ProbabilityNode(frozenset({"X"}), cond=frozenset({"Z"})).marginalize({"X"}).to_latex()
        '\\sum_{X} P(X \\mid Z)'
        """
        sumset = frozenset(sumset)
        if not sumset:
            return self
        return self._marginalize(sumset)

    def _marginalize(self, sumset):
        """Build the marginal of this node over a guaranteed non-empty ``sumset``.

        The general case wraps the node in a sum; subclasses override this to simplify instead.
        """
        return MarginalNode(self, sumset=sumset)

    def simplify(self):
        r"""Return the expression with the sums and factors that cancel taken out.

        Returns a new expression; nodes are immutable. A shape none of the rules matches is returned as it stands.
        Every rule holds for any positive distribution, so none of them needs the graph the expression came from:

        1. A term summed over its whole head sums to one and drops out, and :math:`\sum_{t} P(a, t \mid do(d), c)`
           is :math:`P(a \mid do(d), c)`. Within a product this needs ``t`` in no other factor, as only then does
           the sum distribute.
        2. The chain rule :math:`P(a \mid do(d), c) P(b \mid do(d), a, c) = P(a, b \mid do(d), c)`, applied when it
           moves a summed variable into a head for rule 1.
        3. :math:`P(a, b \mid do(d), c) / P(b \mid do(d), c)` is :math:`P(a \mid do(d), b, c)`, and a denominator
           repeated among the numerator's factors cancels.

        :footcite:t:`tikka_2017` shorten further by rewriting terms under conditional independences read off the
        graph, which the expression on its own does not carry.

        Returns
        -------
        expression: _TreeNode
            The simplified expression.

        Examples
        --------
        >>> from pgmpy.identification.probability_expression import MarginalNode, ProbabilityNode, ProductNode
        >>> chain = MarginalNode(
        ...     ProductNode([
        ...         ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})),
        ...         ProbabilityNode(frozenset({"Z"}), cond=frozenset({"X", "Y"})),
        ...     ]),
        ...     sumset=frozenset({"Y"}),
        ... )
        >>> chain.simplify().to_latex()
        'P(Z \\mid X)'

        References
        ----------
        - :footcite:t:`tikka_2017`
        """
        return self

    def _variables(self):
        """Return every variable name mentioned in this subtree, bound ones included.

        ``simplify`` uses this to spot a variable sitting where it would block a rewrite; over-reporting only costs
        a rewrite.
        """
        names = set()
        for child in self.children:
            names |= child._variables()
        return names

    def to_latex(self):
        """Return a LaTeX string representation of this node.

        Returns
        -------
        str
        """
        raise NotImplementedError

    def __repr__(self):
        """Return an unambiguous string representation of this node.

        Returns
        -------
        str
        """
        raise NotImplementedError


class ProbabilityNode(_TreeNode):
    r"""
    Leaf node of the expression tree.

    Represents an atomic conditional probability term:

    .. math::

        P(\text{variables} \mid do(\text{do\_vars}), \text{cond})

    Parameters
    ----------
    variables : frozenset of hashable
        The random variables in the probability term.
        Example: ``frozenset({"Y"})`` renders as ``P(Y | ...)``.

    do : frozenset of hashable, optional
        Variables being intervened on via the do-operator.
        Example: ``frozenset({"X"})`` renders as ``do(X)`` inside the conditioning bar.
        Default: ``frozenset()``.

    cond : frozenset of hashable, optional
        Variables being passively conditioned on.
        Example: ``frozenset({"Z"})`` renders after the conditioning bar.
        Default: ``frozenset()``.

    Raises
    ------
    ValueError
        If ``variables`` is empty.

    Attributes
    ----------
    children : list
        Always ``[]``. ``ProbabilityNode`` is always a leaf node.

    Examples
    --------
    >>> from pgmpy.identification.probability_expression import ProbabilityNode
    >>> ProbabilityNode(frozenset({"Y"})).to_latex()
    'P(Y)'
    >>> ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})).to_latex()
    'P(Y \\mid X)'
    >>> ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"})).to_latex()
    'P(Y \\mid do(X))'
    """

    def __init__(self, variables, do=frozenset(), cond=frozenset()):
        self.variables = frozenset(variables)
        if not self.variables:
            raise ValueError("A probability term must have at least one variable.")
        self.do = frozenset(do)
        self.cond = frozenset(cond)
        self.children = []

    def _marginalize(self, sumset):
        r"""Shrink a plain joint :math:`P(A)` to :math:`P(A \setminus sumset)`.

        Only a plain joint simplifies this way; summing out of a conditional or interventional term cannot be written
        as a single atomic term.
        """
        if self.do or self.cond:
            return super()._marginalize(sumset)
        return ProbabilityNode(self.variables - sumset)

    def _variables(self):
        """Return the variables of this term, on whichever side of the conditioning bar they sit."""
        return set(self.variables | self.do | self.cond)

    def to_latex(self):
        r"""
        Return LaTeX for this atomic probability term.

        The conditioning bar is omitted when both ``do`` and ``cond`` are empty.
        ``do(...)`` is always rendered before passive conditioning.

        Returns
        -------
        str
            LaTeX string of the form:

            .. math::

                P(\text{variables} \mid do(\ldots), \text{cond})

        Examples
        --------
        >>> from pgmpy.identification.probability_expression import ProbabilityNode
        >>> ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"})).to_latex()
        'P(Y \\mid do(X), Z)'
        """
        variables_str = self._vars_to_latex(self.variables)

        conditioning_parts = []
        if self.do:
            conditioning_parts.append(r"do(" + self._vars_to_latex(self.do) + r")")
        if self.cond:
            conditioning_parts.append(self._vars_to_latex(self.cond))

        if conditioning_parts:
            inner = variables_str + r" \mid " + r", ".join(conditioning_parts)
        else:
            inner = variables_str

        return r"P(" + inner + r")"

    def __repr__(self):
        parts = [f"variables={set(self.variables)!r}"]
        if self.do:
            parts.append(f"do={set(self.do)!r}")
        if self.cond:
            parts.append(f"cond={set(self.cond)!r}")
        return "ProbabilityNode(" + ", ".join(parts) + ")"

    def __eq__(self, other):
        if not isinstance(other, ProbabilityNode):
            return NotImplemented
        return (self.variables, self.do, self.cond) == (other.variables, other.do, other.cond)

    def __hash__(self):
        return hash((self.variables, self.do, self.cond))


class MarginalNode(_TreeNode):
    r"""
    Internal node representing a marginalisation over a sub-expression.

    .. math::

        \sum_{\text{sumset}} \text{child}

    Parameters
    ----------
    child : _TreeNode
        The expression being summed over. Stored as ``children[0]``.

    sumset : frozenset of hashable
        The variables being summed out.

    Raises
    ------
    ValueError
        If ``sumset`` is empty.

    Attributes
    ----------
    children : list of length 1
        ``children[0]`` is the child expression.

    Examples
    --------
    >>> from pgmpy.identification.probability_expression import MarginalNode, ProbabilityNode
    >>> m = MarginalNode(
    ...     ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})),
    ...     sumset=frozenset({"X"}),
    ... )
    >>> m.to_latex()
    '\\sum_{X} P(Y \\mid X)'
    """

    def __init__(self, child, sumset):
        self.sumset = frozenset(sumset)
        if not self.sumset:
            raise ValueError("MarginalNode requires at least one variable to sum over.")
        self.children = [child]

    def _marginalize(self, sumset):
        """Collapse nested sums into a single sum over the union of the two sumsets."""
        return MarginalNode(self.children[0], sumset=self.sumset | sumset)

    def simplify(self):
        """Apply rules 1 and 2 of ``_TreeNode.simplify`` to the factors under this sum.

        The two alternate to a fixed point, rule 2 running only to set rule 1 up, and each pass drops a variable
        from the sumset or a factor from the product, so the loop ends. The sumset binds its variables, so a name
        used above this node is a different variable and these factors are the whole scope the rules look at.

        Returns
        -------
        expression: _TreeNode
            The simplified expression.
        """
        child = self.children[0].simplify()
        sumset = self.sumset
        factors = list(child.children) if isinstance(child, ProductNode) else [child]

        changed = True
        while changed:
            changed = False

            # Rule 1: sum out a variable that this factor alone has in its head.
            for index, factor in enumerate(factors):
                if not isinstance(factor, ProbabilityNode):
                    continue
                others = (other for position, other in enumerate(factors) if position != index)
                elsewhere = set().union(*(other._variables() for other in others))
                summed = (sumset & factor.variables) - elsewhere
                # Summing away the head of the only factor leaves the constant one, which has no node to live in.
                if not summed or (summed == factor.variables and len(factors) == 1):
                    continue
                sumset -= summed
                if summed == factor.variables:
                    del factors[index]
                else:
                    factors[index] = ProbabilityNode(factor.variables - summed, do=factor.do, cond=factor.cond)
                changed = True
                break

            if changed:
                continue

            # Rule 2: merge a pair by the chain rule, so that rule 1 gets a head to work on next pass.
            for (first_index, first), (second_index, second) in permutations(enumerate(factors), 2):
                if not (isinstance(first, ProbabilityNode) and isinstance(second, ProbabilityNode)):
                    continue
                if first.do != second.do or second.cond != first.variables | first.cond:
                    continue
                if not first.variables.isdisjoint(second.variables) or sumset.isdisjoint(first.variables):
                    continue
                joined = ProbabilityNode(first.variables | second.variables, do=first.do, cond=first.cond)
                merged = {first_index, second_index}
                factors = [f for position, f in enumerate(factors) if position not in merged] + [joined]
                changed = True
                break

        product = factors[0] if len(factors) == 1 else ProductNode(factors)
        # As above: a plain joint summed over its own head is the constant one, so that sum is left as it stands.
        if isinstance(product, ProbabilityNode) and not (product.do or product.cond) and product.variables <= sumset:
            return MarginalNode(product, sumset=sumset)
        return product.marginalize(sumset)

    def _variables(self):
        """Return the variables of the summed expression, including the bound ones this node sums over."""
        return super()._variables() | set(self.sumset)

    def to_latex(self):
        r"""
        Return LaTeX for the marginalisation expression.

        .. math::

            \sum_{\text{sumset}} \text{child}

        Calls ``child.to_latex()`` to render the inner expression, then prepends the summation
        symbol and sumset variables.

        Returns
        -------
        str
            A LaTeX string of the form ``\sum_{X} <child_latex>``.
        """
        sumset_str = self._vars_to_latex(self.sumset)
        child_latex = self.children[0].to_latex()
        return r"\sum_{" + sumset_str + r"} " + child_latex

    def __repr__(self):
        return f"MarginalNode(child={self.children[0]!r}, sumset={set(self.sumset)!r})"

    def __eq__(self, other):
        if not isinstance(other, MarginalNode):
            return NotImplemented
        return self.sumset == other.sumset and self.children == other.children

    def __hash__(self):
        return hash((self.sumset, tuple(self.children)))


class ProductNode(_TreeNode):
    r"""
    Internal node representing a product of two or more probability sub-expressions.

    .. math::

        P(Y \mid X) \; P(X \mid do(Z)) \; P(Z)

    Parameters
    ----------
    factors : list of _TreeNode
        The probability expressions being multiplied together. Each element is a node such as
        a ``ProbabilityNode``, ``MarginalNode``, or ``DivisionNode``. Must have at least two
        elements.

    Raises
    ------
    ValueError
        If ``factors`` has fewer than two elements.

    Attributes
    ----------
    children : list of length >= 2
        The factor nodes in the order given.

    Examples
    --------
    >>> from pgmpy.identification.probability_expression import ProbabilityNode, ProductNode
    >>> ProductNode([
    ...     ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"})),
    ...     ProbabilityNode(frozenset({"X"})),
    ... ]).to_latex()
    'P(Y \\mid X) P(X)'
    """

    def __init__(self, factors):
        if len(factors) < 2:
            raise ValueError(f"Product requires at least two factors; got {len(factors)}.")
        self.children = list(factors)

    def simplify(self):
        """Simplify each factor, and flatten a product among them into this one.

        A nested product is only a grouping, multiplication being associative. Flattening puts every factor in one
        list, where the sum above can pair them up.

        Returns
        -------
        product: ProductNode
            The simplified product.
        """
        factors = []
        for child in self.children:
            child = child.simplify()
            factors.extend(child.children if isinstance(child, ProductNode) else [child])
        return ProductNode(factors)

    def to_latex(self):
        r"""
        Return LaTeX for the product of all children, space-separated.

        Composite children (``MarginalNode``, ``ProductNode``, ``DivisionNode``) are wrapped in
        ``\left[ ... \right]`` to make nested grouping unambiguous.

        Returns
        -------
        str
            LaTeX string of space-separated factors, with composite children wrapped in
            ``\left[ ... \right]``.
        """
        parts = []
        for child in self.children:
            child_latex = child.to_latex()
            if isinstance(child, (MarginalNode, ProductNode, DivisionNode)):
                parts.append(r"\left[ " + child_latex + r" \right]")
            else:
                parts.append(child_latex)
        return " ".join(parts)

    def __repr__(self):
        return "ProductNode([" + ", ".join(repr(c) for c in self.children) + "])"

    def __eq__(self, other):
        if not isinstance(other, ProductNode):
            return NotImplemented
        # A product is commutative, so factor order is irrelevant; compare as a multiset
        # (multiplicity still matters: P(X) P(X) != P(X) P(Y)).
        return Counter(self.children) == Counter(other.children)

    def __hash__(self):
        # Order-independent but multiplicity-aware, to stay consistent with __eq__.
        return hash(frozenset(Counter(self.children).items()))


class DivisionNode(_TreeNode):
    r"""
    Internal node representing a ratio of two probability sub-expressions.

    .. math::

        \frac{P(X_1 \mid do(Y_1), Z_1)}{P(X_2 \mid do(Y_2), Z_2)}

    Parameters
    ----------
    numerator : _TreeNode
        The numerator expression. Stored as ``children[0]``.

    denominator : _TreeNode
        The denominator expression. Stored as ``children[1]``.

    Attributes
    ----------
    children : list of length 2
        ``children[0]`` is the numerator; ``children[1]`` is the denominator.

    Examples
    --------
    >>> from pgmpy.identification.probability_expression import DivisionNode, ProbabilityNode
    >>> DivisionNode(
    ...     ProbabilityNode(frozenset({"Y"}), do=frozenset({"X"})),
    ...     ProbabilityNode(frozenset({"Z"}), do=frozenset({"X"})),
    ... ).to_latex()
    '\\frac{P(Y \\mid do(X))}{P(Z \\mid do(X))}'
    """

    def __init__(self, numerator, denominator):
        self.children = [numerator, denominator]

    def simplify(self):
        r"""Apply rule 3 of ``_TreeNode.simplify`` to this ratio.

        A ratio of two terms of one distribution is the conditional :math:`P(a \mid do(d), b, c)`, and a denominator
        repeated among the numerator's factors cancels against it. Both take the denominator to be positive, as
        identification does throughout.

        Returns
        -------
        expression: _TreeNode
            The simplified expression.
        """
        numerator = self.children[0].simplify()
        denominator = self.children[1].simplify()

        if (
            isinstance(numerator, ProbabilityNode)
            and isinstance(denominator, ProbabilityNode)
            and (numerator.do, numerator.cond) == (denominator.do, denominator.cond)
            and denominator.variables < numerator.variables
        ):
            return ProbabilityNode(
                numerator.variables - denominator.variables,
                do=numerator.do,
                cond=numerator.cond | denominator.variables,
            )

        if isinstance(numerator, ProductNode) and denominator in numerator.children:
            remaining = list(numerator.children)
            remaining.remove(denominator)
            return remaining[0] if len(remaining) == 1 else ProductNode(remaining)

        return DivisionNode(numerator, denominator)

    def to_latex(self):
        r"""
        Return LaTeX for the ratio expression.
        .. math::

            \frac{\text{numerator}}{\text{denominator}}

        Returns
        -------
        str
        """
        num_latex = self.children[0].to_latex()
        den_latex = self.children[1].to_latex()
        return r"\frac{" + num_latex + r"}{" + den_latex + r"}"

    def __repr__(self):
        return f"DivisionNode(numerator={self.children[0]!r}, denominator={self.children[1]!r})"

    def __eq__(self, other):
        if not isinstance(other, DivisionNode):
            return NotImplemented
        return self.children == other.children

    def __hash__(self):
        return hash(tuple(self.children))


class ProbabilityExpressionTree:
    r"""
    Container for expression trees produced by identification algorithms.

    ``ProbabilityExpressionTree`` is a lightweight container that holds the root of a symbolic
    probability expression tree. It is the public object returned by causal identification
    routines and is not itself a tree node.

    The expression tree is composed of ``ProbabilityNode``, ``MarginalNode``, ``ProductNode``,
    and ``DivisionNode`` nodes (all subclasses of ``_TreeNode``). Inspect or traverse the tree
    via the ``root`` attribute and the ``children`` lists on nodes.

    Parameters
    ----------
    root : _TreeNode
        Root node of the expression tree. Must be an instance of ``_TreeNode``.

    Raises
    ------
    TypeError
        If ``root`` is not an instance of ``_TreeNode``.

    Attributes
    ----------
    root : _TreeNode
        The expression tree root. Sub-nodes are accessible via ``root.children``.

    Examples
    --------
    Build the bow-arc formula
    :math:`\sum_Z P(Z \mid X) \left[ \sum_X P(Y \mid X, Z) \, P(X) \right]`:

    >>> from pgmpy.identification.probability_expression import (
    ...     MarginalNode, ProbabilityExpressionTree, ProbabilityNode, ProductNode
    ... )
    >>> inner = ProductNode([
    ...     ProbabilityNode(frozenset({"Y"}), cond=frozenset({"Z", "X"})),
    ...     ProbabilityNode(frozenset({"X"})),
    ... ])
    >>> expr = ProbabilityExpressionTree(
    ...     root=MarginalNode(
    ...         ProductNode([
    ...             ProbabilityNode(frozenset({"Z"}), cond=frozenset({"X"})),
    ...             MarginalNode(inner, sumset=frozenset({"X"})),
    ...         ]),
    ...         sumset=frozenset({"Z"}),
    ...     )
    ... )
    >>> expr.to_latex()
    '\\sum_{Z} P(Z \\mid X) \\left[ \\sum_{X} P(Y \\mid X, Z) P(X) \\right]'
    """

    def __init__(self, root):
        if not isinstance(root, _TreeNode):
            raise TypeError(
                f"root must be a _TreeNode instance (one of ProbabilityNode, "
                f"MarginalNode, ProductNode, DivisionNode); got {type(root).__name__}."
            )
        self.root = root

    def to_latex(self):
        r"""
        Return a LaTeX string for the full causal expression.

        Starts the recursive ``to_latex()`` traversal from the root node. Each node in the tree
        renders itself and delegates to its children, so the complete formula is assembled
        bottom-up in a single call.

        Returns
        -------
        str
            A LaTeX string for the full expression, ready for a math environment.

        Examples
        --------
        >>> from pgmpy.identification.probability_expression import (
        ...     ProbabilityExpressionTree, ProbabilityNode
        ... )
        >>> expr = ProbabilityExpressionTree(
        ...     root=ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))
        ... )
        >>> expr.to_latex()
        'P(Y \\mid X)'
        """
        return self.root.to_latex()

    def simplify(self):
        r"""
        Return a new tree holding an equivalent but shorter expression.

        Starts the recursive ``simplify()`` traversal from the root node. Each node applies the rewrites it has a
        rule for and delegates to its children; see ``_TreeNode.simplify`` for what those rules are.

        Returns
        -------
        expression: ProbabilityExpressionTree
            The simplified expression.

        Examples
        --------
        The napkin graph, whose denominator sums out a variable that only its own factor mentions:

        >>> from pgmpy.base import ADMG
        >>> from pgmpy.identification import ID
        >>> admg = ADMG(
        ...     edge_list=[
        ...         ("W", "R", "->"), ("R", "X", "->"), ("X", "Y", "->"),
        ...         ("W", "X", "<>"), ("W", "Y", "<>"),
        ...     ],
        ...     exposures={"X"},
        ...     outcomes={"Y"},
        ... )
        >>> ID().identify(admg).simplify().to_latex()
        '\\frac{\\sum_{W} P(W) P(X \\mid R, W) P(Y \\mid R, W, X)}{\\sum_{W} P(W) P(X \\mid R, W)}'
        """
        return ProbabilityExpressionTree(root=self.root.simplify())

    def collect_node_types(self, node=None):
        """
        Return a depth-first list of node classes for the subtree rooted at ``node``.

        Parameters
        ----------
        node : _TreeNode, optional
            The node to start traversal from. Defaults to ``self.root``.

        Returns
        -------
        list of type
            Node classes in depth-first (pre-order) traversal order.
            Example: ``[MarginalNode, ProductNode, ProbabilityNode, ...]``.
        """
        if node is None:
            node = self.root
        types = [type(node)]
        for child in node.children:
            types.extend(self.collect_node_types(child))
        return types

    def find_leaves(self, node=None):
        """
        Return all leaf ``ProbabilityNode`` instances in the subtree rooted at ``node``.

        Parameters
        ----------
        node : _TreeNode, optional
            The node to start traversal from. Defaults to ``self.root``.

        Returns
        -------
        list of ProbabilityNode
            All leaf nodes in depth-first traversal order.
        """
        if node is None:
            node = self.root
        if isinstance(node, ProbabilityNode):
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.find_leaves(child))
        return leaves

    def __eq__(self, other):
        if not isinstance(other, ProbabilityExpressionTree):
            return NotImplemented
        return self.root == other.root

    def __hash__(self):
        return hash(self.root)

    def __repr__(self):
        return repr(self.root)
