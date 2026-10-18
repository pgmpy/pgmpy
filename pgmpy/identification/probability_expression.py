from collections import Counter


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

    def to_latex(self):
        r"""
        Return LaTeX for this atomic probability term.

        The conditioning bar is omitted when both ``do`` and ``cond`` are empty.
        ``do(·)`` is always rendered before passive conditioning.

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
