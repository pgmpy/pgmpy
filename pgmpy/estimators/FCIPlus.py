"""FCI+ algorithm for causal discovery."""

import collections

import networkx as nx
from tqdm import tqdm

from pgmpy.estimators import PC, StructureEstimator


class FCIPlus(StructureEstimator):
    """Implements the FCI+ algorithm for causal discovery.

    FCI+ is an efficient version of the Fast Causal Inference (FCI) algorithm
    that avoids the exponential complexity of searching Possible-D-SEP sets.
    It produces a Partial Ancestral Graph (PAG).

    Parameters
    ----------
    data: pandas.DataFrame object
        dataframe object where each column represents one variable.
    **kwargs: dict
        Additional arguments for the StructureEstimator.

    """

    def __init__(self, data=None, **kwargs):
        """Initialize the FCI+ estimator."""
        super().__init__(data=data, **kwargs)

    def _get_discriminating_path(
        self, u, b, c, skeleton, edge_marks, max_path_length=10
    ):
        """Find a discriminating path <u, ..., a, b, c> for b via BFS."""
        queue = collections.deque([[u]])
        visited = {u}

        while queue:
            path = queue.popleft()
            curr = path[-1]

            if curr == b:
                if len(path) >= 3:
                    return path
                continue

            if len(path) > max_path_length:
                continue

            for neighbor in skeleton.neighbors(curr):
                if neighbor in visited or neighbor == c:
                    continue

                # neighbor must be a parent of c
                is_parent = (
                    edge_marks.get((neighbor, c)) == ">"
                    and edge_marks.get((c, neighbor)) == "-"
                )

                # curr must be a collider on the path (path[-2] *-> curr <-* neighbor)
                is_collider = True
                if len(path) >= 2:
                    is_collider = (
                        edge_marks.get((path[-2], curr)) == ">"
                        and edge_marks.get((neighbor, curr)) == ">"
                    )

                if is_parent and is_collider:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def estimate(
        self,
        significance_level=0.05,
        test_name="pearsonr",
        show_progress=True,
        **kwargs,
    ):
        """Estimates a Partial Ancestral Graph (PAG) from the data.

        Parameters
        ----------
        significance_level: float (default: 0.05)
            The significance level (alpha) for conditional independence tests.
        test_name: str or function (default: "pearsonr")
            The name of the conditional independence test to use.
            Options: "pearsonr", "chi_square", "g_sq", "log_likelihood".
        show_progress: bool (default: True)
            If True, shows a progress bar for the orientation phase.
        **kwargs: dict
            Additional arguments passed to the build_skeleton method.

        Returns
        -------
        pgmpy.base.DAG: The estimated PAG represented as a NetworkX DiGraph.

        """
        # 1. Fast Adjacency Search (FAS)
        pc_estimator = PC(data=self.data)
        skeleton, sepsets = pc_estimator.build_skeleton(
            ci_test=test_name, significance_level=significance_level, **kwargs
        )

        nodes = list(skeleton.nodes())
        edge_marks = {}
        for u, v in skeleton.edges():
            edge_marks[(u, v)], edge_marks[(v, u)] = "o", "o"

        # 2. Orient Unshielded Colliders
        for b in nodes:
            nb = list(skeleton.neighbors(b))
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    a, c = nb[i], nb[j]
                    if not skeleton.has_edge(a, c):
                        ss = sepsets.get(frozenset([a, c]), tuple())
                        if b not in ss:
                            edge_marks[(a, b)], edge_marks[(c, b)] = ">", ">"

        # 3. Orientation Rules Engine (R1 - R4)
        pbar = tqdm(disable=not show_progress, desc="FCI+ Orienting")
        changed = True
        while changed:
            changed = False
            for b in nodes:
                nb = list(skeleton.neighbors(b))
                for a in nb:
                    for c in nb:
                        if a == c:
                            continue

                        # R1: A *-> B o-* C  => B -> C
                        if (
                            not skeleton.has_edge(a, c)
                            and edge_marks[(a, b)] == ">"
                            and edge_marks[(c, b)] == "o"
                        ):
                            edge_marks[(c, b)], edge_marks[(b, c)] = "-", ">"
                            changed = True

                        # R2: Transitivity (A -> B *-> C or A *-> B -> C) => A *-> C
                        if skeleton.has_edge(a, c) and edge_marks[(a, c)] == "o":
                            c1 = (
                                edge_marks[(a, b)] == ">"
                                and edge_marks[(b, a)] == "-"
                                and edge_marks[(b, c)] == ">"
                            )
                            c2 = (
                                edge_marks[(a, b)] == ">"
                                and edge_marks[(b, c)] == ">"
                                and edge_marks[(c, b)] == "-"
                            )
                            if c1 or c2:
                                edge_marks[(a, c)] = ">"
                                changed = True

                        # R3: Double Collider
                        if (
                            not skeleton.has_edge(a, c)
                            and edge_marks[(a, b)] == ">"
                            and edge_marks[(c, b)] == ">"
                        ):
                            for d in nb:
                                if (
                                    d != a
                                    and d != c
                                    and skeleton.has_edge(a, d)
                                    and skeleton.has_edge(c, d)
                                ):
                                    if (
                                        edge_marks[(a, d)] == "o"
                                        and edge_marks[(c, d)] == "o"
                                        and edge_marks[(d, b)] == "o"
                                    ):
                                        edge_marks[(d, b)] = ">"
                                        changed = True

            # R4: Discriminating Path
            for c in nodes:
                for b in skeleton.neighbors(c):
                    if edge_marks.get((c, b)) == ">":
                        for a in skeleton.neighbors(b):
                            if (
                                a != c
                                and edge_marks.get((a, b)) == ">"
                                and edge_marks.get((c, b)) == ">"
                            ):
                                for u in nodes:
                                    if u not in [a, b, c] and not skeleton.has_edge(
                                        u, c
                                    ):
                                        path = self._get_discriminating_path(
                                            u, b, c, skeleton, edge_marks
                                        )
                                        if path:
                                            ss = sepsets.get(frozenset([u, c]), tuple())
                                            if b in ss:
                                                (
                                                    edge_marks[(a, b)],
                                                    edge_marks[(b, a)],
                                                ) = ("-", ">")
                                                (
                                                    edge_marks[(b, c)],
                                                    edge_marks[(c, b)],
                                                ) = ("-", ">")
                                            else:
                                                (
                                                    edge_marks[(a, b)],
                                                    edge_marks[(b, a)],
                                                ) = (">", ">")
                                                (
                                                    edge_marks[(b, c)],
                                                    edge_marks[(c, b)],
                                                ) = (">", ">")
                                            changed = True
            if changed:
                pbar.update(1)

        pbar.close()

        pag = nx.DiGraph()
        for (u, v), mark_at_v in edge_marks.items():
            if mark_at_v == ">":
                pag.add_edge(u, v)
        return pag
