from pgmpy.estimators import PC


class FCI(StructureEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        """
        FCI algorithm.
        """
        super(FCI, self).__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        variant="orig",
        ci_test="chi_square",
        max_cond_vars=5,
        return_type="mag",
        significance_level=0.01,
        n_jobs=-1,
        show_progress=True,
        **kwargs,
    ):

        # Step 0: Do checks that the specified parameters are correct, else throw meaningful error.
        if variant not in ("orig", "stable", "parallel"):
            raise ValueError(
                f"variant must be one of: orig, stable, or parallel. Got: {variant}"
            )
        elif (not callable(ci_test)) and (
            ci_test not in ("chi_square", "independence_match", "pearsonr")
        ):
            raise ValueError(
                "ci_test must be a callable or one of: chi_square, pearsonr, independence_match"
            )

        if (ci_test == "independence_match") and (self.independencies is None):
            raise ValueError(
                "For using independence_match, independencies argument must be specified"
            )
        elif (ci_test in ("chi_square", "pearsonr")) and (self.data is None):
            raise ValueError(
                "For using Chi Square or Pearsonr, data arguement must be specified"
            )

        # Step 1: Build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            variant=variant,
            n_jobs=n_jobs,
            show_progress=show_progress,
            **kwargs,
        )

        # Step 2: Orient the edges and return
        return self.skeleton_to_pag(skeleton=skel, separating_sets=separating_sets)

    @staticmethod
    def _test_edge(skeleton, test_edge, edge_types, triple=True):
        if triple:
            u, v, w = test_edge

            if (
                (skeleton.edges[(u, v)][u] == edge_types[0])
                and (skeleton.edges[(u, v)][v] == edge_types[1])
                and (skeleton.edges[(v, w)][v] == edge_types[2])
                and (skeleton.edges[(v, w)][w] == edge_types[3])
            ):
                return True
            else:
                return False

        else:
            u, v = test_edge
            if (skeleton.edges[(u, v)][u] == edge_types[0]) and (
                skeleton.edges[(u, v)][v] == edge_types[1]
            ):
                return True
            else:
                return False

    @staticmethod
    def skeleton_to_pag(skeleton, separating_sets):
        """
        Orients the edges to return a pag.
        """
        # Step 0: Orient all the edge edge points as o.
        #         Arrow type map: {o: -1, *: 0, >/<: 1, - : None}
        for u, v in skeleton.edges():
            skeleton.edges[(u, v)][u] = -1
            skeleton.edges[(u, v)][v] = -1

        # Step 1: Step R0 from reference.
        for u, v in skeleton.edges():
            for v_neigh in skeleton.neighbors(v):
                if (
                    (v_neigh != u)
                    and (v_neigh not in skeleton.neighbors(u))
                    and (v not in separating_sets[frozenset((u, v_neigh))])
                ):
                    skeleton.edges[(u, v)][u] = 0
                    skeleton.edges[(u, v)][v] = 1

                    skeleton.edges[(v, v_neigh)][v] = 1
                    skeleton.edges[(v, v_neigh)][v_neigh] = 0

        # Step 2: Steps R1, R2, R3, R4 from reference.
        mod = 1
        while mod:
            mod = 0

            for u, v in skeleton.edges():
                for w in skeleton.neighbors(v):
                    # Step 2.1: Step R1
                    if (
                        (w != u)
                        and (w not in skeleton.neighbors(u))
                        and self._test_edge(skeleton, (u, v, w), (0, 1, -1, 0))
                    ):

                        skeleton.edges[(v, w)][v] = None
                        skeleton.edges[(v, w)][w] = 1
                        mod = 1

                    # Step 2.2: Step R2
                    if (
                        (w != u)
                        and (w in skeleton.neighbors(u))
                        and self._test_edge(skeleton, (u, w), (0, -1), triple=False)
                        and self._test_edge(skeleton, (u, v, w), (None, 1, 0, 1))
                        and self._test_edge(skeleton, (u, v, w), (0, 1, None, 1))
                    ):
                        skeleton.edges[(u, w)][w] = 1
                        mod = 1

                    # Step 2.3: Step R3
                    if (
                        (w != u)
                        and (w not in skeleton.neighbors(u))
                        and self._test_edge(skeleton, (u, v, w), (0, 1, 1, 0))
                    ):
                        for theta in skeleton.neighbors(u):
                            if w in skeleton.neighbors(theta):
                                if (v in skeleton.neighbors(theta)) and self._test_edge(
                                    skeleton, (theta, v), (0, -1), triple=False
                                ):
                                    skeleton.edges[(theta, v)][v] = 1
                                    mod = 1
                    # Step 2.4: Step R4
        return skeleton
