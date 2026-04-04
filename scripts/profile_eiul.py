import cProfile
import io
import pstats
import numpy as np
import pandas as pd

from pgmpy.causal_discovery.ExpertInLoop import ExpertInLoop


def main() -> None:
    np.random.seed(0)
    n = 2000
    p = 12
    cols = [f"X{i}" for i in range(p)]
    X = np.random.randn(n, p)

    # Introduce some dependencies to trigger edges
    X[:, 1] = X[:, 0] + np.random.randn(n) * 0.1
    X[:, 2] = X[:, 1] + np.random.randn(n) * 0.1
    X[:, 3] = X[:, 0] - X[:, 2] + np.random.randn(n) * 0.1

    df = pd.DataFrame(X, columns=cols)

    def orient(u: str, v: str) -> tuple[str, str]:
        return (u, v)

    est = ExpertInLoop(
        orientation_fn=orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
        max_iter=50,
    )

    pr = cProfile.Profile()
    pr.enable()
    est.fit(df)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    main()

