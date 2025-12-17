from __future__ import annotations

from typing import List

import pandas as pd

from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class Lead(_BaseDataset):
    """
    Lead dataset stored as a covariance matrix.

    File format (lead.cov.txt):
      - First line: sample size (e.g., 221)
      - Second line: variable names
      - Next lines: lower-triangular covariance values (including diagonal)
    """

    name = "lead"

    tags = {
        "n_variables": 7,
        "n_samples": 221,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/lead/"
    )

    data_url = base_url + "data/lead.cov.txt"

    ground_truth_url = None
    expert_knowledge_url = None

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        raw = cls._get_raw_data("data", cls.data_url)
        text = raw.decode("utf-8-sig", errors="ignore").strip().splitlines()

        if len(text) < 3:
            raise ValueError("lead.cov.txt looks too short to parse.")

        try:
            n_samples_in_file = int(text[0].strip())
        except Exception as e:
            raise ValueError(
                "First line of lead.cov.txt must be an integer sample size."
            ) from e

        if cls.tags.get("n_samples") and n_samples_in_file != cls.tags["n_samples"]:
            pass

        var_names = text[1].split()
        p = len(var_names)

        values: List[List[float]] = []
        for line in text[2:]:
            if not line.strip():
                continue
            row = [float(x) for x in line.split()]
            values.append(row)

        cov = [[0.0] * p for _ in range(p)]
        for i in range(p):
            if i >= len(values):
                raise ValueError("Not enough covariance rows in lead.cov.txt.")
            if len(values[i]) != i + 1:
                raise ValueError(
                    f"Row {i} in covariance data should have {i + 1} values, got {len(values[i])}."
                )
            for j in range(i + 1):
                cov[i][j] = values[i][j]
                cov[j][i] = values[i][j]  # symmetric

        return pd.DataFrame(cov, index=var_names, columns=var_names)
