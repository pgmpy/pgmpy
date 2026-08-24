from pgmpy.datasets._base import BaseDataset


class TwinsDataset(BaseDataset):
    _tags = {
        "name": "twins",
        "n_variables": 56,
        "n_samples": 71345,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": True,
    }

    base_url = "twins"

    data_url = "twins.txt"

    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = [
        "pldel",
        "birattnd",
        "brstate",
        "stoccfipb",
        "mplbir",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
        "mrace",
        "frace",
        "crace",
        "ormoth",
        "orfath",
        "birmon",
        "data_year",
        "dmar",
        "csex",
        "bord_0",
        "bord_1",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "tobacco",
        "alcohol",
    ]

    ordinal_variables = {
        "mager8": [1, 2, 3, 4, 5, 6, 7, 8],
        "dfageq": [0, 1, 2, 3, 4, 5, 6, 7],
        "meduc6": [1, 2, 3, 4, 5],
        "feduc6": [1, 2, 3, 4, 5],
        "gestat10": list(range(1, 11)),
        "mpre5": [1, 2, 3, 4],
        "adequacy": [1, 2, 3],
        "nprevistq": [0, 1, 2, 3, 4],
        "cigar6": [0, 1, 2, 3, 4, 5],
        "drink5": [0, 1, 2, 3, 4],
        "dlivord_min": list(range(1, 18)),
        "dtotord_min": list(range(1, 22)),
    }
