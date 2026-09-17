from pgmpy.datasets._base import _BaseDataset


class TwinsDataset(_BaseDataset):
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
        "is_ordinal": False,
    }

    base_url = "twins"

    data_url = "twins.txt"

    ground_truth_url = None
    expert_knowledge_url = None

    missing_values_marker = ""

    sep = "\t"

    categorical_variables = [
        "pldel",
        "birattnd",
        "brstate",
        "stoccfipb",
        "mager8",
        "ormoth",
        "mrace",
        "meduc6",
        "dmar",
        "mplbir",
        "mpre5",
        "adequacy",
        "orfath",
        "frace",
        "birmon",
        "gestat10",
        "csex",
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
        "cigar6",
        "drink5",
        "crace",
        "data_year",
        "nprevistq",
        "dfageq",
        "feduc6",
        "dlivord_min",
        "dtotord_min",
        "bord_0",
        "bord_1",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
        "mort_0",
        "mort_1",
    ]

    ordinal_variables = dict()
