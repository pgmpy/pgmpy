from pgmpy.datasets._base import _BaseDataset


class CysticFibrosis(_BaseDataset):
    _tags = {
        "name": "cystic_fibrosis",
        "n_variables": 44,
        "n_samples": 41,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/cystic-fibrosis/"

    data_url = base_url + "data/cystic-fibrosis-20180726-simplified.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "*"

    categorical_variables = [
        "crs_number",
        "sex",
        "cfrd",
        "mutation_508",
        "allergic_rhinitis",
        "ever_on_nasal_steroid",
        "virus",
        "rhinovirus",
        "sinus_exacerbation",
        "pulmonary_exacerbation",
        "on_nasal_cannula_oxygen",
        "hospital_days",
        "sputum_pa",
        "sputum_staph",
        "sinus_pa",
        "current_topabx",
        "current_top_vanco",
        "current_top_gent",
        "current_top_mupirocin",
        "current_top_ciprodex",
        "is_subject_on_systemic_abx",
        "Shannon",
        "Simpson",
        "Evenness",
        "Sheen_LB",
        "Mucoid_LB",
        "Rhamnolipid",
        "Hyper_pigment_binding_VBMM",
        "Twitching",
        "Swimming ",
        "Secreted_Protease_Milk",
        "Pa_Kill_Staph",
        "Pa_kill_Serratia",
        "Pa_Sheen_Serratia",
        "COG_P",
        "COG_V",
    ]
    ordinal_variables = dict()
