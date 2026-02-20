import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset


class LUCAS(_BaseDataset):
    """
    LUCAS (LUng CAncer Simple set) is a lung cancer toy dataset generated
    artificially by a causal Bayesian network with binary variables.

    This dataset is used for illustration purpose only. It models a medical
    application for the diagnosis, prevention, and cure of lung cancer.

    The dataset contains 12 binary variables:
    - Lung_Cancer (target variable)
    - Smoking
    - Yellow_Fingers
    - Anxiety
    - Peer_Pressure
    - Genetics
    - Attention_Disorder
    - Born_an_Even_Day
    - Car_Accident
    - Fatigue
    - Allergy
    - Coughing

    References
    ----------
    .. [1] http://www.causality.inf.ethz.ch/data/LUCAS.html
    """

    _tags = {
        "name": "lucas",
        "n_variables": 12,
        "n_samples": 2000,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "simulated-lucas"

    data_url = None
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "Lung_Cancer",
        "Smoking",
        "Yellow_Fingers",
        "Anxiety",
        "Peer_Pressure",
        "Genetics",
        "Attention_Disorder",
        "Born_an_Even_Day",
        "Car_Accident",
        "Fatigue",
        "Allergy",
        "Coughing",
    ]
    ordinal_variables = dict()

    @classmethod
    def load_dataframe(cls):
        """
        Generate LUCAS data based on the conditional probabilities
        from http://www.causality.inf.ethz.ch/data/LUCAS.html
        """
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n_samples = 2000

        # Generate parent variables first (no parents)
        Anxiety = np.random.choice(
            [True, False], size=n_samples, p=[0.64277, 1 - 0.64277]
        )

        Peer_Pressure = np.random.choice(
            [True, False], size=n_samples, p=[0.32997, 1 - 0.32997]
        )

        Genetics = np.random.choice(
            [True, False], size=n_samples, p=[0.15953, 1 - 0.15953]
        )

        Born_an_Even_Day = np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])

        Allergy = np.random.choice(
            [True, False], size=n_samples, p=[0.32841, 1 - 0.32841]
        )

        # Generate Smoking based on Anxiety and Peer Pressure
        Smoking = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Peer_Pressure[i] and not Anxiety[i]:
                Smoking[i] = np.random.choice([True, False], p=[0.43118, 1 - 0.43118])
            elif Peer_Pressure[i] and not Anxiety[i]:
                Smoking[i] = np.random.choice([True, False], p=[0.74591, 1 - 0.74591])
            elif not Peer_Pressure[i] and Anxiety[i]:
                Smoking[i] = np.random.choice([True, False], p=[0.8686, 1 - 0.8686])
            else:
                Smoking[i] = np.random.choice([True, False], p=[0.91576, 1 - 0.91576])

        # Generate Yellow_Fingers based on Smoking
        Yellow_Fingers = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Smoking[i]:
                Yellow_Fingers[i] = np.random.choice([True, False], p=[0.23119, 1 - 0.23119])
            else:
                Yellow_Fingers[i] = np.random.choice([True, False], p=[0.95372, 1 - 0.95372])

        # Generate Attention_Disorder based on Genetics
        Attention_Disorder = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Genetics[i]:
                Attention_Disorder[i] = np.random.choice(
                    [True, False], p=[0.28956, 1 - 0.28956]
                )
            else:
                Attention_Disorder[i] = np.random.choice(
                    [True, False], p=[0.68706, 1 - 0.68706]
                )

        # Generate Lung_Cancer based on Genetics and Smoking
        Lung_Cancer = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Genetics[i] and not Smoking[i]:
                Lung_Cancer[i] = np.random.choice([True, False], p=[0.23146, 1 - 0.23146])
            elif Genetics[i] and not Smoking[i]:
                Lung_Cancer[i] = np.random.choice([True, False], p=[0.86996, 1 - 0.86996])
            elif not Genetics[i] and Smoking[i]:
                Lung_Cancer[i] = np.random.choice([True, False], p=[0.83934, 1 - 0.83934])
            else:
                Lung_Cancer[i] = np.random.choice([True, False], p=[0.99351, 1 - 0.99351])

        # Generate Coughing based on Allergy and Lung_Cancer
        Coughing = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Allergy[i] and not Lung_Cancer[i]:
                Coughing[i] = np.random.choice([True, False], p=[0.1347, 1 - 0.1347])
            elif Allergy[i] and not Lung_Cancer[i]:
                Coughing[i] = np.random.choice([True, False], p=[0.64592, 1 - 0.64592])
            elif not Allergy[i] and Lung_Cancer[i]:
                Coughing[i] = np.random.choice([True, False], p=[0.7664, 1 - 0.7664])
            else:
                Coughing[i] = np.random.choice([True, False], p=[0.99947, 1 - 0.99947])

        # Generate Fatigue based on Lung_Cancer and Coughing
        Fatigue = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Lung_Cancer[i] and not Coughing[i]:
                Fatigue[i] = np.random.choice([True, False], p=[0.35212, 1 - 0.35212])
            elif Lung_Cancer[i] and not Coughing[i]:
                Fatigue[i] = np.random.choice([True, False], p=[0.56514, 1 - 0.56514])
            elif not Lung_Cancer[i] and Coughing[i]:
                Fatigue[i] = np.random.choice([True, False], p=[0.80016, 1 - 0.80016])
            else:
                Fatigue[i] = np.random.choice([True, False], p=[0.89589, 1 - 0.89589])

        # Generate Car_Accident based on Attention_Disorder and Fatigue
        Car_Accident = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not Attention_Disorder[i] and not Fatigue[i]:
                Car_Accident[i] = np.random.choice([True, False], p=[0.2274, 1 - 0.2274])
            elif Attention_Disorder[i] and not Fatigue[i]:
                Car_Accident[i] = np.random.choice([True, False], p=[0.779, 1 - 0.779])
            elif not Attention_Disorder[i] and Fatigue[i]:
                Car_Accident[i] = np.random.choice([True, False], p=[0.78861, 1 - 0.78861])
            else:
                Car_Accident[i] = np.random.choice([True, False], p=[0.97169, 1 - 0.97169])

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Lung_Cancer": Lung_Cancer,
                "Smoking": Smoking,
                "Yellow_Fingers": Yellow_Fingers,
                "Anxiety": Anxiety,
                "Peer_Pressure": Peer_Pressure,
                "Genetics": Genetics,
                "Attention_Disorder": Attention_Disorder,
                "Born_an_Even_Day": Born_an_Even_Day,
                "Car_Accident": Car_Accident,
                "Fatigue": Fatigue,
                "Allergy": Allergy,
                "Coughing": Coughing,
            }
        )

        return df

    @classmethod
    def load_ground_truth(cls):
        """
        Load the ground truth DAG for the LUCAS dataset.

        The causal structure is:
        - Anxiety -> Smoking
        - Peer_Pressure -> Smoking
        - Smoking -> Yellow_Fingers
        - Genetics -> Lung_Cancer
        - Smoking -> Lung_Cancer
        - Genetics -> Attention_Disorder
        - Lung_Cancer -> Coughing
        - Allergy -> Coughing
        - Lung_Cancer -> Fatigue
        - Coughing -> Fatigue
        - Attention_Disorder -> Car_Accident
        - Fatigue -> Car_Accident
        """
        # Define the edges based on the causal structure
        edges = [
            ("Anxiety", "Smoking"),
            ("Peer_Pressure", "Smoking"),
            ("Smoking", "Yellow_Fingers"),
            ("Genetics", "Lung_Cancer"),
            ("Smoking", "Lung_Cancer"),
            ("Genetics", "Attention_Disorder"),
            ("Lung_Cancer", "Coughing"),
            ("Allergy", "Coughing"),
            ("Lung_Cancer", "Fatigue"),
            ("Coughing", "Fatigue"),
            ("Attention_Disorder", "Car_Accident"),
            ("Fatigue", "Car_Accident"),
        ]

        dag = DAG(ebunch=edges)
        return dag
