_EXDATA_REGISTRY_SCENARIO1 = {
    "airfoil-self-noise": {
        "source": "real",
        "folder": "airfoil-self-noise",
        "subfolder": "data",
        "file": "airfoil-self-noise.continuous.txt",
        "ground_truth": {"graph": "airfoil-self-noise.knowledge.txt"},
    },
    "algerian-forest-fires": {
        "source": "real",
        "folder": "algerian-forest-fires",
        "subfolder": "data",
        "file": "algerian-forest-fires.mixed.maximum.2.txt",
        "ground_truth": {"graph": "algerian-forest-fires.knowledge.txt"},
    },
    "auto-mpg": {
        "source": "real",
        "folder": "auto-mpg",
        "subfolder": "data",
        "file": "auto-mpg.data.mixed.max.3.categories.txt",
        "ground_truth": {"graph": "auto-mpg.knowledge.txt"},
    },
    "boston-housing": {
        "source": "real",
        "folder": "boston-housing",
        "subfolder": "data",
        "file": "boston-housing.continuous.txt",
        # No ground Truth
    },
    "cites": {
        "source": "real",
        "folder": "cites",
        "subfolder": "data",
        "file": "cites.cov.txt",
        "ground_truth": {"graph": "cites.knowledge.txt"},
    },
    "college-plans": {
        "source": "real",
        "folder": "college-plans",
        "subfolder": "data",
        "file": "college-plans.discrete.txt",
        # No ground Truth
    },
    "contraceptive-method": {
        "source": "real",
        "folder": "contraceptive-method",
        "subfolder": "data",
        "file": "contraceptive-method.continuous.txt",
        # No ground Truth
    },
    "credit-approval": {
        "source": "real",
        "folder": "credit-approval",
        "subfolder": "data",
        "file": "crx.data.mixed.maximum.14.txt",
        # No ground Truth
    },
    "cystic-fibrosis": {
        "source": "real",
        "folder": "cystic-fibrosis",
        "subfolder": "data",
        "file": "cystic-fibrosis-20180726-simplified.continuous.txt",
        # No ground Truth
    },
    "depression-coping": {
        "source": "real",
        "folder": "depression-coping",
        "subfolder": "data",
        "file": "depressioncoping.continuous.dat",
        # No ground Truth
    },
    "dropouts": {
        "source": "real",
        "folder": "dropouts",
        "subfolder": "data",
        "file": "dropouts.cov.txt",
        # No ground Truth
    },
    "galton-stature": {
        "source": "real",
        "folder": "galton-stature",
        "subfolder": "data",
        "file": "galton-stature.mixed.txt",
        # No ground Truth
    },
    "goldberg": {
        "source": "real",
        "folder": "goldberg",
        "subfolder": "data",
        "file": "goldberg.cov.txt",
        # No ground Truth
    },
    "hitters": {
        "source": "real",
        "folder": "hitters",
        "subfolder": "data",
        "file": "hitters.txt",
        # No ground Truth
    },
    "iq-brain-size": {
        "source": "real",
        "folder": "iq-brain-size",
        "subfolder": "data",
        "file": "iq_brain_size.continuous.txt",
        # No ground Truth
    },
    "lead": {
        "source": "real",
        "folder": "lead",
        "subfolder": "data",
        "file": "lead.cov.txt",
        # No ground Truth
    },
    "myocardial-infarction-complications": {
        "source": "real",
        "folder": "myocardial-infarction-complications",
        "subfolder": "data",
        "file": "myocarcial-infaraction-complications.continuous.txt",
        "ground_truth": {"graph": "myocarcial-infaraction-complications.knowledge.txt"},
    },
    "pima-diabetes": {
        "source": "real",
        "folder": "pima-diabetes",
        "subfolder": "data",
        "file": "pima-diabetes.mixed.maximum.2.txt",
        "ground_truth": {"graph": "pima-diabetes.knowledge.txt"},
    },
    "pittsburgh-bridges": {
        "source": "real",
        "folder": "pittsburgh-bridges",
        "subfolder": "data",
        "file": "bridges.data.version21.txt",
        # No ground Truth
    },
    "residential-building": {
        "source": "real",
        "folder": "residential-building",
        "subfolder": "data",
        "file": "residential-building.continuous.txt",
        # No ground Truth
    },
    "seoul-bike": {
        "source": "real",
        "folder": "seoul-bike",
        "subfolder": "data",
        "file": "seoul-bike.mixed.maximum.4.txt",
        # No ground Truth
    },
    "south-german-credit": {
        "source": "real",
        "folder": "south-german-credit",
        "subfolder": "data",
        "file": "south-german-credit.data.mixed.txt",
        # No ground Truth
    },
    "spartina": {
        "source": "real",
        "folder": "spartina",
        "subfolder": "data",
        "file": "spartina.cov.txt",
        # No ground Truth
    },
    "student-performance": {
        "source": "real",
        "folder": "student-performance",
        "subfolder": "data",
        "file": "student-performance.data.mixed.maximum.3.txt",
        "ground_truth": {"graph": "student-performance.knowledge.txt"},
    },
    "superconductivity": {
        "source": "real",
        "folder": "superconductivity",
        "subfolder": "data",
        "file": "superconductivity.continuous.txt",
        "ground_truth": {"graph": "superconductivity.knowledge.txt"},
    },
    "uscrime": {
        "source": "real",
        "folder": "uscrime",
        "subfolder": "data",
        "file": "uscrime.continuous.txt",
        # No ground Truth
    },
    "yacht.hydrodynamics": {
        "source": "real",
        "folder": "yacht.hydrodynamics",
        "subfolder": "data",
        "file": "yacht.hydrodynamics.continuous.txt",
        "ground_truth": {"graph": "yacht-hydrodynamics.knowledge.txt"},
    },
}

REGISTRY_MULTI = {
    "blue-driver": {
        "source": "real",
        "folder": "blue-driver",
        "subfolder": "data",
        "default": "bluedata1.edited.continuous.txt",
        "variants": {
            "continuous": "bluedata1.edited.continuous.txt",
            "edited2": "bluedata2.edited.continuous.txt",
            "shortcodes": "blue.driver1.continuous.txt",
            "csvkey": "bluedriver1.key.csv",
        },
        "all_files": [
            "blue.driver1.continuous.txt",
            "bluedata1.edited.continuous.txt",
            "bluedata2.edited.continuous.txt",
            "bluedriver1.key.csv",
        ],
        "header_file": "bluedata1.edited.continuous.txt",
        "key_file": "bluedriver1.key.csv",
        # No ground Truth
        "ground_truth": {},
    }
}
