from ._base import load_dataset, list_datasets, _BaseDataset
from .abalone import AbaloneContinuous, AbaloneMixed  # noqa: F401
from .adult import Adult  # noqa: F401
from .airfoil import Airfoil  # noqa: F401
from .algeria import Algeria  # noqa: F401
from .boston_housing import BostonHousing  # noqa: F401
from .galton_stature import GaltonStature  # noqa: F401
from .sachs import (  # noqa: F401
    SachsContinuous,
    SachsContinuousJittered,
    SachsContinuousJitteredLogScale,
    SachsContinuousLogScale,
    SachsDiscrete,
    SachsMixed,
)
from .depression_coping import DepressionCoping  # noqa: F401
from .college_plans import CollegePlans  # noqa: F401
from .lead import Lead  # noqa: F401
from .goldberg import Goldberg  # noqa: F401
from .hitters import Hitters  # noqa: F401
from .residential_building import ResidentialBuilding  # noqa: F401
from .iq_brain import IQBrainSize  # noqa: F401
from .contraceptive_method import ContraceptiveMethod  # noqa: F401
from .spartina import Spartina  # noqa: F401
from .pittsburgh_bridges import PittsburghBridges  # noqa: F401
from .credit_approval import CreditApproval  # noqa: F401
from .myocardial_infarction import MyocardialInfarction  # noqa: F401
from .htru2 import HTRU2  # noqa: F401
from .yacht_hydrodynamics import YachtHydrodynamics  # noqa: F401
from .dry_bean import DryBean  # noqa: F401
from .cystic_fibrosis import CysticFibrosis  # noqa: F401
from .apple_watch_fitbit import AppleWatchFitbit  # noqa: F401
from .auto_mpg import AutoMpg  # noqa: F401
from .south_german_credit import SouthGermanCredit  # noqa: F401
from .student_performance import StudentPerformance  # noqa: F401
from .pima_diabetes import PimaDiabetes  # noqa: F401
from .superconductivity import Superconductivity  # noqa: F401
from .seoul_bike import SeoulBike  # noqa: F401
from .cities import Cities  # noqa: F401
from .wine_quality import (  # noqa: F401
    WineQualityWhite,
    WineQualityRed,
    WineQualityRedWhiteMixed,
)


__all__ = [
    "_BaseDataset",
    "load_dataset",
    "list_datasets",
]
