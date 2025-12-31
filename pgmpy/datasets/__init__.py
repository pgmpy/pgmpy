from ._base import DATASET_REGISTRY, load_dataset, register_dataset_class
from ._abalone import AbaloneContinuous, AbaloneMixed  # noqa: F401
from ._adult import Adult  # noqa: F401
from ._airfoil import Airfoil  # noqa: F401
from ._algeria import Algeria  # noqa: F401
from ._boston_housing import BostonHousing  # noqa: F401
from ._galton_stature import GaltonStature  # noqa: F401
from ._sachs import (  # noqa: F401
    SachsContinuous,
    SachsContinuousJittered,
    SachsContinuousJitteredLogScale,
    SachsContinuousLogScale,
    SachsDiscrete,
    SachsMixed,
)
from ._lead import Lead  # noqa: F401
from ._goldberg import Goldberg  # noqa: F401
from ._hitters import Hitters  # noqa: F401
from ._residential_building import ResidentialBuilding  # noqa: F401
from ._iq_brain import IQBrainSize  # noqa: F401
from ._contraceptive_method import ContraceptiveMethod  # noqa: F401
from ._spartina import Spartina  # noqa: F401
from ._pittsburgh_bridges import PittsburghBridges  # noqa: F401
from ._credit_approval import CreditApproval  # noqa: F401
from ._myocardial_infarction import MyocardialInfarction  # noqa: F401
from ._htru2 import HTRU2  # noqa: F401
from ._yacht_hydrodynamics import YachtHydrodynamics  # noqa: F401
from ._dry_bean import DryBean  # noqa: F401
from ._cystic_fibrosis import CysticFibrosis  # noqa: F401
from ._apple_watch_fitbit import AppleWatchFitbit  # noqa: F401
from ._auto_mpg import AutoMpg  # noqa: F401
from ._south_german_credit import SouthGermanCredit  # noqa: F401
from ._student_performance import StudentPerformance  # noqa: F401
from ._pima_diabetes import PimaDiabetes  # noqa: F401
from ._superconductivity import Superconductivity  # noqa: F401
from ._seoul_bike import SeoulBike  # noqa: F401
from ._cities import Cities  # noqa: F401
from ._wine_quality import (  # noqa: F401
    WineQualityWhite,
    WineQualityRed,
    WineQualityRedWhiteMixed,
)

__all__ = [
    "_BaseDataset",
    "DATASET_REGISTRY",
    "register_dataset_class",
    "load_dataset",
]
