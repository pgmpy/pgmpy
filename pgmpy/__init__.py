from .config import HAS_PANDAS
from .config import set_device, set_backend, set_dtype

# Initialize configurations
set_device()
set_backend()
set_dtype()

__all__ = ["HAS_PANDAS"]
__version__ = "v0.1.8.dev34"
