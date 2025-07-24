"""
MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr format.

This package provides tools for converting MSI data from various formats (ImzML, Bruker)
into the modern SpatialData/Zarr format with automatic pixel size detection.
"""

# Suppress known warnings from dependencies
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")

__version__ = "1.8.3"

# Import readers and converters to trigger registration
from . import converters  # This triggers converter registrations
from . import readers  # This triggers reader registrations
from .convert import convert_msi

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata_converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter = None

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "SpatialDataConverter",
]
