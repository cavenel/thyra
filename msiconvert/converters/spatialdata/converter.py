# msiconvert/converters/spatialdata/converter.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.base_reader import BaseMSIReader
from ...core.registry import register_converter
from .spatialdata_2d_converter import SpatialData2DConverter
from .spatialdata_3d_converter import SpatialData3DConverter
from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, _import_error_msg


class SpatialDataConverter:
    """Factory converter for MSI data to SpatialData format.
    
    Creates appropriate 2D or 3D converter based on handle_3d parameter and data dimensions.
    """

    def __new__(
        cls,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Create appropriate converter based on handle_3d parameter and data dimensions.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices (False)
            pixel_size_detection_info: Optional metadata about pixel size detection
            **kwargs: Additional keyword arguments

        Returns:
            SpatialData2DConverter or SpatialData3DConverter instance

        Raises:
            ImportError: If SpatialData dependencies are not available
        """
        # Check if we have 3D data and determine converter type
        dimensions = reader.get_essential_metadata().dimensions
        if dimensions[2] > 1 and not handle_3d:
            # 3D data treated as 2D slices
            return SpatialData2DConverter(
                reader=reader,
                output_path=output_path,
                dataset_id=dataset_id,
                pixel_size_um=pixel_size_um,
                pixel_size_detection_info=pixel_size_detection_info,
                **kwargs,
            )
        else:
            # True 3D data or single 2D slice
            return SpatialData3DConverter(
                reader=reader,
                output_path=output_path,
                dataset_id=dataset_id,
                pixel_size_um=pixel_size_um,
                pixel_size_detection_info=pixel_size_detection_info,
                **kwargs,
            )


# Only register the converter if SpatialData dependencies are available
if SPATIALDATA_AVAILABLE:
    register_converter("spatialdata")(SpatialDataConverter)
    logging.debug("SpatialDataConverter registered successfully")
else:
    logging.warning(
        f"SpatialDataConverter not registered due to dependency issues: {_import_error_msg}"
    )