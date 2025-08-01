# msiconvert/converters/spatialdata/base_spatialdata_converter.py

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from ...core.base_converter import BaseMSIConverter
from ...core.base_reader import BaseMSIReader

# Check SpatialData availability (defer imports to avoid issues)
SPATIALDATA_AVAILABLE = False
_import_error_msg = None
try:
    import geopandas as gpd
    from anndata import AnnData  # type: ignore
    from shapely.geometry import box
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity

    SPATIALDATA_AVAILABLE = True
except (ImportError, NotImplementedError) as e:
    _import_error_msg = str(e)
    logging.warning(f"SpatialData dependencies not available: {e}")
    SPATIALDATA_AVAILABLE = False

    # Create dummy classes for registration
    class AnnData:
        pass

    SpatialData = None
    TableModel = None
    ShapesModel = None
    Image2DModel = None
    Identity = None
    box = None
    gpd = None


class BaseSpatialDataConverter(BaseMSIConverter, ABC):
    """Base converter for MSI data to SpatialData format with shared functionality."""

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base SpatialData converter.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices (False)
            pixel_size_detection_info: Optional metadata about pixel size detection
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If SpatialData dependencies are not available
            ValueError: If pixel_size_um is not positive or dataset_id is empty
        """
        # Check if SpatialData is available
        if not SPATIALDATA_AVAILABLE:
            error_msg = (
                f"SpatialData dependencies not available: {_import_error_msg}. "
                f"Please install required packages or fix dependency conflicts."
            )
            raise ImportError(error_msg)

        # Validate inputs
        if pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        # Extract pixel_size_detection_info from kwargs if provided
        kwargs_filtered = dict(kwargs)
        if (
            pixel_size_detection_info is None
            and "pixel_size_detection_info" in kwargs_filtered
        ):
            pixel_size_detection_info = kwargs_filtered.pop("pixel_size_detection_info")

        super().__init__(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs_filtered,
        )

        self._non_empty_pixel_count: int = 0
        self._pixel_size_detection_info = pixel_size_detection_info

    def _create_sparse_matrix(self) -> sparse.lil_matrix:
        """Create sparse matrix for storing intensity values.

        Returns:
            Sparse matrix for storing intensity values

        Raises:
            ValueError: If dimensions or common mass axis are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, n_z = self._dimensions
        n_pixels = n_x * n_y * n_z
        n_masses = len(self._common_mass_axis)

        logging.info(
            f"Creating sparse matrix with {n_pixels} pixels and {n_masses} mass values"
        )
        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)

    def _create_coordinates_dataframe(self) -> pd.DataFrame:
        """Create coordinates dataframe with pixel positions.

        Returns:
            DataFrame with pixel coordinates

        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        # Pre-allocate arrays for better performance
        coords_data = []

        pixel_idx = 0
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    coords_data.append(
                        {
                            "x": x,
                            "y": y,
                            "z": z if n_z > 1 else 0,
                            "instance_id": str(pixel_idx),
                            "region": f"{self.dataset_id}_pixels",
                            "spatial_x": x * self.pixel_size_um,
                            "spatial_y": y * self.pixel_size_um,
                            "spatial_z": (z * self.pixel_size_um if n_z > 1 else 0.0),
                        }
                    )
                    pixel_idx += 1

        coords_df = pd.DataFrame(coords_data)
        coords_df.set_index("instance_id", inplace=True)
        return coords_df

    def _create_mass_dataframe(self) -> pd.DataFrame:
        """Create m/z dataframe for variable metadata.

        Returns:
            DataFrame with m/z values

        Raises:
            ValueError: If common mass axis is not initialized
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        return pd.DataFrame(
            {"mz": self._common_mass_axis},
            index=[f"mz_{i}" for i in range(len(self._common_mass_axis))],
        )

    def _get_pixel_index(self, x: int, y: int, z: int) -> int:
        """Calculate linear pixel index from 3D coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Linear pixel index

        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, _ = self._dimensions
        return z * (n_x * n_y) + y * n_x + x

    def _add_to_sparse_matrix(
        self,
        sparse_matrix: sparse.lil_matrix,
        pixel_idx: int,
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """Add intensity data to sparse matrix.

        Args:
            sparse_matrix: Sparse matrix to update
            pixel_idx: Linear pixel index
            mz_indices: Indices for mass values
            intensities: Intensity values to add
        """
        sparse_matrix[pixel_idx, mz_indices] = intensities

    def _create_pixel_shapes(
        self, adata: AnnData, is_3d: bool = False
    ) -> "ShapesModel":
        """Create geometric shapes for pixels with proper transformations.

        Args:
            adata: AnnData object containing coordinates
            is_3d: Whether to handle as 3D data

        Returns:
            SpatialData shapes model

        Raises:
            ImportError: If required SpatialData dependencies are not available
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        # Extract coordinates directly from obs
        x_coords: NDArray[np.float64] = adata.obs["spatial_x"].values
        y_coords: NDArray[np.float64] = adata.obs["spatial_y"].values

        # Create geometries efficiently - this loop could be optimized but kept for clarity
        half_pixel = self.pixel_size_um / 2
        geometries = []

        for i in range(len(adata)):
            x, y = x_coords[i], y_coords[i]
            pixel_box = box(
                x - half_pixel, y - half_pixel, x + half_pixel, y + half_pixel
            )
            geometries.append(pixel_box)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, index=adata.obs.index)

        # Set up transform
        transform = Identity()
        transformations = {self.dataset_id: transform, "global": transform}

        # Parse shapes
        shapes = ShapesModel.parse(gdf, transformations=transformations)
        return shapes

    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save the data to SpatialData format.

        Args:
            data_structures: Data structures to save

        Returns:
            True if saving was successful, False otherwise
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        try:
            # Create SpatialData object with images included
            sdata = SpatialData(
                tables=data_structures["tables"],
                shapes=data_structures["shapes"],
                images=data_structures["images"],
            )

            # Add metadata
            self.add_metadata(sdata)

            # Write to disk
            sdata.write(str(self.output_path))
            logging.info(f"Successfully saved SpatialData to {self.output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving SpatialData: {e}")
            import traceback

            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

    def add_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive metadata to the SpatialData object.

        Args:
            metadata: SpatialData object to add metadata to
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        # Call parent to prepare structured metadata
        super().add_metadata(metadata)

        # Get comprehensive metadata object for detailed access
        comprehensive_metadata_obj = self.reader.get_comprehensive_metadata()

        # Add explicit pixel size metadata to SpatialData object attributes
        if not hasattr(metadata, "attrs") or metadata.attrs is None:
            metadata.attrs = {}

        logging.info("Adding comprehensive metadata to SpatialData.attrs")

        # Import version dynamically (fix for hard-coded version)
        try:
            from ... import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        # Add pixel size metadata to SpatialData attributes
        pixel_size_attrs = {
            "pixel_size_x_um": float(self.pixel_size_um),
            "pixel_size_y_um": float(self.pixel_size_um),
            "pixel_size_units": "micrometers",
            "coordinate_system": "physical_micrometers",
            "msi_converter_version": version,
            "conversion_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Add pixel size detection provenance if available
        if self._pixel_size_detection_info is not None:
            pixel_size_attrs["pixel_size_detection_info"] = dict(
                self._pixel_size_detection_info
            )
            logging.info(
                f"Added pixel size detection info: {self._pixel_size_detection_info}"
            )

        # Add conversion metadata
        pixel_size_attrs["msi_dataset_info"] = {
            "dataset_id": self.dataset_id,
            "total_grid_pixels": self._dimensions[0]
            * self._dimensions[1]
            * self._dimensions[2],
            "non_empty_pixels": self._non_empty_pixel_count,
            "dimensions_xyz": list(self._dimensions),
        }

        # Add comprehensive metadata sections to SpatialData attributes
        if comprehensive_metadata_obj.format_specific:
            pixel_size_attrs["format_specific_metadata"] = (
                comprehensive_metadata_obj.format_specific
            )

        if comprehensive_metadata_obj.acquisition_params:
            pixel_size_attrs["acquisition_parameters"] = (
                comprehensive_metadata_obj.acquisition_params
            )

        if comprehensive_metadata_obj.instrument_info:
            pixel_size_attrs["instrument_information"] = (
                comprehensive_metadata_obj.instrument_info
            )

        # Update SpatialData attributes
        metadata.attrs.update(pixel_size_attrs)

        # Add comprehensive dataset metadata if SpatialData supports it
        if hasattr(metadata, "metadata"):
            # Start with structured metadata from base class
            metadata_dict = self._structured_metadata.copy()

            # Add SpatialData-specific enhancements
            metadata_dict.update(
                {
                    "non_empty_pixels": self._non_empty_pixel_count,
                    "spatialdata_specific": {
                        "zarr_compression_level": self.compression_level,
                        "tables_count": len(getattr(metadata, "tables", {})),
                        "shapes_count": len(getattr(metadata, "shapes", {})),
                        "images_count": len(getattr(metadata, "images", {})),
                    },
                }
            )

            # Add pixel size detection provenance if available
            if self._pixel_size_detection_info is not None:
                metadata_dict["pixel_size_provenance"] = self._pixel_size_detection_info

            # Add conversion options used
            metadata_dict["conversion_options"] = {
                "handle_3d": self.handle_3d,
                "pixel_size_um": self.pixel_size_um,
                "dataset_id": self.dataset_id,
                **self.options,
            }

            metadata.metadata = metadata_dict

            logging.info(
                f"Comprehensive metadata persisted to SpatialData with "
                f"{len(metadata_dict)} top-level sections"
            )

    @abstractmethod
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for the specific converter type."""
        pass

    @abstractmethod
    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a single spectrum for the specific converter type."""
        pass

    @abstractmethod
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize data structures for the specific converter type."""
        pass
