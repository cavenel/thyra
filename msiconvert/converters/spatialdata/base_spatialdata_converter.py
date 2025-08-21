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
from ...resampling import ResamplingDecisionTree, ResamplingMethod

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
        """Dummy AnnData class for when SpatialData is not available."""

        pass

    SpatialData = None
    TableModel = None
    ShapesModel = None
    Image2DModel = None
    Identity = None
    box = None
    gpd = None


class BaseSpatialDataConverter(BaseMSIConverter, ABC):
    """Base converter for MSI data to SpatialData format with shared
    functionality.
    """

    def __init__(
        self,
        reader: BaseMSIReader,
        output_path: Path,
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        pixel_size_detection_info: Optional[Dict[str, Any]] = None,
        resampling_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base SpatialData converter.

        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices
                (False)
            pixel_size_detection_info: Optional metadata about pixel size
                detection
            resampling_config: Optional resampling configuration dict
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If SpatialData dependencies are not available
            ValueError: If pixel_size_um is not positive or dataset_id is
                empty
        """
        # Check if SpatialData is available
        if not SPATIALDATA_AVAILABLE:
            error_msg = (
                f"SpatialData dependencies not available: "
                f"{_import_error_msg}. "
                f"Please install required packages or fix dependency "
                f"conflicts."
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
        self._resampling_config = resampling_config

        # Set up resampling if enabled
        if self._resampling_config:
            self._setup_resampling()
            # Override the common mass axis with resampled axis
            self._build_resampled_mass_axis()

    def _setup_resampling(self) -> None:
        """Set up resampling configuration and strategy."""
        if not self._resampling_config:
            return

        method = self._resampling_config.get("method", "auto")

        # If method is "auto", use DecisionTree to determine strategy
        if method == "auto":
            try:
                # Get metadata from reader for instrument detection
                metadata = self._get_reader_metadata_for_resampling()
                tree = ResamplingDecisionTree()
                detected_method = tree.select_strategy(metadata)
                logging.info(f"Auto-detected resampling method: {detected_method}")
                self._resampling_method = detected_method
            except NotImplementedError as e:
                logging.error(f"Auto-detection failed: {e}")
                logging.info("Falling back to nearest_neighbor for resampling")
                self._resampling_method = ResamplingMethod.NEAREST_NEIGHBOR
        else:
            # Convert string to enum
            method_map = {
                "nearest_neighbor": ResamplingMethod.NEAREST_NEIGHBOR,
                "tic_preserving": ResamplingMethod.TIC_PRESERVING,
            }
            self._resampling_method = method_map.get(
                method, ResamplingMethod.NEAREST_NEIGHBOR
            )

        logging.info(f"Using resampling method: {self._resampling_method}")

        # Store resampling parameters
        self._target_bins = self._resampling_config.get("target_bins", 5000)
        self._min_mz = self._resampling_config.get("min_mz")
        self._max_mz = self._resampling_config.get("max_mz")

    def _get_reader_metadata_for_resampling(self) -> Dict[str, Any]:
        """Extract metadata from reader for resampling decision tree."""
        metadata = {}

        # Extract different types of metadata
        self._extract_essential_metadata(metadata)
        self._extract_comprehensive_metadata(metadata)
        self._extract_spectrum_metadata(metadata)

        return metadata

    def _extract_essential_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract essential metadata for resampling decisions."""
        try:
            essential = self.reader.get_essential_metadata()
            if hasattr(essential, "source_path"):
                metadata["source_path"] = str(essential.source_path)

            # Add essential metadata for resampling decisions
            metadata["essential_metadata"] = {
                "spectrum_type": getattr(essential, "spectrum_type", None),
                "dimensions": essential.dimensions,
                "mass_range": essential.mass_range,
                "source_path": str(essential.source_path),
            }
        except Exception as e:
            logging.debug(f"Could not extract essential metadata: {e}")

    def _extract_comprehensive_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract comprehensive metadata including Bruker GlobalMetadata."""
        try:
            comp_meta = self.reader.get_comprehensive_metadata()
            self._extract_bruker_metadata(metadata, comp_meta)
            self._extract_instrument_info(metadata, comp_meta)
        except Exception as e:
            logging.debug(f"Could not extract comprehensive metadata: {e}")

    def _extract_bruker_metadata(self, metadata: Dict[str, Any], comp_meta) -> None:
        """Extract Bruker GlobalMetadata from comprehensive metadata."""
        if (
            hasattr(comp_meta, "raw_metadata")
            and "global_metadata" in comp_meta.raw_metadata
        ):
            metadata["GlobalMetadata"] = comp_meta.raw_metadata["global_metadata"]
            logging.debug(
                f"Extracted Bruker GlobalMetadata with keys: "
                f"{list(metadata['GlobalMetadata'].keys())}"
            )

    def _extract_instrument_info(self, metadata: Dict[str, Any], comp_meta) -> None:
        """Extract instrument_info for fallback detection."""
        if hasattr(comp_meta, "instrument_info"):
            metadata["instrument_info"] = comp_meta.instrument_info
            logging.debug(f"Extracted instrument_info: {comp_meta.instrument_info}")

    def _extract_spectrum_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract ImzML-specific spectrum metadata."""
        try:
            if hasattr(self.reader, "get_spectrum_metadata"):
                spec_meta = self.reader.get_spectrum_metadata()
                if spec_meta:
                    metadata.update(spec_meta)
        except Exception as e:
            logging.debug(f"Could not extract spectrum metadata: {e}")

    def _build_resampled_mass_axis(self) -> None:
        """Build resampled mass axis using physics-based generators."""
        from ...resampling.common_axis import CommonAxisBuilder

        # Get the original mass range from the raw data
        mass_range = self.reader.mass_range
        min_mz = mass_range[0] if self._min_mz is None else self._min_mz
        max_mz = mass_range[1] if self._max_mz is None else self._max_mz

        logging.info(
            f"Building resampled mass axis: {min_mz:.2f} - {max_mz:.2f} m/z, "
            f"{self._target_bins} bins"
        )

        # Get metadata for axis type selection
        metadata = self._get_reader_metadata_for_resampling()

        # Select axis type using DecisionTree
        tree = ResamplingDecisionTree()
        axis_type = tree.select_axis_type(metadata)

        logging.info(f"Selected axis type: {axis_type}")

        # Build the physics-based axis
        builder = CommonAxisBuilder()

        if hasattr(axis_type, "value") and axis_type.value != "constant":
            # Use physics-based generator
            mass_axis = builder.build_physics_axis(
                min_mz=min_mz,
                max_mz=max_mz,
                num_bins=self._target_bins,
                axis_type=axis_type,
            )
            logging.info(
                f"Built physics-based {axis_type} mass axis with "
                f"{len(mass_axis.mz_values)} points"
            )
        else:
            # Fall back to uniform axis
            mass_axis = builder.build_uniform_axis(min_mz, max_mz, self._target_bins)
            logging.info(
                f"Built uniform mass axis with " f"{len(mass_axis.mz_values)} points"
            )

        # Override the parent's common mass axis
        self._common_mass_axis = mass_axis.mz_values

        logging.info(
            f"Resampled mass axis: {len(self._common_mass_axis)} bins, "
            f"range {self._common_mass_axis[0]:.2f} - "
            f"{self._common_mass_axis[-1]:.2f}"
        )

    def _initialize_conversion(self) -> None:
        """Override parent initialization to preserve resampled mass axis."""
        logging.info("Loading essential dataset information...")
        try:
            # Load essential metadata first (fast, single query for Bruker)
            essential = self.reader.get_essential_metadata()

            self._dimensions = essential.dimensions
            if any(d <= 0 for d in self._dimensions):
                raise ValueError(
                    f"Invalid dimensions: {self._dimensions}. All dimensions "
                    f"must be positive."
                )

            # Store essential metadata for use throughout conversion
            self._coordinate_bounds = essential.coordinate_bounds
            self._n_spectra = essential.n_spectra
            self._estimated_memory_gb = essential.estimated_memory_gb

            # Override pixel size if not provided and available in metadata
            if self.pixel_size_um is None and essential.pixel_size:
                self.pixel_size_um = essential.pixel_size[0]
                logging.info(f"Using detected pixel size: {self.pixel_size_um} μm")

            # IMPORTANT: Don't overwrite _common_mass_axis if resampling
            # is enabled
            if self._resampling_config and self._common_mass_axis is not None:
                # Already set by _build_resampled_mass_axis() - keep it
                logging.info(
                    f"Preserving resampled mass axis with "
                    f"{len(self._common_mass_axis)} bins"
                )
            else:
                # No resampling - load raw mass axis as usual
                self._common_mass_axis = self.reader.get_common_mass_axis()
                if len(self._common_mass_axis) == 0:
                    raise ValueError(
                        "Common mass axis is empty. Cannot proceed with " "conversion."
                    )
                logging.info(
                    f"Using raw mass axis with "
                    f"{len(self._common_mass_axis)} unique m/z values"
                )

            # Only load comprehensive metadata if needed (lazy loading)
            self._metadata = None  # Will be loaded on demand

            logging.info(f"Dataset dimensions: {self._dimensions}")
            logging.info(f"Coordinate bounds: {self._coordinate_bounds}")
            logging.info(f"Total spectra: {self._n_spectra}")
            logging.info(f"Estimated memory: {self._estimated_memory_gb:.2f} GB")
            logging.info(f"Common mass axis length: {len(self._common_mass_axis)}")
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def _map_mass_to_indices(self, mzs: NDArray[np.float64]) -> NDArray[np.int_]:
        """Override mass mapping to handle resampling with interpolation."""
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized.")

        if mzs.size == 0:
            return np.array([], dtype=int)

        # If resampling is enabled, we need to interpolate instead of exact
        # matching
        if self._resampling_config:
            return self._resample_spectrum_to_indices(mzs)
        else:
            # Use parent's exact matching for non-resampled data
            return super()._map_mass_to_indices(mzs)

    def _resample_spectrum_to_indices(
        self, mzs: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """Map spectrum m/z values to resampled mass axis indices using
        interpolation."""
        # For resampled data, we want to return ALL indices in the resampled
        # axis. The actual resampling/interpolation will be handled in the
        # processing
        return np.arange(len(self._common_mass_axis), dtype=np.int_)

    def _process_single_spectrum(
        self,
        data_structures: Any,
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Override spectrum processing to handle resampling."""
        logging.info(
            f"Processing spectrum at {coords}: {len(mzs)} peaks, "
            f"intensity sum: {np.sum(intensities):.2e}"
        )

        if self._resampling_config:
            # Resample the spectrum onto the common mass axis
            resampled_intensities = self._resample_spectrum(mzs, intensities)
            mz_indices = np.arange(len(self._common_mass_axis), dtype=np.int_)
            logging.info(
                f"Resampled: {len(resampled_intensities)} values, "
                f"sum: {np.sum(resampled_intensities):.2e}"
            )

            # Call the specific converter's processing with resampled data
            self._process_resampled_spectrum(
                data_structures, coords, mz_indices, resampled_intensities
            )
        else:
            # Use standard processing for non-resampled data
            mz_indices = self._map_mass_to_indices(mzs)
            logging.info(
                f"Mapped to {len(mz_indices)} indices, "
                f"intensity sum: {np.sum(intensities):.2e}"
            )
            self._process_resampled_spectrum(
                data_structures, coords, mz_indices, intensities
            )

    def _resample_spectrum(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Resample a single spectrum onto the common mass axis using the
        selected strategy."""
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        # Use the selected resampling method
        if hasattr(self, "_resampling_method"):
            if self._resampling_method == ResamplingMethod.NEAREST_NEIGHBOR:
                return self._nearest_neighbor_resample(mzs, intensities)
            elif self._resampling_method == ResamplingMethod.TIC_PRESERVING:
                return self._tic_preserving_resample(mzs, intensities)

        # Fallback to nearest neighbor
        return self._nearest_neighbor_resample(mzs, intensities)

    def _nearest_neighbor_resample(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Resample using vectorized nearest neighbor - optimized and
        clean."""
        if mzs.size == 0:
            return np.zeros(len(self._common_mass_axis))

        # Find insertion points using vectorized binary search
        indices = np.searchsorted(self._common_mass_axis, mzs)

        # Handle edge cases and find actual nearest neighbors
        indices = np.clip(indices, 0, len(self._common_mass_axis) - 1)

        # Check if left neighbor is closer (when not at boundary)
        mask = indices > 0
        left_indices = indices - 1
        left_dist = np.abs(self._common_mass_axis[left_indices[mask]] - mzs[mask])
        right_dist = np.abs(self._common_mass_axis[indices[mask]] - mzs[mask])
        indices[mask] = np.where(
            left_dist <= right_dist, left_indices[mask], indices[mask]
        )

        # Accumulate intensities using bincount (fastest accumulation)
        resampled = np.bincount(
            indices, weights=intensities, minlength=len(self._common_mass_axis)
        )

        return resampled

    def _tic_preserving_resample(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Resample using TIC-preserving linear interpolation - optimized."""
        if mzs.size == 0:
            return np.zeros(len(self._common_mass_axis))

        # OPTIMIZED: Check if already sorted to avoid unnecessary sorting
        if np.all(mzs[:-1] <= mzs[1:]):
            # Already sorted - use directly
            mzs_sorted = mzs
            intensities_sorted = intensities
        else:
            # Need to sort for interpolation
            sort_indices = np.argsort(mzs)
            mzs_sorted = mzs[sort_indices]
            intensities_sorted = intensities[sort_indices]

        # Interpolate onto the common mass axis (np.interp is highly optimized)
        resampled = np.interp(
            self._common_mass_axis,
            mzs_sorted,
            intensities_sorted,
            left=0,
            right=0,
        )

        return resampled

    def _process_resampled_spectrum(
        self,
        data_structures: Any,
        coords: Tuple[int, int, int],
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """Process a spectrum with resampled intensities - to be overridden
        by subclasses."""
        # This method should be overridden by specific converters (2D/3D)
        pass

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
            f"Creating sparse matrix with {n_pixels} pixels and "
            f"{n_masses} mass values"
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
        """Add intensity data to sparse matrix efficiently.

        Args:
            sparse_matrix: Target sparse matrix (lil_matrix for efficient
                construction)
            pixel_idx: Linear pixel index
            mz_indices: Indices for mass values
            intensities: Intensity values to add
        """
        # Filter out zero intensities to maintain sparsity
        nonzero_mask = intensities != 0.0
        if not np.any(nonzero_mask):
            return

        valid_mz_indices = mz_indices[nonzero_mask]
        valid_intensities = intensities[nonzero_mask]

        # Direct assignment to lil_matrix (simple and robust)
        sparse_matrix[pixel_idx, valid_mz_indices] = valid_intensities

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

        # Create geometries efficiently - this loop could be optimized but
        # kept for clarity
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

        # Setup attributes and add pixel size metadata
        self._setup_spatialdata_attrs(metadata, comprehensive_metadata_obj)

        # Add comprehensive dataset metadata if supported
        self._add_comprehensive_metadata(metadata)

    def _setup_spatialdata_attrs(
        self, metadata: "SpatialData", comprehensive_metadata_obj
    ) -> None:
        """Setup SpatialData attributes with pixel size and metadata."""
        if not hasattr(metadata, "attrs") or metadata.attrs is None:
            metadata.attrs = {}

        logging.info("Adding comprehensive metadata to SpatialData.attrs")

        # Create pixel size attributes
        pixel_size_attrs = self._create_pixel_size_attrs()

        # Add comprehensive metadata sections
        self._add_comprehensive_sections(pixel_size_attrs, comprehensive_metadata_obj)

        # Update SpatialData attributes
        metadata.attrs.update(pixel_size_attrs)

    def _create_pixel_size_attrs(self) -> Dict[str, Any]:
        """Create pixel size and conversion metadata attributes."""
        # Import version dynamically
        try:
            from ... import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        # Base pixel size metadata
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
                f"Added pixel size detection info: "
                f"{self._pixel_size_detection_info}"
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

        return pixel_size_attrs

    def _add_comprehensive_sections(
        self, pixel_size_attrs: Dict[str, Any], comprehensive_metadata_obj
    ) -> None:
        """Add comprehensive metadata sections to attributes."""
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

    def _add_comprehensive_metadata(self, metadata: "SpatialData") -> None:
        """Add comprehensive dataset metadata if SpatialData supports it."""
        if not hasattr(metadata, "metadata"):
            return

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
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize data structures for the specific converter type."""
        pass
