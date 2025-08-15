# Mass Axis Harmonization for MSI Data - Technical Specification

## 1. Overview

### 1.1 Purpose
This module provides automated mass axis harmonization for Mass Spectrometry Imaging (MSI) datasets from multiple vendors and instrument types. The system ensures all spectra in a dataset share a common mass axis, enabling consistent analysis across pixels.

### 1.2 Problem Statement
MSI datasets inherently contain multiple mass axes (one per pixel/spectrum) due to:
- Instrumental drift during acquisition
- Temperature variations
- Calibration differences between pixels
- Vendor-specific data formats and characteristics

### 1.3 Solution Approach
Implement an intelligent resampling system that:
- Automatically detects data characteristics
- Selects appropriate resampling strategy
- Creates a common mass axis
- Preserves data integrity and quality

## 2. Architecture

### 2.1 Module Structure
```
msi_processing/
├── resampling/
│   ├── __init__.py
│   ├── axis_detector.py       # Axis type and characteristic detection
│   ├── strategies/             # Resampling strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base strategy
│   │   ├── nearest_neighbor.py
│   │   ├── tic_preserving.py
│   │   └── linear_interpolation.py
│   ├── common_axis.py          # Common axis creation and management
│   ├── decision_tree.py        # Strategy selection logic
│   └── resampler.py           # Main orchestration class
```

### 2.2 Dependencies
```python
# Core dependencies
numpy >= 1.20.0
scipy >= 1.7.0
numba >= 0.54.0  # For performance-critical operations
```

## 3. Data Structures

### 3.1 Core Data Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import numpy as np

class DataType(Enum):
    """Spectrum data type classification"""
    CENTROID = "centroid"
    PROFILE = "profile"
    UNKNOWN = "unknown"

class AxisType(Enum):
    """Mass axis spacing characteristics"""
    CONSTANT = "constant"          # Equidistant spacing
    LINEAR_TOF = "linear_tof"       # sqrt(m/z) relationship
    REFLECTOR_TOF = "reflector_tof" # Linear m/z relationship
    ORBITRAP = "orbitrap"           # m/z^(3/2) relationship
    FTICR = "fticr"                 # m/z^2 relationship
    UNKNOWN = "unknown"

class ResamplingMethod(Enum):
    """Available resampling strategies"""
    NONE = "none"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TIC_PRESERVING = "tic_preserving"
    LINEAR_INTERPOLATION = "linear_interpolation"

@dataclass
class MassAxis:
    """Represents a mass axis"""
    mz_values: np.ndarray
    min_mz: float
    max_mz: float
    num_bins: int
    axis_type: AxisType

    @property
    def spacing(self) -> np.ndarray:
        """Calculate spacing between consecutive m/z values"""
        return np.diff(self.mz_values)

    @property
    def resolution_at(self, mz: float) -> float:
        """Calculate resolution at given m/z"""
        idx = np.searchsorted(self.mz_values, mz)
        if idx > 0 and idx < len(self.mz_values):
            return mz / (self.mz_values[idx] - self.mz_values[idx-1])
        return 0.0

@dataclass
class Spectrum:
    """Single mass spectrum"""
    mz: np.ndarray
    intensity: np.ndarray
    coordinates: Tuple[int, int]  # (x, y) pixel coordinates
    metadata: dict = None

    @property
    def is_centroid(self) -> bool:
        """Heuristic to detect centroid data"""
        if len(self.mz) < 100:
            return True
        # Check for zero-intensity gaps typical of centroid data
        zero_count = np.sum(self.intensity == 0)
        return zero_count / len(self.intensity) > 0.5

@dataclass
class MSIDataset:
    """Collection of spectra with metadata"""
    spectra: List[Spectrum]
    vendor: str
    instrument_type: str
    common_axis: Optional[MassAxis] = None
    resampling_applied: Optional[ResamplingMethod] = None
```

## 4. Decision Tree Implementation

### 4.1 Decision Logic

```
┌─────────────────┐
│ MSI Dataset     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Yes    ┌──────────────┐
│ Single Mass     │────────────►│ No Resampling│
│ Axis?           │             └──────────────┘
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐     Yes    ┌──────────────────┐
│ Centroid Data?  │────────────►│ Nearest Neighbor │
└────────┬────────┘             └──────────────────┘
         │ No (Profile)
         ▼
┌─────────────────┐     Yes    ┌──────────────────┐
│ Identical Axis  │────────────►│ TIC Preserving   │
│ Types?          │             └──────────────────┘
└────────┬────────┘
         │ No
         ▼
┌──────────────────────┐
│ Linear Interpolation │
└──────────────────────┘
```

### 4.2 Decision Tree Class

```python
class ResamplingDecisionTree:
    """Implements SCiLS-like decision tree for resampling strategy selection"""

    def select_strategy(self, dataset: MSIDataset) -> ResamplingMethod:
        """
        Automatically select appropriate resampling method

        Parameters
        ----------
        dataset : MSIDataset
            Input dataset to analyze

        Returns
        -------
        ResamplingMethod
            Selected resampling strategy
        """
        # Step 1: Check for single mass axis
        if self._has_single_mass_axis(dataset):
            return ResamplingMethod.NONE

        # Step 2: Check data type
        if self._is_centroid_data(dataset):
            return ResamplingMethod.NEAREST_NEIGHBOR

        # Step 3: Profile data - check axis consistency
        if self._has_identical_axis_types(dataset):
            return ResamplingMethod.TIC_PRESERVING
        else:
            return ResamplingMethod.LINEAR_INTERPOLATION
```

## 5. Component Specifications

### 5.1 Axis Detector

```python
class AxisDetector:
    """Detects and characterizes mass axes"""

    def detect_axis_type(self, mz_values: np.ndarray) -> AxisType:
        """
        Detect axis type from m/z spacing pattern

        Algorithm:
        1. Calculate spacing between consecutive m/z values
        2. Fit different models (constant, sqrt, linear, etc.)
        3. Return best-fitting model based on R² score
        """
        pass

    def detect_data_type(self, spectrum: Spectrum) -> DataType:
        """
        Determine if spectrum is centroid or profile

        Heuristics:
        - Centroid: Sparse, discrete peaks with zeros between
        - Profile: Continuous intensity values
        """
        pass

    def compare_axes(self, axis1: MassAxis, axis2: MassAxis,
                    tolerance: float = 1e-6) -> bool:
        """
        Check if two axes are identical within tolerance
        """
        pass
```

### 5.2 Common Axis Builder

```python
class CommonAxisBuilder:
    """Creates optimized common mass axis for dataset"""

    def build_from_dataset(self, dataset: MSIDataset,
                          min_mz: Optional[float] = None,
                          max_mz: Optional[float] = None,
                          num_bins: Optional[int] = None) -> MassAxis:
        """
        Build common axis from dataset characteristics

        Strategy:
        1. Find global min/max m/z across all spectra
        2. Determine optimal resolution (highest present)
        3. Create appropriate axis based on dominant instrument type
        """
        pass

    def build_uniform_axis(self, min_mz: float, max_mz: float,
                          num_bins: int) -> MassAxis:
        """Create uniform (equidistant) mass axis"""
        pass

    def build_tof_axis(self, min_mz: float, max_mz: float,
                      num_bins: int, reflector: bool = True) -> MassAxis:
        """Create TOF-specific axis with appropriate spacing"""
        pass

    def build_ft_axis(self, min_mz: float, max_mz: float,
                     num_bins: int, instrument: str = "orbitrap") -> MassAxis:
        """Create FT-based instrument axis (Orbitrap/FT-ICR)"""
        pass
```

### 5.3 Resampling Strategies

#### 5.3.1 Base Strategy

```python
from abc import ABC, abstractmethod

class ResamplingStrategy(ABC):
    """Abstract base class for resampling strategies"""

    @abstractmethod
    def resample(self, spectrum: Spectrum,
                target_axis: MassAxis) -> Spectrum:
        """
        Resample spectrum to target mass axis

        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum to resample
        target_axis : MassAxis
            Target mass axis

        Returns
        -------
        Spectrum
            Resampled spectrum
        """
        pass

    def resample_batch(self, spectra: List[Spectrum],
                       target_axis: MassAxis,
                       n_jobs: int = -1) -> List[Spectrum]:
        """
        Parallel resampling of multiple spectra

        Parameters
        ----------
        spectra : List[Spectrum]
            Input spectra
        target_axis : MassAxis
            Target mass axis
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
        """
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(
            delayed(self.resample)(spec, target_axis)
            for spec in spectra
        )
```

#### 5.3.2 Nearest Neighbor Strategy

```python
class NearestNeighborStrategy(ResamplingStrategy):
    """
    Nearest neighbor resampling for centroid data

    Best for:
    - FT-ICR data
    - Orbitrap data (post peak-picking)
    - Any centroided spectra

    Preserves:
    - Peak positions (approximately)
    - Peak intensities (exactly)
    """

    def resample(self, spectrum: Spectrum,
                target_axis: MassAxis) -> Spectrum:
        """
        Implementation using numpy searchsorted for efficiency
        """
        new_intensities = np.zeros(len(target_axis.mz_values))

        for mz, intensity in zip(spectrum.mz, spectrum.intensity):
            # Find nearest target m/z
            idx = np.searchsorted(target_axis.mz_values, mz)
            if idx > 0 and idx < len(target_axis.mz_values):
                # Choose closer neighbor
                if (mz - target_axis.mz_values[idx-1] <
                    target_axis.mz_values[idx] - mz):
                    idx -= 1
            elif idx >= len(target_axis.mz_values):
                idx = len(target_axis.mz_values) - 1

            new_intensities[idx] += intensity  # Accumulate if multiple peaks map to same bin

        return Spectrum(
            mz=target_axis.mz_values,
            intensity=new_intensities,
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata
        )
```

#### 5.3.3 TIC-Preserving Strategy

```python
class TICPreservingStrategy(ResamplingStrategy):
    """
    Resampling that preserves Total Ion Current

    Best for:
    - TOF profile data from same instrument
    - Consistent axis types

    Preserves:
    - Total ion current (area under curve)
    - Peak shapes (approximately)
    """

    def resample(self, spectrum: Spectrum,
                target_axis: MassAxis) -> Spectrum:
        """
        Linear interpolation followed by TIC normalization
        """
        from scipy.interpolate import interp1d

        # Calculate original TIC
        original_tic = np.trapz(spectrum.intensity, spectrum.mz)

        # Linear interpolation
        interpolator = interp1d(
            spectrum.mz, spectrum.intensity,
            kind='linear', bounds_error=False, fill_value=0
        )
        new_intensities = interpolator(target_axis.mz_values)

        # Preserve TIC
        new_tic = np.trapz(new_intensities, target_axis.mz_values)
        if new_tic > 0:
            new_intensities *= (original_tic / new_tic)

        return Spectrum(
            mz=target_axis.mz_values,
            intensity=new_intensities,
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata
        )
```

#### 5.3.4 Linear Interpolation Strategy

```python
class LinearInterpolationStrategy(ResamplingStrategy):
    """
    Standard linear interpolation

    Best for:
    - Mixed instrument types
    - Recalibrated data
    - Different axis types

    Most general approach but may not preserve specific properties
    """

    def resample(self, spectrum: Spectrum,
                target_axis: MassAxis) -> Spectrum:
        """
        Simple linear interpolation using scipy
        """
        from scipy.interpolate import interp1d

        interpolator = interp1d(
            spectrum.mz, spectrum.intensity,
            kind='linear', bounds_error=False, fill_value=0
        )
        new_intensities = interpolator(target_axis.mz_values)

        return Spectrum(
            mz=target_axis.mz_values,
            intensity=new_intensities,
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata
        )
```

### 5.4 Main Resampler Class

```python
class MassAxisResampler:
    """
    Main orchestrator for mass axis harmonization

    Example
    -------
    >>> resampler = MassAxisResampler()
    >>> harmonized_dataset = resampler.harmonize(dataset, auto_detect=True)
    """

    def __init__(self):
        self.detector = AxisDetector()
        self.builder = CommonAxisBuilder()
        self.decision_tree = ResamplingDecisionTree()
        self.strategies = {
            ResamplingMethod.NEAREST_NEIGHBOR: NearestNeighborStrategy(),
            ResamplingMethod.TIC_PRESERVING: TICPreservingStrategy(),
            ResamplingMethod.LINEAR_INTERPOLATION: LinearInterpolationStrategy()
        }

    def harmonize(self, dataset: MSIDataset,
                 method: Optional[ResamplingMethod] = None,
                 min_mz: Optional[float] = None,
                 max_mz: Optional[float] = None,
                 num_bins: Optional[int] = None,
                 auto_detect: bool = True,
                 n_jobs: int = -1) -> MSIDataset:
        """
        Harmonize dataset to common mass axis

        Parameters
        ----------
        dataset : MSIDataset
            Input dataset
        method : ResamplingMethod, optional
            Force specific method (overrides auto_detect)
        min_mz, max_mz : float, optional
            Mass range limits
        num_bins : int, optional
            Number of bins in target axis
        auto_detect : bool
            Use automatic method selection
        n_jobs : int
            Parallel processing (-1 for all CPUs)

        Returns
        -------
        MSIDataset
            Harmonized dataset with common mass axis
        """
        # Select resampling method
        if method is None and auto_detect:
            method = self.decision_tree.select_strategy(dataset)
        elif method is None:
            method = ResamplingMethod.LINEAR_INTERPOLATION

        # No resampling needed
        if method == ResamplingMethod.NONE:
            return dataset

        # Build common axis
        common_axis = self.builder.build_from_dataset(
            dataset, min_mz, max_mz, num_bins
        )

        # Apply resampling
        strategy = self.strategies[method]
        resampled_spectra = strategy.resample_batch(
            dataset.spectra, common_axis, n_jobs
        )

        # Create new dataset
        return MSIDataset(
            spectra=resampled_spectra,
            vendor=dataset.vendor,
            instrument_type=dataset.instrument_type,
            common_axis=common_axis,
            resampling_applied=method
        )
```

## 6. Performance Considerations

### 6.1 Memory Management
- **Streaming Processing**: Process spectra one at a time when possible
- **Memory Mapping**: Use numpy memory maps for large datasets
- **Chunking**: Process dataset in chunks for very large files

### 6.2 Computational Optimization
- **Numba JIT**: Use for inner loops in resampling algorithms
- **Vectorization**: Leverage numpy operations over Python loops
- **Parallel Processing**: Use joblib for embarrassingly parallel operations
- **Caching**: Cache common axis calculations

Example optimization with Numba:
```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def nearest_neighbor_numba(source_mz, source_intensity,
                          target_mz, target_intensity):
    """Optimized nearest neighbor resampling"""
    for i in prange(len(source_mz)):
        # Find nearest target index
        best_idx = 0
        best_dist = abs(source_mz[i] - target_mz[0])

        for j in range(1, len(target_mz)):
            dist = abs(source_mz[i] - target_mz[j])
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            elif dist > best_dist:
                break  # Early stopping for sorted arrays

        target_intensity[best_idx] += source_intensity[i]
```

## 7. Configuration

### 7.1 User Configuration Options

```python
@dataclass
class ResamplingConfig:
    """User-configurable resampling parameters"""

    # Automatic detection
    auto_detect: bool = True
    force_method: Optional[ResamplingMethod] = None

    # Mass range
    min_mz: Optional[float] = None
    max_mz: Optional[float] = None
    num_bins: Optional[int] = None

    # Performance
    n_jobs: int = -1
    chunk_size: int = 1000
    use_numba: bool = True

    # Tolerances
    axis_tolerance: float = 1e-6
    centroid_threshold: float = 0.5

    # Preservation options
    preserve_tic: bool = True
    preserve_original: bool = False  # Keep copy of original data
```

## 8. Vendor-Specific Considerations

### 8.1 Bruker Instruments (Focus on GlobalMetadata)

| InstrumentName (from GlobalMetadata) | Method | Implementation Notes |
|-------------------------------------|--------|----------------------|
| "timsTOF Maldi 2" | Nearest Neighbor | Centroid data, discrete peaks |
| "solarix" | Nearest Neighbor | FT-ICR, very high resolution |
| Contains "fticr" | Nearest Neighbor | FT-ICR family instruments |
| Default/Others | Linear TIC Preserving | Profile data, preserve peak shapes |

### 8.2 Implementation Strategy

```python
# Simple instrument detection based on GlobalMetadata
def detect_bruker_instrument(metadata: ComprehensiveMetadata) -> ResamplingMethod:
    instrument_info = metadata.instrument_info

    if "InstrumentName" in instrument_info:
        name = instrument_info["InstrumentName"].lower()

        if "timstof" in name:
            return ResamplingMethod.NEAREST_NEIGHBOR
        elif "solarix" in name or "fticr" in name:
            return ResamplingMethod.NEAREST_NEIGHBOR

    # Default for other Bruker instruments (rapifleX, autoflex, etc.)
    return ResamplingMethod.LINEAR_TIC_PRESERVING
```

## 9. Error Handling

### 9.1 Common Issues and Solutions

```python
class ResamplingError(Exception):
    """Base exception for resampling errors"""
    pass

class IncompatibleAxesError(ResamplingError):
    """Raised when axes cannot be harmonized"""
    pass

class InvalidDataError(ResamplingError):
    """Raised when data format is invalid"""
    pass

def validate_dataset(dataset: MSIDataset):
    """
    Validate dataset before resampling

    Checks:
    - Non-empty spectra
    - Valid m/z ranges
    - Matching array dimensions
    - Positive intensities
    """
    if not dataset.spectra:
        raise InvalidDataError("Dataset contains no spectra")

    for spec in dataset.spectra:
        if len(spec.mz) != len(spec.intensity):
            raise InvalidDataError("Mismatched m/z and intensity arrays")
        if np.any(spec.mz < 0):
            raise InvalidDataError("Negative m/z values detected")
```

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# test_resampling.py

def test_single_axis_detection():
    """Test that identical axes are correctly identified"""
    pass

def test_centroid_detection():
    """Test centroid vs profile classification"""
    pass

def test_tic_preservation():
    """Verify TIC is preserved within tolerance"""
    pass

def test_nearest_neighbor_accuracy():
    """Test nearest neighbor assigns to correct bins"""
    pass
```

### 10.2 Integration Tests

```python
def test_bruker_tof_workflow():
    """End-to-end test with Bruker TOF data"""
    pass

def test_mixed_vendor_harmonization():
    """Test harmonizing data from multiple vendors"""
    pass

def test_large_dataset_performance():
    """Benchmark performance on realistic dataset sizes"""
    pass
```

## 11. Usage Examples

### 11.1 Integration with MSIConverter

```python
# Example: Add resampling to SpatialDataConverter
from msiconvert.resampling import SimpleResampler, ResamplingConfig
from msiconvert.converters.spatialdata_converter import SpatialDataConverter

class InterpolatedSpatialDataConverter(SpatialDataConverter):
    """SpatialData converter with interpolation support"""

    def __init__(self, reader, output_path, **kwargs):
        super().__init__(reader, output_path, **kwargs)

        # Initialize resampling components
        self.resampler = SimpleResampler()
        self.resampling_config = ResamplingConfig(
            target_bins=kwargs.get('interpolation_bins', 10000)
        )

    def _initialize_conversion(self):
        """Extended initialization with resampling setup"""
        super()._initialize_conversion()

        # Set up target mass axis
        self._target_axis = self.resampler.create_target_axis(
            self.reader, self.resampling_config
        )

        # Select resampling method based on instrument
        self._resampling_method = self.resampler.select_method(
            self.reader, self.resampling_config
        )

    def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
        """Process spectrum with interpolation"""
        # Apply resampling
        interpolated_intensity = self.resampler.resample_spectrum(
            mzs, intensities, self._target_axis, self._resampling_method
        )

        # Use interpolated data instead of original
        super()._process_single_spectrum(
            data_structures, coords, self._target_axis, interpolated_intensity
        )
```

### 11.2 Command Line Integration

```python
# Add interpolation options to CLI
def add_interpolation_args(parser):
    """Add interpolation arguments to CLI parser"""
    parser.add_argument('--interpolate', action='store_true',
                       help='Enable mass axis interpolation')
    parser.add_argument('--interpolation-bins', type=int, default=10000,
                       help='Number of target mass bins')
    parser.add_argument('--interpolation-method',
                       choices=['nearest_neighbor', 'linear_tic_preserving'],
                       help='Force specific interpolation method')
```

### 11.3 Simple Standalone Usage

```python
# Basic resampling outside of converter
from msiconvert.readers.bruker_reader import BrukerReader
from msiconvert.resampling import SimpleResampler, ResamplingConfig

reader = BrukerReader("data/sample.d")
resampler = SimpleResampler()
config = ResamplingConfig(target_bins=5000)

# Create target axis
target_axis = resampler.create_target_axis(reader, config)

# Select method based on instrument
method = resampler.select_method(reader, config)

# Process individual spectra
for coords, mzs, intensities in reader.iter_spectra():
    resampled = resampler.resample_spectrum(mzs, intensities, target_axis, method)
    # Use resampled data...
```

## 12. Implementation Phases

### 12.1 Phase 1: Basic Prototype
- [x] Simple instrument detection from GlobalMetadata
- [x] Linear mass axis generation
- [x] Linear TIC-preserving interpolation (modern scipy)
- [x] Nearest neighbor interpolation
- [x] Integration with existing MSIConverter architecture
- [ ] Basic CLI integration
- [ ] Unit tests for core interpolators

### 12.2 Phase 2: Enhanced Features (Future)
- [ ] More sophisticated instrument detection
- [ ] Non-linear mass axis generation (TOF-specific, FT-specific)
- [ ] Parallel processing integration
- [ ] Memory optimization for large datasets
- [ ] Quality metrics and validation
- [ ] Advanced configuration options

### 12.3 Phase 3: Research Extensions (Future)
- [ ] Adaptive binning strategies
- [ ] Machine learning for method selection
- [ ] Ion mobility dimension handling
- [ ] Isotope pattern preservation
- [ ] GPU acceleration

### 12.4 Implementation Priority
1. **Get prototype working**: Focus on basic functionality first
2. **Validate with real data**: Test with timsTOF and other instruments  
3. **Optimize incrementally**: Add performance improvements after validation
4. **Expand capabilities**: Add advanced features based on user needs
