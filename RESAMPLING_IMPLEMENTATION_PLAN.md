# Mass Axis Resampling Implementation Plan

## Overview

This document outlines the implementation plan for adding mass axis resampling/interpolation functionality to MSIConverter. The feature will enable harmonizing spectra from different pixels to a common mass axis, which is essential for consistent analysis across MSI datasets.

## Goals

- Add instrument-specific interpolation methods based on metadata
- Support timsTOF (nearest neighbor) and profile data (linear TIC-preserving)
- Integrate seamlessly with existing MSIConverter architecture
- Maintain compatibility with current CLI and converter interfaces
- Provide simple, prototype-first approach that can be extended later

## Architecture Overview

```
msiconvert/
├── resampling/
│   ├── __init__.py
│   ├── strategies/             # Resampling strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base strategy
│   │   ├── nearest_neighbor.py
│   │   ├── tic_preserving.py
│   │   └── linear_interpolation.py
│   ├── mass_axis/             # Mass axis generation for different analyzers
│   │   ├── __init__.py
│   │   ├── base_generator.py  # Abstract axis generator
│   │   ├── linear_generator.py    # Simple linear spacing
│   │   ├── tof_generator.py       # TOF-specific non-linear (sqrt/linear)
│   │   ├── fticr_generator.py     # FT-ICR non-linear spacing
│   │   └── orbitrap_generator.py  # Orbitrap non-linear spacing
│   ├── common_axis.py          # Common axis creation and management
│   ├── decision_tree.py        # Strategy selection logic
│   └── resampler.py           # Main orchestration class
```

## Implementation Phases

### Phase 1: Basic Module Structure and Enums
**Duration:** 1-2 hours  
**Priority:** High  
**Dependencies:** None

#### Tasks
- [ ] Create `msi_processing/resampling/` directory structure  
- [ ] Create `strategies/` directory with strategy implementations
- [ ] Create `common_axis.py` for axis creation and management
- [ ] Create `decision_tree.py` for strategy selection logic
- [ ] Create `resampler.py` as main orchestration class
- [ ] Set up module `__init__.py` with proper exports

#### Files to Create
```python
# msi_processing/resampling/__init__.py
from .resampler import MassAxisResampler
from .decision_tree import ResamplingDecisionTree  
from .common_axis import CommonAxisBuilder

# msi_processing/resampling/strategies/__init__.py
from .base import ResamplingStrategy
from .nearest_neighbor import NearestNeighborStrategy
from .tic_preserving import TICPreservingStrategy
from .linear_interpolation import LinearInterpolationStrategy
```

#### Acceptance Criteria
- [ ] Module imports correctly with all strategy classes
- [ ] Directory structure matches architecture overview
- [ ] All base classes and enums are properly defined
- [ ] No import errors in existing codebase

---

### Phase 2: Core Resampling Strategies
**Duration:** 4-6 hours  
**Priority:** High  
**Dependencies:** Phase 1

#### Tasks
- [ ] Implement `NearestNeighborStrategy` in `strategies/nearest_neighbor.py`
- [ ] Implement `TICPreservingStrategy` in `strategies/tic_preserving.py` using `scipy.interpolate.interp1d`
- [ ] Create abstract `ResamplingStrategy` base class in `strategies/base.py`
- [ ] Add comprehensive error handling for edge cases

#### Files to Create

```python
# msiconvert/resampling/base_interpolator.py
from abc import ABC, abstractmethod
import numpy as np

class BaseInterpolator(ABC):
    @abstractmethod
    def interpolate(self, source_mz: np.ndarray, source_intensity: np.ndarray,
                   target_mz: np.ndarray) -> np.ndarray:
        pass
```

```python
# msiconvert/resampling/nearest_neighbor.py
class NearestNeighborInterpolator(BaseInterpolator):
    def interpolate(self, source_mz, source_intensity, target_mz):
        # Implementation for discrete peak assignment
        pass
```

```python
# msiconvert/resampling/linear_tic_preserving.py
from scipy.interpolate import interp1d

class LinearTicPreservingInterpolator(BaseInterpolator):
    def interpolate(self, source_mz, source_intensity, target_mz):
        # Linear interpolation with TIC preservation
        pass
```

#### Acceptance Criteria
- [ ] Both interpolators handle empty/single-point spectra gracefully
- [ ] TIC preservation maintains <1% error for test cases
- [ ] Performance acceptable for typical spectrum sizes (10k-100k points)
- [ ] No memory leaks or excessive allocations

#### Test Cases to Validate
- Empty spectrum (0 points)
- Single peak spectrum (1 point)
- Small spectrum (10 points)
- Large spectrum (100k points)
- TIC preservation accuracy
- Boundary conditions (peaks outside target range)

---

### Phase 3: Instrument-Based Method Selection
**Duration:** 3-4 hours  
**Priority:** High  
**Dependencies:** Phase 2

#### Tasks
- [ ] Create `InstrumentBasedSelector` class
- [ ] Extract instrument name from Bruker `GlobalMetadata`
- [ ] Implement mapping logic: timsTOF → nearest neighbor, others → linear TIC-preserving
- [ ] Handle missing or unknown instrument names gracefully

#### Files to Create
```python
# msiconvert/resampling/instrument_selector.py
from ..metadata.types import ComprehensiveMetadata
from .config import ResamplingMethod

class InstrumentBasedSelector:
    INSTRUMENT_METHODS = {
        "timstof": ResamplingMethod.NEAREST_NEIGHBOR,
        "fticr": ResamplingMethod.NEAREST_NEIGHBOR,
        "solarix": ResamplingMethod.NEAREST_NEIGHBOR,
    }

    def select_method(self, metadata: ComprehensiveMetadata) -> ResamplingMethod:
        # Implementation for metadata-based selection
        pass
```

#### Acceptance Criteria
- [ ] Correctly identifies timsTOF instruments from metadata
- [ ] Falls back to linear TIC-preserving for unknown instruments
- [ ] Handles missing InstrumentName gracefully
- [ ] Works with existing metadata extraction pipeline

#### Test Cases
- Bruker timsTOF data with InstrumentName="timsTOF Maldi 2"
- Bruker FT-ICR data with InstrumentName="solarix"
- Bruker data with missing InstrumentName
- Non-Bruker data (should default to linear TIC-preserving)

---

### Phase 4: Mass Analyzer-Specific Axis Generators
**Duration:** 4-6 hours  
**Priority:** High  
**Dependencies:** None (can run in parallel with Phase 2-3)

#### Tasks
- [ ] Create abstract `BaseAxisGenerator` with common interface supporting both approaches
- [ ] Implement `LinearAxisGenerator` for simple linear spacing (default)
- [ ] Implement `TOFAxisGenerator` with proper TOF resolution equations
- [ ] Implement `FTICRAxisGenerator` with constant resolving power (R ∝ m/z)
- [ ] Implement `OrbitrapAxisGenerator` with Orbitrap resolution scaling (R ∝ sqrt(m/z))
- [ ] Add physics-based axis generation from mass width at reference m/z
- [ ] Add instrument detection logic to select appropriate generator

#### Files to Create
```python
# msi_processing/resampling/mass_axis/base_generator.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseAxisGenerator(ABC):
    @abstractmethod
    def generate_axis_bins(self, min_mz: float, max_mz: float, num_bins: int) -> np.ndarray:
        """Generate axis with fixed number of bins"""
        pass

    @abstractmethod  
    def generate_axis_width(self, min_mz: float, max_mz: float,
                           width_da: float, reference_mz: float = 500.0) -> np.ndarray:
        """Generate axis based on mass width at reference m/z using analyzer physics"""
        pass

# msi_processing/resampling/mass_axis/tof_generator.py
class TOFAxisGenerator(BaseAxisGenerator):
    def __init__(self, reflector_mode: bool = True):
        self.reflector_mode = reflector_mode

    def generate_axis_bins(self, min_mz: float, max_mz: float, num_bins: int) -> np.ndarray:
        if self.reflector_mode:
            return np.linspace(min_mz, max_mz, num_bins)  # Linear m/z
        else:
            # Linear TOF: constant time resolution -> sqrt(m/z) spacing
            pass

    def generate_axis_width(self, min_mz: float, max_mz: float,
                           width_da: float, reference_mz: float = 500.0) -> np.ndarray:
        """
        TOF physics: For reflector mode, resolution R = m/Δm is approximately constant
        So mass width scales linearly with m/z: width(mz) = width_ref * (mz/mz_ref)
        """
        if self.reflector_mode:
            # Constant resolving power: R = reference_mz / width_da
            resolving_power = reference_mz / width_da
            # Generate axis where width(mz) = mz / resolving_power
            pass
        else:
            # Linear TOF: resolution scales as sqrt(m/z)
            pass

# msi_processing/resampling/mass_axis/fticr_generator.py  
class FTICRAxisGenerator(BaseAxisGenerator):
    def generate_axis_width(self, min_mz: float, max_mz: float,
                           width_da: float, reference_mz: float = 500.0) -> np.ndarray:
        """
        FT-ICR physics: Constant resolving power R = m/Δm
        Mass width is constant across all m/z for constant resolving power
        """
        resolving_power = reference_mz / width_da
        # Generate logarithmic spacing: each bin has width = mz / resolving_power
        pass

# msi_processing/resampling/mass_axis/orbitrap_generator.py
class OrbitrapAxisGenerator(BaseAxisGenerator):
    def generate_axis_width(self, min_mz: float, max_mz: float,
                           width_da: float, reference_mz: float = 500.0) -> np.ndarray:
        """
        Orbitrap physics: Resolution R ∝ sqrt(m/z)
        Mass width increases as sqrt(m/z): width(mz) = width_ref * sqrt(mz/mz_ref)
        """
        # Calculate reference resolving power
        reference_resolving_power = reference_mz / width_da
        # Generate axis where resolution decreases as 1/sqrt(m/z)
        pass
```

#### Acceptance Criteria
- [ ] Each generator supports both fixed-bins and physics-based width approaches
- [ ] TOF generator handles both reflector (constant R) and linear (R ∝ sqrt(m/z)) modes
- [ ] FT-ICR generator implements constant resolving power (logarithmic spacing)
- [ ] Orbitrap generator implements resolution scaling as R ∝ sqrt(m/z)
- [ ] Physics-based generation respects actual instrument resolution characteristics
- [ ] CLI arguments for bins vs. width are mutually exclusive with proper validation
- [ ] Default reference m/z of 500.0 is reasonable for typical MS analysis
- [ ] All generators validate input parameters and handle edge cases
- [ ] Integration with instrument detection from metadata

---

### Phase 5: Main Resampler Class
**Duration:** 3-4 hours  
**Priority:** High  
**Dependencies:** Phases 2, 3, 4

#### Tasks
- [ ] Create `SimpleResampler` orchestration class
- [ ] Wire together all components (selector, generator, interpolators)
- [ ] Implement main workflow methods
- [ ] Add comprehensive logging and error handling

#### Files to Create
```python
# msiconvert/resampling/simple_resampler.py
class SimpleResampler:
    def __init__(self):
        # Initialize all components
        pass

    def create_target_axis(self, reader, config) -> np.ndarray:
        pass

    def select_method(self, reader, config) -> ResamplingMethod:
        pass

    def resample_spectrum(self, source_mz, source_intensity,
                         target_axis, method) -> np.ndarray:
        pass
```

#### Acceptance Criteria
- [ ] Successfully orchestrates end-to-end resampling workflow
- [ ] Handles all supported interpolation methods
- [ ] Provides clear error messages for failures
- [ ] Integrates with existing MSIConverter logging

---

### Phase 6: CLI Integration
**Duration:** 2-3 hours  
**Priority:** Medium  
**Dependencies:** Phase 5

#### Tasks
- [ ] Add CLI arguments for resampling options
- [ ] Update argument parsing in `__main__.py`
- [ ] Pass resampling configuration through to converters
- [ ] Update help documentation

#### Files to Modify
- `msiconvert/__main__.py` - Add argument parsing
- `msiconvert/convert.py` - Pass config to converters

#### New CLI Arguments
```bash
--resample                       # Enable mass axis resampling
--resample-method <choice>      # Override method selection

# Mass axis generation (mutually exclusive):
--resample-bins <int>           # Number of target bins (simple approach)
--resample-width <float>        # Mass width in Da at reference m/z (physics-based)
--resample-reference-mz <float> # Reference m/z for width specification (default: 500.0)
```

#### Acceptance Criteria
- [ ] CLI arguments parse correctly with proper mutual exclusivity
- [ ] Help text clearly explains physics-based vs. simple binning approaches
- [ ] `--resample-bins` and `--resample-width` are mutually exclusive
- [ ] `--resample-reference-mz` only applies when using `--resample-width`
- [ ] Default values are sensible (500.0 Da for reference m/z)
- [ ] Error messages are helpful when conflicting arguments are provided
- [ ] Arguments are properly validated (positive values, reasonable ranges)
- [ ] Backward compatibility maintained

---

### Phase 7: Reader Integration (Simplified Approach)
**Duration:** 3-4 hours  
**Priority:** High  
**Dependencies:** Phases 5, 6

#### Tasks
- [ ] Create `ResamplingReader` wrapper that applies resampling to any BaseMSIReader
- [ ] Override `get_common_mass_axis()` to return resampled axis
- [ ] Override `iter_spectra()` to yield resampled spectra
- [ ] Maintain full compatibility with existing converter interface
- [ ] No need to modify existing converters - they just receive resampled data

#### Files to Create
```python
# msi_processing/resampling/resampling_reader.py
from msiconvert.core.base_reader import BaseMSIReader
from .resampler import MassAxisResampler

class ResamplingReader(BaseMSIReader):
    """Wrapper that applies resampling to any MSI reader"""

    def __init__(self, base_reader: BaseMSIReader, resampling_config: dict):
        self.base_reader = base_reader
        self.resampler = MassAxisResampler()
        self.resampling_config = resampling_config
        self._resampled_axis = None

    def get_common_mass_axis(self) -> np.ndarray:
        # Return the resampled mass axis instead of original
        if self._resampled_axis is None:
            self._resampled_axis = self.resampler.create_target_axis(
                self.base_reader, self.resampling_config
            )
        return self._resampled_axis

    def iter_spectra(self, batch_size=None):
        # Yield resampled spectra instead of original
        target_axis = self.get_common_mass_axis()
        method = self.resampler.select_method(self.base_reader, self.resampling_config)

        for coords, mzs, intensities in self.base_reader.iter_spectra(batch_size):
            resampled_intensities = self.resampler.resample_spectrum(
                mzs, intensities, target_axis, method
            )
            yield coords, target_axis, resampled_intensities
```

#### Integration with CLI
```python
# In msiconvert/convert.py - modify reader creation
def create_reader(input_path, resampling_config=None):
    base_reader = # ... existing reader creation logic

    if resampling_config and resampling_config.get('enable_resampling'):
        return ResamplingReader(base_reader, resampling_config)
    else:
        return base_reader

# Example CLI integration with mutual exclusivity validation
def main():
    parser.add_argument('--resample', action='store_true',
                       help='Enable mass axis resampling')
    parser.add_argument('--resample-method',
                       choices=['nearest_neighbor', 'tic_preserving'],
                       help='Override resampling method selection')

    # Mutually exclusive group for axis generation
    axis_group = parser.add_mutually_exclusive_group()
    axis_group.add_argument('--resample-bins', type=int,
                           help='Number of target mass bins (simple approach)')
    axis_group.add_argument('--resample-width', type=float,
                           help='Mass width in Da at reference m/z (physics-based)')
    parser.add_argument('--resample-reference-mz', type=float, default=500.0,
                       help='Reference m/z for width specification (default: 500.0)')

# Example usage:
# Physics-based (SCiLS-style):
#   msiconvert data.d output.zarr spatialdata --resample --resample-width 0.1
# Simple bins approach:
#   msiconvert data.d output.zarr spatialdata --resample --resample-bins 10000
```

#### Acceptance Criteria
- [ ] **No changes needed to existing converters** - they work transparently
- [ ] Resampled data flows through normal conversion pipeline
- [ ] All existing converter features work (metadata, 3D handling, etc.)
- [ ] Performance overhead only when resampling is enabled
- [ ] CLI arguments use consistent "resample" terminology
- [ ] Memory usage reasonable - no double storage of spectral data

---

### Phase 8: Testing and Validation
**Duration:** 6-8 hours  
**Priority:** High  
**Dependencies:** All previous phases

#### Tasks
- [ ] Create comprehensive unit test suite
- [ ] Add integration tests with real data
- [ ] Validate TIC preservation accuracy
- [ ] Performance benchmarking
- [ ] CLI testing

#### Files to Create
```
tests/unit/resampling/
├── __init__.py
├── test_decision_tree.py
├── test_common_axis.py
├── test_resampler.py
├── test_resampling_reader.py
├── strategies/
│   ├── test_nearest_neighbor.py
│   ├── test_tic_preserving.py
│   └── test_linear_interpolation.py
├── mass_axis/
│   ├── test_linear_generator.py
│   ├── test_tof_generator.py
│   ├── test_fticr_generator.py
│   └── test_orbitrap_generator.py
└── test_integration.py
```

#### Test Coverage Targets
- [ ] Unit tests: >90% code coverage
- [ ] Integration tests with real timsTOF and Bruker profile data
- [ ] CLI argument validation
- [ ] Error condition handling

#### Performance Benchmarks
- [ ] Interpolation time for 10k spectrum: <1ms
- [ ] Interpolation time for 100k spectrum: <10ms
- [ ] Memory usage linear with spectrum size
- [ ] TIC preservation error: <1%

---

## Quick Start Approach (Minimal Viable Product)

For fastest prototyping with mass analyzer-specific axes, implement in this order:

1. **Phase 1 (1-2 hours):** Basic module structure and strategy pattern setup
2. **Phase 4 (4-6 hours):** Mass analyzer-specific axis generators - **this is critical for quality**
3. **Phase 2 (4-6 hours):** Core resampling strategies
4. **Mini Phase 5 (2 hours):** Basic resampler with hardcoded method selection
5. **Manual testing (1 hour):** Validate with real timsTOF and profile data
6. **Phase 3 (3-4 hours):** Add instrument detection and decision tree
7. **Phase 7 (3-4 hours):** Reader wrapper integration (much simpler than new converter)

**Total MVP time: 18-25 hours**

**Key insight:** Starting with proper mass axis generation (Phase 4) early ensures the resampling produces scientifically accurate results from the beginning.

## Implementation Guidelines

### Code Standards
- Follow existing MSIConverter code patterns and style
- Use type hints throughout
- Comprehensive docstrings with examples
- Error handling with informative messages
- Logging for debugging and progress tracking

### Performance Considerations
- Use NumPy vectorized operations where possible
- Avoid copying large arrays unnecessarily
- Consider memory usage for large datasets
- Profile critical paths and optimize bottlenecks

### Testing Strategy
- Write tests before implementation (TDD approach)
- Test edge cases and error conditions
- Use real MSI data for integration testing
- Validate scientific correctness (TIC preservation)

## Success Metrics

### Functional Requirements
- [ ] timsTOF data uses nearest neighbor resampling
- [ ] Profile data uses linear TIC-preserving resampling
- [ ] TIC preservation accuracy <1% error
- [ ] CLI integration works seamlessly with `--resample` arguments
- [ ] Output compatible with existing SpatialData format

### Performance Requirements
- [ ] Resampling adds <20% overhead to conversion time
- [ ] Memory usage scales linearly with spectrum size
- [ ] Handles datasets up to 10GB without issues

### Quality Requirements
- [ ] >90% test coverage
- [ ] No memory leaks
- [ ] Clear error messages and handling
- [ ] Comprehensive documentation

## Future Enhancements (Post-MVP)

### Phase 9: Advanced Features (Future)
- Non-linear mass axis generation (TOF-specific, FT-specific)
- Parallel processing for multiple spectra
- Advanced interpolation methods (spline, cubic)
- Quality metrics and validation reports

### Phase 10: Optimization (Future)
- Numba JIT compilation for performance-critical loops
- Memory mapping for very large datasets
- Adaptive binning strategies
- GPU acceleration exploration

## Collaboration Notes

### Git Workflow
- Use feature branch: `feature/mass-axis-resampling`
- Create PRs for each phase for review
- Maintain clean commit history with descriptive messages

### Division of Work
- **Developer A:** Phases 1-2 (Module structure and core interpolators)
- **Developer B:** Phases 3-4 (Instrument detection and axis generation)  
- **Both:** Phase 5 (Integration and orchestration)
- **Developer A:** Phase 6-7 (CLI and converter integration)
- **Developer B:** Phase 8 (Testing and validation)

### Communication
- Daily standup to coordinate progress
- Document any architecture decisions or changes
- Share test datasets and validation results
- Code review all phases before merging

---

## Getting Started

1. **Create feature branch:** `git checkout -b feature/mass-axis-resampling`
2. **Set up development environment:** Ensure scipy is available
3. **Start with Phase 1:** Create basic module structure
4. **Test early and often:** Validate each component individually
5. **Document as you go:** Keep this plan updated with actual progress

## Questions and Decisions

### Open Questions
- Should we support other interpolation methods beyond linear and nearest neighbor?
- How should we handle datasets with mixed instrument types?
- What's the appropriate default number of mass bins (currently 10000)?

### Decisions Made
- Use linear interpolation instead of PCHIP for simplicity
- Focus on Bruker instruments initially (can extend to other vendors later)
- Implement physics-based mass axis generation following SCiLS approach
- Support both fixed bins (simple) and mass width at reference m/z (physics-based)
- Use ResamplingReader wrapper instead of creating new converters
- Default reference m/z of 500.0 Da (typical for small molecule and peptide analysis)

### Typical Usage Patterns
- **Research/exploration**: `--resample-bins 10000` (simple, fast)
- **Publication/analysis**: `--resample-width 0.1` (physics-based, accurate)
- **High-resolution instruments**: `--resample-width 0.01 --resample-reference-mz 1000`
- **Low-resolution instruments**: `--resample-width 0.5 --resample-reference-mz 300`

---

*This plan should be updated as implementation progresses to reflect actual progress, discoveries, and any necessary changes to approach or timeline.*
