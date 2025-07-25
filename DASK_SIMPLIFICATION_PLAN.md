# Bruker Reader Simplification for Dask-Based Interpolation

## Executive Summary

This plan simplifies the Bruker reader to support efficient dask-based interpolation workflows. The core philosophy: **let dask handle optimization, make the reader as simple and reliable as possible**.

## Current State vs. Dask Workflow Vision

### Current Complex Architecture
```
BrukerReader
├── MassAxisBuilder (391 lines) - 3 strategies for memory optimization
├── MemoryManager (392 lines) - Buffer pools, memory monitoring
├── BatchProcessor (434 lines) - Batch optimization for processing
├── CoordinateCache (428 lines) - Coordinate caching and preloading
└── SDK + Core reading logic
```

### Simplified Dask-Ready Architecture  
```
BrukerReader
├── Simple mass axis builder (~30 lines) - Basic unique mass collection
├── SDK + Core reading logic - Reliable frame-by-frame reading
└── Minimal coordinate handling - Sequential access only
```

### Dask Workflow Vision
```
Reader (Sequential)          Dask Cluster (Parallel)
├── Read frame 1      ──────► Worker 1: Interpolate pixel 1
├── Read frame 2      ──────► Worker 2: Interpolate pixel 2  
├── Read frame 3      ──────► Worker 3: Interpolate pixel 3
└── ...               ──────► Worker N: Interpolate pixel N...
```

## Components Analysis & Decisions

### 🗑️ REMOVE: BatchProcessor (434 lines)

**Reasoning:**
- Dask will handle batching and parallel processing optimally
- Reader batching adds unnecessary complexity and potential failure points
- Sequential reading is simpler and more reliable
- Dask workers can process pixels in parallel regardless of how reader feeds data

**Current Usage:** Used in `bruker_reader.py` for spectrum processing optimization

### 🗑️ REMOVE: MemoryManager (392 lines)

**Reasoning:**
- Dask workers manage their own memory efficiently
- Buffer pooling adds thread-safety overhead for single-threaded reader
- Reader memory usage is predictable and bounded (one frame at a time)
- psutil dependency and complex monitoring unnecessary for simple sequential reading
- Memory limits should be handled at dask cluster level, not reader level

**Current Usage:** Buffer management and memory monitoring

### 🗑️ REMOVE: MassAxisBuilder Complexity (350+ lines → ~30 lines)

**Reasoning:**
- **Future interpolation module** will create its own optimized mass axis using min/max mass + bin width for non-linear mass-dependent bins
- No need for 3 different strategies when result gets replaced by calculated axis
- Simple set-based collection sufficient for raw mass axis in case user wants original data (not recommended)
- Eliminates complex auto-strategy selection logic

**Simplified Implementation:**
```python
def build_raw_mass_axis(spectra_iterator):
    """Raw Mass axis in case the user wants the full data. Not recommended to use"""
    unique_mzs = set()
    for coords, mzs, intensities in spectra_iterator:
        if mzs.size > 0:
            unique_mzs.update(mzs)
    return np.array(sorted(unique_mzs))
```

### 🗑️ REMOVE: CoordinateCache (428 lines) - **I agree with your assessment**

**Reasoning:**
- **Sequential access pattern**: Going through every frame anyway for dask processing
- **No random access needed**: Reader feeds frames sequentially to dask
- **Single-pass processing**: Each frame read once, coordinates extracted once
- **Caching overhead unnecessary**: No benefit from preloading when accessing sequentially
- **Eliminates complexity**: Database caching, preloading logic, cache invalidation

**Alternative:** Simple coordinate extraction per frame as needed

### ✅ KEEP: SDK Integration & Core Reading

**Reasoning:**
- Essential for actual data access
- Proven reliable in current implementation
- SDK complexity unavoidable (Bruker's interface)

## Future Interpolation Context

**Separate Interpolation Module (Future):**
- Will use **min mass, max mass, and specific bin width** to calculate non-linear mass-dependent bins
- **No raw mass data needed** for interpolation calculations
- Raw mass axis (`build_raw_mass_axis`) only for users who explicitly want original data
- Interpolation module will be completely separate from reader for clean architecture

## Implementation Plan

### Phase 1: Remove BatchProcessor & MemoryManager (Week 1)

**Step 1.1: Remove BatchProcessor Integration**
```python
# In bruker_reader.py, replace:
# self.batch_processor = BatchProcessor(...)
# results = self.batch_processor.process_spectra(...)

# With simple iteration:
for coords, mzs, intensities in self._iter_spectra_raw():
    # Direct processing
    yield coords, mzs, intensities
```

**Step 1.2: Remove MemoryManager Integration**
```python
# Remove: self.memory_manager = MemoryManager(...)
# Remove: memory checks, buffer pooling calls
# Keep: Simple numpy array creation as needed
```

**Step 1.3: Clean Up Imports & Files**
- Delete `utils/batch_processor.py`
- Delete `utils/memory_manager.py`
- Update imports in `bruker_reader.py`

### Phase 2: Simplify MassAxisBuilder (Week 1)

**Step 2.1: Replace Complex Builder**
```python
# Replace utils/mass_axis_builder.py (391 lines) with:
def build_raw_mass_axis(spectra_iterator, progress_callback=None):
    unique_mzs = set()
    count = 0
    
    for coords, mzs, intensities in spectra_iterator:
        if mzs.size > 0:
            unique_mzs.update(mzs)
        count += 1
        if progress_callback and count % 100 == 0:
            progress_callback(count)
    
    return np.array(sorted(unique_mzs))
```

**Step 2.2: Update Integration**
- Replace `self.mass_axis_builder.build_from_spectra_iterator()` calls with `build_raw_mass_axis()`
- Simplify progress reporting
- Add docstring warning that raw mass axis is not recommended for normal use

### Phase 3: Remove CoordinateCache (Week 2)

**Step 3.1: Replace Cached Coordinate Access**
```python
# Replace: self.coordinate_cache.get_coordinates(frame_id)
# With: Direct coordinate extraction during frame iteration

def _get_frame_coordinates(self, frame_id):
    """Extract coordinates directly from frame data."""
    # Simple coordinate extraction without caching
    return self.sdk.get_frame_coordinates(frame_id)
```

**Step 3.2: Update Frame Iteration**
- Remove coordinate cache initialization
- Extract coordinates on-demand during frame processing
- Delete `utils/coordinate_cache.py`

### Phase 4: Testing & Validation (Week 2)

**Step 4.1: Functional Testing**
- Ensure all frames still readable
- Verify coordinate extraction accuracy
- Validate mass axis correctness (even if temporary)

**Step 4.2: Performance Testing**
- Compare reading speed (should be similar or faster)
- Measure memory usage (should be much lower)
- Test with various dataset sizes

**Step 4.3: Integration Testing**
- Ensure compatibility with existing conversion pipeline
- Prepare for dask integration testing

## Expected Benefits

### Code Simplification
- **~1,600+ lines removed** from utils directory
- **~75% reduction** in Bruker reader complexity
- **Eliminates** 4 major components with complex interdependencies

### Reliability Improvements
- **Fewer failure points** - less code means less can go wrong
- **Simpler error handling** - no complex memory/batch/cache errors
- **Predictable behavior** - straightforward sequential processing

### Dask Integration Benefits
- **Clean separation** - reader focuses on frame/pixel reading, dask handles pixel-by-pixel interpolation
- **Optimal resource usage** - dask manages memory/parallelization efficiently for embarrassingly parallel interpolation
- **Scalable architecture** - dask can handle any dataset size with pixel-level parallelization
- **Debuggability** - easier to isolate reader vs interpolation issues
- **Future-ready** - prepared for separate interpolation module with calculated mass axes

### Performance Expectations
- **Reading speed**: Same or faster (less overhead)
- **Memory usage**: Significantly lower (no caching/pooling)
- **Overall throughput**: Much higher (dask parallelization)

## Risk Assessment & Mitigation

### Low Risk
- **MassAxisBuilder simplification**: Result gets replaced by separate interpolation module anyway
- **BatchProcessor removal**: Dask handles pixel-by-pixel processing optimally
- **MemoryManager removal**: Unnecessary for sequential frame/pixel reading

### Medium Risk  
- **CoordinateCache removal**: Could impact performance if coordinates accessed multiple times
- **Mitigation**: Profile coordinate access patterns before removal

### Risk Mitigation Strategy
1. **Gradual implementation**: One component at a time
2. **Performance benchmarking**: Before/after comparisons
3. **Rollback capability**: Keep old code temporarily
4. **Real dataset testing**: Test with actual user data

## Success Metrics

- [ ] **Code size**: >75% reduction in utils directory
- [ ] **Memory usage**: <50% of current reader memory footprint
- [ ] **Reading speed**: No regression, ideally faster
- [ ] **Reliability**: No new errors or failure modes
- [ ] **Dask readiness**: Clean interface for dask integration

## Next Steps for Approval

Please review and approve/modify:

1. **Component removal decisions** - especially CoordinateCache
2. **Implementation timeline** - adjust phases as needed  
3. **Success metrics** - add/modify as appropriate
4. **Risk assessment** - any concerns about specific components

Once approved, we can proceed with Phase 1 implementation.