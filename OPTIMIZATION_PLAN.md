# MSIConverter Optimization Plan

## Overview

This document outlines a targeted optimization strategy for the MSIConverter codebase, focusing on simplifying over-engineered components while preserving necessary abstractions for future extensibility.

## Revised Strategy

After analysis and discussion, the following areas have been identified for optimization, with reasoning adjusted for planned future developments:

### ✅ Keep These Abstractions (Future-Proofing)

**Registry System (`core/registry.py`)**
- **Keep**: 4+ additional file formats planned
- **Minor optimization**: Review thread-safety needs, but maintain decorator pattern
- **Reasoning**: Registry pattern justified with multiple formats coming

**Base Reader (`core/base_reader.py`)**
- **Keep**: Essential for upcoming file format diversity
- **Reasoning**: Abstract interface will prevent code duplication across readers

**Base Converter (`core/base_converter.py`)**
- **Keep**: Future converter types not likely but you never know
- **Minor review**: Ensure methods are actually needed, but maintain structure
- **Reasoning**: Flexibility for future conversion targets

## 🎯 Priority Optimizations

### 1. Bruker Mass Axis Over-Engineering

**Current Issue**: `readers/bruker/utils/mass_axis_builder.py` (391 lines)
- Complex implementations for rare memory scenarios
- Over-engineered for current interpolation plans

**Optimization Plan**:
```python
# Instead of multiple MassAxisBuilder implementations
# Simplify to single approach since:
# 1. Common mass axis memory issues are rare
# 2. Future interpolation will reduce axis elements significantly
# 3. Raw Bruker tens of millions → interpolated much smaller
```

**Action Items**:
- [ ] Consolidate mass axis building into single, simple implementation
- [ ] Remove memory-optimization complexity until actually needed
- [ ] Prepare for interpolation integration (smaller axis sizes)

### 2. SpatialData Converter Preparation

**Current Issue**: `converters/spatialdata_converter.py` (746 lines)
- Monolithic 2D/3D handling
- Better SpatialData support coming soon

**Optimization Plan**:
```
spatialdata_converter.py
├── base_spatialdata_converter.py    # Shared logic
├── spatialdata_2d_converter.py      # 2D-specific
└── spatialdata_3d_converter.py      # 3D-specific
```

**Benefits**:
- Easier to adapt to upcoming SpatialData improvements
- Cleaner separation of concerns
- Simplified testing and maintenance

### 3. Bruker Utils Consolidation

**Current Issue**: 5 separate utility classes (~2000+ lines total)
- `batch_processor.py` (434 lines)
- `coordinate_cache.py` (428 lines) 
- `memory_manager.py` (392 lines)
- Others...

**Optimization Plan**:
- **Keep separate**: If memory management complexity is actually needed
- **Simplify**: Remove premature optimizations
- **Focus**: Mass axis builder is the main over-engineering target

## 🤔 Discussion Points

### Metadata Strategy

**Current Approach**:
- **Essential Metadata**: Required for conversion
- **Comprehensive Metadata**: Instrument parameters, nice-to-have

**Questions for Discussion**:

1. **Community Integration**: How to structure comprehensive metadata without official standards?
   - Wait for community consensus?
   - Create opinionated structure that can be adapted?
   - Store as unstructured but validated JSON?

2. **Essential vs Comprehensive Boundary**: 
   - Is current split appropriate?
   - Should some "comprehensive" items become "essential" for better conversions?

3. **Ontology Complexity**: `_ms.py` (3426 lines)
   - Is this level of ontology detail necessary?
   - Can we subset to MSI-relevant terms only?
   - Should this be optional/pluggable?

**Proposed Approach**:
```python
# Keep essential/comprehensive split
# But make comprehensive metadata more flexible:

class EssentialMetadata:
    """Structured, required for conversion"""
    pass

class ComprehensiveMetadata:
    """Flexible dict-like with validation"""
    def __init__(self):
        self._raw_metadata = {}
        self._structured_metadata = {}
    
    def add_structured(self, key, value, schema=None):
        """For known structures"""
        pass
    
    def add_raw(self, key, value):
        """For arbitrary instrument data"""
        pass
```

## 📋 Implementation Timeline

### Phase 1: Mass Axis Simplification (Week 1)
- [ ] Analyze current mass axis memory usage patterns
- [ ] Simplify to single implementation
- [ ] Update tests
- [ ] Performance validation

### Phase 2: SpatialData Converter Split (Week 2-3)
- [ ] Extract common base functionality
- [ ] Split 2D/3D specific logic
- [ ] Update registration system
- [ ] Comprehensive testing

### Phase 3: Metadata Strategy (Week 4)
- [ ] Finalize essential/comprehensive approach
- [ ] Community feedback on comprehensive structure
- [ ] Implement flexible comprehensive metadata
- [ ] Documentation updates

### Phase 4: Final Bruker Utils Review (Week 5)
- [ ] Post-mass-axis review of other utilities
- [ ] Identify additional simplification opportunities
- [ ] Performance benchmarking
- [ ] Documentation

## Success Metrics

- **Code Reduction**: Target 20-30% reduction in Bruker utilities
- **Maintainability**: Easier to add new file formats
- **Performance**: No regression in conversion speed
- **Flexibility**: Ready for upcoming SpatialData improvements
- **Community**: Clear path for comprehensive metadata integration

## Questions for Team Discussion

1. **Mass Axis**: Are there specific large datasets where current complexity is needed?
2. **Metadata**: Preference for comprehensive metadata structure approach?
3. **Timeline**: Are there any upcoming format additions that should influence priorities?
4. **SpatialData**: Timeline for upcoming improvements to coordinate preparation?