# Pre-commit Manual Fixes Required

This document lists all the manual fixes needed to pass the pre-commit hooks. The automatic formatting fixes (whitespace, line endings, etc.) have already been applied.

## Flake8 Violations

### Cyclomatic Complexity (C901) - Functions Too Complex

These functions exceed the complexity threshold and need refactoring:

1. **msiconvert/__main__.py:11:1** - `main` function (complexity: 12)
2. **msiconvert/convert.py:22:1** - `convert_msi` function (complexity: 20)
3. **msiconvert/converters/spatialdata/base_spatialdata_converter.py:310:5** - `BaseSpatialDataConverter.add_metadata` method (complexity: 11)
4. **msiconvert/metadata/extractors/bruker_extractor.py:221:5** - `BrukerMetadataExtractor._extract_acquisition_params` method (complexity: 12)
5. **msiconvert/readers/bruker/utils/batch_processor.py:334:5** - `BatchProcessor.process_with_memory_monitoring` method (complexity: 12)
6. **msiconvert/readers/imzml_reader.py:149:5** - `ImzMLReader.get_common_mass_axis` method (complexity: 14)
7. **msiconvert/readers/imzml_reader.py:230:5** - `ImzMLReader.iter_spectra` method (complexity: 14)

### Test File Complexity

8. **tests/integration/test_full_spatialdata_conversion.py:24:1** - `test_spatialdata_integration` function (complexity: 40)
9. **tests/unit/metadata/extractors/test_bruker_extractor.py:14:5** - `TestBrukerMetadataExtractor.create_mock_connection` method (complexity: 16)

### Unused Variables (F841)

10. **tests/integration/test_cli.py:113:9** - `original_convert_msi` assigned but never used
11. **tests/integration/test_convert_bruker.py:70:9** - `spatialdata` assigned but never used
12. **tests/unit/converters/test_spatialdata_converter.py:343:9** - `converter` assigned but never used
13. **tests/unit/metadata/extractors/test_bruker_extractor.py:206:9** - `sample_data` assigned but never used
14. **tests/unit/readers/test_bruker_reader_comprehensive.py:129:13** - `invalid_reader` assigned but never used
15. **tests/unit/readers/test_bruker_reader_comprehensive.py:299:13** - `dimensions` assigned but never used
16. **tests/unit/readers/test_bruker_reader_comprehensive.py:441:13** - `mz_values` assigned but never used
17. **tests/unit/readers/test_bruker_reader_comprehensive.py:442:13** - `intensity_values` assigned but never used
18. **tests/unit/readers/test_bruker_reader_phase1.py:269:13** - `spectrum` assigned but never used
19. **tests/unit/readers/test_bruker_reader_phase1.py:368:13** - `spectrum` assigned but never used

### Unused Imports (F401)

20. **tests/integration/test_full_spatialdata_conversion.py:69:17** - `spatialdata.models.ShapesModel` imported but unused
21. **tests/integration/test_full_spatialdata_conversion.py:69:17** - `spatialdata.models.TableModel` imported but unused
22. **tests/integration/test_full_spatialdata_conversion.py:70:17** - `spatialdata.transformations.Identity` imported but unused
23. **tests/unit/converters/test_spatialdata_converter.py:5:1** - `pathlib.Path` imported but unused
24. **tests/unit/converters/test_spatialdata_converter.py:8:1** - `geopandas as gpd` imported but unused
25. **tests/unit/converters/test_spatialdata_converter.py:11:1** - `pytest` imported but unused
26. **tests/unit/metadata/core/test_base_extractor.py:2:1** - `unittest.mock.Mock` imported but unused
27. **tests/unit/metadata/core/test_base_extractor.py:2:1** - `unittest.mock.patch` imported but unused
28. **tests/unit/readers/test_bruker_reader_comprehensive.py:13:1** - `os` imported but unused
29. **tests/unit/readers/test_bruker_reader_comprehensive.py:14:1** - `sys` imported but unused
30. **tests/unit/readers/test_bruker_reader_comprehensive.py:17:1** - `typing.List` imported but unused
31. **tests/unit/readers/test_bruker_reader_comprehensive.py:17:1** - `typing.Tuple` imported but unused
32. **tests/unit/readers/test_bruker_reader_comprehensive.py:23:1** - `msiconvert.utils.bruker_exceptions.DataError` imported but unused
33. **tests/unit/readers/test_bruker_reader_comprehensive.py:23:1** - `msiconvert.utils.bruker_exceptions.SDKError` imported but unused

### Undefined Names (F821)

34. **tests/integration/test_full_spatialdata_conversion.py:246:25** - undefined name `SpatialDataConverter`

### Function Redefinition (F811)

35. **tests/unit/readers/test_bruker_reader_comprehensive.py:251:5** - redefinition of unused `test_spectrum_reading` from line 200

### Line Too Long (E501) - Over 100 Characters

36. **tests/integration/test_full_spatialdata_conversion.py:301:101** - line too long (119 > 100 characters)
37. **tests/integration/test_full_spatialdata_conversion.py:356:101** - line too long (112 > 100 characters)
38. **tests/unit/converters/test_spatialdata_converter.py:347:101** - line too long (120 > 100 characters)
39. **tests/unit/metadata/extractors/test_bruker_extractor.py:52:101** - line too long (110 > 100 characters)

## Pydocstyle Violations (D212)

All violations are "Multi-line docstring summary should start at the first line":

### Core Module Files

40. **msiconvert/core/base_reader.py** - Multiple methods (lines 19, 51, 65, 87)
41. **msiconvert/core/registry.py** - Multiple functions and methods
42. **msiconvert/core/base_converter.py** - Multiple methods

### Reader Files

43. **msiconvert/readers/imzml_reader.py** - Multiple methods
44. **msiconvert/readers/bruker/bruker_reader.py** - Multiple methods
45. **msiconvert/readers/bruker/sdk/dll_manager.py** - Multiple methods (lines 27, 41, 56, 69, 119, 139, 160, 184, 195)
46. **msiconvert/readers/bruker/sdk/platform_detector.py** - Multiple functions (lines 1, 21, 49, 167, 186)
47. **msiconvert/readers/bruker/utils/mass_axis_builder.py** - Multiple methods (lines 1, 18, 31, 56, 83, 112, 165, 230, 289, 314, 344)
48. **msiconvert/readers/bruker/utils/coordinate_cache.py** - Multiple methods
49. **msiconvert/readers/bruker/utils/memory_manager.py** - Multiple methods
50. **msiconvert/readers/bruker/utils/batch_processor.py** - Multiple methods

### Converter Files

51. **msiconvert/converters/spatialdata/base_spatialdata_converter.py** - Multiple methods
52. **msiconvert/converters/spatialdata/spatialdata_converter.py** - Multiple methods

### Metadata Files

53. **msiconvert/metadata/extractors/imzml_extractor.py** - Multiple methods
54. **msiconvert/metadata/extractors/bruker_extractor.py** - Multiple methods
55. **msiconvert/metadata/core/base_extractor.py** - Multiple methods

### Utility and Tool Files

56. **msiconvert/utils/data_processors.py** - Multiple functions
57. **msiconvert/utils/logging_config.py** - Multiple functions
58. **msiconvert/tools/__init__.py** - Module level (line 1)
59. **msiconvert/tools/check_ontology.py** - Multiple functions

## Priority for Fixes

### High Priority (Production Code Issues)
1. **Cyclomatic Complexity** - These should be refactored for maintainability
2. **Undefined Names** - These will cause runtime errors
3. **Function Redefinition** - These can cause unexpected behavior

### Medium Priority (Code Quality)
1. **Unused Imports/Variables** - These can be safely removed
2. **Line Length** - These should be wrapped for readability

### Low Priority (Documentation Style)
1. **Pydocstyle D212** - These are style issues that don't affect functionality

## Recommended Approach

1. Start with undefined names and function redefinitions (critical bugs)
2. Remove unused imports and variables (easy wins)
3. Fix line length violations (formatting)
4. Refactor complex functions (requires more thought)
5. Fix docstring formatting (bulk find/replace possible)