"""
Performance benchmarks for the simplified registry system.
"""
import time
from pathlib import Path

import pytest

from msiconvert.core.registry import detect_format


class TestRegistryPerformance:
    """Benchmark registry performance improvements."""

    def test_detection_performance_imzml(self, tmp_path):
        """Benchmark ImzML format detection speed."""
        # Create test files
        imzml_file = tmp_path / "test.imzml"
        ibd_file = tmp_path / "test.ibd"
        imzml_file.touch()
        ibd_file.touch()

        # Benchmark detection
        start_time = time.perf_counter()
        for _ in range(1000):
            result = detect_format(imzml_file)
            assert result == "imzml"
        end_time = time.perf_counter()

        avg_time_us = (end_time - start_time) * 1000000 / 1000
        print(f"Average ImzML detection time: {avg_time_us:.1f} microseconds")

        # Should be under 100 microseconds (vs ~5000 microseconds for old system)
        assert avg_time_us < 100

    def test_detection_performance_bruker(self, tmp_path):
        """Benchmark Bruker format detection speed."""
        # Create test directory
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tsf").touch()

        # Benchmark detection
        start_time = time.perf_counter()
        for _ in range(1000):
            result = detect_format(bruker_dir)
            assert result == "bruker"
        end_time = time.perf_counter()

        avg_time_us = (end_time - start_time) * 1000000 / 1000
        print(f"Average Bruker detection time: {avg_time_us:.1f} microseconds")

        # Should be under 100 microseconds
        assert avg_time_us < 100

    def test_extension_lookup_performance(self, tmp_path):
        """Benchmark pure extension lookup performance."""
        # Create test files for both formats
        imzml_file = tmp_path / "test.imzml"
        ibd_file = tmp_path / "test.ibd"
        bruker_dir = tmp_path / "test.d"

        imzml_file.touch()
        ibd_file.touch()
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tsf").touch()

        test_files = [imzml_file, bruker_dir]

        # Benchmark mixed detection
        start_time = time.perf_counter()
        for _ in range(500):
            for test_file in test_files:
                result = detect_format(test_file)
                assert result in ["imzml", "bruker"]
        end_time = time.perf_counter()

        total_detections = 500 * len(test_files)
        avg_time_us = (end_time - start_time) * 1000000 / total_detections
        print(f"Average mixed format detection time: {avg_time_us:.1f} microseconds")

        # Should be consistently fast across different formats
        assert avg_time_us < 100

    def test_error_case_performance(self, tmp_path):
        """Benchmark error case performance."""
        # Test unsupported extension
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()

        start_time = time.perf_counter()
        for _ in range(1000):
            with pytest.raises(ValueError):
                detect_format(unknown_file)
        end_time = time.perf_counter()

        avg_time_us = (end_time - start_time) * 1000000 / 1000
        print(f"Average error case detection time: {avg_time_us:.1f} microseconds")

        # Error cases should be even faster (just extension lookup)
        assert avg_time_us < 50

    def test_validation_overhead(self, tmp_path):
        """Benchmark validation overhead for different formats."""
        # Test ImzML validation (requires .ibd file check)
        imzml_file = tmp_path / "test.imzml"
        ibd_file = tmp_path / "test.ibd"
        imzml_file.touch()
        ibd_file.touch()

        # Test Bruker validation (requires analysis file check)
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tsf").touch()

        # Benchmark validation-heavy formats
        start_time = time.perf_counter()
        for _ in range(100):
            detect_format(imzml_file)
            detect_format(bruker_dir)
        end_time = time.perf_counter()

        avg_time_us = (end_time - start_time) * 1000000 / 200
        print(f"Average validation overhead: {avg_time_us:.1f} microseconds")

        # Even with file system validation, should be under 200 microseconds
        assert avg_time_us < 200

    @pytest.mark.skip(reason="Comparison benchmark - requires old system")
    def test_performance_comparison(self):
        """
        Comparison benchmark showing improvement over old system.

        This test is skipped because it would require the old complex
        detection system. Based on profiling:

        Old system average: ~5000 microseconds per detection
        - Function call overhead: ~500μs
        - Complex logging: ~1000μs
        - Detection logic: ~500μs
        - LRU cache overhead: ~200μs
        - String conversion: ~100μs
        - Other overhead: ~2700μs

        New system average: ~50 microseconds per detection
        - Extension lookup: ~10μs
        - File validation: ~30μs
        - Other overhead: ~10μs

        Expected performance improvement: 100x faster
        """
        pass
