"""
Tests for DecisionTree instrument-based method selection - Phase 3.
"""

import pytest

from msiconvert.resampling.decision_tree import ResamplingDecisionTree
from msiconvert.resampling.types import ResamplingMethod


class TestDecisionTreeInstrumentDetection:
    """Test instrument-based resampling strategy selection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tree = ResamplingDecisionTree()
    
    def test_no_metadata_default(self):
        """Test default behavior with no metadata."""
        method = self.tree.select_strategy(None)
        assert method == ResamplingMethod.TIC_PRESERVING
        
        method = self.tree.select_strategy({})
        assert method == ResamplingMethod.TIC_PRESERVING
    
    def test_timstof_detection_by_name(self):
        """Test timsTOF detection by instrument name."""
        # Various timsTOF name variants
        timstof_names = [
            "timsTOF Pro 2",
            "Bruker timsTOF",
            "TIMS-TOF CCS",
            "tims tof flex",
            "TimsTOF Impact II",
            "maxis impact hd",
            "compact timstof"
        ]
        
        for name in timstof_names:
            metadata = {"instrument_name": name}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.NEAREST_NEIGHBOR, f"Failed for {name}"
    
    def test_orbitrap_detection_by_name(self):
        """Test Orbitrap detection by instrument name."""
        orbitrap_names = [
            "Orbitrap Fusion Lumos",
            "Q Exactive HF-X",
            "QExactive Plus",
            "Orbitrap Exploris 480",
            "LTQ Orbitrap Velos",
            "Orbitrap Eclipse Tribrid"
        ]
        
        for name in orbitrap_names:
            metadata = {"instrument_name": name}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.TIC_PRESERVING, f"Failed for {name}"
    
    def test_fticr_detection_by_name(self):
        """Test FT-ICR detection by instrument name."""
        fticr_names = [
            "Bruker 12T FT-ICR",
            "FTICR-MS SolariX",
            "FT-ICR 15 Tesla"
        ]
        
        for name in fticr_names:
            metadata = {"instrument_name": name}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.TIC_PRESERVING, f"Failed for {name}"
    
    def test_waters_instruments(self):
        """Test Waters instrument detection."""
        waters_names = [
            "Waters Synapt G2-Si",
            "Vion IMS QTof",
            "Select Series Cyclic IMS"
        ]
        
        for name in waters_names:
            metadata = {"instrument_name": name}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.TIC_PRESERVING, f"Failed for {name}"
    
    def test_acquisition_mode_detection(self):
        """Test acquisition mode-based detection."""
        # Centroid mode should trigger nearest neighbor
        centroid_modes = ["centroid", "centroided", "peak picked", "peaks"]
        for mode in centroid_modes:
            metadata = {"acquisition_mode": mode}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.NEAREST_NEIGHBOR, f"Failed for {mode}"
        
        # Profile mode should trigger TIC preserving
        profile_modes = ["profile", "continuum", "raw spectrum"]
        for mode in profile_modes:
            metadata = {"acquisition_mode": mode}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.TIC_PRESERVING, f"Failed for {mode}"
    
    def test_bruker_metadata_timstof_detection(self):
        """Test timsTOF detection from Bruker-specific metadata."""
        # Simulate Bruker GlobalMetadata structure
        bruker_timstof_metadata = {
            "GlobalMetadata": {
                "InstrumentFamily": "TOF",
                "InstrumentName": "timsTOF Pro 2",
                "Method": "TIMS-DDA with mobility separation"
            },
            "AcquisitionKeys": {
                "TIMSCalibrationParameters": {"Beta": 0.12345},
                "MobilitySettings": {"Range": "0.6-1.8"}
            }
        }
        
        method = self.tree.select_strategy(bruker_timstof_metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR
        
        # Test with method-based detection
        metadata_method_based = {
            "GlobalMetadata": {
                "Method": "PASEF LC-MS/MS with ion mobility"
            }
        }
        method = self.tree.select_strategy(metadata_method_based)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR
        
        # Test with acquisition keys
        metadata_acq_keys = {
            "AcquisitionKeys": {
                "IMSSettings": {"enabled": True}
            }
        }
        method = self.tree.select_strategy(metadata_acq_keys)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR
    
    def test_metadata_key_variants(self):
        """Test different metadata key variants."""
        # Test various instrument name keys
        instrument_key_variants = [
            "instrument_name",
            "instrument", 
            "instrument_model",
            "InstrumentName",
            "Instrument",
            "InstrumentModel",
            "ms_instrument_name",
            "device_name"
        ]
        
        for key in instrument_key_variants:
            metadata = {key: "timsTOF Pro"}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.NEAREST_NEIGHBOR, f"Failed for key {key}"
        
        # Test acquisition mode key variants
        mode_key_variants = [
            "acquisition_mode",
            "mode", 
            "spectrum_type",
            "AcquisitionMode",
            "Mode",
            "SpectrumType",
            "ms_acquisition_mode",
            "data_type"
        ]
        
        for key in mode_key_variants:
            metadata = {key: "centroid"}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.NEAREST_NEIGHBOR, f"Failed for key {key}"
    
    def test_priority_hierarchy(self):
        """Test that instrument name takes priority over acquisition mode."""
        # timsTOF with profile mode should still use nearest neighbor
        metadata = {
            "instrument_name": "timsTOF Pro 2",
            "acquisition_mode": "profile"  # Conflicting signal
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR
        
        # Orbitrap with centroid mode should use TIC preserving
        metadata = {
            "instrument_name": "Orbitrap Fusion",
            "acquisition_mode": "centroid"  # Conflicting signal
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.TIC_PRESERVING
    
    def test_case_insensitive_detection(self):
        """Test case insensitive detection."""
        # Various case variants
        case_variants = [
            "TIMSTOF PRO",
            "timstof pro", 
            "TimsTof Pro",
            "TIMS-TOF PRO",
            "tImStOf PrO"
        ]
        
        for variant in case_variants:
            metadata = {"instrument_name": variant}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.NEAREST_NEIGHBOR, f"Failed for {variant}"
    
    def test_unknown_instrument_default(self):
        """Test that unknown instruments default to TIC preserving."""
        unknown_instruments = [
            "Unknown MS Instrument",
            "Custom Quadrupole",
            "Prototype TOF-MS",
            "",  # Empty string
            "   "  # Whitespace
        ]
        
        for instrument in unknown_instruments:
            metadata = {"instrument_name": instrument}
            method = self.tree.select_strategy(metadata)
            assert method == ResamplingMethod.TIC_PRESERVING, f"Failed for {instrument}"
    
    def test_complex_metadata_structures(self):
        """Test with complex nested metadata structures."""
        complex_metadata = {
            "instrument_info": {
                "manufacturer": "Bruker",
                "model": "timsTOF Pro 2",
                "serial": "12345"
            },
            "acquisition_parameters": {
                "mode": "PASEF",
                "mobility_range": [0.6, 1.8],
                "ms_range": [50, 2000]
            },
            "file_info": {
                "format": "Bruker .d",
                "size_mb": 1500
            }
        }
        
        # Should not detect timsTOF from nested structure without proper key
        method = self.tree.select_strategy(complex_metadata)
        assert method == ResamplingMethod.TIC_PRESERVING
        
        # Add proper instrument key
        complex_metadata["instrument_name"] = "timsTOF Pro 2"
        method = self.tree.select_strategy(complex_metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR


class TestDecisionTreeHelperMethods:
    """Test helper methods of DecisionTree."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tree = ResamplingDecisionTree()
    
    def test_extract_instrument_name(self):
        """Test instrument name extraction."""
        metadata = {"instrument_name": "  timsTOF Pro 2  "}
        name = self.tree._extract_instrument_name(metadata)
        assert name == "timsTOF Pro 2"  # Should be stripped
        
        # Test None values
        metadata = {"instrument_name": None}
        name = self.tree._extract_instrument_name(metadata)
        assert name is None
        
        # Test empty metadata
        name = self.tree._extract_instrument_name({})
        assert name is None
    
    def test_extract_acquisition_mode(self):
        """Test acquisition mode extraction."""
        metadata = {"acquisition_mode": "  CENTROID  "}
        mode = self.tree._extract_acquisition_mode(metadata)
        assert mode == "centroid"  # Should be stripped and lowercased
        
        # Test None values
        metadata = {"acquisition_mode": None}
        mode = self.tree._extract_acquisition_mode(metadata)
        assert mode is None
    
    def test_is_timstof(self):
        """Test timsTOF detection logic."""
        timstof_names = [
            "timsTOF", "TIMS-TOF", "tims tof", "TimsTOF Pro",
            "Impact II", "maxis 4G", "compact TOF", "flex TOF"
        ]
        
        for name in timstof_names:
            assert self.tree._is_timstof(name), f"Should detect {name} as timsTOF"
        
        non_timstof_names = [
            "Orbitrap", "LTQ", "QTOF", "Synapt"
        ]
        
        for name in non_timstof_names:
            assert not self.tree._is_timstof(name), f"Should NOT detect {name} as timsTOF"
    
    def test_is_profile_instrument(self):
        """Test profile instrument detection."""
        profile_instruments = [
            "Orbitrap Fusion", "Q Exactive", "LTQ Velos",
            "FT-ICR", "Synapt G2", "Waters Vion"
        ]
        
        for instrument in profile_instruments:
            assert self.tree._is_profile_instrument(instrument), f"Should detect {instrument} as profile"
    
    def test_is_bruker_metadata(self):
        """Test Bruker metadata detection."""
        bruker_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF"},
            "AcquisitionKeys": {}
        }
        assert self.tree._is_bruker_metadata(bruker_metadata)
        
        non_bruker_metadata = {
            "instrument_name": "Orbitrap",
            "file_format": "mzML"
        }
        assert not self.tree._is_bruker_metadata(non_bruker_metadata)