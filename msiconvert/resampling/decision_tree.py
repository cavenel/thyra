"""
Decision tree for automatic resampling strategy selection.
"""

import logging
from typing import Any, Dict, Optional

from .types import ResamplingMethod

logger = logging.getLogger(__name__)


class ResamplingDecisionTree:
    """Implements decision tree for resampling strategy selection based on instrument metadata."""

    def select_strategy(self, metadata: Optional[Dict[str, Any]] = None) -> ResamplingMethod:
        """
        Automatically select appropriate resampling method based on instrument metadata.

        Decision logic:
        1. If timsTOF detected -> NEAREST_NEIGHBOR (optimal for centroid data)
        2. If other Bruker instrument -> TIC_PRESERVING (profile data)
        3. If Orbitrap/FT-ICR detected -> TIC_PRESERVING (profile data)
        4. If centroid data detected -> NEAREST_NEIGHBOR
        5. Default fallback -> TIC_PRESERVING (most common case)

        Parameters
        ----------
        metadata : Optional[Dict[str, Any]]
            Metadata dictionary containing instrument information

        Returns
        -------
        ResamplingMethod
            Selected resampling strategy
        """
        if metadata is None:
            logger.info("No metadata provided, defaulting to TIC_PRESERVING strategy")
            return ResamplingMethod.TIC_PRESERVING

        # Check for instrument type in metadata
        instrument_name = self._extract_instrument_name(metadata)
        
        if instrument_name:
            logger.debug(f"Detected instrument: {instrument_name}")
            
            # TimsTOF detection (Bruker timsTOF instruments)
            if self._is_timstof(instrument_name):
                logger.info(f"timsTOF instrument detected ({instrument_name}), using NEAREST_NEIGHBOR strategy")
                return ResamplingMethod.NEAREST_NEIGHBOR
            
            # Other high-resolution instruments that typically produce profile data
            if self._is_profile_instrument(instrument_name):
                logger.info(f"Profile instrument detected ({instrument_name}), using TIC_PRESERVING strategy")
                return ResamplingMethod.TIC_PRESERVING
        
        # Check for acquisition mode if available
        acq_mode = self._extract_acquisition_mode(metadata)
        if acq_mode:
            logger.debug(f"Detected acquisition mode: {acq_mode}")
            
            if self._is_centroid_mode(acq_mode):
                logger.info(f"Centroid acquisition mode detected ({acq_mode}), using NEAREST_NEIGHBOR strategy")
                return ResamplingMethod.NEAREST_NEIGHBOR
            
            if self._is_profile_mode(acq_mode):
                logger.info(f"Profile acquisition mode detected ({acq_mode}), using TIC_PRESERVING strategy")
                return ResamplingMethod.TIC_PRESERVING
        
        # Check Bruker GlobalMetadata for specific instrument detection
        if self._is_bruker_metadata(metadata):
            # Check if it's a timsTOF based on method or instrument model
            if self._detect_timstof_from_bruker_metadata(metadata):
                logger.info("timsTOF detected from Bruker metadata, using NEAREST_NEIGHBOR strategy")
                return ResamplingMethod.NEAREST_NEIGHBOR
        
        # Default fallback - most MS instruments produce profile data
        logger.info("No specific instrument detected, defaulting to TIC_PRESERVING strategy")
        return ResamplingMethod.TIC_PRESERVING

    def _extract_instrument_name(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract instrument name from metadata."""
        # Common metadata keys for instrument information
        instrument_keys = [
            'instrument_name', 'instrument', 'instrument_model', 
            'InstrumentName', 'Instrument', 'InstrumentModel',
            'ms_instrument_name', 'ms_instrument', 'device_name'
        ]
        
        for key in instrument_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key]).strip()
        
        return None

    def _extract_acquisition_mode(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract acquisition mode from metadata."""
        # Common keys for acquisition mode
        mode_keys = [
            'acquisition_mode', 'mode', 'spectrum_type',
            'AcquisitionMode', 'Mode', 'SpectrumType', 
            'ms_acquisition_mode', 'data_type'
        ]
        
        for key in mode_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key]).strip().lower()
        
        return None

    def _is_timstof(self, instrument_name: str) -> bool:
        """Check if instrument is a Bruker timsTOF."""
        instrument_lower = instrument_name.lower()
        timstof_keywords = [
            'timstof', 'tims-tof', 'tims tof', 'tofpro', 
            'impact', 'maxis', 'compact', 'flex'
        ]
        
        return any(keyword in instrument_lower for keyword in timstof_keywords)

    def _is_profile_instrument(self, instrument_name: str) -> bool:
        """Check if instrument typically produces profile data."""
        instrument_lower = instrument_name.lower()
        profile_keywords = [
            'orbitrap', 'exactive', 'q exactive', 'qexactive', 
            'ltq', 'velos', 'elite', 'fusion', 'lumos', 
            'exploris', 'eclipse', 'fticr', 'ft-icr',
            'synapt', 'vion', 'select', 'qtof',  # Waters
            'maxi', 'impact', 'compact'  # Some Bruker QTOF
        ]
        
        return any(keyword in instrument_lower for keyword in profile_keywords)

    def _is_centroid_mode(self, acq_mode: str) -> bool:
        """Check if acquisition mode indicates centroid data."""
        centroid_keywords = ['centroid', 'centroided', 'peak', 'peaks']
        return any(keyword in acq_mode for keyword in centroid_keywords)

    def _is_profile_mode(self, acq_mode: str) -> bool:
        """Check if acquisition mode indicates profile data.""" 
        profile_keywords = ['profile', 'continuum', 'raw']
        return any(keyword in acq_mode for keyword in profile_keywords)

    def _is_bruker_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata appears to be from Bruker instruments."""
        bruker_keys = [
            'GlobalMetadata', 'AcquisitionKeys', 'Method',
            'InstrumentFamily', 'InstrumentName'
        ]
        
        return any(key in metadata for key in bruker_keys)

    def _detect_timstof_from_bruker_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Detect timsTOF from Bruker-specific metadata structure."""
        # Check GlobalMetadata for timsTOF indicators
        if 'GlobalMetadata' in metadata:
            global_meta = metadata['GlobalMetadata']
            
            # Check instrument family
            if 'InstrumentFamily' in global_meta:
                family = str(global_meta['InstrumentFamily']).lower()
                if 'tof' in family or 'tims' in family:
                    return True
            
            # Check method parameters for TIMS-specific settings
            if 'Method' in global_meta:
                method = str(global_meta['Method']).lower()
                tims_indicators = ['tims', 'mobility', 'ims', 'ccs']
                if any(indicator in method for indicator in tims_indicators):
                    return True
        
        # Check AcquisitionKeys for TIMS parameters
        if 'AcquisitionKeys' in metadata:
            acq_keys = metadata['AcquisitionKeys']
            tims_keys = ['TIMSCalibrationParameters', 'MobilitySettings', 'IMSSettings']
            if any(key in acq_keys for key in tims_keys):
                return True
        
        return False
