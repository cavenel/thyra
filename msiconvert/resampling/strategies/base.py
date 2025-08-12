"""
Abstract base class for resampling strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Spectrum:
    """Single mass spectrum with coordinates and metadata."""
    mz: np.ndarray
    intensity: np.ndarray
    coordinates: Tuple[int, int, int]  # (x, y, z) coordinates
    metadata: Optional[Dict] = None
    
    @property
    def is_centroid(self) -> bool:
        """Heuristic to detect centroid data."""
        if len(self.mz) < 100:
            return True
        # Check for zero-intensity gaps typical of centroid data
        zero_count = np.sum(self.intensity == 0)
        return bool(zero_count / len(self.intensity) > 0.5)


class ResamplingStrategy(ABC):
    """Abstract base class for resampling strategies."""
    
    @abstractmethod
    def resample(self, spectrum: Spectrum, target_axis: np.ndarray) -> Spectrum:
        """
        Resample spectrum to target mass axis.
        
        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum to resample
        target_axis : np.ndarray
            Target mass axis values
            
        Returns
        -------
        Spectrum
            Resampled spectrum with target_axis as mz values
        """
        pass