"""
Common axis builder for creating unified mass axes.
"""

import numpy as np

from .types import AxisType, MassAxis


class CommonAxisBuilder:
    """Creates optimized common mass axis for datasets."""

    def build_uniform_axis(
        self, min_mz: float, max_mz: float, num_bins: int
    ) -> MassAxis:
        """
        Create uniform (equidistant) mass axis.

        This is a placeholder implementation that will be expanded
        in Phase 4.

        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        num_bins : int
            Number of bins

        Returns
        -------
        MassAxis
            Generated mass axis
        """
        mz_values = np.linspace(min_mz, max_mz, num_bins)

        return MassAxis(
            mz_values=mz_values,
            min_mz=min_mz,
            max_mz=max_mz,
            num_bins=num_bins,
            axis_type=AxisType.CONSTANT,
        )
