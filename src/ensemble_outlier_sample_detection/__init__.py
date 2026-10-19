"""Ensemble-based outlier sample detection for regression data.

The package exports one estimator, :class:`EnsembleOutlierSampleDetector`.
``__version__`` comes from the ``_version.py`` file that setuptools-scm
writes at build time, and is ``"unknown"`` when that file is absent.
"""

from .ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

try:
    from ._version import __version__
except ModuleNotFoundError:
    # Only a missing _version.py falls back; a broken one still raises.
    __version__ = "unknown"

__license__ = "Apache-2.0"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"

__all__ = ["EnsembleOutlierSampleDetector"]
