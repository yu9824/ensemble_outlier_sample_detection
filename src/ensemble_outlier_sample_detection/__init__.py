"""Ensemble-based outlier sample detection for regression data."""

from pathlib import Path

from .ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

_version_file = Path(__file__).parent / "_version.py"

if _version_file.is_file():
    from ._version import __version__
else:
    __version__ = "unknown"

__license__ = "Apache-2.0"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"

__all__ = ["EnsembleOutlierSampleDetector"]
