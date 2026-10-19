"""Public import, version and PEP 561 marker (3.6, 3.8, 3.10)."""

from importlib.resources import files

import ensemble_outlier_sample_detection as package


def test_detector_is_importable_from_top_level() -> None:
    from ensemble_outlier_sample_detection import (
        EnsembleOutlierSampleDetector,
    )

    assert isinstance(EnsembleOutlierSampleDetector, type)
    assert package.__all__ == ["EnsembleOutlierSampleDetector"]


def test_version_is_set_by_installation() -> None:
    assert isinstance(package.__version__, str)
    assert package.__version__ not in ("", "unknown")


def test_package_ships_py_typed_marker() -> None:
    marker = files("ensemble_outlier_sample_detection").joinpath("py.typed")
    assert marker.is_file()
