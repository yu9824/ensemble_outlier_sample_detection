"""Runtime helpers: Optuna log suppression and the progress bar.

This module has no dependency on the rest of the package. The detector
uses it so that ``fit`` leaves no global side effect behind and shows a
progress bar that suits the running environment.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import optuna
from tqdm.auto import tqdm

__all__ = ["make_progress_bar", "suppress_optuna_logging"]


@contextmanager
def suppress_optuna_logging() -> Iterator[None]:
    """Hide Optuna's per-trial logs while the block runs.

    The global Optuna verbosity is saved on entry and set to
    ``optuna.logging.WARNING``, so the INFO messages written for every
    trial are not emitted. On exit the saved verbosity is restored, both
    after a normal exit and when the block raises. Optuna's default
    handler is never enabled, disabled or otherwise changed.

    Yields
    ------
    None
        Control is passed to the ``with`` block.
    """
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous)


def make_progress_bar(enabled: bool, total: int) -> tqdm:
    """Create a progress bar that suits the running environment.

    The bar comes from :mod:`tqdm.auto`: it is the notebook widget on a
    Jupyter kernel and the text bar elsewhere. A disabled bar is the same
    object with ``disable=True``, so callers use one code path either way
    and a disabled bar writes nothing.

    Parameters
    ----------
    enabled : bool
        Whether the bar is shown. ``False`` disables it.
    total : int
        Expected number of updates.

    Returns
    -------
    tqdm.auto.tqdm
        The progress bar. Use it as a context manager so that it is
        closed even when an exception is raised.
    """
    return tqdm(total=total, disable=not enabled)
