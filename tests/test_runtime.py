"""Unit tests for the runtime helpers (5.1, 5.2, 5.4, 5.6, 7.1, 7.3).

Checks that run ``fit`` itself come later (task 3.5) and extend this file.
"""

import logging
import sys
from collections.abc import Iterator

import optuna
import pytest
import tqdm.auto

from ensemble_outlier_sample_detection._runtime import (
    make_progress_bar,
    suppress_optuna_logging,
)

VERBOSITIES = [
    optuna.logging.DEBUG,
    optuna.logging.INFO,
    optuna.logging.WARNING,
    optuna.logging.ERROR,
    optuna.logging.CRITICAL,
]


def _logger_state(
    logger: logging.Logger,
) -> tuple[list[tuple[logging.Handler, int]], bool]:
    # logging.Handler defines no __eq__, so == compares handlers by
    # identity: same objects, same order, same levels.
    handlers = [(handler, handler.level) for handler in logger.handlers]
    return handlers, logger.propagate


@pytest.fixture(autouse=True)
def restore_optuna_logging() -> Iterator[None]:
    """Put Optuna's global logging state back after every test.

    The verbosity, the handlers with their levels and ``propagate`` are
    restored, so no test starts from what an earlier test (or a faulty
    implementation) left behind.
    """
    logger = logging.getLogger("optuna")
    verbosity = optuna.logging.get_verbosity()
    handlers, propagate = _logger_state(logger)
    yield
    for current in list(logger.handlers):
        logger.removeHandler(current)
    for handler, level in handlers:
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = propagate
    optuna.logging.set_verbosity(verbosity)


@pytest.fixture
def optuna_stderr(
    capsys: pytest.CaptureFixture[str],
) -> Iterator[pytest.CaptureFixture[str]]:
    """Copy every record the ``optuna`` logger emits to captured stderr.

    Optuna's default handler binds ``sys.stderr`` when optuna is imported,
    which is pytest's session-wide capture, so capsys and capfd never see
    its output, and ``disable_default_handler`` may have removed it.
    A handler on the same logger is gated by the same verbosity, so what
    it writes is exactly what the default handler would print.
    """
    logger = logging.getLogger("optuna")
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    try:
        yield capsys
    finally:
        logger.removeHandler(handler)


@pytest.fixture(params=[True, False], ids=["attached", "detached"])
def default_handler(
    request: pytest.FixtureRequest,
) -> tuple[logging.Handler, bool]:
    """Attach or detach Optuna's default handler for the test.

    Returns the default handler and whether it is attached. Only public
    API is used to find it; ``restore_optuna_logging`` puts back the
    handler list seen before the test, in the same order.
    """
    logger = logging.getLogger("optuna")
    optuna.logging.disable_default_handler()
    without = list(logger.handlers)
    optuna.logging.enable_default_handler()
    (handler,) = [h for h in logger.handlers if h not in without]
    attached: bool = request.param
    if not attached:
        optuna.logging.disable_default_handler()
    return handler, attached


def _run_trivial_study() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=2
    )


@pytest.mark.parametrize("previous", VERBOSITIES)
def test_verbosity_is_warning_inside_and_restored_after(
    previous: int,
) -> None:
    optuna.logging.set_verbosity(previous)
    with suppress_optuna_logging():
        inside = optuna.logging.get_verbosity()
    assert inside == optuna.logging.WARNING
    assert optuna.logging.get_verbosity() == previous


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), KeyboardInterrupt()],
    ids=lambda error: type(error).__name__,
)
@pytest.mark.parametrize("previous", VERBOSITIES)
def test_verbosity_is_restored_after_an_exception(
    previous: int, error: BaseException
) -> None:
    optuna.logging.set_verbosity(previous)
    inside: list[int] = []
    with pytest.raises(type(error)) as raised, suppress_optuna_logging():
        inside.append(optuna.logging.get_verbosity())
        raise error
    assert raised.value is error
    assert inside == [optuna.logging.WARNING]
    assert optuna.logging.get_verbosity() == previous


def test_verbosity_changed_inside_is_reverted() -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    with suppress_optuna_logging():
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


def test_trial_logs_are_emitted_outside_the_context(
    optuna_stderr: pytest.CaptureFixture[str],
) -> None:
    # Guards the next test: without it, an empty stderr proves nothing.
    optuna.logging.set_verbosity(optuna.logging.INFO)
    _run_trivial_study()
    assert "Trial 0 finished" in optuna_stderr.readouterr().err


@pytest.mark.parametrize(
    "previous", [optuna.logging.DEBUG, optuna.logging.INFO]
)
def test_trial_logs_are_not_emitted_inside_the_context(
    optuna_stderr: pytest.CaptureFixture[str], previous: int
) -> None:
    optuna.logging.set_verbosity(previous)
    with suppress_optuna_logging():
        _run_trivial_study()
    assert optuna_stderr.readouterr().err == ""


def test_default_handler_is_left_alone(
    default_handler: tuple[logging.Handler, bool],
) -> None:
    # Detached is the state a user who routes Optuna logs elsewhere is
    # in; re-attaching the handler would duplicate their output.
    handler, attached = default_handler
    logger = logging.getLogger("optuna")
    assert (handler in logger.handlers) is attached
    before = _logger_state(logger)
    with suppress_optuna_logging():
        inside = _logger_state(logger)
    assert inside == before
    assert _logger_state(logger) == before


@pytest.mark.parametrize("enabled", [True, False])
def test_progress_bar_is_auto_tqdm_and_a_context_manager(
    enabled: bool,
) -> None:
    bar = make_progress_bar(enabled, 7)
    with bar as entered:
        assert entered is bar
        assert isinstance(bar, tqdm.auto.tqdm)
        assert bar.total == 7


def test_enabled_progress_bar_writes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with make_progress_bar(True, 3) as bar:
        bar.set_description("[iter 0 / 3]")
        bar.update(3)
    captured = capsys.readouterr()
    assert "3/3" in captured.err
    assert "[iter 0 / 3]" in captured.err
    assert captured.out == ""


def test_disabled_progress_bar_writes_nothing(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with make_progress_bar(False, 3) as bar:
        bar.set_description("[iter 0 / 3]")
        bar.update(1)
        bar.set_description("Finished! [iter 1 / 3]")
        bar.update(2)
    assert capfd.readouterr() == ("", "")
