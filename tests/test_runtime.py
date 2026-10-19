"""Runtime side effects and the progress bar (5.1-5.7, 7.1, 7.3).

The first part tests the helpers of ``_runtime`` on their own. The
second part runs ``fit`` with small parameters and checks what it leaves
behind and what it writes.
"""

import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import optuna
import pandas as pd
import pytest
import tqdm.auto
from sklearn.exceptions import ConvergenceWarning

from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector
from ensemble_outlier_sample_detection._runtime import (
    make_progress_bar,
    suppress_optuna_logging,
)

# The module that defines the detector: fit looks its helpers up here.
detector_module = sys.modules[EnsembleOutlierSampleDetector.__module__]

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


class _CurrentStderrHandler(logging.Handler):
    """Write each record to ``sys.stderr`` as it is at emit time.

    pytest gives capsys a new stream for the call phase and closes the
    one of the setup phase, so a ``StreamHandler`` created in a fixture
    would write to a closed file during the test. A failed write raises
    instead of going to ``handleError``, which would print the message
    in a report that the tests could mistake for the log line.
    """

    def emit(self, record: logging.LogRecord) -> None:
        sys.stderr.write(self.format(record) + "\n")


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
    handler = _CurrentStderrHandler()
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


# --- through fit (3.5) -----------------------------------------------------

Data = tuple[pd.DataFrame, pd.Series]
LoggerState = tuple[list[tuple[logging.Handler, int]], bool]

# Small parameters that keep a fit well under a second. With them the
# reference data converges after 2 iterations (the second repeats the
# mask of the first), so max_iter=3 converges and max_iter=1 stops short.
FAST: dict[str, Any] = {
    "method": "pls",
    "n_estimators": 2,
    "max_iter": 3,
    "random_state": 12,
    "n_jobs": 1,
}
N_ITER_FAST = 2
# Start of the detector's own ConvergenceWarning message. Matching it keeps
# the tests independent of ConvergenceWarnings from the submodels.
NOT_CONVERGED = "EnsembleOutlierSampleDetector did not converge"
# What Optuna logs at INFO for every study and every trial.
TRIAL_LOG = re.compile(r"A new study created|Trial \d+ finished")
# The descriptions fit gives its bar, at the start of a rendered line.
DESCRIPTION = re.compile(r"(?:Finished! )?\[iter \d+ / \d+\]")


class CreatedBar(NamedTuple):
    bar: Any
    disabled_at_creation: bool


class SubmodelError(RuntimeError):
    """Raised by a patched submodel to stop ``fit`` midway."""


class Failure(NamedTuple):
    overrides: dict[str, Any]
    error: type[Exception]
    match: str


def fit_fast(
    X: Any, y: Any, **overrides: Any
) -> EnsembleOutlierSampleDetector:
    return EnsembleOutlierSampleDetector(**{**FAST, **overrides}).fit(X, y)


def rendered_lines(err: str) -> list[str]:
    """Split tqdm output into the lines it rendered, in order."""
    return [line for line in re.split(r"[\r\n]", err) if line]


def descriptions_shown(lines: list[str]) -> list[str]:
    """Return the bar descriptions in the order they were first shown."""
    shown: list[str] = []
    for line in lines:
        match = DESCRIPTION.match(line)
        if match is not None and (not shown or shown[-1] != match.group()):
            shown.append(match.group())
    return shown


@pytest.fixture
def created_bars(monkeypatch: pytest.MonkeyPatch) -> list[CreatedBar]:
    """Record every progress bar that ``fit`` creates.

    ``fit`` calls ``_runtime.make_progress_bar`` through the ``_runtime``
    module it imported, so the recording wrapper replaces the function on
    that module object and returns the real bar.
    """
    runtime = detector_module._runtime
    real = runtime.make_progress_bar
    created: list[CreatedBar] = []

    def recording(enabled: bool, total: int) -> Any:
        bar = real(enabled, total)
        created.append(CreatedBar(bar, bar.disable))
        return bar

    monkeypatch.setattr(runtime, "make_progress_bar", recording)
    return created


@pytest.fixture
def states_in_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, LoggerState]]:
    """Record Optuna's verbosity and logger state at every new study."""
    logger = logging.getLogger("optuna")
    real = optuna.create_study
    seen: list[tuple[int, LoggerState]] = []

    def recording(*args: Any, **kwargs: Any) -> optuna.Study:
        seen.append((optuna.logging.get_verbosity(), _logger_state(logger)))
        return real(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", recording)
    return seen


def invalid_cv(monkeypatch: pytest.MonkeyPatch) -> Failure:
    # The first trial's cross-validation raises inside study.optimize;
    # Optuna logs the failed trial and re-raises.
    return Failure({"cv": 1000}, ValueError, "n_splits=1000")


def second_submodel_raises(monkeypatch: pytest.MonkeyPatch) -> Failure:
    # The first submodel is tuned as usual, so fit fails midway.
    real = detector_module._fit_predict_submodel
    calls: list[None] = []

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        if len(calls) == 2:
            raise SubmodelError("second submodel failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(detector_module, "_fit_predict_submodel", failing)
    return Failure({}, SubmodelError, "second submodel failed")


# 5.1


@pytest.mark.parametrize(
    "previous", [optuna.logging.DEBUG, optuna.logging.INFO]
)
def test_detector_like_study_logs_trials_outside_fit(
    optuna_stderr: pytest.CaptureFixture[str], previous: int
) -> None:
    # Guards the next test: a study created the way fit creates its
    # studies logs both kinds of line at this verbosity when nothing
    # hides them, so an empty result there is not vacuous.
    optuna.logging.set_verbosity(previous)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=2
    )
    lines = optuna_stderr.readouterr().err.splitlines()
    # Lines must start with the message, as the handler writes them; a
    # logging error report would only quote it after "Message: ".
    starts = [match.group() for match in map(TRIAL_LOG.match, lines) if match]
    assert starts == [
        "A new study created",
        "Trial 0 finished",
        "Trial 1 finished",
    ]


@pytest.mark.parametrize(
    "previous", [optuna.logging.DEBUG, optuna.logging.INFO]
)
def test_fit_emits_no_trial_logs(
    optuna_stderr: pytest.CaptureFixture[str],
    reference_data: Data,
    previous: int,
) -> None:
    X, y = reference_data
    optuna.logging.set_verbosity(previous)
    detector = fit_fast(X, y, progress_bar=False)
    # 2 iterations of 2 submodels: 4 studies were tuned.
    assert detector.n_iter_ == N_ITER_FAST
    err = optuna_stderr.readouterr().err
    assert TRIAL_LOG.search(err) is None, err


# 5.2


@pytest.mark.parametrize("previous", VERBOSITIES)
def test_fit_restores_logging_state(
    reference_data: Data,
    default_handler: tuple[logging.Handler, bool],
    states_in_fit: list[tuple[int, LoggerState]],
    previous: int,
) -> None:
    handler, attached = default_handler
    X, y = reference_data
    logger = logging.getLogger("optuna")
    optuna.logging.set_verbosity(previous)
    before = _logger_state(logger)
    fit_fast(X, y, progress_bar=False)
    # While fit runs, only the verbosity is lowered.
    assert len(states_in_fit) == N_ITER_FAST * FAST["n_estimators"]
    for state in states_in_fit:
        assert state == (optuna.logging.WARNING, before)
    assert optuna.logging.get_verbosity() == previous
    assert _logger_state(logger) == before
    assert (handler in logger.handlers) is attached


@pytest.mark.parametrize(
    "failure",
    [invalid_cv, second_submodel_raises],
    ids=lambda failure: failure.__name__,
)
@pytest.mark.parametrize("previous", VERBOSITIES)
def test_fit_restores_logging_state_after_an_exception(
    reference_data: Data,
    default_handler: tuple[logging.Handler, bool],
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[pytest.MonkeyPatch], Failure],
    previous: int,
) -> None:
    handler, attached = default_handler
    X, y = reference_data
    logger = logging.getLogger("optuna")
    overrides, error, match = failure(monkeypatch)
    optuna.logging.set_verbosity(previous)
    before = _logger_state(logger)
    with pytest.raises(error, match=match):
        fit_fast(X, y, progress_bar=False, **overrides)
    assert optuna.logging.get_verbosity() == previous
    assert _logger_state(logger) == before
    assert (handler in logger.handlers) is attached


# 5.3


def test_progress_bar_shows_iterations_and_finishes(
    reference_data: Data,
    created_bars: list[CreatedBar],
    capsys: pytest.CaptureFixture[str],
) -> None:
    X, y = reference_data
    detector = fit_fast(X, y, progress_bar=True)
    max_iter, n_iter = FAST["max_iter"], detector.n_iter_
    total = max_iter * FAST["n_estimators"]
    # Converging before max_iter shows the early stop in the description.
    assert n_iter == N_ITER_FAST < max_iter
    captured = capsys.readouterr()
    lines = rendered_lines(captured.err)
    # Iterations are counted from 1, and the last description says so.
    assert descriptions_shown(lines) == [
        *(f"[iter {k} / {max_iter}]" for k in range(1, n_iter + 1)),
        f"Finished! [iter {n_iter} / {max_iter}]",
    ]
    last = lines[-1]
    assert last.startswith(f"Finished! [iter {n_iter} / {max_iter}]: 100%|")
    assert f"| {total}/{total} " in last
    assert captured.out == ""
    (created,) = created_bars
    assert created.bar.n == created.bar.total == total
    # tqdm sets disable when the bar is closed.
    assert created.bar.disable


# 5.4


@pytest.mark.parametrize("progress_bar", [True, False])
def test_fit_creates_its_bar_with_auto_tqdm(
    reference_data: Data,
    created_bars: list[CreatedBar],
    progress_bar: bool,
) -> None:
    X, y = reference_data
    fit_fast(X, y, progress_bar=progress_bar)
    # A shown and a hidden bar both come from make_progress_bar.
    assert len(created_bars) == 1, "fit bypassed _runtime.make_progress_bar"
    (created,) = created_bars
    # tqdm.auto picks the notebook widget on a Jupyter kernel.
    assert isinstance(created.bar, tqdm.auto.tqdm)
    assert created.disabled_at_creation is not progress_bar
    assert created.bar.total == FAST["max_iter"] * FAST["n_estimators"]


# 5.5


def test_progress_bar_closes_when_fit_does_not_converge(
    reference_data: Data,
    created_bars: list[CreatedBar],
    capsys: pytest.CaptureFixture[str],
) -> None:
    X, y = reference_data
    with pytest.warns(ConvergenceWarning, match=NOT_CONVERGED) as record:
        detector = fit_fast(X, y, progress_bar=True, max_iter=1)
    own = [w for w in record if str(w.message).startswith(NOT_CONVERGED)]
    assert len(own) == 1
    assert detector.n_iter_ == 1
    total = FAST["n_estimators"]
    lines = rendered_lines(capsys.readouterr().err)
    assert lines[-1].startswith("Finished! [iter 1 / 1]: 100%|")
    assert f"| {total}/{total} " in lines[-1]
    (created,) = created_bars
    assert not created.disabled_at_creation
    assert created.bar.disable, "the bar was left open"


def test_progress_bar_closes_when_fit_raises(
    reference_data: Data, created_bars: list[CreatedBar]
) -> None:
    X, y = reference_data
    with pytest.raises(ValueError, match="n_splits=1000"):
        fit_fast(X, y, progress_bar=True, cv=1000)
    (created,) = created_bars
    assert not created.disabled_at_creation
    assert created.bar.disable, "the bar was left open"


# 5.6


def test_fit_without_progress_bar_writes_nothing(
    reference_data: Data, capfd: pytest.CaptureFixture[str]
) -> None:
    X, y = reference_data
    detector = fit_fast(X, y, progress_bar=False)
    assert detector.n_iter_ == N_ITER_FAST
    assert capfd.readouterr() == ("", "")


# 5.7


def test_running_the_module_starts_no_debugger(tmp_path: Path) -> None:
    # PYTHONBREAKPOINT=0 would turn a breakpoint() call into a no-op.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONBREAKPOINT"}
    completed = subprocess.run(
        [sys.executable, "-m", detector_module.__name__],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    # stderr is not required to be empty: runpy warns that the module is
    # already imported, because the package __init__ imports it first.
    assert completed.returncode == 0, completed.stderr
    assert "(Pdb)" not in completed.stdout
    assert "(Pdb)" not in completed.stderr
