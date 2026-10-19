"""Generate the golden case files from the baseline commit.

Run from the repository root with the interpreter of the matching
environment (see tests/golden/README.md)::

    python tests/golden/generate_golden.py --env legacy
    python tests/golden/generate_golden.py --env modern

The detector source is read from the fixed baseline commit with
``git show``, the named patches are applied to it, and the result is
imported as ``eosd_baseline`` from a temporary directory. The installed
package and the working tree are never imported.

Each case file holds a ``legacy`` and a ``modern`` block. A run writes
only the block of its ``--env`` and keeps the other one. A block whose
result already matches is left untouched; a block whose result differs
is overwritten only with ``--force --reason TEXT``.

This script must stay compatible with Python 3.8 (the legacy env).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

BASELINE_COMMIT = "072d1dfe7ff79b4ebcc411f2b5c13269697e7eac"
BASELINE_SOURCE = (
    "ensemble_outlier_sample_detection/ensemble_outlier_sample_detection.py"
)
MODULE_NAME = "eosd_baseline"
SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = ("schema_version", "case", "params", "data")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = "example/example_chache.csv"
DATA_TARGET = "y"
DEFAULT_CASES_DIR = REPO_ROOT / "tests" / "golden" / "cases"
UPDATE_LOG = "tests/golden/README.md"
DEFAULT_REASON = "initial baseline"

ENVIRONMENTS = ("legacy", "modern")
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "optuna", "tqdm")

# Keep in sync with tests/golden/environment-legacy.yml.
LEGACY_PYTHON = "3.8.20"
LEGACY_VERSIONS = {
    "numpy": "1.20.1",
    "scipy": "1.6.2",
    "scikit-learn": "0.24.1",
    "pandas": "1.2.4",
    "optuna": "2.8.0",
    "tqdm": "4.59.0",
}
# Keep in sync with tests/golden/requirements-modern.txt.
MODERN_MIN_PYTHON = (3, 10)
MODERN_VERSIONS = {
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "pandas": "3.0.6",
    "optuna": "5.0.0",
    "tqdm": "4.70.1",
}
EXPECTED_VERSIONS = {"legacy": LEGACY_VERSIONS, "modern": MODERN_VERSIONS}

METHODS = ("pls", "svr")
METRICS = ("r2", "mse", "mae", "rmse")
# Number of submodels per method, sized for the CI time budget.
N_ESTIMATORS = {"pls": 30, "svr": 5}
# Parameter adjustments for cases that do not converge within max_iter
# with the common parameters, e.g. {"pls-r2": {"max_iter": 50}}.
# Record every entry and its reason in the update log of UPDATE_LOG.
PARAM_OVERRIDES: dict[str, dict[str, Any]] = {}


def case_params(method: str, metric: str) -> dict[str, Any]:
    """Return the detector parameters of one case."""
    params = {
        "method": method,
        "metric": metric,
        "random_state": 334,
        "n_estimators": N_ESTIMATORS[method],
        "max_iter": 30,
        "cv": 5,
        "max_components": 30,
        "n_jobs": 1,
        "progress_bar": False,
    }
    return {**params, **PARAM_OVERRIDES.get(f"{method}-{metric}", {})}


CASES = {
    f"{method}-{metric}": case_params(method, metric)
    for method in METHODS
    for metric in METRICS
}


class Patch(NamedTuple):
    """Replace ``old``, which must occur exactly once, with ``new``."""

    name: str
    old: str
    new: str
    envs: tuple[str, ...]
    metrics: tuple[str, ...]
    optional: bool


# Indentation of the body of the outer loop in ``fit``.
_LOOP_INDENT = " " * 12

# Applied in this order. Each patch either keeps the computation as it
# is or is the minimal fix listed in design.md ("Named patches").
PATCHES = (
    Patch(
        name="svr-prediction-shape",
        old="y_preds.append(self.scaler_y.inverse_transform(y_pred))",
        new=(
            "y_preds.append(self.scaler_y.inverse_transform("
            "np.reshape(y_pred, (-1, 1))))"
        ),
        envs=ENVIRONMENTS,
        metrics=METRICS,
        optional=False,
    ),
    Patch(
        name="modern-objective-shape",
        old="self.elo.scaler_y.inverse_transform(y_pred_oof)",
        new=(
            "self.elo.scaler_y.inverse_transform("
            "np.reshape(y_pred_oof, (-1, 1)))"
        ),
        envs=("modern",),
        metrics=METRICS,
        optional=False,
    ),
    Patch(
        name="modern-rmse",
        old=(
            "self.metric = lambda x, y:"
            "mean_squared_error(x, y, squared = False)"
        ),
        new="self.metric = lambda x, y: np.sqrt(mean_squared_error(x, y))",
        envs=("modern",),
        metrics=METRICS,
        optional=False,
    ),
    Patch(
        name="record-median",
        old=_LOOP_INDENT + "df_y_preds = pd.DataFrame(np.hstack(y_preds))\n",
        new=(
            _LOOP_INDENT
            + "df_y_preds = pd.DataFrame(np.hstack(y_preds))\n"
            + _LOOP_INDENT
            + "self._median_prediction = "
            + "np.median(np.hstack(y_preds), axis=1)\n"
        ),
        envs=ENVIRONMENTS,
        metrics=METRICS,
        optional=False,
    ),
    Patch(
        name="r2-argument-order",
        old="self.metric = r2_score",
        new="self.metric = lambda y_pred, y_true: r2_score(y_true, y_pred)",
        envs=ENVIRONMENTS,
        metrics=("r2",),
        optional=True,
    ),
)
PATCHES_BY_NAME = {patch.name: patch for patch in PATCHES}
OPTIONAL_PATCHES = tuple(patch.name for patch in PATCHES if patch.optional)
# Removed for the PLS neutrality check in the legacy env.
NEUTRALITY_PATCH = "svr-prediction-shape"


class GoldenError(Exception):
    """An error that aborts the run with a message and exit code 1."""


class Update(NamedTuple):
    """The planned change of one case file."""

    name: str
    path: Path
    status: str  # "unchanged", "added" or "changed"
    text: str | None
    changes: tuple[str, ...]
    stale_blocks: tuple[str, ...]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline commit "
            f"{BASELINE_COMMIT[:7]} with the named patches and write the "
            "legacy or modern block of the golden case files."
        ),
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=ENVIRONMENTS,
        help="block to generate; the interpreter must match its pins",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="NAME",
        help="cases to generate (default: all): " + ", ".join(CASES),
    )
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        choices=OPTIONAL_PATCHES,
        help="optional patch to apply (only to the cases it targets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite blocks whose result differs (requires --reason)",
    )
    parser.add_argument(
        "--reason",
        metavar="TEXT",
        help=f"reason recorded in provenance (default: {DEFAULT_REASON!r})",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=DEFAULT_CASES_DIR,
        metavar="DIR",
        help="directory of the case files (default: tests/golden/cases)",
    )
    return parser.parse_args(argv)


def resolve_cases(requested: list[str] | None) -> tuple[str, ...]:
    """Return the selected case names in table order."""
    if not requested:
        return tuple(CASES)
    unknown = [name for name in requested if name not in CASES]
    if unknown:
        raise GoldenError(
            f"unknown case(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(CASES)}"
        )
    return tuple(name for name in CASES if name in requested)


def installed_versions() -> dict[str, str | None]:
    """Return the versions of Python and the checked packages."""

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        **{name: version(name) for name in PACKAGES},
    }


def environment_problems(
    env: str, versions: dict[str, str | None]
) -> list[str]:
    """Return how ``versions`` deviate from the pins of ``env``."""
    python = versions["python"] or ""
    if env == "legacy":
        python_ok = python == LEGACY_PYTHON
        python_expected = LEGACY_PYTHON
    else:
        major_minor = tuple(int(part) for part in python.split(".")[:2])
        python_ok = major_minor >= MODERN_MIN_PYTHON
        python_expected = ">= {}.{}".format(*MODERN_MIN_PYTHON)
    python_problems = (
        []
        if python_ok
        else [f"python: expected {python_expected}, found {python}"]
    )
    return python_problems + [
        f"{name}: expected {expected}, "
        f"found {versions[name] or 'not installed'}"
        for name, expected in EXPECTED_VERSIONS[env].items()
        if versions[name] != expected
    ]


def select_patches(
    env: str, metric: str, requested: list[str]
) -> tuple[str, ...]:
    """Return the names of the patches that apply to a case, in order."""
    return tuple(
        patch.name
        for patch in PATCHES
        if env in patch.envs
        and metric in patch.metrics
        and (not patch.optional or patch.name in requested)
    )


def apply_patches(source: str, names: tuple[str, ...]) -> str:
    """Apply the named patches to ``source`` in the given order."""
    for name in names:
        patch = PATCHES_BY_NAME[name]
        count = source.count(patch.old)
        if count != 1:
            raise GoldenError(
                f"patch {name!r}: expected exactly one occurrence of "
                f"{patch.old!r} in the baseline source, found {count}. "
                "Fix the patch definition in generate_golden.py."
            )
        source = source.replace(patch.old, patch.new)
    return source


def read_baseline_source() -> str:
    """Return the detector source of the baseline commit."""
    command = [
        "git",
        "-C",
        str(REPO_ROOT),
        "show",
        f"{BASELINE_COMMIT}:{BASELINE_SOURCE}",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise GoldenError(
            f"cannot run git ({exc}); git is required to read the "
            "baseline source"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GoldenError(
            f"`{' '.join(command)}` failed: {stderr}. The baseline commit "
            "must be available in the local clone (fetch the full history)."
        )
    return completed.stdout.decode("utf-8")


def import_source(source: str, directory: Path) -> Any:
    """Write ``source`` as ``eosd_baseline.py`` and import it."""
    path = directory / f"{MODULE_NAME}.py"
    path.write_bytes(source.encode("utf-8"))
    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(path))
    if spec is None or spec.loader is None:
        raise GoldenError(f"cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def neutral_patches(patches: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``patches`` without the SVR shape patch."""
    return tuple(name for name in patches if name != NEUTRALITY_PATCH)


def needs_neutrality_check(env: str, params: dict[str, Any]) -> bool:
    """Return whether a case is re-run without the SVR shape patch."""
    return env == "legacy" and params["method"] == "pls"


def required_patch_sets(
    env: str, names: tuple[str, ...], requested: list[str]
) -> tuple[tuple[str, ...], ...]:
    """Return every patch combination the run needs, without duplicates."""
    main_sets = [
        select_patches(env, CASES[name]["metric"], requested) for name in names
    ]
    neutral_sets = [
        neutral_patches(patches)
        for name, patches in zip(names, main_sets)
        if needs_neutrality_check(env, CASES[name])
    ]
    return tuple(dict.fromkeys(main_sets + neutral_sets))


def load_reference_data() -> tuple[Any, Any, dict[str, Any]]:
    """Return ``X``, ``y`` and the ``data`` entry of the case files."""
    # Imported here so that --help and the environment check do not
    # depend on pandas.
    import pandas as pd

    path = REPO_ROOT / DATA_PATH
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GoldenError(f"cannot read the reference data: {exc}") from exc
    frame = pd.read_csv(io.BytesIO(raw), index_col=0, comment="#")
    X, y = frame.iloc[:, 1:], frame.iloc[:, 0]
    if y.name != DATA_TARGET:
        raise GoldenError(
            f"{DATA_PATH}: expected the first column to be "
            f"{DATA_TARGET!r}, found {y.name!r}"
        )
    info = {
        "path": DATA_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "target": DATA_TARGET,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X, y, info


def first_line(message: object) -> str:
    """Return the first line of a warning message."""
    lines = str(message).splitlines()
    return lines[0] if lines else ""


class Run(NamedTuple):
    """The outcome of fitting one case."""

    result: dict[str, Any] | None  # None if the loop did not converge
    warning_counts: dict[str, int]  # "Category: message" -> count


def run_case(module: Any, params: dict[str, Any], X: Any, y: Any) -> Run:
    """Fit the baseline detector and collect the warnings it emits."""
    detector = module.EnsembleOutlierSampleDetector(**params)
    # The baseline warns once per submodel fit (e.g. SVR receives a
    # column-vector y), so the warnings are counted instead of printed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.fit(X.copy(), y.copy())
    counts = Counter(
        f"{item.category.__name__}: {first_line(item.message)}"
        for item in caught
    )
    # The baseline sets outlier_support_ only when the loop converges.
    if not hasattr(detector, "outlier_support_"):
        return Run(None, dict(counts))
    result = {
        "outlier_support": detector.outlier_support_.tolist(),
        "inlier_support": detector.inlier_support_.tolist(),
        # n_iter_finished is the 0-based index of the last iteration.
        "n_iter": int(detector.n_iter_finished) + 1,
        "median_prediction": detector._median_prediction.tolist(),
    }
    return Run(result, dict(counts))


def checked_run(
    name: str,
    env: str,
    patches: tuple[str, ...],
    module: Any,
    X: Any,
    y: Any,
) -> Run:
    """Run one case and attach the case to any exception it raises."""
    try:
        return run_case(module, CASES[name], X, y)
    except Exception as exc:
        raise GoldenError(
            f"case {name!r} raised {type(exc).__name__} under --env {env} "
            f"with patches {list(patches)}: {exc}"
        ) from exc


def report_warnings(label: str, counts: dict[str, int]) -> None:
    """Print the warnings collected during a fit, one line per kind."""
    for message, count in sorted(counts.items()):
        print(f"{label}: warning x{count}: {message}", file=sys.stderr)


def result_differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Describe how two results differ, one line per field."""

    def differing(key: str) -> str | None:
        before, after = old.get(key), new.get(key)
        if before == after:
            return None
        if not (isinstance(before, list) and isinstance(after, list)):
            return f"{key}: {before!r} -> {after!r}"
        if len(before) != len(after):
            return f"{key}: length {len(before)} -> {len(after)}"
        indices = [
            index for index, (a, b) in enumerate(zip(before, after)) if a != b
        ]
        shown = ", ".join(str(index) for index in indices[:20])
        more = ", ..." if len(indices) > 20 else ""
        return (
            f"{key}: differs at {len(indices)} sample(s), "
            f"index [{shown}{more}]"
        )

    keys = ("outlier_support", "inlier_support", "n_iter", "median_prediction")
    return [line for line in map(differing, keys) if line is not None]


def neutrality_problem(
    name: str,
    env: str,
    patches: tuple[str, ...],
    result: dict[str, Any],
    variants: dict[tuple[str, ...], Any],
    X: Any,
    y: Any,
) -> str | None:
    """Re-run a PLS case without the SVR shape patch and compare."""
    without = neutral_patches(patches)
    run = checked_run(name, env, without, variants[without], X, y)
    report_warnings(f"[{env}] {name} (neutrality)", run.warning_counts)
    differences = (
        ["did not converge"]
        if run.result is None
        else result_differences(result, run.result)
    )
    if not differences:
        return None
    return (
        f"PLS patch neutrality check failed for {name!r}: removing "
        f"{NEUTRALITY_PATCH!r} changed the result "
        f"({'; '.join(differences)}). The SVR shape patch must not change "
        "PLS results; fix the patch definition."
    )


def nonconvergence_problem(name: str, env: str) -> str:
    """Explain which case to adjust when the baseline did not converge."""
    return (
        f"case {name!r} did not converge within max_iter="
        f"{CASES[name]['max_iter']} under --env {env}, so the baseline set "
        "no outlier_support_. Adjust the parameters of this case in "
        "PARAM_OVERRIDES in tests/golden/generate_golden.py and record the "
        f"change and its reason in the update log of {UPDATE_LOG}."
    )


def compute_results(
    env: str,
    names: tuple[str, ...],
    requested: list[str],
    variants: dict[tuple[str, ...], Any],
    X: Any,
    y: Any,
) -> dict[str, tuple[tuple[str, ...], dict[str, Any]]]:
    """Run every case; abort if one did not converge or was not neutral.

    All cases are run before aborting so that every case to adjust is
    reported at once.
    """
    results = {}
    problems = []
    for name in names:
        patches = select_patches(env, CASES[name]["metric"], requested)
        label = f"[{env}] {name}"
        start = time.perf_counter()
        run = checked_run(name, env, patches, variants[patches], X, y)
        elapsed = time.perf_counter() - start
        report_warnings(label, run.warning_counts)
        if run.result is None:
            print(f"{label}: did not converge", flush=True)
            problems.append(nonconvergence_problem(name, env))
            continue
        print(
            f"{label}: {sum(run.result['outlier_support'])} outlier(s), "
            f"n_iter={run.result['n_iter']}, patches={list(patches)} "
            f"({elapsed:.1f} s)",
            flush=True,
        )
        if needs_neutrality_check(env, CASES[name]):
            problem = neutrality_problem(
                name, env, patches, run.result, variants, X, y
            )
            if problem is not None:
                problems.append(problem)
                continue
            print(
                f"{label}: PLS patch neutrality check passed (identical "
                f"mask, n_iter and median without {NEUTRALITY_PATCH!r})",
                flush=True,
            )
        results[name] = (patches, run.result)
    if problems:
        raise GoldenError(
            "no case file was written:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    return results


def read_case_file(path: Path, name: str) -> dict[str, Any] | None:
    """Return the existing case file, or None if there is none."""
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GoldenError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise GoldenError(f"{path}: expected a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise GoldenError(
            f"{path}: unsupported schema_version "
            f"{document.get('schema_version')!r} (this script writes "
            f"{SCHEMA_VERSION})"
        )
    if document.get("case") != name:
        raise GoldenError(
            f"{path}: expected case {name!r}, found {document.get('case')!r}"
        )
    unknown = sorted(set(document) - set(TOP_LEVEL_KEYS + ENVIRONMENTS))
    if unknown:
        raise GoldenError(
            f"{path}: unknown top-level key(s) {unknown}; refusing to "
            "rewrite a file this script does not fully understand"
        )
    return document


def render(path: Path, document: dict[str, Any]) -> str:
    """Serialize a case file as UTF-8 JSON with indent 2."""
    try:
        text = json.dumps(
            document, indent=2, ensure_ascii=False, allow_nan=False
        )
    except ValueError as exc:
        raise GoldenError(
            f"{path}: cannot serialize the result: {exc}"
        ) from exc
    return text + "\n"


def same_block(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    """Return whether a block needs no rewrite."""
    if not isinstance(old, dict):
        return False
    old_provenance = old.get("provenance") or {}
    return (
        old.get("result") == new["result"]
        and old_provenance.get("patches") == new["provenance"]["patches"]
        and old_provenance.get("baseline_commit") == BASELINE_COMMIT
    )


def plan_update(
    name: str,
    path: Path,
    existing: dict[str, Any] | None,
    header: dict[str, Any],
    env: str,
    block: dict[str, Any],
    reason: str | None,
) -> Update:
    """Return what writing ``block`` into the case file would change.

    ``block["provenance"]["reason"]`` is filled in here: ``reason`` or
    the default for a new block, ``reason`` for a changed one.
    """
    old = existing or {}
    header_changes = [
        f"{key}: {old.get(key)!r} -> {header[key]!r}"
        for key in ("params", "data")
        if existing is not None and old.get(key) != header[key]
    ]
    old_block = old.get(env)
    if not header_changes and same_block(old_block, block):
        return Update(name, path, "unchanged", None, (), ())
    is_new = not header_changes and old_block is None
    status = "added" if is_new else "changed"
    final_reason = (reason or DEFAULT_REASON) if is_new else reason
    final_block = {
        "result": block["result"],
        "provenance": {**block["provenance"], "reason": final_reason},
    }
    blocks = {key: old[key] for key in ENVIRONMENTS if key in old}
    blocks[env] = final_block
    document = {
        **header,
        **{key: blocks[key] for key in ENVIRONMENTS if key in blocks},
    }
    block_changes: list[str] = []
    if isinstance(old_block, dict):
        block_changes = result_differences(
            old_block.get("result") or {}, block["result"]
        )
        old_patches = (old_block.get("provenance") or {}).get("patches")
        if old_patches != block["provenance"]["patches"]:
            block_changes.append(
                f"patches: {old_patches} -> {block['provenance']['patches']}"
            )
    stale = tuple(key for key in blocks if key != env and header_changes)
    return Update(
        name,
        path,
        status,
        render(path, document),
        tuple(header_changes + block_changes),
        stale,
    )


def write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8, LF) in one rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(str(temporary), str(path))


def conflict_message(env: str, conflicts: list[Update]) -> str:
    """Explain why differing blocks are not overwritten."""
    lines = [
        (
            f"the {env} block of {len(conflicts)} case file(s) differs from "
            "the new result; nothing was written:"
        )
    ]
    for update in conflicts:
        lines.append(f"  - {update.name} ({update.path})")
        lines.extend(f"      {change}" for change in update.changes)
    lines.append(
        "If the change comes from this package's code, fix the code and "
        "keep the golden. If it comes from a dependency and the maintainer "
        "approved the update, rerun with --force --reason TEXT and add a "
        f"row to the update log in {UPDATE_LOG}."
    )
    return "\n".join(lines)


def check_environment(env: str) -> dict[str, str | None]:
    """Return the installed versions; abort if they miss the pins."""
    versions = installed_versions()
    problems = environment_problems(env, versions)
    if problems:
        pins = (
            "tests/golden/environment-legacy.yml"
            if env == "legacy"
            else "tests/golden/requirements-modern.txt"
        )
        raise GoldenError(
            f"this interpreter ({sys.executable}) does not match the "
            f"pins of --env {env} ({pins}):\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\nRun the script with the interpreter of that environment."
        )
    return versions


def plan_updates(
    args: argparse.Namespace,
    results: dict[str, tuple[tuple[str, ...], dict[str, Any]]],
    data: dict[str, Any],
    environment: dict[str, str | None],
) -> list[Update]:
    """Plan the change of every case file; nothing is written here."""
    generated_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    def plan(name: str, patches: tuple[str, ...], result: dict) -> Update:
        path = args.cases_dir / f"{name}.json"
        header = {
            "schema_version": SCHEMA_VERSION,
            "case": name,
            "params": CASES[name],
            "data": data,
        }
        provenance = {
            "baseline_commit": BASELINE_COMMIT,
            "patches": list(patches),
            "environment": environment,
            "generated_at": generated_at,
        }
        return plan_update(
            name=name,
            path=path,
            existing=read_case_file(path, name),
            header=header,
            env=args.env,
            block={"result": result, "provenance": provenance},
            reason=args.reason,
        )

    return [
        plan(name, patches, result)
        for name, (patches, result) in results.items()
    ]


def write_updates(env: str, updates: list[Update]) -> None:
    """Write the planned case files and report what changed."""
    for update in updates:
        if update.text is None:
            print(f"unchanged: {update.path} ({env} block matches)")
            continue
        write_text(update.path, update.text)
        print(f"{update.status}: {update.path} ({env} block)")
        for change in update.changes:
            print(f"    {change}")
        for block in update.stale_blocks:
            print(
                f"warning: {update.name}: params/data changed, so the "
                f"{block} block is stale; regenerate it with --env {block} "
                f"--cases {update.name} --force --reason TEXT",
                file=sys.stderr,
            )
    written = [update.name for update in updates if update.text is not None]
    if written:
        print(
            f"Reminder: add a row for each written block ({env}: "
            f"{', '.join(written)}) to the update log in {UPDATE_LOG}."
        )


def generate(args: argparse.Namespace) -> None:
    """Run the whole generation; raise GoldenError on any refusal."""
    if args.force and not args.reason:
        raise GoldenError(
            "--force requires --reason TEXT (the reason is recorded in "
            "the provenance of the overwritten block)"
        )
    names = resolve_cases(args.cases)
    versions = check_environment(args.env)
    source = read_baseline_source()
    X, y, data = load_reference_data()
    print(
        f"[{args.env}] baseline {BASELINE_COMMIT[:12]}, "
        f"{len(names)} case(s) -> {args.cases_dir}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="eosd-golden-") as workdir:
        # One module per patch combination, each in its own directory.
        variants = {
            patches: import_source(
                apply_patches(source, patches),
                Path(tempfile.mkdtemp(dir=workdir)),
            )
            for patches in required_patch_sets(args.env, names, args.patch)
        }
        results = compute_results(args.env, names, args.patch, variants, X, y)
    environment = {**versions, "platform": platform.platform()}
    updates = plan_updates(args, results, data, environment)
    conflicts = [update for update in updates if update.status == "changed"]
    if conflicts and not args.force:
        raise GoldenError(conflict_message(args.env, conflicts))
    write_updates(args.env, updates)


def main(argv: list[str] | None = None) -> int:
    """Entry point; return the process exit code."""
    args = parse_args(argv)
    try:
        generate(args)
    except GoldenError as exc:
        cause = exc.__cause__
        if cause is not None and not isinstance(cause, GoldenError):
            traceback.print_exception(type(cause), cause, cause.__traceback__)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
