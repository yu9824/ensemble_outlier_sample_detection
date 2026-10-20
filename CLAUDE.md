# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python library (PyPI: `ensemble-outlier-sample-detection`, import name `ensemble_outlier_sample_detection`) implementing ensemble-based outlier sample detection for regression data (Kaneko 2018, Chemometr. Intell. Lab. Syst. 177, 74-82; reference impl: https://github.com/hkaneko1985/ensemble_outlier_sample_detection). The only public API is `EnsembleOutlierSampleDetector`, a scikit-learn estimator. Version 1.0.0 (see `CHANGELOG.md`) modernized packaging, dependencies and the API; project context lives in `.kiro/steering/` and the completed spec in `.kiro/specs/modernize-with-golden-tests/`.

## Commands

```bash
pip install -e ".[test,dev]"          # dev install (extras: test, dev, example)
pytest                                # full suite incl. golden tests (~30-60 s)
ruff check . && ruff format --check .
mypy                                  # mypy.ini: files = src
python -m build                       # sdist + wheel (version from git tags via setuptools-scm)
```

Local conda envs used for golden data: `eosd-golden-legacy` (Python 3.8, 2021-era deps, osx-arm64; from `tests/golden/environment-legacy.yml`) and `eosd-modern` (Python 3.12 + `tests/golden/requirements-modern.txt`, plus pytest/build/ruff/mypy). The example notebook needs the `example` extra and is run from inside `example/` (`jupyter nbconvert --to notebook --execute --inplace example.ipynb`).

## Architecture

`src/ensemble_outlier_sample_detection/`; dependency direction is `_runtime` / `_submodels` -> `ensemble_outlier_sample_detection` (detector) -> `__init__`; the private modules never import each other.

- `ensemble_outlier_sample_detection.py`: `EnsembleOutlierSampleDetector.fit(X, y)` validates inputs up front (`ValueError`), then runs an outer loop (up to `max_iter`) until the outlier mask stops changing:
  1. Take samples not flagged in the previous iteration.
  2. For each of `n_estimators` bootstraps: drop zero-variance features, standardize X and y, tune PLS (`n_components`) or RBF-SVR (`C`, `epsilon`, `gamma`) with Optuna TPE on the `cross_val_predict` OOF score, refit, and predict on **all** samples into one column of a prediction matrix.
  3. Flag `|y - median(preds)| > 3 * 1.4826 * MAD`; converged when the mask repeats or no inliers remain. Otherwise the last judgment is kept and `ConvergenceWarning` is emitted.
  Fitted attributes: `outlier_support_`, `inlier_support_`, `n_iter_` (executed iterations), `n_features_in_`, `feature_names_in_`, plus private `_median_prediction` used by the golden tests. `fit` returns `self` and never mutates constructor params.
- `_submodels.py`: method/metric normalization, search space, model factory, trial counts, and the objective closure (takes everything as arguments; no back-reference to the detector). Metrics are called as `score(y_true, y_pred)`.
- `_runtime.py`: context manager that sets Optuna verbosity to WARNING during fit and restores it (default handler untouched), and a `tqdm.auto` progress-bar factory.

### Invariants guarded by the golden tests

- One `check_random_state` RNG; per bootstrap it draws `randint(2**32)` twice in this order: resample seed, then TPE seed. Changing the order changes results.
- For PLS, `max_components` is a running minimum of the scaled bootstrap's matrix rank, carried across bootstraps and iterations (kept in a local variable).
- MAD = median over previous inliers of |all predictions - median of per-sample medians|; scores are computed on the original y scale.

## Testing and golden data

- `tests/golden/cases/*.json` (8 cases: pls/svr x r2/mse/mae/rmse) hold a `legacy` block (2021 env; masks and `n_iter` compared exactly) and a `modern` block (latest env; masks/`n_iter` exact, `_median_prediction` with tolerance: PLS rtol=1e-7/atol=1e-10, SVR rtol=atol=1e-3 because SVR differs by ~4e-4 between macOS arm64 and Linux CI).
- Never hand-edit case files. They are produced by `tests/golden/generate_golden.py` (Python 3.8-compatible), which runs the baseline commit's code via `git show` with named exact-once patches; r2 cases must be regenerated with `--patch r2-argument-order`. Follow the update rules and log in `tests/golden/README.md` (code-caused mismatch -> fix the code; dependency-caused -> diagnose with `--cases-dir <scratch>` and update only with maintainer approval).
- Seeds in `tests/test_estimator.py` / `tests/test_runtime.py` were measured so that fits converge in a known number of iterations with the default metric (r2); behavior changes to search/scoring require re-measuring them.
- `tests/_golden_cases.py` is imported as a top-level module (no `tests/__init__.py`; default prepend import mode).
- The detector imports private modules as modules (`from . import _runtime, _submodels`) because tests monkeypatch `_runtime.make_progress_bar` etc. on the detector module; keep that style.
- Optuna's default handler is bound to the stderr captured at import, so tests check Optuna output through the `optuna_stderr` fixture in `tests/test_runtime.py`, not capsys.
- pytest turns `DeprecationWarning`/`FutureWarning` raised from this package into errors (pyproject `filterwarnings`).

## Tooling, CI and release

- ruff (`ruff.toml`: line length 79, D + I rules, numpy docstrings) also formats Python snippets inside Markdown files; `.kiro`, `.claude` and `example` are excluded.
- CI (`.github/workflows/CI.yml`) runs only on pull requests and pushes to main: test matrix 3.10-3.14 (10 min timeout), build (wheel must not contain `example/` or `tests/`, must contain `py.typed`), lint (ruff + mypy).
- Releases: push a tag `vX.Y.Z` (rc as `vX.Y.Z-rc.N`; `vX.Y.ZrcN` does not trigger). `release.yml` builds, signs with sigstore, creates a GitHub Release, and publishes via trusted publishing (stable/rc -> PyPI, all tags -> TestPyPI). See `HOW_TO_RELEASE.md`; date the `Unreleased` CHANGELOG heading before tagging.

## Packaging

- `pyproject.toml` is the single source of build config and metadata (setuptools + setuptools-scm `version_file` -> `_version.py`, git-ignored); there is no `setup.py` or `requirements.txt`.
- `__init__.py` keeps only `__version__` (fallback `"unknown"`), `__license__`, `__author__`, `__copyright__`; license is Apache-2.0 (do not copy MIT notices from python-template).
- `MANIFEST.in` keeps the notebook and dev directories out of the sdist but includes `example/example_chache.csv` (the golden reference data; its SHA-256 is recorded in every case file, and it must be read with `comment="#"`).

# Agentic SDLC and Spec-Driven Development

Kiro-style Spec-Driven Development on an agentic SDLC

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `/kiro-spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in Japanese. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `/kiro-steering`, `/kiro-steering-custom`
- Discovery: `/kiro-discovery "idea"` — determines action path, writes brief.md + roadmap.md for multi-spec projects
- Phase 1 (Specification):
  - Single spec: `/kiro-spec-quick {feature} [--auto]` or step by step:
    - `/kiro-spec-init "description"`
    - `/kiro-spec-requirements {feature}`
    - `/kiro-validate-gap {feature}` (optional: for existing codebase)
    - `/kiro-spec-design {feature} [-y]`
    - `/kiro-validate-design {feature}` (optional: design review)
    - `/kiro-spec-tasks {feature} [-y]`
  - Multi-spec: `/kiro-spec-batch` — creates all specs from roadmap.md in parallel by dependency wave
- Phase 2 (Implementation): `/kiro-impl {feature} [tasks] [--review required|inline|off]`
  - Without task numbers: autonomous mode (subagent per task + independent review + final validation)
  - With task numbers: manual mode (selected tasks in main context, still reviewer-gated before completion)
  - `--review off` skips task-local review; use it intentionally and keep `/kiro-validate-impl {feature}` as the final quality gate
  - `/kiro-validate-impl {feature}` (standalone re-validation)
- Progress check: `/kiro-spec-status {feature}` (use anytime)

## Skills Structure
Skills are located in `.claude/skills/kiro-*/SKILL.md`
- Each skill is a directory with a `SKILL.md` file
- Skills run inline with access to conversation context
- Skills may delegate parallel research to subagents for efficiency
- Additional files (templates, examples) can be added to skill directories
- `kiro-review` — task-local adversarial review protocol used by reviewer subagents
- `kiro-debug` — root-cause-first debug protocol used by debugger subagents
- `kiro-verify-completion` — fresh-evidence gate before success or completion claims
- Use skills explicitly requested by the user and skills relevant to the task's domain, including design, accessibility, and UX.
- Select skills from their descriptions or metadata first, then read only the selected skills and the references needed for the task.
- Follow explicit host and project rules and retain required workflow checks. Do not skip relevant skills just because the task is small.

## Development Rules
- 3-phase approval workflow: Requirements → Design → Tasks → Implementation
- Human review required each phase; use `-y` only for intentional fast-track
- Keep steering current and verify alignment with `/kiro-spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- For spec and implementation work, load the core steering files below from `.kiro/steering/`. Reuse current context rather than rereading unchanged files.
- Load additional steering only when required by project rules or relevant to the task.
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `/kiro-steering-custom`)
