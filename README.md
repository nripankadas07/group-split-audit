# Group Split Audit

A plausible validation score can hide shared entities, repeated samples or overlapping label windows between training and test sets.

Inspect saved fold IDs and report concrete group, fingerprint, inclusive time-window and class-coverage findings. Optionally generate KFold or GroupKFold splits with scikit-learn.

## Try the example

Python 3.11+ on Linux. No account or remote service is needed.

```sh
git clone https://github.com/nripankadas07/group-split-audit.git
cd group-split-audit
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade "pip>=26.2" "setuptools>=84.1"
python -m pip install .
group-split-audit example.json --html report.html
```

The invented 18-row, six-asset example exposes group overlap in shuffled KFold. Changing generate to groupkfold keeps the supplied asset groups apart.

The JSON goes to stdout; open `report.html` in a browser for searchable evidence
tables. The bundled [HTML report](demo.html) and [expected JSON](example-output.json)
come from [example.json](example.json), which uses invented data.

Exit 0 means a completed report without the tool's documented review findings;
exit 1 means findings require review; exit 2 means invalid input or an operational
error, with stderr and no partial JSON report. Some bundled examples intentionally
exit 1 to demonstrate a failure diagnosis. Do not discard those findings.


To generate a group-aware comparison, change `generate` in the example to `groupkfold`. Both modes use the pinned scikit-learn implementation.

## Input and supported behavior

Each `rows` entry has unique string `id`, nonempty `group` and `label`; optional `fingerprint` is a supplied content identifier. Optional integer `start` and `end` are inclusive and use one declared time axis. Supply `folds` with train/test ID arrays, or `generate` as kfold/groupkfold with n_splits (default 3) and seed (default 0).

## Limits and compatibility

2..20000 rows, 1..20 supplied folds, 4 MiB JSON. Reports only observable declared identifiers/windows. Fingerprints are not computed or verified. All declared windows share a time axis; overlaps across unrelated groups can be legitimate and require review. No model training, score adjustment, embargo/purge implementation or proof that all leakage is absent. Missing training labels are review findings, not automatic evidence of leakage.

scikit-learn 1.9.1, installed from the official Python package registry

Pinned scikit-learn with NumPy/SciPy transitive wheels; heavier install than the standard-library tools. Keep generated fold versions in the report and rerun group/window oracles before upgrading.

## Verify

```sh
python -m unittest -v
python -m compileall -q .
python -m pip wheel --no-deps -w dist .
python -m venv /tmp/group-split-audit-check
/tmp/group-split-audit-check/bin/python -m pip install dist/*.whl
/tmp/group-split-audit-check/bin/python verify_installed.py
```

The installed check executes the real console command from a temporary working
directory and checks malformed-input exit behavior. CI repeats tests and clean
installation on Python 3.11/3.13, then runs a vulnerability audit. See
[validation coverage](VALIDATION.md) for tested behavior and remaining limits.

## Why this project

Existing prompt-eval and climate-evidence-bench assess responses; notebook-state-check concerns execution state. This tool audits tabular train/test membership and declared leakage evidence.

scikit-learn supplies splitters; Cleanlab investigates data/label quality; Deepchecks provides broader validation suites. This report ties declared fold membership to specific overlapping identifiers and windows without fitting a model. It does not replace their broader capabilities.

[Research and alternatives](RESEARCH.md) records dated sources, stars, licenses,
maintenance observations and demand limits. No adoption or performance advantage
is claimed. Original code and invented fixtures, developed with AI assistance.
No affiliation with or endorsement by the upstream project is implied.

[License](LICENSE) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Brief](BRIEF.json)
