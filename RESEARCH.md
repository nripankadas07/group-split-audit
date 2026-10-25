# Research and alternatives

Observed on 25 October 2026; repository metadata snapshot at 06:05 UTC, with narrower
comparison searches and README reads later in the same review. Exact receipts
are retained in the portfolio ledger. Searches request GitHub star ordering;
topic/keyword coverage is limited and does not establish a global ranking.

Queries: `machine learning stars:>10000 archived:false sort:stars; data leakage validation sort:stars`.

| Repository | Observed stars | Last push observed UTC | API license classification |
| --- | ---: | --- | --- |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | 67899 | 2026-10-24T21:40:06Z | BSD-3-Clause |
| [cleanlab/cleanlab](https://github.com/cleanlab/cleanlab) | 11720 | 2026-01-13T17:39:04Z | Apache-2.0 |
| [deepchecks/deepchecks](https://github.com/deepchecks/deepchecks) | 4067 | 2025-12-28T12:07:44Z | NOASSERTION |

The highest-star relevant comparable found among these is recorded in the table.
Stars indicate an audience to investigate, not user demand or technical quality.
NOASSERTION/unclassified licenses require separate review before reusing code.
These comparables were read as evidence; no competitor code/prose was reused.

## User and evidence

ML practitioners evaluating repeated observations from the same asset, person or experiment. A plausible validation score can hide shared entities, repeated samples or overlapping label windows between training and test sets.

- https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
- https://github.com/scikit-learn/scikit-learn/issues/32076
- https://github.com/scikit-learn/scikit-learn/issues/13621

Verified historical concern about grouped evaluation in upstream discussions; the cited issues are closed and existing GroupKFold/StratifiedGroupKFold are acknowledged. Demand for this particular file-based audit is inferred.

## Workflow comparison

scikit-learn supplies splitters; Cleanlab investigates data/label quality; Deepchecks provides broader validation suites. This report ties declared fold membership to specific overlapping identifiers and windows without fitting a model. It does not replace their broader capabilities.

Our first useful result is the bundled install-and-report example. The other
projects have their own setup, examples and support channels; their installation
times were not benchmarked here. No speed, memory, reliability or feature-absence
superiority claim follows from our successful example. Their broader workflows
may be more appropriate for larger or different tasks.

## Scope and portfolio comparison

Existing prompt-eval and climate-evidence-bench assess responses; notebook-state-check concerns execution state. This tool audits tabular train/test membership and declared leakage evidence.

The five selected projects address separate tasks: join cardinality, fold leakage,
ASGI URL wiring, container startup history and workflow export relocation. They
are not forks or subdivisions of one application. A portfolio-wide inventory of
218 repositories was compared before selection. Reusable packaging/report
boilerplate does not count as the original capability.

Pinned scikit-learn with NumPy/SciPy transitive wheels; heavier install than the standard-library tools. Keep generated fold versions in the report and rerun group/window oracles before upgrading.

Discovery path: A repeat-observation asset example, KFold versus GroupKFold evidence and searchable split-audit documentation.

Candidate rank 2 of 5 reflects practical usefulness, problem evidence,
distinctness, demonstrability and maintenance burden. It is a judgment, not a
measured score or a prediction of attention. Playwright/browser-agent ideas were
set aside because the existing portfolio already has tracing/replay tools.
Generic top-star lists/tutorial collections were not selected as product anchors.
Polars is used for a Polars-native workflow even though pandas has more stars;
scikit-learn fits the selected evaluation task rather than larger general ML repos.
