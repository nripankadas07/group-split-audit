# Validation coverage

Local Linux/Python 3.12 validation includes core unit/integration cases, real
ecosystem APIs where used, runnable invented example, wheel build and clean
installed console-command checks outside the source tree. The original failure
expectations are preserved while implementation defects are corrected.

## Acceptance checks

- Actual scikit-learn GroupKFold output has no train/test group intersection.
- Shuffled KFold exposes the synthetic repeated-asset overlap.
- An interval-counting sweep agrees with a direct pairwise oracle, including touching endpoints.
- Unknown, duplicate or empty fold IDs and reversed windows reject.

CI must pass on the intended public default-branch commit before this project is
counted as a verified launch. Public repository existence alone is insufficient.
Remote CI links and commit receipts belong in the portfolio ledger; this text
does not claim an unobserved run passed.

Windows/macOS, production data and live external service operation are untested.
2..20000 rows, 1..20 supplied folds, 4 MiB JSON. Reports only observable declared identifiers/windows. Fingerprints are not computed or verified. All declared windows share a time axis; overlaps across unrelated groups can be legitimate and require review. No model training, score adjustment, embargo/purge implementation or proof that all leakage is absent. Missing training labels are review findings, not automatic evidence of leakage.

The cloud browser rejects local file URLs, so the generated HTML report was
checked structurally and for escaped output but was not visually inspected in
that browser. Console behavior and installed commands are independently tested.

The initial new-project environment audit flagged its preinstalled pip 25.0.1.
The installer is upgraded to pip 26.2+ before dependency installation and the
full environment audit is rerun. Project runtime dependencies were not the
source of those installer advisories. No advisory is suppressed.
