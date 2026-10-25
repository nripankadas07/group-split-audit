"""Audit externally supplied train/test folds for observable leakage."""

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path


def analyze(data):
    if not isinstance(data, dict) or set(data) - {
        "rows",
        "folds",
        "generate",
        "n_splits",
        "seed",
    }:
        raise ValueError("expected rows and folds, or rows and generate")
    rows = data.get("rows")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 20000:
        raise ValueError("rows must contain 2..20000 observations")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {
            "id",
            "group",
            "label",
            "fingerprint",
            "start",
            "end",
        }:
            raise ValueError("unsupported row fields")
        if any(
            not isinstance(row.get(k), str) or not row[k]
            for k in ("id", "group", "label")
        ):
            raise ValueError("id, group and label must be nonempty strings")
        if "fingerprint" in row and (
            not isinstance(row["fingerprint"], str) or not row["fingerprint"]
        ):
            raise ValueError(
                "fingerprint must be a nonempty supplied content identifier"
            )
        if ("start" in row) != ("end" in row):
            raise ValueError("time windows require both start and end")
        if "start" in row and (
            type(row["start"]) is not int
            or type(row["end"]) is not int
            or row["start"] > row["end"]
        ):
            raise ValueError("time windows must be ordered integer ticks")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("row IDs must be unique")
    generated = None
    if "generate" in data:
        if "folds" in data:
            raise ValueError("choose supplied folds or generation, not both")
        mode = data["generate"]
        n = data.get("n_splits", 3)
        seed = data.get("seed", 0)
        if (
            mode not in ("kfold", "groupkfold")
            or type(n) is not int
            or not 2 <= n <= min(20, len(rows))
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError("invalid splitter, split count or seed")
        import sklearn
        from sklearn.model_selection import GroupKFold, KFold

        splitter = (
            KFold(n_splits=n, shuffle=True, random_state=seed)
            if mode == "kfold"
            else GroupKFold(n_splits=n)
        )
        groups = [r["group"] for r in rows]
        if mode == "groupkfold" and len(set(groups)) < n:
            raise ValueError("not enough groups")
        folds = [
            {"train": [ids[int(i)] for i in tr], "test": [ids[int(i)] for i in te]}
            for tr, te in (
                splitter.split(rows)
                if mode == "kfold"
                else splitter.split(rows, groups=groups)
            )
        ]
        generated = {"splitter": mode, "version": sklearn.__version__, "folds": folds}
    else:
        folds = data.get("folds")
    if not isinstance(folds, list) or not 1 <= len(folds) <= 20:
        raise ValueError("folds must contain 1..20 splits")
    lookup = dict(zip(ids, rows))
    results = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict) or set(fold) != {"train", "test"}:
            raise ValueError("each fold requires train and test")
        for k in ("train", "test"):
            v = fold[k]
            if (
                not isinstance(v, list)
                or not v
                or not all(isinstance(x, str) and x in lookup for x in v)
                or len(set(v)) != len(v)
            ):
                raise ValueError("fold IDs must be nonempty, unique and refer to rows")
        train, test = [[lookup[x] for x in fold[k]] for k in ("train", "test")]
        same = sorted(set(fold["train"]) & set(fold["test"]))
        group_overlap = sorted({r["group"] for r in train} & {r["group"] for r in test})
        duplicate_overlap = sorted(
            {r["fingerprint"] for r in train if "fingerprint" in r}
            & {r["fingerprint"] for r in test if "fingerprint" in r}
        )
        # Interval overlaps are checked on declared common time axes, independently of groups.
        # A sorted sweep counts pairs in O((train+test) log(train)), without pair materialization.
        import bisect

        starts = sorted(r["start"] for r in train if "start" in r)
        ends = sorted(r["end"] for r in train if "end" in r)
        overlaps = sum(
            bisect.bisect_right(starts, r["end"]) - bisect.bisect_left(ends, r["start"])
            for r in test
            if "start" in r
        )
        missing = sorted({r["label"] for r in test} - {r["label"] for r in train})
        findings = bool(
            same or group_overlap or duplicate_overlap or overlaps or missing
        )
        results.append(
            {
                "fold": index,
                "train_rows": len(train),
                "test_rows": len(test),
                "same_row_ids": same,
                "shared_groups": group_overlap,
                "shared_fingerprints": duplicate_overlap,
                "overlapping_window_pairs": overlaps,
                "test_labels_absent_from_train": missing,
                "train_labels": dict(
                    sorted(Counter(r["label"] for r in train).items())
                ),
                "test_labels": dict(sorted(Counter(r["label"] for r in test).items())),
                "needs_review": findings,
            }
        )
    return {
        "schema": "group-split-audit/v1",
        "rows": len(rows),
        "folds": results,
        "generated": generated,
        "needs_review": any(r["needs_review"] for r in results),
        "limits": "Observable identifiers and declared inclusive time windows only; no model training or proof that all leakage is absent.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--html")
    args = p.parse_args(argv)
    try:
        result = analyze(load(args.input))
        if args.html:
            write_html(result, args.html)
        print(json.dumps(result, sort_keys=True))
        return int(result["needs_review"])
    except (ValueError, OSError, TypeError) as e:
        print(str(e), file=sys.stderr)
        return 2


def load(path):
    p = Path(path)
    if p.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("input exceeds 4 MiB")

    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate JSON key: " + k)
            d[k] = v
        return d

    return json.loads(
        p.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite number")),
    )


def write_html(report, path):
    """Render escaped, searchable evidence tables without external assets."""
    esc = lambda x: html.escape(str(x))

    def table(headers, rows):
        return (
            "<table><thead><tr>"
            + "".join("<th>" + esc(x) + "</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join("<td>" + esc(x) + "</td>" for x in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    schema = report["schema"]
    sections = []
    cards = []
    if schema == "join-fanout/v1":
        title = "Where the extra join rows come from"
        subtitle = "Polars equi-join rehearsal. Group counts explain row growth before an expanded join is materialized."
        cards = [
            ("Left rows", report["left_rows"]),
            ("Right rows", report["right_rows"]),
            ("Output rows", report["output_rows"]),
            ("Row budget", report["max_rows"]),
        ]
        sections.append(
            "<h2>Keys contributing the most rows</h2>"
            + table(
                ["Key", "Left", "Right", "Matched pairs", "Output"],
                [
                    (
                        json.dumps(x["key"]),
                        x["left_rows"],
                        x["right_rows"],
                        x["matched_pairs"],
                        x["output_rows"],
                    )
                    for x in report["top_keys"]
                ],
            )
        )
        state = "Over row budget" if report["over_budget"] else "Within row budget"
    elif schema == "group-split-audit/v1":
        title = "Cross-validation split review"
        subtitle = "Review observable group, duplicate and time-window overlap before interpreting a validation score."
        cards = [
            ("Observations", report["rows"]),
            ("Folds", len(report["folds"])),
            ("Folds to review", sum(f["needs_review"] for f in report["folds"])),
        ]
        sections.append(
            "<h2>Fold findings</h2>"
            + table(
                [
                    "Fold",
                    "Train / test",
                    "Shared groups",
                    "Shared fingerprints",
                    "Window pairs",
                    "Unseen labels",
                ],
                [
                    (
                        f["fold"],
                        str(f["train_rows"]) + " / " + str(f["test_rows"]),
                        ", ".join(f["shared_groups"]) or "None",
                        ", ".join(f["shared_fingerprints"]) or "None",
                        f["overlapping_window_pairs"],
                        ", ".join(f["test_labels_absent_from_train"]) or "None",
                    )
                    for f in report["folds"]
                ],
            )
        )
        state = (
            "Review findings" if report["needs_review"] else "No declared overlap found"
        )
    elif schema == "compose-readiness-replay/v1":
        title = "What blocked startup"
        subtitle = "One ordered startup attempt, replayed against declared Compose dependency barriers."
        cards = [
            ("Events", len(report["timeline"])),
            ("Early starts", len(report["violations"])),
            ("Not started", len(report["not_started"])),
        ]
        sections.append(
            "<h2>Starts before their barriers were met</h2>"
            + table(
                ["Tick", "Service", "Waiting for"],
                [
                    (x["at"], x["service"], json.dumps(x["blocked_by"]))
                    for x in report["violations"]
                ],
            )
        )
        sections.append(
            "<h2>Observed event timeline</h2>"
            + table(
                ["Tick", "Service", "Event", "Ready to start after event"],
                [
                    (
                        x["at"],
                        x["service"],
                        x["event"],
                        ", ".join(x["ready_to_start"]) or "None",
                    )
                    for x in report["timeline"]
                ],
            )
        )
        state = (
            "Review findings"
            if report["needs_review"]
            else "Observed barriers respected"
        )
    elif schema == "n8n-relocation/v1":
        title = "Workflow relocation review"
        subtitle = "Local export reference mappings only. No server access, credential payloads or workflow activation."
        cards = [
            ("Planned mappings", len(report["operations"])),
            ("Unresolved references", len(report["unresolved"])),
        ]
        sections.append(
            "<h2>Reference changes</h2>"
            + table(
                ["Node", "Reference", "Before", "After"],
                [
                    (
                        x["node_id"],
                        x["kind"],
                        json.dumps(x["before"]),
                        json.dumps(x["after"]),
                    )
                    for x in report["operations"]
                ],
            )
        )
        sections.append(
            "<h2>Unresolved references</h2>"
            + table(
                ["Node", "Kind", "Reason"],
                [(x["node_id"], x["kind"], x["reason"]) for x in report["unresolved"]],
            )
        )
        state = (
            "Ready for review and local application"
            if report["ready"]
            else "Mapping is incomplete"
        )
    else:
        title = "Proxy wiring rehearsal"
        subtitle = "Local ASGI probes compare public redirects, documentation URLs and OpenAPI server prefixes."
        cards = [
            ("Scenarios", len(report["scenarios"])),
            ("Passing scenarios", sum(x["passed"] for x in report["scenarios"])),
        ]
        for scenario in report["scenarios"]:
            sections.append(
                "<h2>"
                + esc(scenario["name"])
                + (" · Pass" if scenario["passed"] else " · Review")
                + "</h2>"
                + table(
                    ["Public path", "Status", "Redirect / schema URL"],
                    [
                        (
                            x["path"],
                            x["status"],
                            x.get("location", x.get("docs_schema_url", "")),
                        )
                        for x in scenario["responses"]
                    ],
                )
            )
            sections.append(
                table(
                    ["Finding", "Path", "Evidence"],
                    [
                        (x["kind"], x["path"], json.dumps(x))
                        for x in scenario["findings"]
                    ],
                )
            )
        state = (
            "Review scenario differences"
            if report["needs_review"]
            else "All probe contracts passed"
        )
    body = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    css = "body{font:16px/1.6 system-ui;background:#f3f6fa;color:#182438;max-width:1100px;margin:40px auto;padding:0 24px}header{border-bottom:3px solid #137a68;padding-bottom:24px}h1{font-size:2.25rem;line-height:1.15;max-width:800px}h2{margin-top:40px;font-size:1.2rem}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.card{background:white;padding:16px 22px;border:1px solid #d2dae5;border-radius:10px;min-width:140px}.card b{display:block;font-size:1.7rem;color:#0b6555}table{border-collapse:collapse;width:100%;background:white;margin-bottom:22px;display:block;overflow:auto}th,td{text-align:left;padding:12px;border-bottom:1px solid #d6deea;vertical-align:top}th{background:#e3eaf2}td{overflow-wrap:anywhere}input{font:inherit;padding:10px;width:min(90%,600px);border:1px solid #8593a6;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #d2dae5}.status{font-weight:700;color:#0b6555}footer{margin-top:40px;color:#4a5b73}details{margin-top:30px}label{display:block}"
    card_html = (
        '<div class="cards">'
        + "".join(
            '<div class="card">' + esc(k) + "<b>" + esc(v) + "</b></div>"
            for k, v in cards
        )
        + "</div>"
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + esc(title)
        + "</title><style>"
        + css
        + "</style><header><p>LOCAL DIAGNOSTICS · SYNTHETIC EXAMPLE WHEN USING THE BUNDLED FIXTURE</p><h1>"
        + esc(title)
        + "</h1><p>"
        + esc(subtitle)
        + '</p><p class="status">'
        + esc(state)
        + "</p></header>"
        + card_html
        + '<label>Filter evidence rows <input id="q" type="search" placeholder="Service, key, group, path or finding"></label>'
        + "".join(sections)
        + "<details><summary>Complete JSON evidence</summary><pre>"
        + body
        + '</pre></details><footer>Read the repository limitations before acting. These results describe supplied inputs; they do not certify a live system.</footer><script>document.querySelector("#q").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});</script></html>'
    )
    Path(path).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
