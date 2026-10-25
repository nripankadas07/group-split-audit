import unittest

from group_split_audit import analyze


class SplitTest(unittest.TestCase):
    def data(self):
        rows = [
            {
                "id": str(i),
                "group": str(i // 3),
                "label": str(i % 2),
                "fingerprint": str(i),
            }
            for i in range(18)
        ]
        return {"rows": rows, "generate": "groupkfold", "n_splits": 3}

    def test_groupkfold_actually_keeps_entities_apart(self):
        d = self.data()
        r = analyze(d)
        self.assertFalse(r["needs_review"])
        self.assertEqual(r["generated"]["version"], "1.9.1")
        by_id = {x["id"]: x for x in d["rows"]}
        for f in r["generated"]["folds"]:
            self.assertFalse(
                {by_id[x]["group"] for x in f["train"]}
                & {by_id[x]["group"] for x in f["test"]}
            )

    def test_kfold_exposes_group_leakage(self):
        d = self.data()
        d["generate"] = "kfold"
        r = analyze(d)
        self.assertTrue(r["needs_review"])
        self.assertTrue(any(f["shared_groups"] for f in r["folds"]))

    def test_exact_duplicate_across_distinct_groups(self):
        d = self.data()
        d.pop("generate")
        d["rows"][0]["fingerprint"] = "copy"
        d["rows"][3]["fingerprint"] = "copy"
        d["folds"] = [{"train": ["0", "1", "2"], "test": ["3", "4", "5"]}]
        f = analyze(d)["folds"][0]
        self.assertEqual(f["shared_groups"], [])
        self.assertEqual(f["shared_fingerprints"], ["copy"])

    def test_inclusive_interval_oracle(self):
        d = {
            "rows": [
                {"id": str(i), "group": str(i), "label": "ok", "start": i, "end": i + 2}
                for i in range(8)
            ],
            "folds": [{"train": ["0", "2", "4", "6"], "test": ["1", "3", "5", "7"]}],
        }
        tr = d["rows"][::2]
        te = d["rows"][1::2]
        oracle = sum(
            a["start"] <= b["end"] and b["start"] <= a["end"] for a in tr for b in te
        )
        self.assertEqual(analyze(d)["folds"][0]["overlapping_window_pairs"], oracle)

    def test_same_row_and_unseen_label(self):
        d = self.data()
        d.pop("generate")
        d["rows"][3]["label"] = "novel"
        d["folds"] = [{"train": ["0", "1"], "test": ["0", "3"]}]
        f = analyze(d)["folds"][0]
        self.assertEqual(f["same_row_ids"], ["0"])
        self.assertEqual(f["test_labels_absent_from_train"], ["novel"])

    def test_unknown_duplicate_and_empty_indices_reject(self):
        for train in (["unknown"], ["0", "0"], []):
            d = self.data()
            d.pop("generate")
            d["folds"] = [{"train": train, "test": ["3"]}]
            with self.assertRaises(ValueError):
                analyze(d)

    def test_insufficient_groups_and_bad_window(self):
        d = self.data()
        d["n_splits"] = 7
        with self.assertRaises(ValueError):
            analyze(d)
        d = self.data()
        d["rows"][0]["start"] = 2
        d["rows"][0]["end"] = 1
        with self.assertRaises(ValueError):
            analyze(d)


if __name__ == "__main__":
    unittest.main()
