import json
from pathlib import Path
from statistics import mean
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.make_report import summarize, average
ROOT=Path(__file__).resolve().parents[1]

class ResultsTests(unittest.TestCase):
    def test_best_two_match_recorded_selection(self):
        d=json.loads((ROOT/"results/metrics.json").read_text())
        rows=sorted((r for r in d["overall"] if r["phase"]=="recovery"),key=lambda r:r["student_gt_mse"])
        self.assertEqual([r["update"] for r in rows[:2]],[1000,950])
    def test_macro_average_recomputed(self):
        d=json.loads((ROOT/"results/metrics.json").read_text())
        rows=[r for r in d["tasks"] if r["phase"]=="recovery" and r["update"]==1000]
        summary=next(r for r in d["overall"] if r["phase"]=="recovery" and r["update"]==1000)
        self.assertEqual(len(rows),35)
        self.assertAlmostEqual(mean(r["student_gt_mse"] for r in rows),summary["student_gt_mse"],places=15)
        self.assertAlmostEqual(summary["student_gt_mse"],.0005726480308242441,places=15)
    def test_teacher_constant_and_validations_complete(self):
        d=json.loads((ROOT/"results/metrics.json").read_text())
        values=[r for r in d["overall"] if r["phase"]=="recovery"]
        self.assertEqual([r["update"] for r in values],[55]+list(range(100,1001,50)))
        for r in values:self.assertAlmostEqual(r["teacher_gt_mse"],.0002650727891461559,places=15)
    def test_missing_comparison_is_not_invented(self):
        d=json.loads((ROOT/"results/metrics.json").read_text())
        for r in d["overall"]:
            if r["phase"]=="mask_learning":self.assertIsNone(r["student_teacher_mse"])
    def test_duplicate_windows_rejected(self):
        p=ROOT/"results/evidence/recovery_1000/native_001000.json"
        d=json.loads(p.read_text());d["rows"][1]=dict(d["rows"][0])
        with self.assertRaises(ValueError):summarize(d,"recovery","fixture")
    def test_missing_task_rejected(self):
        p=ROOT/"results/evidence/recovery_1000/native_001000.json"
        d=json.loads(p.read_text());d["rows"].pop()
        with self.assertRaises(ValueError):summarize(d,"recovery","fixture")
if __name__=="__main__":unittest.main()
