"""Paired official RoboDojo aggregation and zero-margin retention checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import write_json
from .native.official_metrics import DIMENSIONS


def expected_entries():
    return {task:25 if dimension == "Generalization" else 50
            for dimension, tasks in DIMENSIONS.items() for task in tasks} | {
                task+"_random":25 for task in DIMENSIONS["Generalization"]}


def load_run(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"]:
        raise ValueError("An evaluation protocol fingerprint is required")
    if set(manifest["results"]) != set(expected_entries()):
        raise ValueError("Evaluation must contain the exact official 54-entry inventory")
    result = {}
    for task, count in expected_entries().items():
        target = Path(manifest["results"][task])
        if not target.is_absolute():
            target = path.parent / target
        payload = json.loads(target.read_text())
        details = payload["details"]
        rows = list(details.values())
        if len(rows) != count or any("layout_id" not in row or "score" not in row or "success" not in row for row in rows):
            raise ValueError(f"Incomplete result for {task}")
        ids = [str(row["layout_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate layouts for {task}")
        records = {}
        for key, row in zip(ids, rows):
            if not isinstance(row["success"], (bool,int)) or row["success"] not in (0,1):
                raise ValueError("Success must be a boolean or binary integer")
            score = float(row["score"])
            if not np.isfinite(score):
                raise ValueError("Non-finite official episode score")
            records[key] = (score, float(row["success"]))
        result[task] = records
    return manifest["protocol"], result


def compare(teacher, student, *, draws=10000, seed=100008):
    if set(teacher) != set(expected_entries()) or set(student) != set(teacher):
        raise ValueError("Incomplete official inventory")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((draws,2), dtype=np.float64)
    overall_teacher = np.zeros(2)
    overall_student = np.zeros(2)
    cells, dimensions = {}, {}
    for dimension, tasks in DIMENSIONS.items():
        dimension_teacher, dimension_student = [], []
        for task in tasks:
            entry_names = [task] + ([task+"_random"] if dimension == "Generalization" else [])
            all_teacher, all_student = [], []
            for entry in entry_names:
                expected = expected_entries()[entry]
                if set(teacher[entry]) != set(student[entry]) or len(teacher[entry]) != expected:
                    raise ValueError(f"Unpaired/incomplete layout identities: {entry}")
                ids = sorted(teacher[entry])
                t = np.array([teacher[entry][key] for key in ids], dtype=np.float64)
                s = np.array([student[entry][key] for key in ids], dtype=np.float64)
                if not np.isfinite(t).all() or not np.isfinite(s).all():
                    raise ValueError("Non-finite paired metrics")
                delta = s-t
                # Each cell has 50 episodes; five dimensions have equal weight.
                weight = expected / (50*len(DIMENSIONS)*len(tasks))
                indices = rng.integers(expected, size=(draws,expected))
                bootstrap += 100*weight*delta[indices].mean(axis=1)
                overall_teacher += 100*weight*t.mean(axis=0)
                overall_student += 100*weight*s.mean(axis=0)
                all_teacher.extend(t)
                all_student.extend(s)
            t, s = 100*np.mean(all_teacher,axis=0), 100*np.mean(all_student,axis=0)
            cells[task] = {"teacher":t.tolist(), "student":s.tolist(), "difference":(s-t).tolist()}
            dimension_teacher.append(t)
            dimension_student.append(s)
        t, s = np.mean(dimension_teacher,axis=0), np.mean(dimension_student,axis=0)
        dimensions[dimension] = {"teacher":t.tolist(), "student":s.tolist(), "difference":(s-t).tolist()}
    lower = np.quantile(bootstrap, .05, axis=0)
    return {"metric_order":["official_score","official_success_rate"], "units":"percentage points",
            "teacher":overall_teacher.tolist(), "student":overall_student.tolist(),
            "difference":(overall_student-overall_teacher).tolist(),
            "one_sided_95_lower":lower.tolist(), "two_sided_95_interval":np.quantile(bootstrap,[.025,.975],axis=0).tolist(),
            "aggregate_retention_established":bool((lower >= 0).all()), "margin":0,
            "bootstrap_draws":draws, "seed":seed, "episodes":2100, "cells":cells, "dimensions":dimensions,
            "limitation":"Aggregate evidence under this evaluation protocol; not a guarantee for every task or unseen condition."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="JSON protocol + mapping of all 54 entries to exact _result.json paths")
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tp, teacher = load_run(args.teacher)
    sp, student = load_run(args.student)
    if tp != sp:
        raise ValueError("Teacher and student evaluation protocols differ")
    result = compare(teacher, student)
    write_json(args.output, {**result, "protocol":tp})
    print(json.dumps({k:v for k,v in result.items() if k not in {"cells","dimensions"}},indent=2))


if __name__ == "__main__":
    main()

