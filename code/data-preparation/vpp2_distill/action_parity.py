"""Check the student runtime against one native teacher trajectory before training."""
from __future__ import annotations

import argparse
import json

import torch

from .action import FrozenPolicyCore
from .contracts import file_digest, write_json


@torch.no_grad()
def compare(core, fixture, device):
    if fixture["format"] != "vpp2-runtime-action-parity-v1":
        raise ValueError("Unrecognized runtime parity fixture")
    context = fixture["context"].to(device)
    mask = fixture["mask"].to(device)
    # Include the frozen proprio encoder in this cross-runtime comparison.
    actual_context, actual_mask = core.append_proprio(context[:, :-1], mask[:, :-1], fixture["state"].to(device))
    context_equal = torch.equal(context, actual_context) and torch.equal(mask, actual_mask)
    memory = [{kind:layer[kind].to(device) for kind in ("k", "v")} for layer in fixture["kv"]]
    x = fixture["initial"].to(device)
    times, deltas = core.scheduler.build_inference_schedule(10, x.device, x.dtype, shift_override=1.)
    schedule_equal = torch.equal(times.cpu(), fixture["times"]) and torch.equal(deltas.cpu(), fixture["deltas"])
    records = []
    for i, (t, delta) in enumerate(zip(times, deltas)):
        prediction = core(x, t[None], actual_context, actual_mask, memory)
        expected = fixture["velocities"][i].to(device)
        records.append(dict(step=i, equal=torch.equal(prediction, expected),
                            max_abs=float((prediction.float()-expected.float()).abs().max())))
        x = core.scheduler.step(prediction, delta, x)
    final_equal = torch.equal(x.cpu(), fixture["final"])
    passed = context_equal and schedule_equal and final_equal and len(records)==10 and all(r["equal"] for r in records)
    return dict(passed=passed, context_equal=context_equal, schedule_equal=schedule_equal, final_equal=final_equal,
                checked_steps=len(records), steps=records, teacher_identity=fixture["teacher_identity"],
                teacher_torch=fixture["teacher_torch"], student_torch=str(torch.__version__), device=str(device),
                scope="Identical native teacher K/V, frozen Action2B/proprio, and all ten BF16 trajectory steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if file_digest(args.action) != fixture["teacher_identity"]["action_sha256"]:
        raise ValueError("Parity fixture and student Action2B checkpoint differ")
    core = FrozenPolicyCore.from_checkpoint(args.action, checkpoint=False).to(args.device).eval().requires_grad_(False)
    report = compare(core, fixture, args.device)
    write_json(args.report, report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Student-runtime Action2B parity failed; inspect the report before changing the gate")


if __name__ == "__main__":
    main()
