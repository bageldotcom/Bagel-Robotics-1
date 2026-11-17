# Release status and handoff

The source is published in [bageldotcom/Bagel-Robotics-1](https://github.com/bageldotcom/Bagel-Robotics-1), which is **public**. The evaluated checkpoint is also [public and ungated](https://huggingface.co/bageldotcom/Bagel-Robotics-1). Repository creation, Write access and the initial source upload are complete.

## Published artifacts

- InternW0-style project organization and concise installation/data/training/evaluation guides.
- Explicit no-K/V-matching mask-learning and recovery launch contracts; recovery 0–55 omitted from the exposed workflow.
- Dataset selection source, exact selection metadata, text conditioning, public-data conversion and checksums.
- Physically compact export code, architecture verification and the checkpoint-1000 XPolicyLab adapter.
- Unchanged public checkpoint pins, full hash verification and training/validation provenance.
- Existing three-seed results and generated summaries, with original replacement details retained.
- Recorded source, data, training-contract and adapter checks under `verification/release/`.

## Remaining before declaring the complete training replay publicly available

1. **Recovery initialization distribution:** the exact full checkpoint-55 training state and matching compact export are present locally and verified by the input checks, but are not published as downloadable training assets. They are required to replay recovery starting at 55; the public inference checkpoint is not a substitute. Agree on their release location or supply them directly to a training reviewer. Git should not contain these large optimizer tensors.
2. **New-environment GPU qualification:** historical eight-GPU checks are preserved, and CPU/data/package checks have been rerun. The newly packaged container has not been built and exercised for a complete GPU training/resume/export sequence. The launcher requires its verification actions before training.

The technical report remains planned for **November 20, 2026**. The selected no-K/V mask-learning recipe is documented as a distinct experiment; changing a training recipe cannot change the history of the already evaluated checkpoint.

Evaluation-video publication is deferred. This release contains scores and summaries; rollout videos are not included.

## Reviewer route

For model reevaluation, start at `docs/EVALUATION.md`. Reviewers use the existing public six-file checkpoint bundle and the fixed adapter. They do not need the training-state assets above.

For training, start at `docs/TRAINING.md` and use `run.py doctor` to verify inputs before launching. Never bypass a missing input or identity mismatch. The original source snapshots and manifests are preserved so the exact implementation can be inspected independently of the wrappers.

## Publication procedure

When the remaining training assets and qualification are complete:

1. Review the validation receipts and immutable artifact manifest.
2. Record the reviewed Git commit and create a source release at that commit.
3. Publish the remaining training inputs when authorized; the source and evaluated checkpoint are already public.
4. Keep PR #234 and the current Hugging Face model card aligned with the reviewed code release; retain the evaluated checkpoint revision and hashes.
5. Give the committee one PR link and one detailed-results link; attach the unchanged native summary if requested.

No tool in this package automatically changes repository visibility or sends messages to the committee.
