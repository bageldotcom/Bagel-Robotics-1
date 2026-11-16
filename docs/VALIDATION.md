# Release validation

Checks completed on November 13, 2026:

| Check | Result | Evidence |
| --- | --- | --- |
| Checkpoint 1000, all six local artifact hashes | Passed, including both large tensor files | `verification/release/checkpoint1000-identity.json` |
| Actual video tensor shapes against compact loader | 1,303 tensors match; 12,468,409,664 parameters | `verification/release/actual-checkpoint-shapes.json` |
| Original inference and generated result files | Unchanged; only stale replacement wording corrected in `REEVALUATION.md` | `verification/release/inference-copy-comparison.json` |
| No-K/V mask objectives and physical pruning contracts | 172 CPU tests passed | `verification/release/no-kv-cpu-tests.xml` |
| Recovery, resume and export contracts | 215 CPU tests passed | `verification/release/recovery-cpu-tests.xml` |
| Adapter settings, history/reset, temporal blending, real architecture | 15 CPU tests passed | `verification/release/adapter-cpu-tests.xml` |
| Packaging and no-K/V guard checks | 24 tests passed | `verification/release/package-tests.log` |
| Source/evidence integrity | 2,828 recorded files verified | `verification/release/source-check.json` |
| Real public-data conversion | Episode 0: exact frame/action/state hashes | `verification/release/data-sample.json` |
| Released selection | 30,000 fit / 1,119 monitor windows; no split leakage | `verification/release/data-selection.log` |
| Stage input checks | All four pass against existing local assets | `verification/release/doctor-*.json` |
| K/V loss removal in checkpoint1000 recovery | All 945 updates 56–1000 report zero weight/comparisons | `verification/release/kv-loss-lineage.json` |

The wrapper changes have not rerun full GPU training or all evaluation episodes. Original GPU integration and training checks remain preserved evidence. The full public training replay still needs distribution of the checkpoint-55 initialization assets and qualification of the packaged training environment. See [RELEASE.md](RELEASE.md).
