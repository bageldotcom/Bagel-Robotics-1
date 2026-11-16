# Checkpoint identity

The evaluated model is **Bagel-Robotics-1, recovery update 1000**.

| Item | Immutable identifier |
| --- | --- |
| Hugging Face repository | `bageldotcom/Bagel-Robotics-1` |
| Evaluated revision | `2651631c5d08819088475e8778676f5cf6d9afcf` |
| Training identity | `790bffc9464fa7a026aa43b217aec2033d186895cad643166daa0bcc01914d5c` |
| Video weights SHA-256 | `fb62db0b3a74a53aeab60f4b1acc8a5be47274f5c9db166684043521fe2a40bb` |
| Action weights SHA-256 | `9b04836abe5a30972e69ee84e5f5e39c09bf53dd45865551e1d04b42ead90ec3` |
| Compact video parameters | 12,468,409,664 |

All six exact sizes and hashes are recorded in the adapter's `artifact-manifest.json`. The release check rehashed the original exported bundle, including both large weight files; every file matched. See `verification/release/checkpoint1000-identity.json`.

## Training provenance

The evaluated checkpoint's selected mask came from the earlier mask-learning run with K/V-matching weight **0.05**. Its recovery branch from **55 to 1000** used weight **0**, retaining the compact architecture and frozen action expert. Recovery before 55 is outside this package's requested workflow.

The separate fresh mask-learning run has K/V-matching weight **0** and identity `d337dd2cd1a27c733930a947bb8ca48718d8b19b1b917c37b5d48184e6910f78`. Its configuration explicitly prohibits resuming the previous mask/video state. It must not be described as the source of checkpoint 1000 or assigned checkpoint 1000's evaluation scores.

The released training workflows explicitly disable K/V matching in both mask learning and recovery, as requested. This is the selected training recipe for new runs. Checkpoint 1000 remains the original evaluated artifact with the history above; changing the released recipe does not retroactively change its weights. No public inference code has been replaced.

## Loading and validation

The adapter checks the checkpoint identity, update, compact architecture schema and parameter count before constructing the policy. It then loads the compact tensors and the matching action expert and normalization. It rejects changes to the fixed inference contract rather than silently accepting another recipe.

The stored `validation.json` describes training-time generated-action validation. The simulation results are separate records under `evaluation/`; the simulator scores are not stored in that training diagnostic file. Historical training diagnostics and closed-loop evaluation serve different purposes.

The serving contract remains: predict 32 actions, execute 16, preserve observation history, use 10 Euler steps with shift 1, BF16, and 90/10 temporal blending. Seeds are supplied by the evaluator. The experimental optimized inference backends are outside this release.

The transition was verified against every recorded recovery update: all 945 updates from 56 through 1000 report zero matching weight and zero measured K/V comparisons. See `verification/release/kv-loss-lineage.json`.
