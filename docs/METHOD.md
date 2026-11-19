# Method and implementation

DEAP connects **data selection**, **adaptive structured pruning**, and **recovery training**. A video expert supplies visual context to an action expert. The method reduces the video expert and trains it to retain useful action-facing features.

## Recipe and backbone

DEAP is designed as a training and compression recipe for world-action models. Bagel-Robotics-1's contributions are the data-selection strategy, learned structured masks and post-pruning recovery procedure.

This implementation uses [VPP2](https://github.com/roboterax/video-prediction-policy-2) as its pretrained base model for implementation and evaluation. Applying the recipe to another WAM requires adapting its observation/action interface, mask locations, tensor-slicing rules and recovery objectives to that architecture. The reported results describe the supplied implementation and released weights.

## 1. Select informative demonstration windows

Selection operates on complete action windows within robot demonstrations. The selector combines:

- task-balanced capacity allocation and episode limits;
- temporal coverage and motion-based candidates;
- frozen image and observation-history features for diversity;
- episode quality estimates and a seeded random component.

Selected action windows do not overlap within an episode. Fit, monitor and audit roles are separated by episode. The released panel contains 30,000 fitting windows and 1,119 monitoring windows across 35 tasks. [DATA.md](DATA.md) explains the format and exact reconstruction.

Visual descriptors capture image/history features; motion and quality estimates guide selection without requiring exhaustive contact or success/failure annotations. Controlled ablations can compare selection criteria with matched data budgets, training and evaluation.

Read [`data_selection/`](../src/bagel_robotics/data_selection) for candidate generation, selection and split checks.

## 2. Learn masks and export a smaller model

The masks allocate capacity across three structures in the video expert:

| Mask family | Scope | Export effect |
| --- | --- | --- |
| Hidden channels | Shared hidden width | Slices all dependent hidden axes consistently |
| FFN units | Per eligible transformer layer | Produces nonuniform FFN widths |
| Cross-attention heads | Per eligible transformer layer | Produces nonuniform cross-attention head counts |

The action expert stays fixed. Self-attention head structure and protected terminal components follow the model geometry contract. Mask scores become continuous gates during learning; the configured schedule sharpens them toward discrete choices. These are learned architecture decisions, held fixed when the exported policy runs.

The training objective combines ground-truth action error, reliability-weighted teacher feature losses, and size, entropy and discreteness penalties. The exact coefficients, active teacher terms and schedules are specified by the selected recipe. Reliability suppresses a teacher contribution when it performs poorly against the student or a zero-action baseline, including checks on action components.

The size objective follows the actual tensor-slicing geometry, including products of hidden dimensions. Binary selection applies deterministic ranking, width floors and packing constraints. Physical export slices the **trained student weights** with those selected identities, writes the compact architecture, and compares the compact model against the corresponding masked model. A gate set on a dense tensor is not a physically compact export.

Read [`adaptive_pruning/`](../src/bagel_robotics/adaptive_pruning) for gates, costs, selection, slicing and export checks; [`models/`](../src/bagel_robotics/models) contains the shared model implementation.

## 3. Recover action quality after pruning

Recovery fixes the exported architecture and optimizes the compact video expert. The action expert remains frozen, while action-loss gradients can pass through it into the video expert. The recipe combines action supervision with configured reliable teacher features. The optimizer, random state and sample position belong to the recovery run and must be restored together when resuming.

The recovery input must match the compact architecture and normalization. Resuming a run restores the complete training state; starting from model weights alone creates a new optimization run.

Read [`recovery/`](../src/bagel_robotics/recovery) for training and restart handling, and [REPRODUCE.md](REPRODUCE.md) for required assets.

## Evaluate the compact policy

The deployed policy uses BF16, ten Euler denoising steps with shift one, a 32-action prediction horizon, and 16 executed actions before replanning. Successive predictions are blended with 90% new and 10% previous prediction. Observation history is reset between episodes.

The selected video expert has 12,468,409,664 parameters, 23.95% fewer than the base video expert. This is a video-expert parameter reduction; it does not establish the same percentage reduction in total policy memory or latency. Offline action error and simulator task success are separate measurements.

[EVALUATION.md](EVALUATION.md) describes the benchmark interface and model settings.
