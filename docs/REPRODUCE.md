# Reproduce the pipeline

Run commands from the repository root. Recipes keep the data, architecture, objectives and schedules together. Start with the supplied configuration and set local input/output paths in a copy.

## Environment

Use Linux and Python 3.10. Training uses PyTorch 2.10.0, torchvision 0.25.0 and CUDA 12.8 wheels:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
python -m pip install -e .
python -m pip check
```

Commands use the active training environment by default (`execution.engine: native`). Set `execution.python` to your training environment's Python executable when using a separate CLI environment. For Apptainer execution, set `execution.engine` to `apptainer` and `execution.image` to the image path; the recipes and allocated GPU requirements stay the same.

Use a separate inference environment as described in [EVALUATION.md](EVALUATION.md). The training recipe uses eight GPUs with 80 GB-class memory, global batch 256, substantial host RAM and shared storage. Full restart states require roughly 1.2 TiB of free storage. Preserve the configured effective batch and schedules when reproducing a result.

## Required inputs

| Input | Availability |
| --- | --- |
| Demonstrations | Public RoboDojo dataset; [DATA.md](DATA.md) explains conversion |
| Selection and normalization | Small released manifests and data files |
| Base video/action weights and encoders | [Pinned files, revisions and checksums](../src/bagel_robotics/resources/artifacts.json) |
| Mask ranking initialization | Included small tensor input, pinned by configuration |
| Recovery initialization | Matching compact model and normalization |
| Resume state, if resuming | Complete model/optimizer state, random state and sample position |
| Evaluated inference weights | Public bundle; [MODELS.md](MODELS.md) |

The public inference download contains policy weights. To resume training, supply a complete matching training state; a policy-only export cannot restore optimizer state. Starting recovery from a new compact export initializes a new optimization run.

Download and verify the public base model and encoders:

```bash
python scripts/download_training_assets.py --output local/model-assets --dry-run
python scripts/download_training_assets.py --output local/model-assets
```

Set `paths.base_model` and `paths.encoders` to the downloaded directories identified in the model configuration. Dataset preparation is described in [DATA.md](DATA.md).

## Configure and check

Set `workspace` and the local paths in [`configs/research.json`](../configs/research.json). These identify demonstrations, base weights, encoders and compact/recovery inputs. The mask-learning and recovery recipes live under `configs/recipes/`.

```bash
python -m bagel_robotics init --config configs/research.json
python -m bagel_robotics doctor --config configs/research.json --stage learn-masks
```

Input checks identify missing files before training. Run a GPU smoke check in the environment that will execute the model.

## Run stages

Prepare demonstrations and materialize the default selection:

```bash
python -m bagel_robotics prepare-data --config configs/research.json --dry-run
python -m bagel_robotics select-data --config configs/research.json --dry-run
```

Inspect the resolved inputs and outputs, then remove `--dry-run`. Start conversion with one episode before converting the full dataset. [DATA.md](DATA.md) describes storage and format checks. `select-data` materializes the released panel by default; resampling it changes the experiment.

For mask learning, inspect the launch and run its GPU verification before training:

```bash
python -m bagel_robotics learn-masks --config configs/research.json --action prepare --dry-run
python -m bagel_robotics learn-masks --config configs/research.json --action verify --dry-run
python -m bagel_robotics learn-masks --config configs/research.json --action train --dry-run
```

Execute the verification command without `--dry-run` in the allocated environment, then train. Mask identities and trained video weights must remain paired through physical export:

```bash
python -m bagel_robotics prune --config configs/research.json --action select --dry-run
python -m bagel_robotics prune --config configs/research.json --action export --dry-run
python -m bagel_robotics prune --config configs/research.json --action verify --dry-run
```

Use a separate workspace for recovery. Copy the research configuration, set its `workspace` to a new directory and `paths.compact_model` to the verified export or supported compact policy bundle:

```bash
cp configs/research.json configs/recovery.local.json
```

After setting those paths in the copy, initialize and check it:

```bash
python -m bagel_robotics init --config configs/recovery.local.json
python -m bagel_robotics doctor --config configs/recovery.local.json --stage recover
python -m bagel_robotics recover --config configs/recovery.local.json --action verify --dry-run
python -m bagel_robotics recover --config configs/recovery.local.json --action train --dry-run
```

When resuming a run, set `paths.resume` to its complete training state and use the matching recipe. Resume restores optimizer state, random state and sample position together.

Use a GPU smoke check before a full training run. Keep source revision, configuration, runtime versions, data manifest, seed and output directory with the run. New configurations and model inputs require their own verification.

## Evaluate and interpret

Follow [EVALUATION.md](EVALUATION.md) to run the compact policy. Keep model settings, task coverage, seeds and simulator revision matched when comparing models. Offline generated-action error does not substitute for closed-loop success.

Run the package checks and a short model smoke check before committing resources to full training or benchmark evaluation.
