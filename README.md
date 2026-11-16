<div align="center">

# Bagel-Robotics-1

### Data-efficient adaptive pruning for compact world-action models

[Checkpoint](https://huggingface.co/bageldotcom/Bagel-Robotics-1) · [RoboDojo submission](https://github.com/XPolicyLab/XPolicyLab/pull/234) · [Detailed results](experiments/robodojo/policy/Bagel_Robotics_1/EVALUATION_RESULTS.md)

</div>

Bagel-Robotics-1 studies adaptive pruning of a world-action model's video expert. We select informative segments from demonstrations, learn which parameters to retain, physically remove the selected channels, and recover action-prediction quality through further training. VPP2 serves as the base model for evaluating this approach.

**Private release preparation.** The requested training scope is dataset selection, mask learning without K/V-matching loss, physical pruning, and recovery without K/V-matching loss. Recovery begins at checkpoint 55. The technical report is prepared separately.

## Contents

- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Model assets and checkpoints](#model-assets-and-checkpoints)
- [Data preparation](#data-preparation)
- [Training](#training)
- [RoboDojo evaluation](#robodojo-evaluation)
- [Results](#results)
- [Acknowledgements and licenses](#acknowledgements-and-licenses)

## Repository layout

```text
Bagel-Robotics-1/
├── configs/                  Stage configurations and artifact pins
├── scripts/                  Preparation, training and installation commands
├── experiments/robodojo/      Submitted XPolicyLab policy and evaluation records
├── src/                      Release utilities and implementation entry point
├── code/                     Preserved implementations for each training stage
├── data/                     Selected windows, normalization and text conditioning
├── assets/                   Small mask initialization artifacts
├── docs/                     Data, training, evaluation and release instructions
├── tests/                    Packaging and data-contract checks
├── reproduction/             Exact stage sources, configurations and receipts
└── verification/             Recorded checks and release validation results
```

The organization follows [InternW0](https://github.com/InternRobotics/InternW0). Model code, configurations and checkpoint identities remain specific to Bagel-Robotics-1.

## Environment

Use Linux and Python 3.10. Training and inference retain separate environments because their recorded PyTorch versions differ.

| Use | PyTorch / torchvision | CUDA wheels | Instructions |
| --- | --- | --- | --- |
| Mask learning and recovery | 2.10.0 / 0.25.0 | 12.8 | [Training](docs/TRAINING.md) |
| Submitted policy inference | 2.11.0 / 0.26.0 | 13.0 | [Evaluation](docs/EVALUATION.md) |

Basic package checks do not need a GPU:

```bash
python run.py verify-source
python -m unittest discover -s tests -v
python scripts/train.py --help
python scripts/prepare_data.py --help
```

## Model assets and checkpoints

The evaluated checkpoint is public at **`bageldotcom/Bagel-Robotics-1`**, pinned to **`2651631c5d08819088475e8778676f5cf6d9afcf`**. Its six-file bundle contains the video/action weights, compact architecture, normalization and verification records. The adapter verifies sizes and SHA-256 hashes before loading tensors.

```bash
python experiments/robodojo/policy/Bagel_Robotics_1/download_checkpoint.py --help
```

See [checkpoint identity and training provenance](docs/CHECKPOINT_IDENTITY.md). Large model weights, optimizer states and demonstration videos are stored outside Git. The training-input manifest identifies the required resumable checkpoint-55 state; an inference-only checkpoint cannot substitute for it.

## Data preparation

The data workflow retains native 25 Hz RGB observations and absolute dual-arm EE16 actions. The pinned panel contains **30,000 fitting windows**, **1,119 monitoring windows**, and a separate audit partition across 35 training tasks. A window contains consecutive frames; it is not a complete demonstration episode.

```bash
python scripts/prepare_data.py --output /path/to/new/native-data --dry-run
```

[Data instructions](docs/DATA.md) cover the public dataset, exact conversion, selection manifest, split checks and a small conversion test. Data hashes are checked before the reconstructed dataset is marked complete.

## Training

The requested release has four components:

| Component | Implementation |
| --- | --- |
| Select informative training segments | `full_mask/curation_v3*.py` and the pinned selected panel |
| Learn adaptive masks without K/V matching | `code/mask-no-kv/src/vpp2/flexible_mask/` |
| Physically export compact tensors | `flexible_mask/export.py` and `verify_export.py` |
| Recovery without K/V matching, starting at 55 | `code/recovery-pilot/` and `code/recovery/` |

“No K/V matching” means the auxiliary K/V comparison loss is disabled. The action model still consumes video K/V features. Inference preserves that conditioning.

The no-K/V mask-learning experiment and the evaluated checkpoint have separate histories. The source record in [CHECKPOINT_IDENTITY.md](docs/CHECKPOINT_IDENTITY.md) prevents accidentally treating the former as the origin of the latter. See [training instructions](docs/TRAINING.md) for stage commands and required inputs.

## RoboDojo evaluation

Install the preserved adapter into a separate XPolicyLab checkout:

```bash
python scripts/install_adapter.py --xpolicylab /path/to/XPolicyLab --dry-run
python scripts/install_adapter.py --xpolicylab /path/to/XPolicyLab
```

Then use its standard install, checkpoint-download and evaluation scripts. [Evaluation instructions](docs/EVALUATION.md) give the full sequence. The fixed serving settings are: BF16, 32 predicted actions, 16 executed actions, 10 denoising steps, and 90% new / 10% previous prediction blending.

## Results

The evaluated checkpoint's local three-seed, five-suite result is **31.59 ± 0.33% success** and **38.36 ± 0.46 partial score /100**, with equal weighting of the five suites. The complete suite/task tables, original generated summaries and scene replacement appendix remain together in the [Detailed results](experiments/robodojo/policy/Bagel_Robotics_1/EVALUATION_RESULTS.md) document. These scores belong to checkpoint 1000, not to the separate no-K/V mask-learning run.

## Acknowledgements and licenses

We thank the VPP2, Wan, RoboDojo and XPolicyLab authors for their released models, components and evaluation infrastructure. InternW0 is the reference for this repository's presentation. Existing third-party code headers and licenses are preserved. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); checkpoints, datasets and simulator assets retain their own terms.
