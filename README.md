<h1 align="center">Bagel-Robotics-1</h1>

<p align="center"><strong>Data-efficient adaptive pruning for compact world-action models</strong></p>

<p align="center">
  <a href="#quick-start">🚀 Quick start</a> ·
  <a href="#training">🔥 Training</a> ·
  <a href="#evaluation-and-deployment">🦾 Evaluation &amp; deployment</a> ·
  <a href="docs/MODELS.md">📦 Models</a>
</p>

<p align="center">
  <a href="#overview"><b>🌐 Project overview</b></a> ·
  <b>📄 Paper / arXiv</b> (coming soon) ·
  <a href="https://huggingface.co/bageldotcom/Bagel-Robotics-1"><b>🤗 Hugging Face</b></a>
</p>

<p align="center">
  <img src="assets/readme/approach.svg" alt="Bagel-Robotics-1: select informative demonstration segments, learn adaptive masks, physically prune the video expert, and apply recovery training." width="100%">
</p>

<a name="overview"></a>

## ✨ Overview

**Bagel-Robotics-1** combines data selection, adaptive pruning and recovery training to build a compact world-action model. VPP2 serves as the base model for evaluating the approach.

- 🗂️ **Focused training data.** Select informative segments across robot demonstrations, balancing task coverage and data quality.
- ✂️ **Adaptive structured pruning.** Learn which video-expert features, FFN units and cross-attention heads to retain, then export physically smaller tensors. The evaluated video expert has **12.47B parameters — 23.95% fewer** than the base video expert.
- 🔄 **Recovery and action prediction.** Train the compact video expert to recover action quality while keeping the action expert fixed. Inference predicts 32 actions, executes 16 before replanning, and blends successive predictions 90/10.

This repository provides data preparation, mask learning, physical pruning, recovery training and RoboDojo evaluation. The [Models guide](docs/MODELS.md) identifies the evaluated checkpoint and its training history.

<a name="quick-start"></a>

## 🚀 Quick start

### 1. Get the code

Use Linux and Python 3.10. This repository and the evaluated checkpoint are public; no account or access token is required to download them.

```bash
git clone https://github.com/bageldotcom/Bagel-Robotics-1.git
cd Bagel-Robotics-1
python3.10 run.py verify-source
```

The source check runs on CPU. Training and inference use separate environments:

| Workflow | PyTorch / torchvision | CUDA wheels | Setup |
| --- | --- | --- | --- |
| Mask learning and recovery | 2.10.0 / 0.25.0 | 12.8 | [Training guide](docs/TRAINING.md) |
| Policy inference | 2.11.0 / 0.26.0 | 13.0 | Installer below |

### 2. Install the policy and download the model

Prepare the RoboDojo workspace using the [installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/), including `env_cfg/` alongside XPolicyLab. Choose a **new XPolicyLab checkout** in that workspace. Run these commands from the Bagel-Robotics-1 repository root, replacing the example path:

```bash
BAGEL_XPL_ROOT=/absolute/path/to/RoboDojo/XPolicyLab
git clone https://github.com/XPolicyLab/XPolicyLab.git "$BAGEL_XPL_ROOT"
git -C "$BAGEL_XPL_ROOT" checkout --detach 6baccd57ff958570d0db1588370289ee36c8bb70
python3.10 scripts/install_adapter.py --xpolicylab "$BAGEL_XPL_ROOT"

BAGEL_POLICY_DIR="$BAGEL_XPL_ROOT/policy/Bagel_Robotics_1"
BAGEL_POLICY_ENV="$BAGEL_POLICY_DIR/.venv"
PYTHON_BIN=python3.10 bash "$BAGEL_POLICY_DIR/install.sh" "$BAGEL_POLICY_ENV"
PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash "$BAGEL_POLICY_DIR/download_checkpoint.sh"
```

Keep these variables in the same shell. The installer creates the policy environment; the downloader fetches the pinned checkpoint and required encoders and checks their hashes. Allow approximately **45.7 GB** for model assets, plus installation and cache space. No Hugging Face token is required. See [Models](docs/MODELS.md) to reuse existing files.

### 3. Check inference

After downloading, run both observation transports with the real model. These checks use generated observations, without starting the simulator. The qualified setup used a **96 GB RTX PRO 6000 Blackwell**, 128 GiB host RAM and eight CPUs.

```bash
# Dedicated workstation: choose an available GPU.
# Under a scheduler, use the allocated GPU and preserve its visibility settings.
BAGEL_GPU_ID=0
cd "$BAGEL_POLICY_DIR"
for BAGEL_RGB_ENCODED in 0 1; do
  EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED="$BAGEL_RGB_ENCODED" \
    DEBUG_EVAL_EPISODES=1 bash eval.sh \
    RoboDojo stack_blocks cp1000 arx_x5 ee 1 \
    "$BAGEL_GPU_ID" "$BAGEL_GPU_ID" "$BAGEL_POLICY_ENV" "$BAGEL_POLICY_ENV"
done
```

Each check should finish with `[TestEnv] Episode finished` and `[MAIN] eval finished`. Continue with [RoboDojo evaluation](docs/EVALUATION.md) for simulator episodes and complete suites.

<a name="training"></a>

## 🔥 Training

Run preparation commands from this repository root. Follow the [training guide](docs/TRAINING.md) to build the environment, configure an isolated workspace and verify its inputs before GPU training.

| Stage | Entry point / configuration | Guide |
| --- | --- | --- |
| Select demonstration segments | [`scripts/select_data.py`](scripts/select_data.py), [`data/selected-panel.json`](data/selected-panel.json) | [Data preparation](docs/DATA.md) |
| Learn adaptive masks | `mask-no-kv`, [`mask_no_kv.json`](configs/stages/mask_no_kv.json) | [Mask learning](docs/TRAINING.md#2-mask-learning-without-kv-matching) |
| Export the compact model | `select` → `prune` → `verify-export` | [Physical pruning](docs/TRAINING.md#3-physical-pruning) |
| Recover from checkpoint 55 | `recovery-55-to-250` → `recovery-250-to-1000` | [Recovery](docs/TRAINING.md#4-recovery-without-kv-matching-starting-at-55) |

Both exposed training workflows disable the auxiliary K/V-matching loss; video features remain available to the action expert. The separate no-K/V mask-learning experiment and the evaluated checkpoint have distinct histories, recorded in [checkpoint provenance](docs/CHECKPOINT_IDENTITY.md). Recovery initialization requirements and current GPU qualification status are in the training guide.

<a name="evaluation-and-deployment"></a>

## 🦾 Evaluation and deployment

Use the installed policy with RoboDojo's unified XPolicyLab interface:

| Guide | Use |
| --- | --- |
| [RoboDojo setup and settings](docs/EVALUATION.md) | Environment setup, model settings and benchmark revisions |
| [Task, suite and three-seed commands](experiments/robodojo/policy/Bagel_Robotics_1/REEVALUATION.md) | Run native evaluations and generate the official summary |
| [Adapter validation](experiments/robodojo/policy/Bagel_Robotics_1/VALIDATION.md) | Recorded model-loading, action-comparison and simulator checks |

The fixed inference settings are **BF16, 10 denoising steps, 32 predicted / 16 executed actions, and 90% new / 10% previous prediction blending**. Observation history is maintained and reset for each episode. The RoboDojo submission is [PR #234](https://github.com/XPolicyLab/XPolicyLab/pull/234).

<a name="data-and-tools"></a>

## 🗂️ Data and tools

The selected panel contains **30,000 training windows** and **1,119 monitoring windows** across 35 tasks, with separate episode splits. [Data preparation](docs/DATA.md) covers the public dataset, conversion and selection checks.

| Path | Contents |
| --- | --- |
| `scripts/`, `configs/` | Preparation and training commands, stage settings and artifact pins |
| `code/`, `src/` | Stage implementations and package utilities |
| `experiments/robodojo/` | XPolicyLab adapter and evaluation records |
| `data/`, `assets/` | Selection metadata, normalization and small initialization files |
| `reproduction/`, `verification/` | Preserved source snapshots and validation evidence |

<a name="observation-and-action-format"></a>

## 📐 Observation and action format

The RoboDojo pipeline uses native **25 Hz** observations with three-camera RGB composition at **240 × 416**. Actions and robot state use **16 values**: position, quaternion and gripper for each arm, `[xyz, qwxyz, gripper] × 2`, with absolute end-effector targets. Use the supplied normalization and adapter to preserve these conventions.

<a name="results"></a>

## 📊 Results

Checkpoint 1000's local three-seed results are **31.59 ± 0.33% success** and **38.36 ± 0.46 partial score /100**, with equal weighting of five suites. The [Detailed results](experiments/robodojo/policy/Bagel_Robotics_1/EVALUATION_RESULTS.md) contains all suite/task tables, native summaries and the replacement/repair appendix. These scores refer to the evaluated checkpoint.

<a name="citation"></a>

## 📚 Citation

**Paper / arXiv: coming soon.** The technical report and citation will be added when available.

<a name="acknowledgements"></a>

## Acknowledgements

Bagel-Robotics-1 builds on [VPP2](https://github.com/roboterax/video-prediction-policy-2). We thank the VPP2, Wan, RoboDojo and XPolicyLab authors for their contributions.
