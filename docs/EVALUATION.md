# Evaluate on RoboDojo

Evaluation uses the public policy bundle and a thin XPolicyLab adapter under [`benchmarks/robodojo/`](../benchmarks/robodojo). Training demonstrations and full recovery states are not required for inference.

## Install the adapter and assets

Install the simulator and assets with the [RoboDojo guide](https://robodojo-benchmark.com/doc/usage/install-and-download/). Use a dedicated XPolicyLab checkout alongside the benchmark's `env_cfg/` directory, at the revision specified by the evaluation configuration.

From this repository root:

```bash
python scripts/install_adapter.py --xpolicylab /absolute/path/to/XPolicyLab

BAGEL_POLICY_DIR=/absolute/path/to/XPolicyLab/policy/Bagel_Robotics_1
BAGEL_POLICY_ENV="$BAGEL_POLICY_DIR/.venv"
cd "$BAGEL_POLICY_DIR"
PYTHON_BIN=python3.10 bash install.sh "$BAGEL_POLICY_ENV"
PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash download_checkpoint.sh
```

Run installation from the adapter inside XPolicyLab so its server package is available. Use a separate simulator environment. Inference uses PyTorch 2.11.0, torchvision 0.26.0 and CUDA 13.0 wheels. A reference setup is a 96 GB RTX PRO 6000 Blackwell, 128 GiB host RAM and eight CPUs.

The downloader verifies the configured policy and encoder files. See [MODELS.md](MODELS.md) for storage and artifact requirements.

## Policy settings

| Setting | Value |
| --- | --- |
| Prediction horizon | 32 actions |
| Execute before replanning | 16 actions |
| Denoising | Ten Euler steps, shift one |
| Precision | BF16 |
| Temporal blending | 90% new prediction, 10% previous prediction |
| Observation history | Retained during each episode, reset between episodes |

Return to the repository root and set local adapter, model, encoder and Python-executable paths in [`configs/evaluation.json`](../configs/evaluation.json). The CLI can download the same pinned assets and run both generated-observation transports:

```bash
python -m bagel_robotics evaluate --config configs/evaluation.json --operation download --dry-run
python -m bagel_robotics evaluate --config configs/evaluation.json --operation smoke --dry-run
```

Inspect the resolved commands, then remove `--dry-run`. Run the GPU smoke operation inside the required allocation. It checks model loading and action prediction in both RGB transport modes; it does not measure task success. Use the simulator's native evaluation entrypoint for full episodes and retain native outputs, failures, task identities and seeds.

## Reported results

The [official RoboDojo-Sim leaderboard](https://robodojo-benchmark.com/leaderboard?bench=sim), excluding the Agent track, lists **38.65 partial score /100** and **31.91% success**, placing Bagel-Robotics-1 **3rd overall by partial score** as of November 20, 2026.

Memory achieves **57.21 partial score /100 (2nd)**. Generalization Standard achieves **52.45 partial score /100 (2nd)** and **46.44% success (tied 1st)**. See the [full results table](../README.md#results) and [machine-readable summary](../results/summary.json) for all dimensions and the dated rankings. Refer to the official leaderboard for current standings.

Compare new models using the same inference settings, task coverage, seeds and simulator revision.
