# Models and checkpoints

Use the evaluated policy weights for inference. Training uses additional inputs described below.

## Evaluated policy

| Item | Value |
| --- | --- |
| Model | Bagel-Robotics-1, recovery checkpoint 1000 |
| Hugging Face | [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1) — public and ungated |
| Pinned revision | `2651631c5d08819088475e8778676f5cf6d9afcf` |
| Compact video expert | 12,468,409,664 parameters |
| XPolicyLab checkpoint name | `cp1000` |
| Reproducible file identities | [artifact-manifest.json](../experiments/robodojo/policy/Bagel_Robotics_1/artifact-manifest.json) |

The six-file policy bundle contains `video.pt`, `action.pt`, `compact_architecture.json`, `dataset_stats.json`, `COMPLETE.json` and `validation.json`. Load the matching architecture, action expert and normalization together.

## Download and verify

First complete the [README installation](../README.md#quick-start), which defines `BAGEL_POLICY_DIR` and `BAGEL_POLICY_ENV`:

```bash
PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash "$BAGEL_POLICY_DIR/download_checkpoint.sh"
```

The script downloads the six policy files plus separately pinned Wan encoder/tokenizer assets from `Haodong082399/VPP2` at revision `04824ddb3aae11f13db50984ed7d540c3b77dd9f`. Every file is verified. Allow approximately **45.7 GB**, plus cache and installation space. No access token is required.

Default locations inside the installed policy:

```text
checkpoints/
├── cp1000/                  Compact policy, architecture and normalization
└── Wan2.1-I2V-14B-480P/     Required encoder/tokenizer assets
```

To reuse existing assets, set their paths and verify them before evaluation:

```bash
export BAGEL_CP1000_BUNDLE=/absolute/path/to/update_001000
export BAGEL_CP1000_WAN_ROOT=/absolute/path/to/Wan2.1-I2V-14B-480P
"$BAGEL_POLICY_ENV/bin/python" "$BAGEL_POLICY_DIR/download_checkpoint.py" \
  --verify-only --bundle "$BAGEL_CP1000_BUNDLE" --wan-root "$BAGEL_CP1000_WAN_ROOT"
```

The adapter uses these environment variables when provided. Otherwise, it resolves the default policy checkpoint paths. Keep the pinned revision and checksum checks enabled.

## Training inputs

| Input | Source |
| --- | --- |
| Base video/action weights and Wan assets | Pinned public assets; use [`download_training_assets.py`](../scripts/download_training_assets.py) |
| Mask initialization | Included: [`initial-rankings-no-kv.pt`](../assets/initial-rankings-no-kv.pt) |
| Selected data and normalization | Included manifests; reconstruct demonstrations using the [data guide](DATA.md) |
| Recovery initialization | Full matching checkpoint-55 training state and compact export; see [training inputs](TRAINING.md#inputs) |

The large intermediate recovery states are currently available locally and are not part of the public inference bundle. Replaying the historical recovery branch requires those states; reevaluating the submitted policy does not.

The prepared no-K/V mask-learning recipe is a separate experiment from the earlier mask-learning run that produced checkpoint 1000's compact parent. The exact evaluated checkpoint is preserved. See [checkpoint identity and history](CHECKPOINT_IDENTITY.md) for the recorded lineage and [training](TRAINING.md) for stage commands.
