# Model inputs

The evaluated weights are public and ungated at [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1). The model configuration pins the exact revision and complete file hashes. Keep the video weights, action weights, compact architecture and normalization from the same bundle.

The policy bundle contains `video.pt`, `action.pt`, `compact_architecture.json`, `dataset_stats.json`, `COMPLETE.json` and `validation.json`. Encoder and tokenizer assets are downloaded separately from the configured upstream source. Together the model assets require approximately **45.7 GB**, plus download cache and installation space. No account or access token is required.

Use the installed adapter's download script as described in [EVALUATION.md](EVALUATION.md). It verifies downloaded files; metadata-only checks do not qualify a complete model installation. Existing assets can be reused by configuring their local paths and verifying their hashes.

| Use | Required assets |
| --- | --- |
| Evaluate the released policy | Evaluated policy bundle, encoders and tokenizer |
| Learn masks | Demonstrations, selected panel, base weights, encoders, normalization and mask initialization |
| Recover a compact model | Matching physically compact export, architecture and normalization |
| Continue recovery | Complete matching model/optimizer state, random state and sample position |

The inference bundle contains model weights. Resuming training also requires its complete optimizer and run state; an inference export cannot replace them. [REPRODUCE.md](REPRODUCE.md) describes the training inputs.
