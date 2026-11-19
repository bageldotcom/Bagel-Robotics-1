# Demonstrations and selection

Download the public [RoboDojo dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) at the dataset revision and directory in [`configs/artifacts.json`](../configs/artifacts.json). Clone the preprocessing reader at `reader_revision` from [`configs/data.json`](../configs/data.json), then set `data.source` and `data.reader` in the research configuration to those local directories.

The data configuration also pins normalization and selection hashes. Packaged `resources/native_manifest.json.gz` records each episode's expected arrays and text-context identity. Keep these checks when rebuilding the released inputs; text contexts are downloaded and verified during preparation.

## Data contract

| Field | Representation |
| --- | --- |
| Timing | Native 25 Hz |
| RGB | Three-camera composition, 240 × 416, including the BF16 round-trip |
| Actions and state | Absolute dual-arm end-effector targets, 16 values: `[xyz, qwxyz, gripper] × 2` |
| Episodes | 3,500 source identities; 3,466 training and 34 official held-out episodes |
| Task conditioning | 35 released text contexts |

Conversion uses the native arithmetic and verifies episode hashes. Complete decoded RGB arrays require roughly 556 GB, plus labels and reserve space. Start with one episode in a new output directory; a sample conversion is not a complete dataset. Resume only when source identities and existing episode hashes match.

Run conversion in the [inference environment](EVALUATION.md) with PyTorch 2.11 and CUDA 13.0 dependencies. Conversion itself uses CPU. Mask learning and recovery use the separate training environment.

Configure `data.source`, the preprocessing reader, output path and worker count in [`configs/research.json`](../configs/research.json). Use `--episodes` to select a sample, then inspect and run:

```bash
python -m bagel_robotics prepare-data --config configs/research.json --episodes 0 --dry-run
python -m bagel_robotics prepare-data --config configs/research.json --episodes 0
```

Omit `--episodes` to convert the full dataset. Set `paths.demonstrations` to the completed `data.output` directory before training.

## Released panel

The default panel contains **30,000 fitting windows and 1,119 monitoring windows across 35 tasks**. Fit, monitor and audit roles are disjoint by episode. The selector combines task balance, temporal coverage, motion, frozen image/history descriptors, episode quality and seeded random selection. Windows are bounded and non-overlapping within each episode.

```bash
python -m bagel_robotics select-data --config configs/research.json --dry-run
python -m bagel_robotics select-data --config configs/research.json
```

The default command validates and materializes the exact released panel as readable window lists and a receipt. It does not rerun the selection algorithm. The implementation under [`data_selection/`](../src/bagel_robotics/data_selection) includes candidate generation, frozen-feature extraction, episode quality and final selection.

Resampling, changing an episode split or changing normalization creates a new experiment with new checks. Hold the selected window budget and training schedule fixed when comparing selection methods.
