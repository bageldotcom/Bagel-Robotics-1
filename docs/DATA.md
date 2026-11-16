# Data preparation and selection

Use the public `RoboDojo-Benchmark/RoboDojo` dataset at revision `cefcfbbf2497103fe46b99039dbd381976fe4a42`, directory `data/RoboDojo_ee_lerobot_v30_video`. The release reuses the original conversion arithmetic and checks its output against the recorded episode hashes.

```bash
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision cefcfbbf2497103fe46b99039dbd381976fe4a42 \
  --include 'data/RoboDojo_ee_lerobot_v30_video/**' --local-dir /path/to/dataset

git clone https://github.com/roboterax/video-prediction-policy-2.git external/data-reader
git -C external/data-reader checkout --detach 72c17acf022494ff63617be86079bb92e92f5bf8
```

The data-reader revision is a preprocessing dependency. It does not replace the adaptive pruning or compact inference implementation. Run conversion in the recorded preprocessing environment (Python 3.10, PyTorch 2.11/CUDA 13.0 dependencies); conversion itself uses CPU.

Start with one episode in a new directory:

```bash
python scripts/prepare_data.py \
  --source /path/to/dataset/data/RoboDojo_ee_lerobot_v30_video \
  --vpp2-root external/data-reader --output /path/to/native-sample \
  --episodes 0 --workers 1 --min-free-gib 10
```

For the complete dataset omit `--episodes`. The complete decoded RGB arrays occupy roughly 556 GB, plus labels and reserve space. The script checks available storage before conversion and never marks a sample as a complete dataset. Completed episodes can be resumed only with matching source identities and checksums.

## Exact data contract

- All 3,500 original episode identities and 25 Hz timing are preserved.
- RGB uses the native three-camera composition at 240 × 416, including the original BF16 round-trip.
- Actions and states are absolute dual-arm EE16: `[xyz, qwxyz, gripper] × 2`.
- The released normalization and 35 text contexts are verified against their original hashes.
- The split retains 3,466 training and 34 official held-out episodes.

`data/native-manifest.json` contains expected episode hashes. `data/selected-panel.json` points to the unchanged panel with 30,000 fitting windows and 1,119 monitoring windows. Fitting, monitoring and audit episodes are disjoint. The panel's original `training_ready` field is historical metadata; training additionally verifies the subsequent visual-review and data-check receipts. It is not silently overwritten.

## Selection method

The preserved selector combines temporal coverage, motion-based candidates, frozen image/history features and episode quality checks, with task balancing and a seeded random component. It retains bounded, non-overlapping action windows per episode. The source records coarse visual review; it does not label every selected frame as a verified contact or success/failure event.

Implementation entry points are `vpp2.full_mask.curation_v3` (inventory, candidates, select) and companion `curation_v3_*` modules. The exact released panel is the reproducible default. Regenerating a different panel creates a new experiment and requires new data/model checks rather than reusing historical approval receipts.

Materialize the exact fitting and monitoring selections:

```bash
python scripts/select_data.py --output /path/to/new-selected-windows --dry-run
python scripts/select_data.py --output /path/to/new-selected-windows
```

This checks the panel identity and split disjointness, and writes readable CSVs, complete JSONL rows and a selection receipt. It does not resample or change the selected windows.
