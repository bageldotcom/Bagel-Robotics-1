# Demonstrations and selection

Download the public [RoboDojo dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) at the dataset revision and directory in [`configs/artifacts.json`](../configs/artifacts.json). Clone the preprocessing reader at `reader_revision` from [`configs/data.json`](../configs/data.json), then set `data.source` and `data.reader` in the research configuration to those local directories.

The data configuration also pins normalization and selection hashes. Packaged `resources/native_manifest.json.gz` records each episode's expected arrays and text-context identity. Keep these checks when rebuilding the released inputs; text contexts are downloaded and verified during preparation.

For a complete dataset prepared with the earlier release, run `python -m bagel_robotics prepare-data --config configs/research.json --verify-only` to verify the existing arrays and update the local manifest.

## Data contract

| Field | Representation |
| --- | --- |
| Timing | Native 25 Hz |
| RGB | Three-camera composition, 240 × 416, including the BF16 round-trip |
| Actions and state | Absolute dual-arm end-effector targets, 16 values: `[xyz, qwxyz, gripper] × 2` |
| Episodes | 3,500 source identities; 3,466 training and 34 official held-out episodes |
| Task conditioning | 35 released text contexts |

Conversion uses the native arithmetic and verifies episode hashes. Complete decoded RGB arrays require roughly 556 GB, plus labels and reserve space. Start with one episode in a new output directory; a sample conversion is not a complete dataset. Resume only when source identities and existing episode hashes match.

Run conversion in the [inference environment](EVALUATION.md) with PyTorch 2.11 and CUDA 13.0 dependencies. From this repository root, install the additional video and Parquet reader dependencies:

```bash
python -m pip install -e '.[data]'
```

This adds PyAV, PyArrow and pandas, which are required by the conversion code and pinned preprocessing reader. Conversion itself uses CPU. Mask learning and recovery use the separate training environment.

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

## Optional: regenerate a selection

The lower-level selector stages remain available for research. Run them in the [training environment](REPRODUCE.md#environment), which includes their SciPy and vision dependencies. These stages do not change the default `select-data` panel. On a shared GPU system, use an allocation and preserve its device visibility.

The recipes retain the original frozen ConvNeXt-Tiny feature settings. Its weights are a separate public input: [download](https://download.pytorch.org/models/convnext_tiny-983f1562.pth). The full configured checksum is required; the feature stage verifies it before loading.

From the repository root, replace the two example input paths below. This creates a raw method configuration for the selector modules, rather than a CLI research configuration. It supplies a new output directory, the released episode roles and task-suite mapping, and a local feature-weight path:

```bash
python - <<'PY'
import gzip
import json
from pathlib import Path
import urllib.request
from bagel_robotics.configuration import resource
from bagel_robotics.data_selection.manifest import file_hash

run = Path("local/reselection").resolve()
run.mkdir(parents=True, exist_ok=True)
cfg = json.loads(Path("configs/recipes/mask_learning.json").read_text())
cfg["demonstrations"] = "/absolute/path/to/prepared-native-demonstrations"
cfg["teacher_bundle"] = "/absolute/path/to/base-model"
panel = run / "input-panel.json"
panel.write_bytes(gzip.decompress(resource("selected_panel.json.gz").read_bytes()))
cfg["panel_file"] = str(panel)
cfg["curation"]["output"] = str(run / "artifacts")
weights = run / "convnext_tiny-983f1562.pth"
if not weights.exists():
    urllib.request.urlretrieve("https://download.pytorch.org/models/convnext_tiny-983f1562.pth", weights)
if file_hash(weights) != cfg["curation"]["vision"]["sha256"]:
    raise ValueError("Feature-weight checksum mismatch")
cfg["curation"]["vision"]["weights"] = str(weights)
(run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY

python -m bagel_robotics.data_selection.curation --config local/reselection/config.json --stage inventory
python -m bagel_robotics.data_selection.curation --config local/reselection/config.json --stage candidates
python -m bagel_robotics.data_selection.features --config local/reselection/config.json
python -m bagel_robotics.data_selection.quality --config local/reselection/config.json
python -m bagel_robotics.data_selection.curation --config local/reselection/config.json --stage select
```

`panel_file` supplies the published fit/monitor/audit roles and task-suite mapping; it must match the prepared dataset. For a different dataset in the native data contract, set it to `null`, supply `curation.task_suites` for every task, and adjust window budgets to its available capacity. The existing inventory stage then creates a seeded episode split using the configured monitor and audit counts.

Check the generated split and feature/quality receipts with the existing validator:

```bash
python - <<'PY'
import json
from pathlib import Path
from bagel_robotics.data_selection.checks import verify_roles
cfg = json.loads(Path("local/reselection/config.json").read_text())
panel = json.loads((Path(cfg["curation"]["output"]) / "panel.json").read_text())
print(json.dumps(verify_roles(cfg, panel), indent=2))
PY
```

Review `coverage.json`, `near_duplicate_review.jsonl` and `review_requirements.json` in the new artifact directory before using the panel for training. Automatic checks do not replace the required visual review and model diagnostics. A regenerated panel is a new experiment; its checks and results do not transfer from the released panel.
