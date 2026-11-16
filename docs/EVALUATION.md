# RoboDojo evaluation

Use the existing public checkpoint and the preserved XPolicyLab adapter. No training assets are required for model reevaluation.

## Install

Create a separate XPolicyLab checkout and install this package's adapter into it:

```bash
git clone https://github.com/XPolicyLab/XPolicyLab.git /path/to/XPolicyLab
python scripts/install_adapter.py --xpolicylab /path/to/XPolicyLab
cd /path/to/XPolicyLab/policy/Bagel_Robotics_1
PYTHON_BIN=python3.10 bash install.sh /path/to/bagel-policy-env
PYTHON_BIN=/path/to/bagel-policy-env/bin/python bash download_checkpoint.sh
```

The installer must run from the installed adapter within XPolicyLab. Running it directly inside this research repository would omit XPolicyLab's server package. The downloader uses the pinned public checkpoint and encoder assets and verifies their complete hashes. The simulator is installed in a separate environment following RoboDojo's installation guide.

## Fixed model settings

| Setting | Value |
| --- | --- |
| Predicted action chunk | 32 |
| Actions executed before replanning | 16 |
| Denoising | 10 Euler steps, shift 1 |
| Precision | BF16 |
| Temporal blending | 90% new prediction, 10% previous prediction |
| Observation history | Preserved through the deployment adapter; reset each episode |
| Checkpoint | Recovery update 1000; pinned revision in `artifact-manifest.json` |

Use the same settings for all tasks. K/V matching is a training loss; disabling it does not remove the video K/V features used by the action model.

## Debug, simulation and summaries

The adapter's [README](../experiments/robodojo/policy/Bagel_Robotics_1/README.md) supplies the standard ten-argument `eval.sh` call and both RGB debug modes. [REEVALUATION.md](../experiments/robodojo/policy/Bagel_Robotics_1/REEVALUATION.md) provides native task, suite and three-seed commands, along with summary generation. Use the simulator's own `scripts/robodojo.sh` rather than a similarly named research helper.

## Version provenance

| Purpose | RoboDojo revision |
| --- | --- |
| Historical research episodes | `08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d` plus recorded local runtime patches |
| Generated summary scripts and recorded adapter qualification | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| Committee's stated official evaluation target | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |

The summary-script revision does not change the simulator version used for old episodes. Fresh committee-aligned runs should use the agreed official version and retain native outputs. Preserve any startup failure for review; replacement layouts are never enabled silently.

The unchanged [Detailed results](../experiments/robodojo/policy/Bagel_Robotics_1/EVALUATION_RESULTS.md) contains all three-seed task results and the final appendix for 28 later replacements, one repaired Memory episode and four earlier Seed-1 Precision substitutions.
