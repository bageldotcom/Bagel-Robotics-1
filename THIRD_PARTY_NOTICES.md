# Third-party notices

Bagel-Robotics-1's contribution is data selection, adaptive pruning, compact export and recovery training. VPP2 is the base model used to evaluate this approach. Reused implementation components retain their original attribution and license notices; their historical Python namespace remains `vpp2` to preserve source and checkpoint compatibility.

- **VPP2 / FastWAM:** preserved source headers and `LICENSES/VPP2-upstream-MIT.txt`.
- **Wan and related encoder components:** the policy's `licenses/` directory and notices, plus the model repository's asset licenses.
- **RoboDojo and XPolicyLab:** external benchmark/integration dependencies with their own licenses.
- **InternW0:** repository organization and documentation reference; no InternW0 model architecture or checkpoint is included.

Dataset media, pretrained weights and simulator assets retain their respective terms. The original small mask-ranking initialization and generated text contexts are included with hashes; large model and optimizer tensors remain external.
