# Bagel-Robotics-1

Bagel-Robotics-1 is Bagel Labs’ open-weight world-action model for bimanual robot manipulation. It predicts movements for both arms and grippers from a task instruction, three camera views and the robot’s current state.

The `bagel-robotics` package provides the Python implementation and command-line tools for data selection, structured pruning, recovery training and evaluation.

## Install

```bash
python -m pip install bagel-robotics
bagel-robotics --help
```

The package requires Python 3.10 or newer. The documented model workflows use Linux and Python 3.10. The model weights and encoder assets are downloaded separately.

- [Inference and RoboDojo-Sim evaluation](https://github.com/bageldotcom/Bagel-Robotics-1/blob/main/docs/EVALUATION.md)
- [Training and reproduction](https://github.com/bageldotcom/Bagel-Robotics-1/blob/main/docs/REPRODUCE.md)
- [Model weights and model card](https://huggingface.co/bageldotcom/Bagel-Robotics-1)
- [Model release and robot demos](https://blog.bagel.com/p/bagel-robotics-1)

The package source is distributed under the MIT and Apache 2.0 licenses. See the [Bagel Labs Research License](https://github.com/bageldotcom/Bagel-Robotics-1/blob/main/WEIGHTS-LICENSE.md) for the model weights.
