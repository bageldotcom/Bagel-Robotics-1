"""Download the configured, hash-pinned policy and encoder assets."""
from pathlib import Path

from bagel_robotics.inference.download import main


if __name__ == "__main__":
    main(policy_dir=Path(__file__).resolve().parent)
