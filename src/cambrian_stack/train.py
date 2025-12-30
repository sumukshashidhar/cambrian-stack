"""Main training entry point.

Usage:
    python -m cambrian_stack.train                           # Use default config
    python -m cambrian_stack.train --config-name=my_config   # Use custom config
    python -m cambrian_stack.train model.depth=6             # Override params
"""
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()

from cambrian_stack.training import train


@hydra.main(version_base=None, config_path="conf", config_name="baseline_transformer")
def main(cfg: DictConfig) -> None:
    """Entry point for training."""
    train(cfg)


if __name__ == "__main__":
    main()
