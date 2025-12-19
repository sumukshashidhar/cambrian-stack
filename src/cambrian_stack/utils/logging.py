"""Logging utilities - wandb and loguru setup."""
import sys
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()  # Load WANDB_API_KEY
from omegaconf import OmegaConf


class DummyWandb:
    """No-op wandb replacement when logging is disabled."""
    
    def log(self, *args, **kwargs) -> None:
        pass
    
    def finish(self) -> None:
        pass


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure loguru for console and file logging.
    
    Args:
        log_dir: Directory for log files (optional)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        colorize=True,
    )
    
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(log_dir) / "train.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
            enqueue=True,
        )


def setup_wandb(cfg, is_master: bool):
    """Initialize wandb or return dummy if disabled.
    
    Args:
        cfg: Config with logging.wandb_project and logging.wandb_run
        is_master: Whether this is the main process
    
    Returns:
        wandb.Run or DummyWandb
    """
    wandb_run = cfg.logging.wandb_run
    if (wandb_run is None) or (str(wandb_run).lower() in {"dummy", "none", "null"}) or (not is_master):
        return DummyWandb()
    
    import wandb
    
    return wandb.init(
        project=cfg.logging.wandb_project,
        name=wandb_run,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
